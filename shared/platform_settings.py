"""Operator-chosen platform settings, read live from Redis.

THE CROSS-REPO CONTRACT
    Written by   warptalk-backend's workspace service (PlatformSettingsPublisherWorker), the only
                 writer. Everything here only reads.
    Mirrors      WarpTalk.Shared/PlatformSettings/ — PlatformSettingsRedisKeys.cs,
                 PlatformSettingsReader.cs, SettingValueValidator.cs, FeatureFlagValue.cs and
                 PlatformSettingsCatalog.cs. The semantics below are theirs; a change there must
                 land here on the same day.

    platform:settings:v1:platform            hash, field = setting key, value = JSON text. Always
                                             carries `__version` once published, so "nothing
                                             set" (hash with only the version) is different
                                             from "missing" (evicted, or Redis restarted).
    platform:settings:v1:plan:{slug}         overrides for one plan
    platform:settings:v1:workspace:{uuid}    overrides for one workspace (lower-case D format)

RESOLUTION, FIRST HIT WINS
    workspace override -> plan override -> platform value -> the caller's fallback -> the
    registry default. The fallback is the worker's own env/pydantic setting, so a key nobody has
    set changes NOTHING about how the worker behaved before this module existed. Overrides are
    only consulted for a key whose definition allows that scope.

    A stored value that fails its definition (a bound was tightened, or someone wrote Redis by
    hand) is ignored with a warning, and the fallback applies.

FAILURE IS SOFT, ALWAYS
    Each hash is cached in process for ten seconds on a monotonic clock. On a Redis error the last
    snapshot is kept and the next attempt waits one TTL; a missing platform hash also keeps the
    last snapshot (Redis runs allkeys-lru, and eviction is not a reset). Nothing here raises: a
    settings outage must never stop audio or a meeting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeGuard

from shared.logger import get_logger

logger = get_logger(__name__)

PREFIX = "platform:settings:v1:"
PLATFORM_HASH = PREFIX + "platform"
CHANGED_CHANNEL = PREFIX + "changed"
VERSION_FIELD = "__version"

DEFAULT_CACHE_TTL_SECONDS = 10.0
# How long a room -> workspace lookup is remembered. The projection it reads lives 24h and a room
# never changes workspace, so this only bounds memory and how soon a late-written projection is
# noticed.
_ROOM_WORKSPACE_TTL_SECONDS = 60.0
_ROOM_WORKSPACE_CACHE_MAX = 4096


def plan_hash(plan_slug: str) -> str:
    return PREFIX + "plan:" + plan_slug.strip().lower()


def workspace_hash(workspace_id: uuid.UUID) -> str:
    # str(UUID) is .NET's Guid.ToString("D"): lower-case, hyphenated.
    return PREFIX + "workspace:" + str(workspace_id)


# ---------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------

SettingType = Literal["boolean", "integer", "decimal", "string", "feature_flag"]

SCOPE_PLATFORM = "platform"
SCOPE_PLAN = "plan"
SCOPE_WORKSPACE = "workspace"


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    type: SettingType
    default: Any
    min: float | None = None
    max: float | None = None
    max_length: int | None = None
    scopes: frozenset[str] = frozenset({SCOPE_PLATFORM})

    def allows(self, scope: str) -> bool:
        return scope in self.scopes


# Keys this repo reads. KEYS ARE A CONTRACT: add, never rename — a renamed key is a setting every
# operator's stored value silently stops applying to.
FLASH_MODE_DEFAULT = "meetings.flash_mode_default"
CHUNK_DURATION_MS = "meetings.chunk_duration_ms"
SUGGEST_MIN_WORDS = "meetings.ai_suggest.min_words"
SUGGEST_MIN_CONFIDENCE = "meetings.ai_suggest.min_confidence"
SUGGEST_COOLDOWN_SECONDS = "meetings.ai_suggest.cooldown_seconds"
SUGGEST_MAX_PER_MEETING = "meetings.ai_suggest.max_per_meeting"
SUGGEST_MIN_STT_CONFIDENCE = "meetings.ai_suggest.min_stt_confidence"
VOICE_CLONE_MIN_SECONDS = "meetings.voice_clone.min_sample_seconds"
VOICE_CLONE_UPGRADE_MARGIN = "meetings.voice_clone.upgrade_margin"
FLAG_AI_SUGGEST = "flags.ai_suggest"
FLAG_VOICE_CLONE = "flags.voice_clone"
FLAG_WARPBOT_WEB_SEARCH = "flags.warpbot_web_search"
FLAG_GLOBAL_GLOSSARY = "flags.global_glossary"

_FLAG_ON: Mapping[str, Any] = {"enabled": True, "rolloutPercent": 100}

# Mirrors PlatformSettingsCatalog.cs for the keys the AI workers own (types, bounds, defaults and
# scopes copied exactly). Only what this repo reads is listed; an unknown key reads as "not set".
REGISTRY: dict[str, SettingDefinition] = {
    d.key: d
    for d in (
        SettingDefinition(FLASH_MODE_DEFAULT, "boolean", True),
        SettingDefinition(CHUNK_DURATION_MS, "integer", 6000, min=2000, max=15000),
        SettingDefinition(SUGGEST_MIN_WORDS, "integer", 4, min=1, max=20),
        SettingDefinition(SUGGEST_MIN_CONFIDENCE, "decimal", 0.55, min=0, max=1),
        SettingDefinition(SUGGEST_COOLDOWN_SECONDS, "integer", 20, min=0, max=600),
        SettingDefinition(SUGGEST_MAX_PER_MEETING, "integer", 30, min=0, max=200),
        SettingDefinition(SUGGEST_MIN_STT_CONFIDENCE, "decimal", -0.5, min=-5, max=0),
        SettingDefinition(VOICE_CLONE_MIN_SECONDS, "decimal", 20.0, min=5, max=90),
        SettingDefinition(VOICE_CLONE_UPGRADE_MARGIN, "decimal", 0.15, min=0, max=1),
        SettingDefinition(FLAG_AI_SUGGEST, "feature_flag", dict(_FLAG_ON)),
        SettingDefinition(FLAG_VOICE_CLONE, "feature_flag", dict(_FLAG_ON)),
        SettingDefinition(FLAG_WARPBOT_WEB_SEARCH, "feature_flag", dict(_FLAG_ON)),
        SettingDefinition(FLAG_GLOBAL_GLOSSARY, "feature_flag", dict(_FLAG_ON)),
    )
}


# ---------------------------------------------------------------------------------------------
# Validation — SettingValueValidator.cs
# ---------------------------------------------------------------------------------------------

_MAX_FLAG_LIST_ITEMS = 500
_FLAG_FIELDS = frozenset(
    {"enabled", "rolloutPercent", "allowPlans", "allowWorkspaces", "denyWorkspaces"}
)
_PLAN_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}$")


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(definition: SettingDefinition, value: Any) -> str | None:
    """None when `value` is acceptable for `definition`, otherwise why not."""
    if definition.type == "boolean":
        return None if isinstance(value, bool) else "Expected true or false."
    if definition.type in ("integer", "decimal"):
        integral = definition.type == "integer"
        if not _is_number(value):
            return "Expected a whole number." if integral else "Expected a number."
        if integral and value != math.trunc(value):
            return "Expected a whole number."
        if definition.min is not None and value < definition.min:
            return f"Must be at least {definition.min:g}."
        if definition.max is not None and value > definition.max:
            return f"Must be at most {definition.max:g}."
        return None
    if definition.type == "string":
        if not isinstance(value, str):
            return "Expected text."
        if definition.max_length is not None and len(value) > definition.max_length:
            return f"The value must be at most {definition.max_length} characters."
        if len(value) > 2000:
            return "The value is too long."
        if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
            return "The value contains control characters."
        return None
    if definition.type == "feature_flag":
        return _validate_flag(value)
    return "This setting has an unknown type."


def _validate_flag(value: Any) -> str | None:
    if not isinstance(value, dict):
        return "Expected a feature flag."
    for name in value:
        if name not in _FLAG_FIELDS:
            return f"Unknown flag field '{str(name)[:60]}'."
    if not isinstance(value.get("enabled"), bool):
        return "A flag needs enabled: true or false."
    if "rolloutPercent" in value:
        percent = value["rolloutPercent"]
        if not isinstance(percent, int) or isinstance(percent, bool) or not 0 <= percent <= 100:
            return "Rollout must be a whole percentage from 0 to 100."
    for name in ("allowPlans", "allowWorkspaces", "denyWorkspaces"):
        if name not in value:
            continue
        items = value[name]
        if not isinstance(items, list):
            return f"{name} must be a list."
        if len(items) > _MAX_FLAG_LIST_ITEMS:
            return f"{name} has more than {_MAX_FLAG_LIST_ITEMS} entries."
        for item in items:
            if not isinstance(item, str) or not item.strip():
                return f"{name} may only contain text."
            if name != "allowPlans" and _parse_uuid(item) is None:
                return f"{name} must list workspace ids."
            if name == "allowPlans" and not _PLAN_SLUG.match(item):
                return "allowPlans must list plan slugs."
    return None


def _reject_constant(name: str) -> Any:
    # NaN/Infinity are not JSON; System.Text.Json refuses them, so this reader does too.
    raise ValueError(f"{name} is not JSON")


def _parse_json(raw: str) -> Any:
    return json.loads(raw, parse_constant=_reject_constant)


def _parse_uuid(value: str | None) -> uuid.UUID | None:
    if not value:
        return None
    try:
        return uuid.UUID(str(value).strip())
    except (ValueError, AttributeError, TypeError):
        return None


# ---------------------------------------------------------------------------------------------
# Feature flags — FeatureFlagValue.cs
# ---------------------------------------------------------------------------------------------


def bucket(flag_key: str, subject: str) -> int:
    """0-99. First four bytes of SHA-256("{flag}:{subject lower-cased}"), big-endian, mod 100.

    The .NET side uses the same formula and both repos test the same vectors, so a workspace is
    either inside a rollout everywhere or nowhere.
    """
    digest = hashlib.sha256(f"{flag_key}:{subject.lower()}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


@dataclass(frozen=True)
class FeatureFlag:
    enabled: bool
    rollout_percent: int = 100
    allow_plans: tuple[str, ...] = ()
    allow_workspaces: tuple[str, ...] = ()
    deny_workspaces: tuple[str, ...] = ()

    @classmethod
    def off(cls) -> FeatureFlag:
        return cls(enabled=False, rollout_percent=0)

    @classmethod
    def parse(cls, value: Any) -> FeatureFlag | None:
        if not isinstance(value, dict):
            return None
        enabled = value.get("enabled", False)
        percent = value.get("rolloutPercent", 100)

        def _strings(name: str) -> tuple[str, ...]:
            items = value.get(name) or []
            return tuple(str(item) for item in items) if isinstance(items, list) else ()

        return cls(
            enabled=enabled is True,
            rollout_percent=percent
            if isinstance(percent, int) and not isinstance(percent, bool)
            else 100,
            allow_plans=_strings("allowPlans"),
            allow_workspaces=_strings("allowWorkspaces"),
            deny_workspaces=_strings("denyWorkspaces"),
        )

    def depends_on_workspace(self) -> bool:
        """Whether the workspace could change the answer (so a caller can skip the lookup)."""
        if not self.enabled:
            return False
        return self.rollout_percent < 100 or bool(self.deny_workspaces)

    def is_enabled_for(
        self, flag_key: str, workspace_id: str | None, plan_slug: str | None = None
    ) -> bool:
        if not self.enabled:
            return False
        workspace = _parse_uuid(workspace_id)
        if workspace is None:
            return self.rollout_percent >= 100

        subject = str(workspace)
        if any(entry.lower() == subject for entry in self.deny_workspaces):
            return False
        if any(entry.lower() == subject for entry in self.allow_workspaces):
            return True
        if plan_slug is not None and any(
            entry.lower() == plan_slug.lower() for entry in self.allow_plans
        ):
            return True
        if self.rollout_percent >= 100:
            return True
        if self.rollout_percent <= 0:
            return False
        return bucket(flag_key, subject) < self.rollout_percent


# ---------------------------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------------------------


class SettingsSource(Protocol):
    """What the reader needs from Redis. RedisStreamClient satisfies it."""

    async def hgetall(self, key: str) -> dict[bytes | str, bytes | str]: ...

    async def get(self, key: str) -> bytes | str | None: ...


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


@dataclass
class _CachedHash:
    values: dict[str, str] | None
    fetched_at: float
    parsed: dict[str, Any] = field(default_factory=dict)


_UNPARSED = object()

WorkspaceResolver = Callable[[], Awaitable[str | None]]


class PlatformSettings:
    """Live platform settings over the published Redis hashes. Never raises from a read."""

    def __init__(
        self,
        source: SettingsSource | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
        registry: Mapping[str, SettingDefinition] | None = None,
    ) -> None:
        self.source = source
        self._clock = clock
        self.cache_ttl_seconds = cache_ttl_seconds
        self._registry = registry if registry is not None else REGISTRY
        self._cache: dict[str, _CachedHash] = {}
        self._warned: set[tuple[str, str, str]] = set()
        self._room_workspaces: dict[str, tuple[str | None, float]] = {}

    # -- stored values -----------------------------------------------------------------------

    async def get_stored(
        self,
        key: str,
        *,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
    ) -> Any | None:
        """The stored value after override resolution, or None when nothing valid is set."""
        try:
            definition = self._registry.get(key)
            if definition is None:
                return None

            workspace = _parse_uuid(workspace_id)
            if workspace is not None and definition.allows(SCOPE_WORKSPACE):
                hash_key = workspace_hash(workspace)
                value = self._value(
                    definition, hash_key, await self._read(hash_key, keep_stale=False)
                )
                if value is not None:
                    return value

            if plan_slug and plan_slug.strip() and definition.allows(SCOPE_PLAN):
                hash_key = plan_hash(plan_slug)
                value = self._value(
                    definition, hash_key, await self._read(hash_key, keep_stale=False)
                )
                if value is not None:
                    return value

            return self._value(
                definition, PLATFORM_HASH, await self._read(PLATFORM_HASH, keep_stale=True)
            )
        except Exception:
            # Belt and braces: _read already swallows Redis failures. Nothing on this path is
            # allowed to reach a caller that is in the middle of processing audio.
            logger.warning("platform_setting_read_failed", setting_key=key, exc_info=True)
            return None

    def _default(self, key: str) -> Any:
        definition = self._registry.get(key)
        return definition.default if definition is not None else None

    async def get_bool(
        self,
        key: str,
        fallback: bool | None = None,
        *,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
    ) -> bool:
        value = await self.get_stored(key, workspace_id=workspace_id, plan_slug=plan_slug)
        if isinstance(value, bool):
            return value
        if fallback is not None:
            return fallback
        return self._default(key) is True

    async def get_int(
        self,
        key: str,
        fallback: int | None = None,
        *,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
    ) -> int:
        value = await self.get_stored(key, workspace_id=workspace_id, plan_slug=plan_slug)
        if _is_number(value) and value == math.trunc(value):
            return int(value)
        if fallback is not None:
            return int(fallback)
        default = self._default(key)
        return int(default) if _is_number(default) else 0

    async def get_float(
        self,
        key: str,
        fallback: float | None = None,
        *,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
    ) -> float:
        value = await self.get_stored(key, workspace_id=workspace_id, plan_slug=plan_slug)
        if _is_number(value):
            return float(value)
        if fallback is not None:
            return float(fallback)
        default = self._default(key)
        return float(default) if _is_number(default) else 0.0

    async def get_str(
        self,
        key: str,
        fallback: str | None = None,
        *,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
    ) -> str:
        value = await self.get_stored(key, workspace_id=workspace_id, plan_slug=plan_slug)
        if isinstance(value, str):
            return value
        if fallback is not None:
            return fallback
        default = self._default(key)
        return default if isinstance(default, str) else ""

    async def is_enabled(
        self,
        flag_key: str,
        workspace_id: str | None = None,
        plan_slug: str | None = None,
        fallback: bool | None = None,
        *,
        resolve_workspace: WorkspaceResolver | None = None,
    ) -> bool:
        """A feature flag for a workspace (or, with none, for the platform as a whole).

        `fallback` applies only when the flag is not set. `resolve_workspace` is called only when
        the flag's value could actually depend on the workspace, so the common case — a flag that
        is plainly on or plainly off — costs no lookup at all.
        """
        # Flags are platform-scoped: overrides never apply to them (IsEnabledAsync reads with an
        # empty context), the workspace and plan only feed the evaluation.
        stored = await self.get_stored(flag_key)
        if stored is None and fallback is not None:
            return fallback
        flag = FeatureFlag.parse(stored if stored is not None else self._default(flag_key))
        if flag is None:
            flag = FeatureFlag.off()

        if workspace_id is None and resolve_workspace is not None and flag.depends_on_workspace():
            try:
                workspace_id = await resolve_workspace()
            except Exception:
                logger.warning(
                    "platform_flag_workspace_unresolved", flag_key=flag_key, exc_info=True
                )
                workspace_id = None
        return flag.is_enabled_for(flag_key, workspace_id, plan_slug)

    # -- room -> workspace -------------------------------------------------------------------

    async def room_workspace_id(self, room_id: str) -> str | None:
        """The workspace a translation room belongs to, from MeetingService's room projection.

        `meeting:room:v2:{room}` is the projection billing_worker already reads for the same
        question (see _resolve_subscription there — the `v2` is load-bearing). None when it is
        missing or unreadable, which evaluates a flag at platform level: the safe reading of
        "we do not know whose room this is".
        """
        now = self._clock()
        cached = self._room_workspaces.get(room_id)
        if cached is not None and now - cached[1] < _ROOM_WORKSPACE_TTL_SECONDS:
            return cached[0]

        workspace: str | None = None
        try:
            if self.source is not None:
                raw = await self.source.get(f"meeting:room:v2:{room_id}")
                if isinstance(raw, (bytes, str)) and raw:
                    projection = json.loads(_text(raw))
                    if isinstance(projection, dict):
                        candidate = projection.get("WorkspaceId") or projection.get("workspaceId")
                        parsed = _parse_uuid(candidate if isinstance(candidate, str) else None)
                        workspace = str(parsed) if parsed is not None else None
        except Exception:
            logger.warning("platform_room_workspace_unreadable", room_id=room_id, exc_info=True)

        if len(self._room_workspaces) >= _ROOM_WORKSPACE_CACHE_MAX:
            self._room_workspaces.clear()
        self._room_workspaces[room_id] = (workspace, now)
        return workspace

    # -- cache -------------------------------------------------------------------------------

    def invalidate(self) -> None:
        """Forget every cached hash, so the next read goes to Redis."""
        self._cache.clear()

    def _value(
        self, definition: SettingDefinition, hash_key: str, cached: _CachedHash | None
    ) -> Any | None:
        if cached is None or cached.values is None:
            return None
        raw = cached.values.get(definition.key)
        if raw is None:
            return None
        memo = cached.parsed.get(definition.key, _UNPARSED)
        if memo is not _UNPARSED:
            return memo

        value: Any | None
        try:
            value = _parse_json(raw)
            error = validate(definition, value)
        except ValueError:
            value, error = None, "it is not JSON"
        if error is not None:
            marker = (hash_key, definition.key, raw)
            if marker not in self._warned:
                if len(self._warned) > 1000:
                    self._warned.clear()
                self._warned.add(marker)
                logger.warning(
                    "platform_setting_ignored",
                    setting_key=definition.key,
                    hash_key=hash_key,
                    error=error,
                )
            value = None
        cached.parsed[definition.key] = value
        return value

    async def _read(self, hash_key: str, *, keep_stale: bool) -> _CachedHash | None:
        now = self._clock()
        cached = self._cache.get(hash_key)
        if cached is not None and now - cached.fetched_at < self.cache_ttl_seconds:
            return cached

        if self.source is None:
            entry = _CachedHash(None, now)
            self._cache[hash_key] = entry
            return entry

        try:
            raw = await self.source.hgetall(hash_key)
            fresh: dict[str, str] | None = None
            # HGETALL on a missing key is an empty reply. Anything that is not a mapping (a test
            # double, a protocol surprise) is treated the same way rather than trusted.
            if isinstance(raw, Mapping) and len(raw) > 0:
                fresh = {_text(k): _text(v) for k, v in raw.items()}
            if fresh is None and keep_stale and cached is not None and cached.values is not None:
                # Evicted or restarted, not reset: a reset leaves the version field behind. Keep
                # what we had until the writer's next re-publish, and keep the parse memo with it.
                entry = _CachedHash(cached.values, now, cached.parsed)
            else:
                entry = _CachedHash(fresh, now)
        except Exception:
            logger.warning("platform_settings_unreadable", hash_key=hash_key, exc_info=True)
            # Back off one TTL, so an outage costs one failed call per hash per TTL rather than one
            # per read.
            if cached is not None:
                entry = _CachedHash(cached.values, now, cached.parsed)
            else:
                entry = _CachedHash(None, now)
        self._cache[hash_key] = entry
        return entry


def reader_for(owner: object) -> PlatformSettings:
    """The PlatformSettings reader of a worker, over that worker's own Redis client.

    Lazy and attribute-tolerant on purpose: workers are routinely built with `__new__` in tests,
    with only the attributes a test needs, and sometimes have their `redis` swapped after the
    fact. The reader is rebuilt whenever the client it was built over is no longer the worker's,
    so a swapped client is never read through a stale reader.
    """
    source = getattr(owner, "redis", None)
    reader = getattr(owner, "_platform_settings", None)
    if isinstance(reader, PlatformSettings) and reader.source is source:
        return reader
    reader = PlatformSettings(source)
    try:
        owner._platform_settings = reader  # type: ignore[attr-defined]
    except Exception:
        # Cannot cache on this owner (slots, a frozen double): a fresh reader per call still
        # answers correctly, just without the cache.
        pass
    return reader
