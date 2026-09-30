"""The platform settings reader: resolution order, fallbacks, cache and failure behaviour.

These semantics are a contract with WarpTalk.Shared/PlatformSettings in the backend. The bucket
vectors at the bottom are asserted by the .NET tests too, so a workspace is inside a rollout on
both sides or on neither.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from shared import platform_settings as ps
from shared.platform_settings import (
    PLATFORM_HASH,
    FeatureFlag,
    PlatformSettings,
    bucket,
    plan_hash,
    reader_for,
    workspace_hash,
)
from shared.redis_client import RedisStreamClient
from tests.conftest import FakeClock

WS = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"


async def publish(client: RedisStreamClient, hash_key: str = PLATFORM_HASH) -> None:
    """A published snapshot with nothing set: only the version field."""
    await client.redis.hset(hash_key, mapping={ps.VERSION_FIELD: "1"})


async def put(
    client: RedisStreamClient, key: str, value: Any, hash_key: str = PLATFORM_HASH
) -> None:
    await client.redis.hset(hash_key, mapping={ps.VERSION_FIELD: "1", key: json.dumps(value)})


class ExplodingSource:
    """A Redis that fails every call, or only when told to."""

    def __init__(self, values: dict[str, dict[str, str]] | None = None) -> None:
        self.values = values or {}
        self.fail = False
        self.calls = 0

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        self.calls += 1
        if self.fail:
            raise ConnectionError("redis down")
        return {k.encode(): v.encode() for k, v in self.values.get(key, {}).items()}

    async def get(self, key: str) -> bytes | None:
        if self.fail:
            raise ConnectionError("redis down")
        return None


def reader(source: Any, clock: FakeClock) -> PlatformSettings:
    return PlatformSettings(source, clock=clock)


# ── resolution ──────────────────────────────────────────────────────────────────────────────


async def test_unset_key_returns_the_callers_fallback_not_the_registry_default(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)
    await publish(settings_redis)  # a published snapshot with nothing set

    assert await r.get_int(ps.CHUNK_DURATION_MS, 1234) == 1234
    assert await r.get_bool(ps.FLASH_MODE_DEFAULT, False) is False
    assert await r.get_float(ps.VOICE_CLONE_MIN_SECONDS, 11.5) == 11.5


async def test_unset_key_without_fallback_returns_the_registry_default(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)

    assert await r.get_int(ps.CHUNK_DURATION_MS) == 6000
    assert await r.get_bool(ps.FLASH_MODE_DEFAULT) is True
    assert await r.get_float(ps.SUGGEST_MIN_CONFIDENCE) == 0.55
    assert await r.get_float(ps.SUGGEST_MIN_STT_CONFIDENCE) == -0.5
    assert await r.get_str("not.a.key", "x") == "x"


async def test_a_stored_value_beats_the_fallback(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000)
    r = reader(settings_redis, fake_clock)

    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000


async def test_workspace_override_beats_plan_beats_platform(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    # None of the AI-owned keys allow overrides, so the generic path is exercised through a
    # registry that does — the resolution code is the same for every key.
    registry = {
        "t.value": ps.SettingDefinition(
            "t.value",
            "integer",
            1,
            min=0,
            max=100,
            scopes=frozenset({ps.SCOPE_PLATFORM, ps.SCOPE_PLAN, ps.SCOPE_WORKSPACE}),
        )
    }
    r = PlatformSettings(settings_redis, clock=fake_clock, registry=registry)
    await put(settings_redis, "t.value", 10)
    await put(settings_redis, "t.value", 20, plan_hash("Pro"))
    await put(settings_redis, "t.value", 30, workspace_hash(uuid.UUID(WS)))

    assert await r.get_int("t.value", 0) == 10
    assert await r.get_int("t.value", 0, plan_slug="pro") == 20
    assert await r.get_int("t.value", 0, plan_slug="pro", workspace_id=WS.upper()) == 30
    assert await r.get_int("t.value", 0, workspace_id=WS) == 30
    # Another workspace has no override: its plan, then the platform, apply.
    other = "00000000-0000-0000-0000-000000000001"
    assert await r.get_int("t.value", 0, plan_slug="pro", workspace_id=other) == 20


async def test_overrides_are_ignored_for_a_platform_only_key(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 5000)
    await put(settings_redis, ps.CHUNK_DURATION_MS, 3000, workspace_hash(uuid.UUID(WS)))
    r = reader(settings_redis, fake_clock)

    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000, workspace_id=WS) == 5000


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        (ps.CHUNK_DURATION_MS, "1999"),  # below min
        (ps.CHUNK_DURATION_MS, "15001"),  # above max
        (ps.CHUNK_DURATION_MS, "6000.5"),  # not whole
        (ps.CHUNK_DURATION_MS, '"6000"'),  # a string
        (ps.CHUNK_DURATION_MS, "true"),  # a bool is not a number
        (ps.FLASH_MODE_DEFAULT, '"true"'),
        (ps.SUGGEST_MIN_CONFIDENCE, "1.5"),
        (ps.SUGGEST_MIN_CONFIDENCE, "NaN"),
        (ps.SUGGEST_MIN_STT_CONFIDENCE, "0.1"),
        (ps.CHUNK_DURATION_MS, "{not json"),
        (ps.FLAG_AI_SUGGEST, '{"rolloutPercent": 50}'),  # no enabled
        (ps.FLAG_AI_SUGGEST, '{"enabled": true, "rolloutPercent": 101}'),
        (ps.FLAG_AI_SUGGEST, '{"enabled": true, "surprise": 1}'),
        (ps.FLAG_AI_SUGGEST, '{"enabled": true, "denyWorkspaces": ["not-a-guid"]}'),
        (ps.FLAG_AI_SUGGEST, '{"enabled": true, "allowPlans": ["Pro Plan"]}'),
    ],
)
async def test_an_invalid_stored_value_is_ignored_and_the_fallback_applies(
    settings_redis: RedisStreamClient, fake_clock: FakeClock, key: str, raw: str
) -> None:
    await settings_redis.redis.hset(PLATFORM_HASH, mapping={ps.VERSION_FIELD: "1", key: raw})
    r = reader(settings_redis, fake_clock)

    assert await r.get_stored(key) is None
    if key == ps.CHUNK_DURATION_MS:
        assert await r.get_int(key, 7000) == 7000
    if key == ps.FLAG_AI_SUGGEST:
        assert await r.is_enabled(key, fallback=False) is False


async def test_integral_float_is_accepted_for_an_integer(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000.0)
    assert await reader(settings_redis, fake_clock).get_int(ps.CHUNK_DURATION_MS, 1) == 4000


# ── cache and failure ───────────────────────────────────────────────────────────────────────


async def test_a_change_is_seen_after_the_ttl_and_not_before(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000)
    r = reader(settings_redis, fake_clock)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000

    await put(settings_redis, ps.CHUNK_DURATION_MS, 8000)
    fake_clock.advance(9.9)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000

    fake_clock.advance(0.2)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 8000


async def test_a_missing_platform_hash_keeps_the_last_snapshot(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000)
    r = reader(settings_redis, fake_clock)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000

    await settings_redis.redis.delete(PLATFORM_HASH)  # evicted
    fake_clock.advance(11)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000


async def test_a_reset_that_leaves_the_version_field_does_reset(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000)
    r = reader(settings_redis, fake_clock)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000

    await settings_redis.redis.delete(PLATFORM_HASH)
    await publish(settings_redis)  # only __version: "nothing set"
    fake_clock.advance(11)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 6000


async def test_a_redis_error_keeps_the_snapshot_and_backs_off_one_ttl(
    fake_clock: FakeClock,
) -> None:
    source = ExplodingSource({PLATFORM_HASH: {ps.CHUNK_DURATION_MS: "4000"}})
    r = reader(source, fake_clock)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000
    assert source.calls == 1

    source.fail = True
    fake_clock.advance(11)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000
    assert source.calls == 2
    # Backing off: no further call inside the next TTL.
    fake_clock.advance(5)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 4000
    assert source.calls == 2

    source.fail = False
    source.values[PLATFORM_HASH][ps.CHUNK_DURATION_MS] = "9000"
    fake_clock.advance(6)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 9000


async def test_a_redis_error_before_any_snapshot_uses_the_fallback(fake_clock: FakeClock) -> None:
    source = ExplodingSource()
    source.fail = True
    r = reader(source, fake_clock)

    assert await r.get_int(ps.CHUNK_DURATION_MS, 6000) == 6000
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, WS) is True
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, WS, fallback=False) is False


@pytest.mark.parametrize("source", [None, object(), "not redis"])
async def test_a_reader_over_nothing_usable_never_raises(
    source: Any, fake_clock: FakeClock
) -> None:
    r = reader(source, fake_clock)
    assert await r.get_int(ps.CHUNK_DURATION_MS, 1) == 1
    assert await r.get_bool(ps.FLASH_MODE_DEFAULT, False) is False
    assert await r.is_enabled(ps.FLAG_VOICE_CLONE) is True
    assert await r.room_workspace_id("room-1") is None


async def test_reader_for_follows_a_swapped_client(settings_redis: RedisStreamClient) -> None:
    class Owner:
        pass

    owner = Owner()
    owner.redis = settings_redis  # type: ignore[attr-defined]
    first = reader_for(owner)
    assert reader_for(owner) is first

    owner.redis = object()  # type: ignore[attr-defined]
    assert reader_for(owner) is not first
    assert reader_for(owner).source is owner.redis  # type: ignore[attr-defined]


# ── feature flags ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("flag_key", "subject", "expected"),
    [
        ("flags.ai_suggest", "00000000-0000-0000-0000-000000000001", 76),
        ("flags.ai_suggest", "3f2504e0-4f89-11d3-9a0c-0305e82c3301", 14),
        ("flags.voice_clone", "9b2d7c1e-8a4f-4e3b-b5d6-1c2e3f4a5b6c", 61),
        ("flags.global_glossary", "00000000-0000-0000-0000-000000000001", 83),
        ("flags.warpbot_web_search", "ffffffff-ffff-ffff-ffff-ffffffffffff", 4),
    ],
)
def test_bucket_matches_the_dotnet_vectors(flag_key: str, subject: str, expected: int) -> None:
    assert bucket(flag_key, subject) == expected
    assert bucket(flag_key, subject.upper()) == expected


async def test_flag_evaluation_order(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)
    ws_in = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"  # ai_suggest bucket 14
    ws_out = "00000000-0000-0000-0000-000000000001"  # ai_suggest bucket 76

    async def set_flag(value: dict[str, Any]) -> None:
        await put(settings_redis, ps.FLAG_AI_SUGGEST, value)
        r.invalidate()

    # Kill switch beats every allow list.
    await set_flag({"enabled": False, "allowWorkspaces": [ws_in], "allowPlans": ["pro"]})
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_in, "pro") is False
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST) is False

    # Deny beats allow.
    await set_flag({"enabled": True, "denyWorkspaces": [ws_in.upper()], "allowWorkspaces": [ws_in]})
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_in) is False
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_out) is True

    # Allow lists beat a zero rollout.
    await set_flag(
        {"enabled": True, "rolloutPercent": 0, "allowWorkspaces": [ws_out], "allowPlans": ["pro"]}
    )
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_out) is True
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_in) is False
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_in, "PRO") is True

    # Partial rollout by bucket: 14 < 50, 76 >= 50. No workspace: only at 100.
    await set_flag({"enabled": True, "rolloutPercent": 50})
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_in) is True
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_out) is False
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST) is False
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, "not-a-uuid") is False

    await set_flag({"enabled": True})
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST) is True
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, ws_out) is True


async def test_an_unset_flag_uses_the_fallback_then_the_default(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)
    assert await r.is_enabled(ps.FLAG_WARPBOT_WEB_SEARCH) is True
    assert await r.is_enabled(ps.FLAG_WARPBOT_WEB_SEARCH, fallback=False) is False
    assert await r.is_enabled("flags.unknown") is False


async def test_workspace_resolver_is_only_called_when_the_answer_depends_on_it(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)
    calls: list[int] = []

    async def resolve() -> str | None:
        calls.append(1)
        return "3f2504e0-4f89-11d3-9a0c-0305e82c3301"

    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, resolve_workspace=resolve) is True
    assert calls == []

    await put(settings_redis, ps.FLAG_AI_SUGGEST, {"enabled": True, "rolloutPercent": 50})
    r.invalidate()
    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, resolve_workspace=resolve) is True  # bucket 14
    assert calls == [1]


async def test_a_resolver_that_raises_evaluates_at_platform_level(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    await put(settings_redis, ps.FLAG_AI_SUGGEST, {"enabled": True, "rolloutPercent": 99})
    r = reader(settings_redis, fake_clock)

    async def resolve() -> str | None:
        raise RuntimeError("boom")

    assert await r.is_enabled(ps.FLAG_AI_SUGGEST, resolve_workspace=resolve) is False


async def test_room_workspace_comes_from_the_meeting_projection(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    r = reader(settings_redis, fake_clock)
    await settings_redis.redis.set(
        "meeting:room:v2:room-1", json.dumps({"WorkspaceId": WS.upper(), "Status": "ACTIVE"})
    )
    assert await r.room_workspace_id("room-1") == WS
    assert await r.room_workspace_id("room-2") is None

    await settings_redis.redis.set("meeting:room:v2:room-3", "not json")
    assert await r.room_workspace_id("room-3") is None


def test_flag_parse_defaults_rollout_to_100() -> None:
    flag = FeatureFlag.parse({"enabled": True})
    assert flag is not None
    assert flag.rollout_percent == 100
    assert flag.is_enabled_for("flags.ai_suggest", None) is True


def test_registry_mirrors_the_catalog() -> None:
    """Types, bounds and defaults copied from PlatformSettingsCatalog.cs. A drift here is a bug."""
    expected = {
        ps.FLASH_MODE_DEFAULT: ("boolean", True, None, None),
        ps.CHUNK_DURATION_MS: ("integer", 6000, 2000, 15000),
        ps.SUGGEST_MIN_WORDS: ("integer", 4, 1, 20),
        ps.SUGGEST_MIN_CONFIDENCE: ("decimal", 0.55, 0, 1),
        ps.SUGGEST_COOLDOWN_SECONDS: ("integer", 20, 0, 600),
        ps.SUGGEST_MAX_PER_MEETING: ("integer", 30, 0, 200),
        ps.SUGGEST_MIN_STT_CONFIDENCE: ("decimal", -0.5, -5, 0),
        ps.VOICE_CLONE_MIN_SECONDS: ("decimal", 20.0, 5, 90),
        ps.VOICE_CLONE_UPGRADE_MARGIN: ("decimal", 0.15, 0, 1),
    }
    for key, (type_, default, low, high) in expected.items():
        d = ps.REGISTRY[key]
        assert (d.type, d.default, d.min, d.max) == (type_, default, low, high), key
        assert d.scopes == frozenset({ps.SCOPE_PLATFORM}), key
    for key in (ps.FLAG_AI_SUGGEST, ps.FLAG_VOICE_CLONE, ps.FLAG_WARPBOT_WEB_SEARCH):
        assert ps.REGISTRY[key].default == {"enabled": True, "rolloutPercent": 100}
