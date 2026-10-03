"""Everyone on the Meet side of a bridge room is ONE seat — and was therefore one voice.

WT-932. An EXTERNAL_BRIDGE room publishes the whole far side of the call under a single stand-in
speaker_id, and the hashed stock voice is chosen per speaker_id, so three people taking turns in
Meet were dubbed as one person talking to themselves. The caption name the translation worker
now forwards is the only thing that tells them apart.

What must hold:

    same name            → same voice, for the whole meeting
    two names            → two voices, while the catalogue has any left
    no name / not sure   → exactly the voice the stand-in had before
    a native speaker     → exactly the voice they had before, whatever fields ride along
    everybody in the room → exactly the voice they had before a caption name turned up
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from tts_worker.worker import (
    FAR_SPEAKER_VOICE_MIN_CONFIDENCE,
    TTSWorker,
    far_speaker_voice_key,
)

MEETING = "room-1"
LANG = "en"
STAND_IN = EXTERNAL_BRIDGE_SPEAKER_ID
# Real participants. Ids chosen to sort on BOTH sides of nothing in particular — the stand-in
# GUID is all zeros and sorts ahead of every one of them, which is the case that matters.
NATIVES = ["3f2b0c1e-host", "9a7d4e55-guest", "c0ffee00-third"]

EN: list[dict[str, Any]] = [
    {"id": f"en-{name}", "name": name, "gender": gender}
    for name, gender in [
        ("ada", "feminine"),
        ("bea", "feminine"),
        ("cy", "masculine"),
        ("dov", "masculine"),
        ("eve", "feminine"),
        ("fox", "masculine"),
        ("gil", "masculine"),
        ("hana", "feminine"),
        ("ivo", "masculine"),
        ("june", "feminine"),
        ("kai", "masculine"),
        ("lena", "feminine"),
    ]
]


class _FakeRedis:
    """Just the calls voice resolution makes: GET for the catalogue, and three hashes."""

    def __init__(self, roster: list[str], catalog: list[dict[str, Any]] | None = None) -> None:
        self.strings: dict[str, str] = {}
        if catalog is not None:
            self.strings[f"voice_catalog:{LANG}"] = json.dumps(catalog)
        # Bytes, as the real client returns them.
        self.hashes: dict[str, dict[bytes | str, bytes | str]] = {
            f"translationRoom:{MEETING}:languages": {uid.encode(): b"en" for uid in roster},
        }
        self.expired: dict[str, int] = {}
        self.fail_hash: str | None = None

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def hgetall(self, key: str) -> dict[bytes | str, bytes | str]:
        if self.fail_hash and key.startswith(self.fail_hash):
            raise ConnectionError("redis is down")
        return dict(self.hashes.get(key, {}))

    async def hget(self, key: str, field: str) -> bytes | str | None:
        return self.hashes.get(key, {}).get(field)

    async def hset(self, key: str, field: str, value: bytes | str) -> None:
        self.hashes.setdefault(key, {})[field] = value

    async def expire(self, key: str, ttl_seconds: int) -> None:
        self.expired[key] = ttl_seconds

    def join(self, uid: str) -> None:
        self.hashes[f"translationRoom:{MEETING}:languages"][uid.encode()] = b"en"

    def prefer(self, listener: str, voice_id: str) -> None:
        self.hashes.setdefault(f"translationRoom:{MEETING}:voice_preferences", {})[
            listener.encode()
        ] = voice_id.encode()


def _worker(
    roster: list[str] | None = None, catalog: list[dict[str, Any]] | None = None
) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker.worker_name = "tts"
    worker._consumer_name = "tts-test"
    worker.redis = _FakeRedis(  # type: ignore[assignment]
        roster if roster is not None else [STAND_IN, *NATIVES],
        EN if catalog is None else catalog,
    )
    # No profile pick, no live clone: the stand-in never has either, and the natives here are
    # the un-cloned ones whose hashed voice a caption name could have disturbed.
    worker.chosen_dub_voice = lambda _m, _s: None  # type: ignore[method-assign]

    async def _no_clone(_meeting: str, _speaker: str) -> str | None:
        return None

    worker._get_voice_id = _no_clone  # type: ignore[method-assign]
    return worker


async def _default_voice(
    worker: TTSWorker,
    speaker: str = STAND_IN,
    name: str | None = None,
    confidence: float | None = None,
) -> str:
    variants = await worker._resolve_voice_variants(
        MEETING, speaker, LANG, far_speaker_name=name, far_speaker_confidence=confidence
    )
    voice_id, voice_type, voice_key = variants[0]
    # Whatever the voice, it is still THE default variant on the stand-in's own identity.
    assert (voice_type, voice_key) == ("default", "")
    return voice_id


def _assigned_events(worker: TTSWorker) -> list[dict[str, Any]]:
    return [
        call.kwargs
        for call in worker.logger.info.call_args_list  # type: ignore[attr-defined]
        if call.args and call.args[0] == "far_speaker_voice_assigned"
    ]


# ── the gate ────────────────────────────────────────────────────────────────────────────────


class TestOnlyACertainNameOnTheStandIn:
    def test_the_threshold_is_certainty(self) -> None:
        assert FAR_SPEAKER_VOICE_MIN_CONFIDENCE == 1.0

    def test_a_certain_name_on_the_stand_in_has_a_key(self) -> None:
        assert far_speaker_voice_key(STAND_IN, "An", 1.0) == f"{STAND_IN}:an"

    def test_the_same_person_spelled_differently_is_one_key(self) -> None:
        assert far_speaker_voice_key(STAND_IN, "  Trần   An ", 1.0) == far_speaker_voice_key(
            STAND_IN, "trần an", 1.0
        )

    @pytest.mark.parametrize("confidence", [None, 0.0, 0.5, 0.99])
    def test_anything_short_of_certain_has_none(self, confidence: float | None) -> None:
        assert far_speaker_voice_key(STAND_IN, "An", confidence) is None

    @pytest.mark.parametrize("name", [None, "", "   "])
    def test_no_name_has_none(self, name: str | None) -> None:
        assert far_speaker_voice_key(STAND_IN, name, 1.0) is None

    def test_a_native_speaker_has_none_whatever_rides_along(self) -> None:
        assert far_speaker_voice_key(NATIVES[0], "An", 1.0) is None


# ── same name, different names ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_same_name_is_the_same_voice() -> None:
    worker = _worker()

    first = await _default_voice(worker, name="An", confidence=1.0)
    again = await _default_voice(worker, name="An", confidence=1.0)
    respelled = await _default_voice(worker, name=" an ", confidence=1.0)

    assert first == again == respelled


@pytest.mark.asyncio
async def test_two_names_are_two_voices_and_neither_is_the_shared_one() -> None:
    worker = _worker()
    shared = await _default_voice(worker)

    names = ["An", "Bình", "Chi", "Dũng"]
    voices = [await _default_voice(worker, name=name, confidence=1.0) for name in names]

    assert len(set(voices)) == len(names)
    # A named person must not sound like "somebody on the far side".
    assert shared not in voices


@pytest.mark.asyncio
async def test_a_named_person_does_not_take_a_voice_someone_in_the_room_has() -> None:
    worker = _worker()
    room = {await _default_voice(worker, speaker=uid) for uid in [STAND_IN, *NATIVES]}

    voices = {await _default_voice(worker, name=name, confidence=1.0) for name in "ABCDEFGH"}

    # 12 voices, 4 held by the room, 8 names: everyone fits, nobody overlaps.
    assert len(voices) == 8
    assert not voices & room


@pytest.mark.asyncio
async def test_another_replica_gives_the_same_name_the_same_voice() -> None:
    """The choice lives in Redis, not in the process that made it."""
    first = _worker()
    voice = await _default_voice(first, name="An", confidence=1.0)

    second = _worker()
    second.redis = first.redis

    assert await _default_voice(second, name="An", confidence=1.0) == voice


@pytest.mark.asyncio
async def test_a_name_keeps_its_voice_when_the_roster_changes_under_it() -> None:
    worker = _worker()
    before = await _default_voice(worker, name="An", confidence=1.0)

    for late in ["00000000-early-sorter", "ffffffff-late-sorter"]:
        worker.redis.join(late)  # type: ignore[attr-defined]

    assert await _default_voice(worker, name="An", confidence=1.0) == before


@pytest.mark.asyncio
async def test_more_names_than_voices_still_answers_and_stays_put() -> None:
    worker = _worker(catalog=EN[:5])
    names = [f"person-{i}" for i in range(9)]

    voices = [await _default_voice(worker, name=name, confidence=1.0) for name in names]

    assert all(voice in {v["id"] for v in EN[:5]} for voice in voices)
    assert voices == [await _default_voice(worker, name=name, confidence=1.0) for name in names]


# ── everything else is exactly what it was ──────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "confidence"),
    [("An", 0.5), ("An", None), ("An", 0.99), ("", 1.0), ("   ", 1.0), (None, 1.0)],
)
async def test_an_unsure_or_unnamed_segment_keeps_todays_voice(
    name: str | None, confidence: float | None
) -> None:
    worker = _worker()
    today = await worker._hashed_default_voice_id(LANG, STAND_IN, MEETING)

    assert await _default_voice(worker, name=name, confidence=confidence) == today
    assert _assigned_events(worker) == []
    assert not any(
        key.startswith("tts:far_speaker_voices:")
        for key in worker.redis.hashes  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_a_native_speaker_carrying_far_speaker_fields_keeps_todays_voice() -> None:
    worker = _worker()
    native = NATIVES[1]
    today = await worker._hashed_default_voice_id(LANG, native, MEETING)

    assert await _default_voice(worker, speaker=native, name="An", confidence=1.0) == today
    assert _assigned_events(worker) == []


@pytest.mark.asyncio
async def test_nobody_in_the_room_changes_voice_when_caption_names_appear() -> None:
    worker = _worker()
    room = [STAND_IN, *NATIVES]
    before = {uid: await _default_voice(worker, speaker=uid) for uid in room}

    for name in ["An", "Bình", "Chi", "Dũng", "Em"]:
        await _default_voice(worker, name=name, confidence=1.0)

    after = {uid: await _default_voice(worker, speaker=uid) for uid in room}
    assert after == before


def test_adding_the_key_to_the_roster_is_what_would_have_moved_them() -> None:
    """Why the key claims AFTER the roster instead of joining it.

    The stand-in GUID is all zeros, so its far-speaker keys sort ahead of nearly every real id
    and would claim first. With one voice to fight over, the participant loses theirs.
    """
    one_voice = [{"id": "only", "gender": "feminine"}, {"id": "other", "gender": "feminine"}]
    native = "9a7d4e55-guest"
    alone = TTSWorker._assign_voice(one_voice, [native], native)["id"]

    moved = False
    for i in range(50):
        key = f"{STAND_IN}:name-{i}"
        if TTSWorker._assign_voice(one_voice, sorted([native, key]), native)["id"] != alone:
            moved = True
            break
    assert moved


@pytest.mark.asyncio
async def test_a_profile_or_a_clone_still_wins() -> None:
    worker = _worker()

    async def _cloned(_meeting: str, _speaker: str) -> str | None:
        return "cloned-voice"

    worker._get_voice_id = _cloned  # type: ignore[method-assign]
    variants = await worker._resolve_voice_variants(
        MEETING, STAND_IN, LANG, far_speaker_name="An", far_speaker_confidence=1.0
    )
    assert variants == [("cloned-voice", "cloned", "")]
    assert _assigned_events(worker) == []


# ── a listener's explicit pick ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_listeners_pick_is_still_rendered_for_every_name() -> None:
    worker = _worker()
    picked = "en-june"
    worker.redis.prefer(NATIVES[0], picked)  # type: ignore[attr-defined]

    for name in ["An", "Bình", "Chi"]:
        variants = await worker._resolve_voice_variants(
            MEETING, STAND_IN, LANG, far_speaker_name=name, far_speaker_confidence=1.0
        )
        # One default + the listener's variant, under the same variant key as today.
        assert len(variants) == 2
        assert variants[0][0] != picked
        assert variants[1] == (picked, "preference", f"voice-{picked[:8]}")


# ── caches ──────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_names_saying_the_same_words_do_not_share_cached_audio() -> None:
    worker = _worker()
    an = await _default_voice(worker, name="An", confidence=1.0)
    binh = await _default_voice(worker, name="Bình", confidence=1.0)

    def key(voice_id: str) -> str:
        return TTSWorker._cache_key(STAND_IN, LANG, "Okay, thank you.", f"default:{voice_id}")

    assert key(an) != key(binh)
    assert key(an) == key(an)


# ── logging ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_assignment_is_logged_once_and_never_with_the_name() -> None:
    worker = _worker()
    name = "Nguyễn Văn An"

    voice = await _default_voice(worker, name=name, confidence=1.0)
    await _default_voice(worker, name=name, confidence=1.0)
    await _default_voice(worker, name=name, confidence=1.0)

    events = _assigned_events(worker)
    assert len(events) == 1
    assert events[0]["meeting_id"] == MEETING
    assert events[0]["lang"] == LANG
    assert events[0]["voice_id"] == voice
    assert name.casefold() not in json.dumps(events[0], ensure_ascii=False).casefold()

    # Nor in Redis: the hash field is the same short hash the log line carries.
    stored = worker.redis.hashes[f"tts:far_speaker_voices:{MEETING}:{LANG}"]  # type: ignore[attr-defined]
    assert list(stored) == [events[0]["far_speaker_hash"]]
    assert worker.redis.expired[f"tts:far_speaker_voices:{MEETING}:{LANG}"] > 0  # type: ignore[attr-defined]


# ── failure ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_unreadable_memory_falls_back_to_todays_voice() -> None:
    worker = _worker()
    today = await worker._hashed_default_voice_id(LANG, STAND_IN, MEETING)
    worker.redis.fail_hash = "tts:far_speaker_voices:"  # type: ignore[attr-defined]

    assert await _default_voice(worker, name="An", confidence=1.0) == today
    assert _assigned_events(worker) == []


@pytest.mark.asyncio
async def test_a_remembered_voice_the_catalogue_dropped_is_replaced() -> None:
    worker = _worker()
    first = await _default_voice(worker, name="An", confidence=1.0)

    remaining = [voice for voice in EN if voice["id"] != first]
    worker.redis.strings[f"voice_catalog:{LANG}"] = json.dumps(remaining)  # type: ignore[attr-defined]

    second = await _default_voice(worker, name="An", confidence=1.0)
    assert second != first
    assert second in {voice["id"] for voice in remaining}
    assert await _default_voice(worker, name="An", confidence=1.0) == second
