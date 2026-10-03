"""A Meet-side person who consented for themselves is dubbed in their own cloned voice (WT-933).

Everyone on the Meet side of a bridge room is one stand-in seat on one mixed feed. WT-932 gave
each caption name a stock voice. This lets ONE of them consent to a clone, and voice is
biometric, so what must hold is mostly about what must never happen:

    flag off                       nothing changes, and nothing is even read
    audio nobody certainly said    never enters a clone buffer
    audio of someone not consented never enters a clone buffer
    hints that arrive late         still count, because the chunk waits for them
    the clone                      is spoken only from the 3rd certain sentence in a row
    consent withdrawn              stock voice at once, buffer dropped, voice model deleted
    a native speaker               exactly as before
    a display name                 never in Redis, never in a log
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.far_speaker import far_speaker_hints_key
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import SAMPLE_RATE, _varied
from tts_worker import far_speaker_clone
from tts_worker.far_speaker_clone import (
    CLONE_MIN_STREAK,
    clones_key,
    consent_field,
    consents_key,
    fold,
)
from tts_worker.worker import _VOICE_DELETE_STREAM, TTSWorker

MEETING = "0199aaaa-0000-7000-8000-000000000001"
LANG = "en"
STAND_IN = EXTERNAL_BRIDGE_SPEAKER_ID
NATIVE = "3f2b0c1e-0000-7000-8000-000000000002"
AN = "Trần  An"
LAN = "Lan Pham"
AN_FIELD = consent_field(AN) or ""
LAN_FIELD = consent_field(LAN) or ""
HINT_LAG_MS = 1000  # TTSSettings.far_speaker_clone_hint_lag_ms default

EN: list[dict[str, Any]] = [
    {"id": f"en-{name}", "name": name, "gender": gender}
    for name, gender in [
        ("ada", "feminine"),
        ("bea", "feminine"),
        ("cy", "masculine"),
        ("dov", "masculine"),
        ("eve", "feminine"),
        ("fox", "masculine"),
    ]
]


class _RawRedis:
    """The two calls made on the raw client: reading caption hints, and HDEL."""

    def __init__(self, owner: _FakeRedis) -> None:
        self._owner = owner

    async def xrevrange(self, key: str, count: int = 64) -> list[tuple[bytes, dict[bytes, bytes]]]:
        entries = self._owner.streams.get(key, [])
        return [
            (f"{index}-0".encode(), {k.encode(): v.encode() for k, v in fields.items()})
            for index, fields in reversed(list(enumerate(entries)))
        ][:count]

    async def hdel(self, key: str, field: str) -> int:
        return 1 if self._owner.hashes.get(key, {}).pop(field, None) is not None else 0


class _FakeRedis:
    def __init__(self) -> None:
        self.strings: dict[str, str] = {f"voice_catalog:{LANG}": json.dumps(EN)}
        self.hashes: dict[str, dict[str, str]] = {
            f"translationRoom:{MEETING}:languages": {STAND_IN: "vi", NATIVE: "en"},
        }
        self.streams: dict[str, list[dict[str, str]]] = {}
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.hash_reads: list[str] = []
        self.unreadable: str | None = None
        self.redis = _RawRedis(self)

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        self.hash_reads.append(key)
        if self.unreadable and key == self.unreadable:
            raise ConnectionError("redis is down")
        # Bytes, as the real client returns them.
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}

    async def hget(self, key: str, field: str) -> bytes | None:
        self.hash_reads.append(key)
        if self.unreadable and key == self.unreadable:
            raise ConnectionError("redis is down")
        value = self.hashes.get(key, {}).get(field)
        return None if value is None else value.encode()

    async def hset(self, key: str, field: str, value: bytes | str) -> None:
        self.hashes.setdefault(key, {})[field] = (
            value.decode() if isinstance(value, bytes) else value
        )

    async def expire(self, key: str, ttl_seconds: int) -> None: ...

    async def publish(self, stream: str, data: dict[str, Any]) -> str:
        self.published.append((stream, data))
        return "1-0"

    async def publish_system_event(self, **kwargs: Any) -> None:
        self.published.append(("system_event", kwargs))

    # ── what the backend and the desktop do ──────────────────────────────────────────────
    def consent(self, name: str) -> None:
        self.hashes.setdefault(consents_key(MEETING), {})[consent_field(name) or ""] = (
            "2026-10-03T08:00:00Z"
        )

    def withdraw(self, name: str) -> None:
        self.hashes.get(consents_key(MEETING), {}).pop(consent_field(name) or "", None)

    def hint(self, name: str, spoken_at_ms: int) -> None:
        self.streams.setdefault(far_speaker_hints_key(MEETING), []).append(
            {"name": name, "t_ms": str(spoken_at_ms + HINT_LAG_MS), "source": "meet_caption"}
        )

    def delete_requests(self) -> list[dict[str, Any]]:
        return [data for stream, data in self.published if stream == _VOICE_DELETE_STREAM]


class _FakeCartesia:
    def __init__(self) -> None:
        self.cloned: list[tuple[bytes, str, str]] = []
        self.deleted: list[str] = []
        self.renamed: list[str] = []
        self.before_answer: Any = None

    async def clone_voice(self, wav: bytes, label: str, language: str) -> str:
        self.cloned.append((wav, label, language))
        if self.before_answer is not None:
            self.before_answer()
        return f"far-voice-{len(self.cloned)}"

    async def delete_voice(self, voice_id: str) -> bool:
        self.deleted.append(voice_id)
        return True

    async def rename_voice(self, voice_id: str, _name: str) -> bool:
        self.renamed.append(voice_id)
        return True


def _worker(*, enabled: bool = True, **overrides: Any) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(
        far_speaker_clone_enabled=enabled,
        far_speaker_clone_hint_wait_ms=overrides.pop("wait_ms", 0),
        voice_clone_min_seconds=10.0,
        **overrides,
    )
    worker.logger = MagicMock()
    worker.worker_name = "tts"
    worker._consumer_name = "tts-test"
    worker._running = True
    worker._route_states = {}
    worker._room_routes = {MEETING: [{"SourceUserId": NATIVE, "VoiceCloneEnabled": True}]}
    worker.redis = _FakeRedis()  # type: ignore[assignment]
    worker.cartesia = _FakeCartesia()  # type: ignore[assignment]
    worker.chosen_dub_voice = lambda _m, _s: None  # type: ignore[method-assign]

    async def _allowed(_meeting: str) -> bool:
        return True

    async def _min_seconds() -> float:
        return float(worker.tts_settings.voice_clone_min_seconds)

    async def _margin() -> float:
        return float(worker.tts_settings.voice_clone_upgrade_margin)

    worker._voice_clone_allowed = _allowed  # type: ignore[method-assign]
    worker._clone_min_seconds = _min_seconds  # type: ignore[method-assign]
    worker._clone_upgrade_margin = _margin  # type: ignore[method-assign]
    return worker


def _redis(worker: TTSWorker) -> _FakeRedis:
    return worker.redis  # type: ignore[return-value]


def _cartesia(worker: TTSWorker) -> _FakeCartesia:
    return worker.cartesia  # type: ignore[return-value]


def _chunk(
    speaker: str = STAND_IN, pcm: bytes | None = None, *, speech_ms: int = 0
) -> AudioChunkMessage:
    return AudioChunkMessage(
        meeting_id=MEETING,
        speaker_id=speaker,
        chunk_index=0,
        audio_data=_varied() if pcm is None else pcm,
        language="vi",
        sample_rate=SAMPLE_RATE,
        speech_ms=speech_ms,
        timestamp_ms=int(time.time() * 1000),
    )


def _mid(chunk: AudioChunkMessage) -> int:
    """A moment in the middle of the chunk's audio, in the clock hints are stamped in."""
    return chunk.timestamp_ms - far_speaker_clone.chunk_pcm_ms(chunk) // 2


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _capture(worker: TTSWorker, chunk: AudioChunkMessage) -> None:
    await worker._capture_far_chunk(chunk)
    await _settle()


async def _say(
    worker: TTSWorker,
    name: str | None = AN,
    *,
    confidence: float | None = 1.0,
    duration_ms: int = 2000,
    segment: str | None = None,
    speaker: str = STAND_IN,
    lang: str = LANG,
) -> tuple[str, str]:
    """One dubbed sentence: the (voice_id, voice_type) of its default variant."""
    _say.count += 1  # type: ignore[attr-defined]
    variants = await worker._resolve_voice_variants(
        MEETING,
        speaker,
        lang,
        far_speaker_name=name,
        far_speaker_confidence=confidence,
        far_segment_id=segment or f"seg-{_say.count}",  # type: ignore[attr-defined]
        far_duration_ms=duration_ms,
    )
    voice_id, voice_type, voice_key = variants[0]
    assert voice_key == ""
    return voice_id, voice_type


_say.count = 0  # type: ignore[attr-defined]


def _with_clone(worker: TTSWorker, name: str = AN, voice_id: str = "far-voice-an") -> None:
    _redis(worker).consent(name)
    _redis(worker).hashes.setdefault(clones_key(MEETING), {})[consent_field(name) or ""] = voice_id


def _logged(worker: TTSWorker) -> str:
    logger: Any = worker.logger
    return repr(logger.mock_calls)


# ── fold and consent_field: the contract's vectors ──────────────────────────────────────────


class TestTheContractVectors:
    @pytest.mark.parametrize(
        ("name", "folded"),
        [
            ("Trần  An", "trần an"),
            ("  TÚ Huỳnh ", "tú huỳnh"),
            ("Ａｎ Ｎｇｕｙｅｎ", "an nguyen"),
            ("", ""),
            (" \t\n ", ""),
        ],
    )
    def test_fold(self, name: str, folded: str) -> None:
        assert fold(name) == folded

    def test_consent_field_is_sha256_of_the_stand_in_id_and_the_folded_name(self) -> None:
        expected = hashlib.sha256(
            "00000000-0000-0000-0000-00000000b21d:trần an".encode()
        ).hexdigest()
        assert consent_field("Trần  An") == expected
        assert len(expected) == 64 and expected == expected.lower()

    def test_two_spellings_of_one_name_are_one_field(self) -> None:
        assert consent_field("Trần  An") == consent_field(" trần an ")
        assert consent_field("Ａｎ Ｎｇｕｙｅｎ") == consent_field("an nguyen")

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_an_empty_name_has_no_field_to_consent_under(self, name: str | None) -> None:
        assert consent_field(name) is None

    def test_the_keys_are_the_contracts(self) -> None:
        assert consents_key(MEETING) == f"translationRoom:{MEETING}:far_speaker_clone_consents"


# ── flag off ────────────────────────────────────────────────────────────────────────────────


class TestFlagOff:
    def test_on_is_the_default_because_consent_is_the_real_switch(self) -> None:
        # The per-room consent hash decides who is cloned; this flag is only the kill switch.
        assert TTSSettings().far_speaker_clone_enabled is True

    def test_the_env_name_is_the_contracts_and_turns_it_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TTS_FAR_SPEAKER_CLONE_ENABLED", "false")
        assert TTSSettings().far_speaker_clone_enabled is False

    async def test_a_consented_clone_is_not_spoken(self) -> None:
        off = _worker(enabled=False)
        _with_clone(off)
        untouched = _worker(enabled=False)

        for _ in range(CLONE_MIN_STREAK + 2):
            assert await _say(off) == await _say(untouched)
        assert (await _say(off))[1] == "default"
        # Not merely unused: never asked for, and no state ever made.
        assert consents_key(MEETING) not in _redis(off).hash_reads
        assert clones_key(MEETING) not in _redis(off).hash_reads
        assert getattr(off, "_far_clone_state_impl", None) is None

    async def test_stand_in_audio_goes_where_it_always_went(self) -> None:
        worker = _worker(enabled=False)
        _redis(worker).consent(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))
        states: list[str] = []

        async def _note(_key: tuple[str, str], reason: str, **_kwargs: Any) -> None:
            states.append(reason)

        worker._note_clone_state = _note  # type: ignore[method-assign]
        await _run_loop(worker, [chunk])

        # The ordinary consent gate refused the seat, as it does today.
        assert states and all(state != "capturing" for state in states)
        assert _cartesia(worker).cloned == []
        assert getattr(worker, "_far_clone_state_impl", None) is None


async def _run_loop(worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
    """Drive the real capture loop over a scripted audio:chunks stream."""
    redis = _redis(worker)

    async def consume(**_kwargs: Any) -> Any:
        for index, chunk in enumerate(chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        worker._running = False

    async def _housekeep(*_args: Any, **_kwargs: Any) -> None: ...

    redis.consume = consume  # type: ignore[attr-defined]
    worker._housekeep_side_group = _housekeep  # type: ignore[method-assign]
    await worker._consume_audio_for_cloning()
    await _settle()


# ── capture ─────────────────────────────────────────────────────────────────────────────────


class TestCaptureGate:
    async def test_a_certain_consented_long_chunk_is_cloned_under_the_names_own_key(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))

        await _capture(worker, chunk)

        cloned = _cartesia(worker).cloned
        assert len(cloned) == 1
        assert _redis(worker).hashes[clones_key(MEETING)] == {AN_FIELD: "far-voice-1"}
        # Never the stand-in seat's own voice, which would be spoken for everyone over there.
        assert f"voice:{MEETING}:{STAND_IN}" not in _redis(worker).hashes
        assert await worker._get_voice_id(MEETING, STAND_IN) is None
        # Sweepable, and never handed to AuthService as somebody's profile.
        assert cloned[0][1].startswith("speaker-")
        assert _cartesia(worker).renamed == []
        assert _redis(worker).published == []

    async def test_nobody_named_it(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)

        await _capture(worker, _chunk())

        assert _cartesia(worker).cloned == []
        assert worker._far_clone_state().buffers == {}

    async def test_a_hand_over_inside_the_chunk_is_not_certain(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        _redis(worker).consent(LAN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk) - 1000)
        _redis(worker).hint(LAN, _mid(chunk) + 1000)

        await _capture(worker, chunk)

        assert _cartesia(worker).cloned == []
        assert worker._far_clone_state().buffers == {}

    async def test_a_hint_beside_the_chunk_is_not_inside_it(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, chunk.timestamp_ms + 400)

        await _capture(worker, chunk)

        assert worker._far_clone_state().buffers == {}

    async def test_the_person_did_not_consent(self) -> None:
        worker = _worker()
        _redis(worker).consent(LAN)  # somebody else did
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))

        await _capture(worker, chunk)

        assert _cartesia(worker).cloned == []
        assert worker._far_clone_state().buffers == {}

    async def test_consent_that_cannot_be_read_is_not_consent(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        _redis(worker).unreadable = consents_key(MEETING)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))

        await _capture(worker, chunk)

        assert _cartesia(worker).cloned == []
        assert worker._far_clone_state().buffers == {}

    @pytest.mark.parametrize(
        ("seconds", "speech_ms"),
        [
            (1.4, 0),  # short audio, ingress did not say how much was speech
            (3.0, 900),  # long audio, but VAD called less than 1.5 s of it speech
        ],
    )
    async def test_under_a_second_and_a_half_of_speech_is_not_even_held(
        self, seconds: float, speech_ms: int
    ) -> None:
        worker = _worker(wait_ms=10_000)
        _redis(worker).consent(AN)
        pcm = _varied()[: int(SAMPLE_RATE * seconds) * 2]

        worker._hold_far_chunk(_chunk(pcm=pcm, speech_ms=speech_ms))

        assert not worker._far_clone_state().pending
        assert worker._far_clone_state().drainer is None

    async def test_short_certain_chunks_add_up_to_one_sample(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        pcm = _varied()
        half = len(pcm) // 2

        for part in (pcm[:half], pcm[half:]):
            chunk = _chunk(pcm=part)
            _redis(worker).hint(AN, _mid(chunk))
            await _capture(worker, chunk)

        (wav, _label, _language) = _cartesia(worker).cloned[0]
        assert wav.endswith(pcm)
        assert len(_cartesia(worker).cloned) == 1

    async def test_one_clone_per_name(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        for _ in range(3):
            chunk = _chunk()
            _redis(worker).hint(AN, _mid(chunk))
            await _capture(worker, chunk)

        assert len(_cartesia(worker).cloned) == 1
        assert worker._far_clone_state().buffers == {}

    async def test_two_people_get_two_clones_and_never_each_others(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        _redis(worker).consent(LAN)
        first = _chunk()
        _redis(worker).hint(AN, _mid(first))
        await _capture(worker, first)
        # Hints for An stay in the stream; the next chunk is a later window.
        await asyncio.sleep(0.01)
        second = _chunk(pcm=_varied()[: SAMPLE_RATE * 2 * 11])
        second = second.model_copy(update={"timestamp_ms": first.timestamp_ms + 30_000})
        _redis(worker).hint(LAN, _mid(second))
        worker._far_clone_tracker().forget(MEETING)
        await _capture(worker, second)

        assert _redis(worker).hashes[clones_key(MEETING)] == {
            AN_FIELD: "far-voice-1",
            LAN_FIELD: "far-voice-2",
        }

    async def test_the_platform_kill_switch_still_wins(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)

        async def _off(_meeting: str) -> bool:
            return False

        worker._voice_clone_allowed = _off  # type: ignore[method-assign]
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))
        await _capture(worker, chunk)

        assert _cartesia(worker).cloned == []
        assert worker._far_clone_state().buffers == {}


class TestLateHints:
    async def test_a_hint_written_after_the_chunk_arrived_still_names_it(self) -> None:
        worker = _worker(wait_ms=80)
        _redis(worker).consent(AN)
        chunk = _chunk()

        worker._hold_far_chunk(chunk)
        # Nothing has named the chunk yet. Attributed now, it would be dropped.
        assert len(worker._far_clone_state().pending) == 1
        _redis(worker).hint(AN, _mid(chunk))
        drainer = worker._far_clone_state().drainer
        assert drainer is not None
        await drainer
        await _settle()

        assert len(_cartesia(worker).cloned) == 1
        assert not worker._far_clone_state().pending

    async def test_a_chunk_no_hint_ever_names_is_let_go(self) -> None:
        worker = _worker(wait_ms=30)
        _redis(worker).consent(AN)

        worker._hold_far_chunk(_chunk())
        drainer = worker._far_clone_state().drainer
        assert drainer is not None
        await drainer

        assert _cartesia(worker).cloned == []
        assert not worker._far_clone_state().pending
        assert worker._far_clone_state().buffers == {}

    async def test_a_chunk_that_already_waited_in_the_stream_is_not_held_again(self) -> None:
        worker = _worker(wait_ms=60_000)
        chunk = _chunk().model_copy(update={"timestamp_ms": int(time.time() * 1000) - 120_000})

        worker._hold_far_chunk(chunk)

        assert worker._far_clone_state().pending[0].due <= time.monotonic()
        drainer = worker._far_clone_state().drainer
        assert drainer is not None
        await drainer

    async def test_the_queue_is_bounded(self) -> None:
        worker = _worker(wait_ms=60_000)
        for _ in range(far_speaker_clone.MAX_PENDING_CHUNKS + 5):
            worker._hold_far_chunk(_chunk(pcm=_varied()[: SAMPLE_RATE * 2 * 2]))

        state = worker._far_clone_state()
        assert len(state.pending) == far_speaker_clone.MAX_PENDING_CHUNKS
        assert state.drainer is not None
        state.drainer.cancel()

    async def test_holding_the_stand_in_does_not_hold_anyone_else(self) -> None:
        """The real loop: a stand-in chunk is parked for a minute, the native speaker's chunk
        behind it is buffered and cloned straight away."""
        worker = _worker(wait_ms=60_000)
        natives: list[str] = []

        async def _clone_and_cache(_meeting: str, speaker: str, *_args: Any, **_kw: Any) -> str:
            natives.append(speaker)
            return ""

        async def _no_voice(_meeting: str, _speaker: str) -> str | None:
            return None

        async def _note(*_args: Any, **_kwargs: Any) -> None: ...

        worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
        worker._get_voice_id = _no_voice  # type: ignore[method-assign]
        worker._note_clone_state = _note  # type: ignore[method-assign]

        await _run_loop(worker, [_chunk(), _chunk(NATIVE)])

        assert natives == [NATIVE]
        state = worker._far_clone_state()
        assert [held.chunk.speaker_id for held in state.pending] == [STAND_IN]
        assert state.drainer is not None
        state.drainer.cancel()


# ── use ─────────────────────────────────────────────────────────────────────────────────────


class TestUseGate:
    async def test_the_clone_is_spoken_from_the_third_certain_sentence(self) -> None:
        worker = _worker()
        _with_clone(worker)

        first, second, third, fourth = [await _say(worker) for _ in range(4)]

        assert first[1] == second[1] == "default"
        assert first[0] == second[0], "the WT-932 stock voice, the same one each time"
        assert third == fourth == ("far-voice-an", "cloned")

    async def test_before_the_third_it_is_exactly_the_wt932_voice(self) -> None:
        worker = _worker()
        _with_clone(worker)
        flag_off = _worker(enabled=False)

        assert await _say(worker) == await _say(flag_off)

    @pytest.mark.parametrize(
        "breaker",
        [
            {"confidence": 0.667},
            {"confidence": None},
            {"duration_ms": 1499},
        ],
    )
    async def test_a_sentence_that_does_not_qualify_resets_the_streak(
        self, breaker: dict[str, Any]
    ) -> None:
        worker = _worker()
        _with_clone(worker)
        for _ in range(CLONE_MIN_STREAK):
            await _say(worker)
        assert (await _say(worker))[1] == "cloned"

        assert (await _say(worker, **breaker))[1] == "default"
        # And the count starts again from nothing.
        assert (await _say(worker))[1] == "default"
        assert (await _say(worker))[1] == "default"
        assert (await _say(worker))[1] == "cloned"

    async def test_exactly_a_second_and_a_half_at_certainty_qualifies(self) -> None:
        worker = _worker()
        _with_clone(worker)

        voices = [await _say(worker, duration_ms=1500) for _ in range(CLONE_MIN_STREAK)]

        assert voices[-1][1] == "cloned"

    async def test_consent_is_asked_for_every_sentence(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).withdraw(AN)

        voices = [await _say(worker) for _ in range(CLONE_MIN_STREAK + 1)]

        assert {voice_type for _voice, voice_type in voices} == {"default"}

    async def test_consent_without_a_clone_yet_is_the_stock_voice(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)

        voices = [await _say(worker) for _ in range(CLONE_MIN_STREAK + 1)]

        assert {voice_type for _voice, voice_type in voices} == {"default"}

    async def test_the_streak_is_already_there_when_the_clone_arrives(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        for _ in range(CLONE_MIN_STREAK):
            await _say(worker)

        _with_clone(worker)

        assert (await _say(worker))[1] == "cloned"

    async def test_one_name_never_speaks_in_anothers_clone(self) -> None:
        worker = _worker()
        _with_clone(worker, AN, "far-voice-an")
        for _ in range(CLONE_MIN_STREAK):
            await _say(worker, AN)

        lan = [await _say(worker, LAN) for _ in range(CLONE_MIN_STREAK + 1)]

        assert all(voice != "far-voice-an" and kind == "default" for voice, kind in lan)
        # And Lan speaking in between did not cost An their streak.
        assert await _say(worker, AN) == ("far-voice-an", "cloned")

    async def test_one_sentence_in_three_languages_is_one_sentence(self) -> None:
        worker = _worker()
        _with_clone(worker)
        for lang in ("en", "ja", "ko"):
            _redis(worker).strings[f"voice_catalog:{lang}"] = json.dumps(EN)

        first = [await _say(worker, lang=lang) for lang in ("en", "ja", "ko")]

        assert {voice_type for _voice, voice_type in first} == {"default"}

    async def test_a_redelivered_sentence_is_not_counted_twice(self) -> None:
        worker = _worker()
        _with_clone(worker)

        voices = [await _say(worker, segment="seg-same") for _ in range(CLONE_MIN_STREAK + 1)]

        assert {voice_type for _voice, voice_type in voices} == {"default"}

    async def test_a_listeners_pick_is_still_rendered_beside_the_clone(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).hashes[f"translationRoom:{MEETING}:languages"]["listener-1"] = LANG
        _redis(worker).hashes[f"translationRoom:{MEETING}:voice_preferences"] = {
            "listener-1": "en-fox"
        }
        for _ in range(CLONE_MIN_STREAK - 1):
            await _say(worker)

        variants = await worker._resolve_voice_variants(
            MEETING,
            STAND_IN,
            LANG,
            far_speaker_name=AN,
            far_speaker_confidence=1.0,
            far_segment_id="seg-pick",
            far_duration_ms=2000,
        )

        assert variants[0] == ("far-voice-an", "cloned", "")
        assert [kind for _voice, kind, _key in variants[1:]] == ["preference"]

    async def test_an_unreadable_hash_is_the_stock_voice_and_deletes_nothing(self) -> None:
        worker = _worker()
        _with_clone(worker)
        for _ in range(CLONE_MIN_STREAK):
            await _say(worker)
        _redis(worker).unreadable = consents_key(MEETING)

        assert (await _say(worker))[1] == "default"
        assert _redis(worker).delete_requests() == []
        assert AN_FIELD in _redis(worker).hashes[clones_key(MEETING)]


class TestNativeSpeakersAreUntouched:
    async def test_a_native_speaker_resolves_exactly_as_with_the_flag_off(self) -> None:
        on, off = _worker(), _worker(enabled=False)
        _with_clone(on)

        for _ in range(CLONE_MIN_STREAK + 1):
            # Stray far-speaker fields on a native speaker are ignored, as in WT-932.
            assert await _say(on, speaker=NATIVE) == await _say(off, speaker=NATIVE)
        assert consents_key(MEETING) not in _redis(on).hash_reads
        assert worker_streaks(on) == {}

    async def test_a_native_speakers_own_clone_still_wins(self) -> None:
        worker = _worker()
        _redis(worker).hashes[f"voice:{MEETING}:{NATIVE}"] = {"voice_id": "native-clone"}

        assert await _say(worker, speaker=NATIVE) == ("native-clone", "cloned")

    async def test_a_native_chunk_is_never_held(self) -> None:
        worker = _worker(wait_ms=60_000)
        cloned: list[str] = []

        async def _clone_and_cache(_meeting: str, speaker: str, *_args: Any, **kw: Any) -> str:
            assert "far_field" not in kw
            cloned.append(speaker)
            return ""

        async def _no_voice(_meeting: str, _speaker: str) -> str | None:
            return None

        async def _note(*_args: Any, **_kwargs: Any) -> None: ...

        worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
        worker._get_voice_id = _no_voice  # type: ignore[method-assign]
        worker._note_clone_state = _note  # type: ignore[method-assign]

        await _run_loop(worker, [_chunk(NATIVE)])

        assert cloned == [NATIVE]
        assert getattr(worker, "_far_clone_state_impl", None) is None


def worker_streaks(worker: TTSWorker) -> dict[Any, Any]:
    state = getattr(worker, "_far_clone_state_impl", None)
    return {} if state is None else state.streaks


# ── withdrawal ──────────────────────────────────────────────────────────────────────────────


class TestWithdrawal:
    async def test_the_next_sentence_is_stock_and_the_voice_model_is_deleted(self) -> None:
        worker = _worker()
        _with_clone(worker)
        stock = (await _say(worker))[0]
        for _ in range(CLONE_MIN_STREAK - 1):
            await _say(worker)
        assert (await _say(worker))[1] == "cloned"

        _redis(worker).withdraw(AN)

        assert await _say(worker) == (stock, "default")
        assert AN_FIELD not in _redis(worker).hashes[clones_key(MEETING)]
        requests = _redis(worker).delete_requests()
        assert [request["voice_id"] for request in requests] == ["far-voice-an"]

        # The existing deletion path, end to end: the request this wrote is one the delete
        # consumer carries out at the vendor.
        await worker._handle_voice_delete_request(
            {k.encode(): str(v).encode() for k, v in requests[0].items()}
        )
        assert _cartesia(worker).deleted == ["far-voice-an"]

    async def test_it_is_deleted_once_however_many_paths_notice(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).withdraw(AN)

        await worker._reconcile_far_clones(MEETING)
        await worker._withdraw_far_clone(MEETING, AN_FIELD)
        for _ in range(CLONE_MIN_STREAK + 1):
            await _say(worker)

        assert len(_redis(worker).delete_requests()) == 1

    async def test_someone_who_withdraws_and_says_nothing_more_is_still_deleted(self) -> None:
        worker = _worker()
        _with_clone(worker, AN, "far-voice-an")
        _with_clone(worker, LAN, "far-voice-lan")
        _redis(worker).withdraw(AN)

        await worker._reconcile_far_clones(MEETING)

        assert _redis(worker).hashes[clones_key(MEETING)] == {LAN_FIELD: "far-voice-lan"}
        assert [r["voice_id"] for r in _redis(worker).delete_requests()] == ["far-voice-an"]

    async def test_a_buffer_being_assembled_is_dropped(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        chunk = _chunk(pcm=_varied()[: SAMPLE_RATE * 2 * 4])  # 4 s: held, not yet a sample
        _redis(worker).hint(AN, _mid(chunk))
        await _capture(worker, chunk)
        assert (MEETING, AN_FIELD) in worker._far_clone_state().buffers

        _redis(worker).withdraw(AN)
        await worker._reconcile_far_clones(MEETING)

        assert worker._far_clone_state().buffers == {}
        assert worker._far_clone_state().buffer_seconds == {}

    async def test_their_next_chunk_is_not_buffered_and_takes_the_clone_with_it(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).withdraw(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))

        await _capture(worker, chunk)

        assert worker._far_clone_state().buffers == {}
        assert _cartesia(worker).cloned == []
        assert [r["voice_id"] for r in _redis(worker).delete_requests()] == ["far-voice-an"]

    async def test_withdrawing_while_the_vendor_is_still_cloning(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        _cartesia(worker).before_answer = lambda: _redis(worker).withdraw(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))

        await _capture(worker, chunk)

        # The voice exists at the vendor for a moment, is never reachable, and is deleted.
        assert _redis(worker).hashes.get(clones_key(MEETING), {}) == {}
        assert [r["voice_id"] for r in _redis(worker).delete_requests()] == ["far-voice-1"]
        assert (MEETING, AN_FIELD) not in worker._far_clone_state().in_flight

    async def test_a_redis_blip_is_not_a_withdrawal(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).unreadable = consents_key(MEETING)

        await worker._reconcile_far_clones(MEETING)

        assert _redis(worker).hashes[clones_key(MEETING)] == {AN_FIELD: "far-voice-an"}
        assert _redis(worker).delete_requests() == []

    async def test_consenting_again_starts_from_nothing(self) -> None:
        worker = _worker()
        _with_clone(worker)
        _redis(worker).withdraw(AN)
        await worker._reconcile_far_clones(MEETING)

        _redis(worker).consent(AN)
        voices = [await _say(worker) for _ in range(CLONE_MIN_STREAK + 1)]

        assert {voice_type for _voice, voice_type in voices} == {"default"}


# ── privacy and housekeeping ────────────────────────────────────────────────────────────────


class TestTheNameStaysOut:
    async def test_no_log_line_and_no_redis_entry_carries_the_name(self) -> None:
        worker = _worker()
        _redis(worker).consent(AN)
        chunk = _chunk()
        _redis(worker).hint(AN, _mid(chunk))
        await _capture(worker, chunk)
        for _ in range(CLONE_MIN_STREAK + 1):
            await _say(worker)
        _redis(worker).withdraw(AN)
        await _say(worker)

        logged = _logged(worker)
        assert "far_speaker_voice_cloned" in logged and "far_speaker_clone_withdrawn" in logged
        for spelling in (AN, fold(AN), "Trần", "trần"):
            assert spelling not in logged
        # The hash, shortened; never the whole field either.
        assert AN_FIELD[:12] in logged and AN_FIELD not in logged

        hints = far_speaker_hints_key(MEETING)  # the desktop's own stream, not ours
        written = repr({key: value for key, value in _redis(worker).hashes.items()}) + repr(
            _redis(worker).published
        )
        assert hints not in written
        for spelling in (AN, fold(AN), "Trần", "trần"):
            assert spelling not in written

    async def test_a_room_that_ends_leaves_nothing_in_memory(self) -> None:
        worker = _worker(wait_ms=60_000)
        worker._translation_active = {}
        worker._paused_rooms = set()
        worker._key_locks = {}
        _redis(worker).consent(AN)
        worker._hold_far_chunk(_chunk(pcm=_varied()[: SAMPLE_RATE * 2 * 2]))
        short = _chunk(pcm=_varied()[: SAMPLE_RATE * 2 * 4])
        _redis(worker).hint(AN, _mid(short))
        await _capture(worker, short)
        await _say(worker)
        state = worker._far_clone_state()
        assert state.pending and state.buffers and state.streaks and state.watched

        try:
            worker._cleanup_room(MEETING)
        finally:
            if state.drainer is not None:
                state.drainer.cancel()

        assert not state.pending
        assert state.buffers == {} and state.buffer_seconds == {} and state.buffer_lang == {}
        assert state.streaks == {} and state.watched == set()
