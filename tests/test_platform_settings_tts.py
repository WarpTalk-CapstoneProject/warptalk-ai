"""The TTS worker reads the voice-clone thresholds and `flags.voice_clone` live.

The capture loop is driven for real (`_consume_audio_for_cloning`), with a scripted audio stream
and the platform settings hash beside it. Each test changes the value inside the test and moves
the reader's clock past its TTL; the worker is never rebuilt.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared import platform_settings as ps
from shared.config import TTSSettings, WorkerSettings
from shared.platform_settings import PLATFORM_HASH, PlatformSettings
from shared.schemas import AudioChunkMessage
from tests.conftest import FakeClock
from tests.test_clone_pitch_coverage import _flat, _varied
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000
MEETING = "m1"
WS_IN = "9b2d7c1e-8a4f-4e3b-b5d6-1c2e3f4a5b6c"  # flags.voice_clone bucket 61


class _Redis:
    """Scripted audio for the capture loop, plus the settings hash, the room projection and the
    live-clone cache `_get_voice_id` reads."""

    def __init__(self, worker: TTSWorker) -> None:
        self._worker = worker
        self.chunks: list[AudioChunkMessage] = []
        self.platform: dict[str, str] = {ps.VERSION_FIELD: "1"}
        self.values: dict[str, str] = {}
        self.voice_cache: dict[str, str] = {}

    def put(self, key: str, value: Any) -> None:
        self.platform[key] = json.dumps(value)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.platform) if key == PLATFORM_HASH else {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def hget(self, key: str, field: str) -> str | None:
        return self.voice_cache.get(key) if field == "voice_id" else None

    async def consume(self, **_kwargs: Any) -> Any:
        chunks, self.chunks = self.chunks, []
        for index, chunk in enumerate(chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self._worker._running = False


class _Harness:
    def __init__(self, clock: FakeClock, **env: Any) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.settings = WorkerSettings()
        worker.tts_settings = TTSSettings(**env)
        worker.logger = MagicMock()
        worker._route_states = {}
        worker._room_routes = {
            MEETING: [
                {"SourceUserId": speaker, "VoiceCloneEnabled": True}
                for speaker in ("s1", "s2", "s3", "s4")
            ]
        }
        worker._consumer_name = "test"
        worker.worker_name = "tts"
        self.worker = worker
        self.redis = _Redis(worker)
        worker.redis = self.redis  # type: ignore[assignment]
        worker._platform_settings = PlatformSettings(self.redis, clock=clock)  # type: ignore[arg-type]

        self.cloned: list[str] = []
        self.states: list[tuple[str, str, dict[str, Any]]] = []
        cloned_voices: dict[str, str] = {}

        async def _clone_and_cache(
            _meeting: str, speaker: str, _audio: bytes, *_args: Any, **_kwargs: Any
        ) -> None:
            self.cloned.append(speaker)
            cloned_voices[speaker] = f"voice-{speaker}"

        async def _note(key: tuple[str, str], reason: str, **metrics: Any) -> None:
            self.states.append((key[1], reason, metrics))

        real_get_voice_id = worker._get_voice_id

        async def _get_voice_id(meeting: str, speaker: str) -> str | None:
            # The real gate (consent + platform flag), over clones made in this test.
            if speaker in cloned_voices:
                self.redis.voice_cache[f"voice:{meeting}:{speaker}"] = cloned_voices[speaker]
            return await real_get_voice_id(meeting, speaker)

        worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
        worker._note_clone_state = _note  # type: ignore[method-assign]
        worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]

    async def run(self, speaker: str, *clips: bytes) -> None:
        self.redis.chunks = [
            AudioChunkMessage(
                meeting_id=MEETING,
                speaker_id=speaker,
                chunk_index=index,
                audio_data=clip,
                language="vi",
                sample_rate=SAMPLE_RATE,
            )
            for index, clip in enumerate(clips)
        ]
        self.worker._running = True
        await self.worker._consume_audio_for_cloning()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    def required_seconds(self, speaker: str) -> list[float]:
        return [
            m["required_seconds"]
            for s, reason, m in self.states
            if s == speaker and reason == "capturing"
        ]

    def reasons(self, speaker: str) -> list[str]:
        return [reason for s, reason, _ in self.states if s == speaker]


# ── min_sample_seconds ──────────────────────────────────────────────────────────────────────


async def test_nothing_stored_means_the_env_threshold_applies(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=20.0)
    await h.run("s1", _varied())  # 12s < 20s
    assert h.cloned == []
    assert h.required_seconds("s1") == [20.0]


async def test_min_sample_seconds_is_read_live(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=20.0)
    h.redis.put(ps.VOICE_CLONE_MIN_SECONDS, 10)
    await h.run("s1", _varied())  # 12s >= 10s: cloned
    assert h.cloned == ["s1"]
    assert h.required_seconds("s1") == [10.0]

    h.redis.put(ps.VOICE_CLONE_MIN_SECONDS, 30)
    fake_clock.advance(11)
    await h.run("s2", _varied(), _varied())  # 24s < 30s
    assert h.cloned == ["s1"]
    assert h.required_seconds("s2") == [30.0, 30.0]


# ── upgrade_margin ──────────────────────────────────────────────────────────────────────────


async def test_upgrade_margin_is_read_live(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=10.0)

    # A margin of 1 makes any upgrade unsatisfiable: the narrow opening clip stays the voice.
    h.redis.put(ps.VOICE_CLONE_UPGRADE_MARGIN, 1.0)
    await h.run("s1", _flat(), _varied())
    assert h.cloned == ["s1"]
    assert "cloned_best_possible" in h.reasons("s1")

    # Back to 0: the same pair of clips now earns the upgrade.
    h.redis.put(ps.VOICE_CLONE_UPGRADE_MARGIN, 0.0)
    fake_clock.advance(11)
    await h.run("s2", _flat(), _varied())
    assert h.cloned == ["s1", "s2", "s2"]


async def test_upgrade_margin_falls_back_to_the_env(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=10.0, voice_clone_upgrade_margin=1.0)
    await h.run("s1", _flat(), _varied())
    assert h.cloned == ["s1"]


# ── flags.voice_clone ───────────────────────────────────────────────────────────────────────


async def test_the_flag_stops_new_clones_live(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=10.0)
    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": False})
    await h.run("s1", _varied())
    assert h.cloned == []
    assert h.reasons("s1") == ["disabled_by_platform"]

    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": True})
    fake_clock.advance(11)
    await h.run("s1", _varied())
    assert h.cloned == ["s1"]


async def test_the_flag_stops_a_cloned_voice_being_used_for_synthesis(
    fake_clock: FakeClock,
) -> None:
    h = _Harness(fake_clock)
    h.redis.voice_cache[f"voice:{MEETING}:s1"] = "voice-live"

    async def hashed(*_args: Any) -> str:
        return "catalog-voice"

    h.worker._hashed_default_voice_id = hashed  # type: ignore[method-assign]
    h.worker._get_explicit_voice_choices = MagicMock()  # never reached for a cloned speaker

    variants = await h.worker._resolve_voice_variants(MEETING, "s1", "en")
    assert variants[0][:2] == ("voice-live", "cloned")

    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": False})
    fake_clock.advance(11)

    async def no_choices(*_args: Any) -> set[str]:
        return set()

    h.worker._get_explicit_voice_choices = no_choices  # type: ignore[method-assign]
    variants = await h.worker._resolve_voice_variants(MEETING, "s1", "en")
    assert variants[0][:2] == ("catalog-voice", "default")

    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": True})
    fake_clock.advance(11)
    variants = await h.worker._resolve_voice_variants(MEETING, "s1", "en")
    assert variants[0][:2] == ("voice-live", "cloned")


@pytest.mark.parametrize(("workspace", "expected"), [(WS_IN, "voice-live"), (None, None)])
async def test_a_partial_rollout_is_decided_by_the_rooms_workspace(
    fake_clock: FakeClock, workspace: str | None, expected: str | None
) -> None:
    h = _Harness(fake_clock)
    h.redis.voice_cache[f"voice:{MEETING}:s1"] = "voice-live"
    if workspace is not None:
        h.redis.values[f"meeting:room:v2:{MEETING}"] = json.dumps({"WorkspaceId": workspace})
    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": True, "rolloutPercent": 62})  # bucket 61 is in

    assert await h.worker._get_voice_id(MEETING, "s1") == expected


async def test_the_flag_never_overrides_consent(fake_clock: FakeClock) -> None:
    h = _Harness(fake_clock, voice_clone_min_seconds=10.0)
    h.worker._room_routes = {}
    h.redis.put(ps.FLAG_VOICE_CLONE, {"enabled": True})
    await h.run("s1", _varied())
    assert h.cloned == []
    assert "disabled_by_platform" not in h.reasons("s1")
