"""A speaker with no voice is cloned at once and improved as they talk (progressive clone).

Owner, 3 Oct 2026: "cứ phát giọng chưa đạt chuẩn trước ... rồi từ từ bắt giọng tiếp để nói giọng
càng chuẩn, không để chờ đủ 20s mới phát". Twenty accepted seconds before the first clone meant
the first half-minute to a minute of every new speaker went out in a stranger's voice.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import _varied
from tests.test_clone_upgrade import _run
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000
CHUNK_SECONDS = 1.5


def _worker(**overrides: Any) -> tuple[TTSWorker, list[dict[str, Any]]]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    overrides.setdefault("voice_clone_min_seconds", 20.0)
    overrides.setdefault("voice_clone_ladder_seconds", (3.0, 8.0, 45.0))
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
    worker._consumer_name = "test"
    worker.worker_name = "tts"
    worker._running = True

    clones: list[dict[str, Any]] = []
    voice_id: list[str] = []

    async def _clone_and_cache(
        _meeting: str,
        _speaker: str,
        audio: bytes,
        _language: str = "en",
        _sample_rate: int = 16000,
        _score: float | None = None,
        **kwargs: Any,
    ) -> str:
        clones.append({"seconds": len(audio) / 2 / SAMPLE_RATE, **kwargs})
        voice_id.append(f"voice-{len(clones)}")
        return ""

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return voice_id[-1] if voice_id else None

    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    return worker, clones


def _speech(seconds: float) -> list[AudioChunkMessage]:
    pcm = b""
    while len(pcm) < int(seconds * SAMPLE_RATE * 2):
        pcm += _varied()
    step = int(CHUNK_SECONDS * SAMPLE_RATE) * 2
    pcm = pcm[: int(seconds * SAMPLE_RATE) * 2]
    return [
        AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=index,
            audio_data=pcm[offset : offset + step],
            language="vi",
            sample_rate=SAMPLE_RATE,
        )
        for index, offset in enumerate(range(0, len(pcm), step))
    ]


@pytest.mark.asyncio
async def test_the_first_voice_does_not_wait_for_twenty_seconds() -> None:
    worker, clones = _worker()

    await _run(worker, _speech(4.5))

    assert len(clones) == 1
    assert clones[0]["seconds"] < 5.0
    # Provisional: used in this meeting, never carried into the next one.
    assert clones[0]["offer_carry_over"] is False


@pytest.mark.asyncio
async def test_each_rung_clones_from_a_longer_reference_of_the_same_speech() -> None:
    worker, clones = _worker()

    await _run(worker, _speech(48.0))

    lengths = [clone["seconds"] for clone in clones]
    assert len(lengths) == 4, lengths
    assert lengths == sorted(lengths)
    assert lengths[0] < 5.0 and lengths[-1] >= 45.0
    # Below min_seconds a rung is provisional; at or above it, it may become their voice.
    assert [clone["offer_carry_over"] for clone in clones] == [False, False, True, True]


@pytest.mark.asyncio
async def test_a_speaker_who_already_has_a_voice_does_not_climb() -> None:
    worker, clones = _worker()
    worker._room_routes = {
        "m1": [
            {
                "SourceUserId": "s1",
                "VoiceCloneEnabled": True,
                "SourceAutoCloneVoiceId": "carried-voice",
                "SourceAutoCloneScore": "0.95",
            }
        ]
    }

    await _run(worker, _speech(12.0))

    assert clones == []


@pytest.mark.asyncio
async def test_a_rung_still_at_the_vendor_is_not_raced_by_the_next() -> None:
    worker, clones = _worker()
    release = asyncio.Event()
    started: list[float] = []

    async def _slow_clone(*args: Any, **kwargs: Any) -> str:
        started.append(len(args[2]) / 2 / SAMPLE_RATE)
        await release.wait()
        return ""

    worker._clone_and_cache = _slow_clone  # type: ignore[method-assign]

    await _run(worker, _speech(12.0))

    assert len(started) == 1
    release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_the_first_dub_waits_briefly_for_the_first_voice() -> None:
    worker, _clones = _worker(voice_clone_first_wait_ms=1000)
    done = asyncio.Event()

    async def _clone() -> None:
        await asyncio.sleep(0.05)
        done.set()

    task = asyncio.create_task(_clone())
    worker._mark_first_clone_pending(("m1", "s1"), task)

    await worker._await_first_clone("m1", "s1")

    assert done.is_set()


@pytest.mark.asyncio
async def test_the_wait_for_the_first_voice_is_bounded() -> None:
    worker, _clones = _worker(voice_clone_first_wait_ms=50)
    task = asyncio.create_task(asyncio.sleep(5))
    worker._mark_first_clone_pending(("m1", "s1"), task)

    started = time.monotonic()
    await worker._await_first_clone("m1", "s1")

    assert time.monotonic() - started < 0.5
    assert not task.done()
    task.cancel()


@pytest.mark.asyncio
async def test_no_pending_clone_means_no_wait() -> None:
    worker, _clones = _worker()
    started = time.monotonic()
    await worker._await_first_clone("m1", "s1")
    assert time.monotonic() - started < 0.05
