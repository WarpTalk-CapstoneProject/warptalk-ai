"""A clone may be replaced once, by a materially better sample (WT-371 #9).

The worker used to stop listening the moment it had any clone at all:

    if await self._get_voice_id(meeting_id, speaker_id):
        continue

so a speaker's voice was locked to whatever register they happened to open the meeting in. Raise
your voice, or crack it, and the clone stopped being you — which is the report.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import _flat, _varied
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000


class _ScriptedRedis:
    """Yields one batch of audio chunks, then stops the worker's outer loop.

    The `sleep(0)` after each chunk is load-bearing, not tidiness. The worker dispatches its clone
    with `asyncio.create_task`, and `_get_voice_id` only reports a voice once that task has run.
    Without yielding to the loop between chunks, EVERY chunk sees "not cloned yet" and the
    already-cloned branch — the entire subject of this file — is never reached. These tests
    initially passed with the old `if already cloned: continue` short-circuit put back, which is
    exactly what that looks like from the outside.
    """

    def __init__(self, worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
        self._worker = worker
        self._chunks = chunks

    async def consume(self, **_kwargs: Any) -> Any:
        for index, chunk in enumerate(self._chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self._worker._running = False


def _worker(clones: list[bytes], **overrides: Any) -> tuple[TTSWorker, list[bytes]]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
    worker._consumer_name = "test"
    worker.worker_name = "tts"
    worker._running = True

    # Cloned audio is captured rather than sent; what matters is HOW MANY clones happen and from
    # WHICH clip.
    cloned_from: list[bytes] = []
    voice_id: list[str] = []

    async def _clone_and_cache(
        _meeting: str,
        _speaker: str,
        audio: bytes,
        _language: str = "en",
        _sample_rate: int = 16000,
        _score: float | None = None,
    ) -> None:
        cloned_from.append(audio)
        voice_id.append("voice-1")

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return voice_id[-1] if voice_id else None

    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    return worker, cloned_from


def _chunks(pcm: bytes, count: int = 1) -> list[AudioChunkMessage]:
    return [
        AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=index,
            audio_data=pcm,
            language="vi",
            sample_rate=SAMPLE_RATE,
        )
        for index in range(count)
    ]


async def _run(worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
    worker.redis = _ScriptedRedis(worker, chunks)  # type: ignore[assignment]
    await worker._consume_audio_for_cloning()
    # Drain any clone task dispatched by the final chunk.
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_better_clip_replaces_the_first_clone() -> None:
    # A narrow opening ("alo alo"), then the speaker actually talks and covers their range.
    worker, cloned_from = _worker([], voice_clone_min_seconds=10.0)
    await _run(worker, _chunks(_flat()) + _chunks(_varied()))

    assert len(cloned_from) == 2, (
        "the wider second clip should have replaced the first clone; "
        f"got {len(cloned_from)} clone(s)"
    )


@pytest.mark.asyncio
async def test_an_equally_narrow_clip_does_not_churn_the_voice() -> None:
    # Re-cloning changes the voice people are listening to. It has to buy something.
    worker, cloned_from = _worker([], voice_clone_min_seconds=10.0)
    await _run(worker, _chunks(_flat()) + _chunks(_flat()))

    assert len(cloned_from) == 1


@pytest.mark.asyncio
async def test_upgrades_are_bounded() -> None:
    # Each re-clone is a paid Cartesia call and an audible change. One is enough to escape a bad
    # opening clip; unbounded would be a voice that keeps shifting under the listener.
    worker, cloned_from = _worker([], voice_clone_min_seconds=10.0, voice_clone_max_upgrades=1)
    await _run(worker, _chunks(_flat()) + _chunks(_varied()) + _chunks(_varied(90.0, 300.0)))

    assert len(cloned_from) <= 2


@pytest.mark.asyncio
async def test_revoking_consent_stops_cloning_entirely() -> None:
    # The consent gate must still win over any of the above.
    worker, cloned_from = _worker([], voice_clone_min_seconds=10.0)
    worker._room_routes = {}
    await _run(worker, _chunks(_varied()))

    assert cloned_from == []
