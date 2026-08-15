"""A clip the quality gate turns away must say so.

THE OBSERVATION (2026-08-15, meeting 01a0033f)
    Two speakers, four minutes, 46 dubbed sentences — every one on a stock catalog voice. The
    tester's report was "nói liên tục 2 người thì ko có nhận voice clone được".

    Production logs for that meeting contained, in total:

        voice_clone_state            0 lines
        voice_clone_sample_accepted  0 lines

    Nothing. Consent had passed, so the gate that reports `not_opted_in` / `routes_unknown` was
    never reached. The clip was refused by `assess_clone_sample`, and THAT branch returned in
    silence — buffer slides, try again, forever, for the whole meeting.

WHY THAT IS THE BUG AND NOT JUST A MISSING LOG
    From outside, "every clip failed the bar" and "cloning is switched off" produce byte-for-byte
    identical output: none. tts_worker/worker.py's own header says this in as many words — "Every
    exit before the clone call returned in silence" — as the reason a dead feature survived to be
    found by a tester rather than by a log. The quality gate was added later and reintroduced the
    same blind spot at a new exit.

    The reason matters as much as the fact. "too quiet" is a microphone, "too little speech" is a
    room, "clipped" is a gain setting. Three different conversations with the user, and none of
    them can begin from silence.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import _varied
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000
SECONDS = 12.0


class _ScriptedRedis:
    def __init__(self, worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
        self._worker = worker
        self._chunks = chunks

    async def consume(self, **_kwargs: Any) -> Any:
        for index, chunk in enumerate(self._chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self._worker._running = False


def _worker(**overrides: Any) -> tuple[TTSWorker, list[tuple[tuple[str, str], str]]]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._route_states = {}
    # Consent granted, so the gate above the quality check cannot be what reports anything —
    # this is the production shape, and it is what left the whole path silent.
    worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
    worker._consumer_name = "test"
    worker.worker_name = "tts"
    worker._running = True

    noted: list[tuple[tuple[str, str], str]] = []

    # Async and kwargs-tolerant since WT-420: _note_clone_state now publishes as well as logs,
    # and carries the capture metrics the meeting UI draws its progress bar from.
    async def _note(key: tuple[str, str], reason: str, **_metrics: Any) -> None:
        noted.append((key, reason))

    worker._note_clone_state = _note  # type: ignore[method-assign]

    async def _clone_and_cache(
        _meeting: str, _speaker: str, _audio: bytes, _language: str = "en"
    ) -> None:
        pass

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return None

    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    return worker, noted


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


def _near_silence() -> bytes:
    """Room tone. Fails MIN_RMS — the "too quiet" verdict."""
    n = int(SAMPLE_RATE * SECONDS)
    return (np.zeros(n, dtype=np.float64) * _INT16).astype(np.int16).tobytes()


_INT16 = 32767.0


async def _run(worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
    worker.redis = _ScriptedRedis(worker, chunks)  # type: ignore[assignment]
    await worker._consume_audio_for_cloning()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_rejected_clip_reports_why_instead_of_nothing() -> None:
    worker, noted = _worker(voice_clone_min_seconds=10.0)

    await _run(worker, _chunks(_near_silence()))

    assert noted, (
        "A clip refused by the quality gate reported nothing at all. That is the state "
        "production was in for a whole meeting: no clone, and no way to tell that from the "
        "feature being switched off."
    )
    # Searched rather than indexed: since WT-420 the buffer reports "capturing" on the way to a
    # verdict, so the rejection is no longer the first thing noted. What this test is about — that
    # a refused clip says WHY — is unchanged.
    verdicts = [(key, reason) for key, reason in noted if reason.startswith("clip_rejected:")]
    assert verdicts, f"No verdict was reported at all; only saw {[r for _k, r in noted]}"
    key, reason = verdicts[0]
    assert key == ("m1", "s1")
    assert "quiet" in reason, (
        f"The verdict has to name WHICH bar was missed, not merely that one was; got {reason!r}"
    )


@pytest.mark.asyncio
async def test_an_accepted_clip_is_not_reported_as_rejected() -> None:
    """The new branch must not fire on the happy path — it would make every good clone look
    like a failure, which is the same instrument lying in the other direction."""
    worker, noted = _worker(voice_clone_min_seconds=10.0)

    await _run(worker, _chunks(_varied()))

    rejections = [reason for _key, reason in noted if reason.startswith("clip_rejected:")]
    assert not rejections, f"A clip good enough to clone was reported as refused: {rejections}"
    assert any(reason == "cloning" for _key, reason in noted), (
        "an accepted clip should still report that it is cloning"
    )
