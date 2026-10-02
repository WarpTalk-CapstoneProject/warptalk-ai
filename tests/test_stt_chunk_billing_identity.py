"""Every segment a chunk produces names the chunk, so billing can charge the audio once.

billing_worker used to skip early sentences and let the completed segment pay for the chunk. A
chunk whose every sentence went out early has no completed segment — the remainder is empty and
is never published — so nothing paid for it: 136 of 664 translated chunks on prod over three
days (2 Oct). The fix keys the charge on the chunk, which only works if the early path states
the chunk as well as the completed one does. That is what is pinned here; the charge itself is
pinned in test_billing_worker.TestAChunkIsChargedOnce.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from shared.config import WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker.model import TranscribedSegment
from tests.test_transcript_anchor_on_the_wire import _published_results, _worker


def _chunk(audio: bytes) -> dict[str, str]:
    return AudioChunkMessage(
        meeting_id="meeting-1",
        speaker_id="speaker-1",
        chunk_index=0,
        audio_data=audio,
        is_final_chunk=False,
    ).to_redis()


def _early_then(remainder: list[TranscribedSegment]) -> AsyncMock:
    async def fake_transcribe(*_args: Any, **kwargs: Any) -> list[TranscribedSegment]:
        await kwargs["on_early_segment"](
            TranscribedSegment(
                text="Hello there.", language="en", confidence=0.0, start_ms=0, end_ms=0
            )
        )
        return remainder

    return AsyncMock(side_effect=fake_transcribe)


@pytest.mark.asyncio
async def test_a_chunk_with_only_early_sentences_still_states_what_to_charge(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    worker = _worker(mock_redis_client, worker_settings)
    worker.model.transcribe = _early_then([])

    await worker.process(b"1790926897937-0", _chunk(sample_audio_bytes))

    (only,) = _published_results(mock_redis_client)
    assert only["is_early"] == "1"
    assert only["chunk_id"] == "1790926897937-0"
    # One second of audio, give or take the WAV header counted as samples.
    assert 1000 <= int(only["chunk_duration_ms"]) <= 1010


@pytest.mark.asyncio
async def test_early_and_completed_segments_name_the_same_chunk(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    worker = _worker(mock_redis_client, worker_settings)
    worker.model.transcribe = _early_then(
        [
            TranscribedSegment(
                text="How are you?", language="en", confidence=0.0, start_ms=0, end_ms=1000
            )
        ]
    )

    await worker.process(b"1790926897937-0", _chunk(sample_audio_bytes))

    published = _published_results(mock_redis_client)
    assert len(published) == 2
    assert {(data["chunk_id"], data["chunk_duration_ms"]) for data in published} == {
        ("1790926897937-0", published[0]["chunk_duration_ms"])
    }
