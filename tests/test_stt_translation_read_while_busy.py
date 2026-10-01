"""A slow commit for one speaker must not hold another speaker's chunk unread.

tools/meeting_sim, three people in a meeting: a one-word backchannel's commit took 7.1s to come
back, and the next speaker's chunk — published 180ms after that read — sat unread the whole time,
because the consume loop read a batch and waited for all of it. While it waited, that speaker's
next turn started streaming, found the previous turn still uncommitted and threw the buffer away
(`previous_turn_never_committed`), so the chunk was then re-sent and transcribed from scratch.

These drive the workers' REAL consume loops over the real RedisStreamClient dispatch; only the
stream reads and the per-message work are faked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from stt_worker.worker import STTWorker
from translation_worker.worker import TranslationWorker


def _reads(
    worker: object, batches: list[list[tuple[bytes, dict[bytes, bytes]]]], stream: bytes
) -> object:
    """xreadgroup that hands out one batch per call, then stops the worker."""

    async def xreadgroup(**_kwargs: object) -> list[object]:
        if batches:
            return [(stream, batches.pop(0))]
        await asyncio.sleep(0.01)
        if not batches and getattr(worker, "_done", False):
            worker._shutdown_event.set()  # type: ignore[attr-defined]
        return []

    return xreadgroup


async def test_stt_reads_the_next_speakers_chunk_while_one_speaker_commits(
    mock_redis_client,
) -> None:
    worker = STTWorker.__new__(STTWorker)
    worker.logger = MagicMock()
    worker._shutdown_event = asyncio.Event()
    worker._consumer_name = "test-consumer"
    worker.input_stream = "audio:chunks"
    worker.consumer_group = "stt-workers"
    worker._speaker_locks = {}
    worker.redis = mock_redis_client
    worker._recover_stale_messages = AsyncMock()  # type: ignore[method-assign]

    b_started_while_a_running = asyncio.Event()
    a_running = asyncio.Event()
    a_waited_out: list[bool] = []

    async def process(message_id: bytes, data: dict[bytes, bytes]) -> None:
        if message_id == b"1-0":
            a_running.set()
            # Speaker A's slow commit: only finishes once B has been read and started.
            try:
                await asyncio.wait_for(b_started_while_a_running.wait(), timeout=1.0)
            except TimeoutError:
                a_waited_out.append(True)
        else:
            assert a_running.is_set()
            b_started_while_a_running.set()
        if message_id == b"2-0":
            worker._done = True  # type: ignore[attr-defined]

    worker.process = process  # type: ignore[method-assign]
    a = (b"1-0", {b"meeting_id": b"m1", b"speaker_id": b"A"})
    b = (b"2-0", {b"meeting_id": b"m1", b"speaker_id": b"B"})
    mock_redis_client._redis.xreadgroup = _reads(worker, [[a], [b]], b"audio:chunks")

    await asyncio.wait_for(worker._consume_loop(), timeout=5.0)

    # B was read and started WHILE A was still committing, not after A gave up.
    assert b_started_while_a_running.is_set()
    assert a_waited_out == []


async def test_translation_reads_the_next_line_while_a_slow_one_translates(
    mock_redis_client,
) -> None:
    worker = TranslationWorker.__new__(TranslationWorker)
    worker.logger = MagicMock()
    worker._shutdown_event = asyncio.Event()
    worker._consumer_name = "test-consumer"
    worker.input_stream = "stt:results"
    worker.consumer_group = "translate-workers"
    worker.redis = mock_redis_client
    worker._recover_stale_messages = AsyncMock()  # type: ignore[method-assign]

    second_started = asyncio.Event()

    async def process(message_id: bytes, data: dict[bytes, bytes]) -> None:
        if message_id == b"1-0":
            await asyncio.wait_for(second_started.wait(), timeout=2.0)
            worker._done = True  # type: ignore[attr-defined]
        else:
            second_started.set()

    worker.process = process  # type: ignore[method-assign]
    first = (b"1-0", {b"meeting_id": b"m1", b"speaker_id": b"A"})
    second = (b"2-0", {b"meeting_id": b"m1", b"speaker_id": b"B"})
    mock_redis_client._redis.xreadgroup = _reads(worker, [[first], [second]], b"stt:results")

    await asyncio.wait_for(worker._consume_loop(), timeout=5.0)

    assert second_started.is_set()


async def test_a_chunk_queued_behind_its_speaker_is_not_reclaimed_as_abandoned() -> None:
    worker = STTWorker.__new__(STTWorker)
    worker._in_flight_message_ids().add(b"7-0")

    assert worker._is_in_flight(b"7-0")
    assert not worker._is_in_flight(b"8-0")
