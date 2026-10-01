"""The next turn keeps streaming while the previous one is being transcribed.

tools/meeting_sim, product-launch meeting (three people, 13 min): 99 of 205 chunks lost flash
mode to `commit_in_flight`. The speaker's next turn began while their previous chunk was still
being transcribed, its first frame found the speaker's lock taken, and the whole turn was thrown
away — so its chunk was uploaded whole at commit and decoded from scratch (p95 caption latency
3.0 s against ~0.7 s for a streamed turn). The frames are now held and handed to the session the
moment the previous item has COMPLETED (not merely been committed: appending while it was still
being transcribed once stalled it 12 s and truncated it), for at most _HOLD_FOR_COMPLETION_S.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.schemas import AudioFrameMessage
from stt_worker.worker import STTWorker

KEY = ("m1", "s1")


def _worker() -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.logger = MagicMock()
    worker._speaker_locks = {}
    worker.model = MagicMock()
    worker.model.append_streamed_audio = AsyncMock(return_value=7)
    worker.model.discard_streamed_audio = AsyncMock()
    return worker


def _frame(turn: str, seq: int) -> dict[bytes, bytes]:
    msg = AudioFrameMessage(
        meeting_id=KEY[0],
        speaker_id=KEY[1],
        turn_id=turn,
        seq=seq,
        audio_data=bytes([seq]) * 64,
        sample_rate=16000,
        language="vi",
    )
    return {
        k.encode(): (v if isinstance(v, bytes) else str(v).encode())
        for k, v in msg.to_redis().items()
    }


def _appended(worker: STTWorker) -> list[bytes]:
    return [call.args[1][:1] for call in worker.model.append_streamed_audio.await_args_list]


async def test_frames_wait_for_the_commit_then_go_in_in_order() -> None:
    worker = _worker()
    lock = worker._speaker_locks.setdefault(KEY, asyncio.Lock())
    await lock.acquire()  # the previous chunk is being processed

    await worker._append_speech_frame(_frame("t2", 0))
    await worker._append_speech_frame(_frame("t2", 1))
    assert _appended(worker) == []  # not into the buffer about to be committed

    await worker._release_held_frames(KEY)  # the previous item has completed
    await worker._append_speech_frame(_frame("t2", 2))  # process still running: straight in

    assert _appended(worker) == [b"\x00", b"\x01", b"\x02"]
    assert worker._streamed_turns[KEY] == ("t2", 7, 3)

    worker._end_commit_window(KEY)
    lock.release()
    await worker._append_speech_frame(_frame("t2", 3))
    assert worker._streamed_turns[KEY] == ("t2", 7, 4)
    worker.model.discard_streamed_audio.assert_not_awaited()


async def test_frames_held_for_an_item_that_never_completed_are_not_streamed() -> None:
    worker = _worker()
    lock = worker._speaker_locks.setdefault(KEY, asyncio.Lock())
    await lock.acquire()

    await worker._append_speech_frame(_frame("t2", 0))
    worker._end_commit_window(KEY)  # the chunk failed before its item completed
    lock.release()

    # The turn now has a hole where frame 0 was: it is abandoned to its own chunk's audio
    # rather than committed incomplete.
    await worker._append_speech_frame(_frame("t2", 1))
    assert _appended(worker) == []
    assert KEY not in worker._streamed_turns


async def test_held_frames_wait_out_a_lost_completion_only_up_to_the_bound(
    monkeypatch: Any,
) -> None:
    """A completion that never comes must not stall the speaker: past the bound the turn falls
    back to its own chunk's audio, exactly as every turn did before."""
    from stt_worker import worker as worker_module

    monkeypatch.setattr(worker_module, "_HOLD_FOR_COMPLETION_S", 0.05)
    worker = _worker()
    lock = worker._speaker_locks.setdefault(KEY, asyncio.Lock())
    await lock.acquire()

    await worker._append_speech_frame(_frame("t2", 0))
    await asyncio.sleep(0.1)
    await worker._append_speech_frame(_frame("t2", 1))  # past the bound: give up on streaming
    await worker._append_speech_frame(_frame("t2", 2))

    assert worker._held_frames()[KEY] == []
    await worker._release_held_frames(KEY)  # the late completion hands over nothing
    worker._end_commit_window(KEY)
    lock.release()
    await worker._append_speech_frame(_frame("t2", 3))  # a closed turn's frame: dropped

    assert _appended(worker) == []
    assert KEY not in worker._streamed_turns
    worker.model.discard_streamed_audio.assert_not_awaited()


async def test_the_hand_over_waits_for_the_previous_item_to_complete(
    mock_redis_client: Any, sample_audio_bytes: bytes
) -> None:
    """gpt-live-transcribe stalled a committed item for 12 s (and truncated it) when the next
    turn's audio arrived while it was still being transcribed; frames go in after `completed`."""
    from shared.config import STTSettings, WorkerSettings
    from shared.schemas import AudioChunkMessage
    from stt_worker.model import TranscribedSegment

    order: list[str] = []
    worker = _worker()
    worker.settings = WorkerSettings()
    worker.redis = mock_redis_client
    worker.stt_settings = STTSettings(prosody_enabled=False)
    worker._paused_rooms = set()
    worker._stt_prompts = {}
    worker._room_languages = {}

    async def transcribe(*_args: Any, **_kwargs: Any) -> list[TranscribedSegment]:
        order.append("transcribe started")
        await asyncio.sleep(0.05)
        order.append("completed")
        return [
            TranscribedSegment(
                text="Xin chào.", language="vi", confidence=-1.0, start_ms=0, end_ms=1
            )
        ]

    async def release(key: Any) -> None:
        order.append("held frames released")

    worker.model.transcribe = transcribe
    worker._release_held_frames = release  # type: ignore[method-assign]
    chunk = AudioChunkMessage(
        meeting_id=KEY[0],
        speaker_id=KEY[1],
        chunk_index=0,
        audio_data=sample_audio_bytes,
        language="vi",
    )

    await worker.process(b"1-0", chunk.to_redis())

    assert order == ["transcribe started", "completed", "held frames released"]
