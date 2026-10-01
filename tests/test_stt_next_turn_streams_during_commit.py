"""The next turn keeps streaming while the previous one is being transcribed.

tools/meeting_sim, product-launch meeting (three people, 13 min): 99 of 205 chunks lost flash
mode to `commit_in_flight`. The speaker's next turn began while their previous chunk was still
being transcribed, its first frame found the speaker's lock taken, and the whole turn was thrown
away — so its chunk was uploaded whole at commit and decoded from scratch (p95 caption latency
3.0 s against ~0.7 s for a streamed turn). The frames are now held and handed to the session the
moment the previous commit has been SENT, into the fresh buffer it leaves behind.
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

    await worker._release_held_frames(KEY)  # that commit has now been sent
    await worker._append_speech_frame(_frame("t2", 2))  # still mid-transcription: straight in

    assert _appended(worker) == [b"\x00", b"\x01", b"\x02"]
    assert worker._streamed_turns[KEY] == ("t2", 7, 3)

    worker._end_commit_window(KEY)
    lock.release()
    await worker._append_speech_frame(_frame("t2", 3))
    assert worker._streamed_turns[KEY] == ("t2", 7, 4)
    worker.model.discard_streamed_audio.assert_not_awaited()


async def test_frames_held_for_a_commit_that_never_went_out_are_not_streamed() -> None:
    worker = _worker()
    lock = worker._speaker_locks.setdefault(KEY, asyncio.Lock())
    await lock.acquire()

    await worker._append_speech_frame(_frame("t2", 0))
    worker._end_commit_window(KEY)  # the chunk failed before its commit was sent
    lock.release()

    # The turn now has a hole where frame 0 was: it is abandoned to its own chunk's audio
    # rather than committed incomplete.
    await worker._append_speech_frame(_frame("t2", 1))
    assert _appended(worker) == []
    assert KEY not in worker._streamed_turns


async def test_the_commit_hands_over_before_it_waits_for_the_transcript() -> None:
    """OpenAISTT calls `on_committed` right after sending the commit, not after `completed`."""
    from stt_worker.model import OpenAISTT
    from tests.test_stt_worker import FakeRealtimeConn, FakeRealtimeManager

    order: list[str] = []

    class Conn(FakeRealtimeConn):
        def __aiter__(self) -> Any:
            async def gen() -> Any:
                order.append("reading")
                for event in self._events:
                    yield event

            return gen()

    from types import SimpleNamespace

    conn = Conn(
        [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed", transcript="Xin chào."
            )
        ]
    )
    stt = OpenAISTT.__new__(OpenAISTT)
    stt.api_key = ""
    stt.model = "gpt-live-transcribe"
    stt.noise_reduction = "off"
    stt._sessions = {}
    stt._client = MagicMock()
    stt._client.realtime.connect = MagicMock(return_value=FakeRealtimeManager(conn))

    async def committed() -> None:
        order.append("committed")

    text, _ = await stt._transcribe_via_session(
        ("m1", "s1"), b"\x00\x00" * 2400, on_committed=committed
    )

    assert text == "Xin chào."
    assert order == ["committed", "reading"]
