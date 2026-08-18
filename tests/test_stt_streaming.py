"""Audio reaches STT while the speaker is still talking, and every way that can go wrong
degrades BACKWARDS — to the latency the pipeline had before, never to a lost or wrong sentence.

The pipeline is otherwise deaf until VAD closes a turn: five seconds of speech produce zero
work for five seconds and then ~2.6s of it. The Realtime API separates
`input_audio_buffer.append` from `.commit`, so the audio can arrive as it is spoken.

What is actually dangerous here is not the happy path. It is a commit firing against a buffer
that holds something other than this turn — audio from an abandoned fragment, or from a session
that was recreated underneath. Both produce a confident, fluent, WRONG transcript, which is the
one failure this pipeline has repeatedly proven nobody can spot from the outside. Hence the
epoch, the turn id, and most of the tests below.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import STT_FRAME_STREAM, AudioChunkMessage, AudioFrameMessage
from stt_worker.worker import STTWorker

MEETING = "m1"
SPEAKER = "s1"


class _Model:
    """Records what the streaming path does, without a network."""

    def __init__(self, epoch: int | None = 7) -> None:
        self.epoch = epoch
        self.appended: list[bytes] = []
        self.discards = 0

    async def append_streamed_audio(
        self, _key: tuple[str, str], pcm: bytes, _rate: int
    ) -> int | None:
        self.appended.append(pcm)
        return self.epoch

    async def discard_streamed_audio(self, _key: tuple[str, str]) -> None:
        self.discards += 1


def _worker(model: _Model) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker.worker_name = "stt"
    worker._consumer_name = "stt-test"
    worker.model = model  # type: ignore[assignment]
    worker._streamed_turns = {}
    return worker


def _frame(turn: str, seq: int = 0, audio: bytes = b"\x01\x02") -> dict[bytes, bytes]:
    return AudioFrameMessage(
        meeting_id=MEETING, speaker_id=SPEAKER, turn_id=turn, seq=seq, audio_data=audio
    ).to_redis()  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_a_frame_is_appended_to_the_open_session() -> None:
    model = _Model()
    worker = _worker(model)

    await worker._append_speech_frame(_frame("T1", audio=b"abcd"))

    assert model.appended == [b"abcd"]
    assert worker._streamed_turns[(MEETING, SPEAKER)] == ("T1", 7)


@pytest.mark.asyncio
async def test_frames_of_one_turn_accumulate() -> None:
    model = _Model()
    worker = _worker(model)

    await worker._append_speech_frame(_frame("T1", 0, b"aa"))
    await worker._append_speech_frame(_frame("T1", 1, b"bb"))

    assert model.appended == [b"aa", b"bb"]
    assert model.discards == 0, "one turn must never clear its own buffer mid-way"


@pytest.mark.asyncio
async def test_a_new_turn_while_the_old_one_is_open_clears_the_abandoned_audio() -> None:
    """An utterance too short for ingress to publish leaves samples in the buffer that nothing
    will commit. Left there they are transcribed as the opening of the NEXT turn, and every
    abandoned fragment makes the following utterance wronger."""
    model = _Model()
    worker = _worker(model)
    await worker._append_speech_frame(_frame("T1"))

    await worker._append_speech_frame(_frame("T2"))

    assert model.discards == 1
    assert worker._streamed_turns[(MEETING, SPEAKER)] == ("T2", 7)


@pytest.mark.asyncio
async def test_no_session_yet_means_the_frame_is_simply_dropped() -> None:
    """The prewarm has not opened a session. The turn goes the old way — `process` sends the
    audio itself — which is the behaviour before this feature existed."""
    model = _Model(epoch=None)
    worker = _worker(model)

    await worker._append_speech_frame(_frame("T1"))

    assert worker._streamed_turns == {}, "nothing may be recorded as buffered"


@pytest.mark.asyncio
async def test_an_unreadable_frame_does_not_take_the_consumer_down() -> None:
    model = _Model()
    worker = _worker(model)

    await worker._append_speech_frame({b"nonsense": b"1"})

    assert model.appended == []


def _chunk(turn: str) -> AudioChunkMessage:
    return AudioChunkMessage(
        meeting_id=MEETING,
        speaker_id=SPEAKER,
        chunk_index=0,
        audio_data=b"\x00\x00",
        turn_id=turn,
    )


def test_the_frame_stream_is_not_audio_chunks() -> None:
    """`audio:chunks` means "one closed utterance" and three things read it that way — STT, the
    TTS worker's voice-clone buffer, and prosody. Publishing frames onto it would multiply its
    entry rate ~40x and hand the clone path, which re-runs an FFT over its whole buffer on every
    message, that same multiplier. Neither is a change to STT; both are collateral."""
    assert STT_FRAME_STREAM != "audio:chunks"


def test_a_chunk_from_an_older_ingress_carries_no_turn() -> None:
    """Through a rolling deploy the previous ingress keeps publishing chunks with no turn_id.
    Empty must read as "the audio is in this message", which is the pre-streaming contract."""
    payload = _chunk("T1").to_redis()
    del payload["turn_id"]

    assert AudioChunkMessage.from_redis(payload).turn_id == ""


class _Buffer:
    def __init__(self) -> None:
        self.appends: list[str] = []
        self.commits = 0
        self.clears = 0

    async def append(self, audio: str) -> None:
        self.appends.append(audio)

    async def commit(self) -> None:
        self.commits += 1

    async def clear(self) -> None:
        self.clears += 1


class _Conn:
    def __init__(self) -> None:
        self.input_audio_buffer = _Buffer()

    def __aiter__(self) -> Any:
        async def _events() -> Any:
            yield type(
                "E",
                (),
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "hello there",
                    "logprobs": [],
                },
            )()

        return _events()


def _stt_with_session(epoch: int) -> tuple[Any, _Conn]:
    from stt_worker.model import OpenAISTT

    stt = OpenAISTT.__new__(OpenAISTT)
    conn = _Conn()
    stt._sessions = {(MEETING, SPEAKER): {"conn": conn, "epoch": epoch, "last_used": 0.0}}

    async def _get_or_create_session(key: tuple[str, str], *_a: Any, **_k: Any) -> Any:
        return stt._sessions[key]

    stt._get_or_create_session = _get_or_create_session  # type: ignore[method-assign]
    return stt, conn


@pytest.mark.asyncio
async def test_a_matching_epoch_commits_without_resending_the_audio() -> None:
    """The whole latency win. The model has already heard the turn, so all that is left is to
    say go — instead of sending five seconds of audio and waiting for it to be heard."""
    stt, conn = _stt_with_session(epoch=7)

    text, _logprob = await stt._transcribe_via_session(
        (MEETING, SPEAKER), b"\x00\x00" * 4000, streamed_epoch=7
    )

    assert conn.input_audio_buffer.appends == [], "the audio was already in the buffer"
    assert conn.input_audio_buffer.commits == 1
    assert text == "hello there"


@pytest.mark.asyncio
async def test_a_stale_epoch_sends_the_audio_rather_than_committing_a_stranger() -> None:
    """A session recreated since the frames were appended — a language change, an idle sweep, a
    restart — took that buffer with it. Committing anyway would transcribe whatever happens to
    be in the NEW session's buffer and publish it as this speaker's sentence."""
    stt, conn = _stt_with_session(epoch=9)

    await stt._transcribe_via_session((MEETING, SPEAKER), b"\x00\x00" * 4000, streamed_epoch=7)

    assert conn.input_audio_buffer.appends, "fell back to sending the audio"
    assert conn.input_audio_buffer.commits == 1


@pytest.mark.asyncio
async def test_streaming_off_behaves_exactly_as_before() -> None:
    stt, conn = _stt_with_session(epoch=7)

    await stt._transcribe_via_session((MEETING, SPEAKER), b"\x00\x00" * 4000)

    assert conn.input_audio_buffer.appends
    assert conn.input_audio_buffer.commits == 1
