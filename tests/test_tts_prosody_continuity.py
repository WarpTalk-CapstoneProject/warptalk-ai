"""How the worker decides between a continuous turn and an independent generation.

The sentence-boundary protocol itself is covered in test_prosody_context.py. This covers the
wiring around it: when a context is opened, when it is reused, when it is closed, and — the part
that matters most in a live meeting — that every failure still produces audio.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import TranslationResultMessage
from tts_worker.worker import TTSWorker


class _FakeTurn:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, Any]] = []
        self.closed = False
        self.abandoned = False
        self.fail = False

    async def speak(self, text: str, generation_config: Any = None) -> tuple[bytes, int]:
        if self.fail:
            raise RuntimeError("socket died")
        self.spoken.append((text, generation_config))
        return b"\x00" * 44 + b"\x01\x02" * 100, 12

    async def aclose(self) -> None:
        self.closed = True

    async def abandon(self) -> None:
        self.abandoned = True


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeSynthesizer:
    """Records which path was taken."""

    def __init__(self) -> None:
        self.one_shot_calls: list[str] = []
        self.opened: list[str] = []
        self.turns: list[_FakeTurn] = []
        self.connections: list[_FakeConnection] = []

    async def synthesize(
        self, *, text: str, language: str, voice_id: str | None, generation_config: Any
    ) -> tuple[bytes, int, str]:
        self.one_shot_calls.append(text)
        return b"\x00" * 44 + b"\xaa\xbb" * 10, 5, voice_id or "default-voice"

    async def open_prosody_context(
        self, *, context_id: str, language: str, voice_id: str | None
    ) -> tuple[_FakeTurn, _FakeConnection]:
        self.opened.append(context_id)
        turn, connection = _FakeTurn(), _FakeConnection()
        self.turns.append(turn)
        self.connections.append(connection)
        return turn, connection


def _worker(*, continuity: bool) -> tuple[TTSWorker, _FakeSynthesizer]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(prosody_continuity=continuity)
    worker.logger = MagicMock()
    worker._turns = {}
    worker._turn_connections = {}
    synthesizer = _FakeSynthesizer()
    worker.cartesia = synthesizer  # type: ignore[assignment]
    return worker, synthesizer


def _msg(text: str, *, final: bool = False, chunk: int = 0) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id="seg-1",
        meeting_id="m1",
        speaker_id="s1",
        original_text="src",
        translated_text=text,
        source_lang="en",
        target_lang="vi",
        chunk_index=chunk,
        is_final_chunk=final,
    )


async def _say(worker: TTSWorker, message: TranslationResultMessage, voice: str = "v1") -> Any:
    return await worker._synthesize_sentence(
        translation=message,
        text=message.translated_text,
        voice_id=voice,
        voice_key="",
        generation_config={"speed": 1.0, "volume": 1.0},
    )


@pytest.mark.asyncio
async def test_off_by_default_uses_the_proven_one_shot_path() -> None:
    # The setting ships OFF: the WebSocket path has never run against the real API from this
    # codebase, and a dub that fails is silence in a live meeting.
    assert TTSSettings().prosody_continuity is False

    worker, synth = _worker(continuity=False)
    await _say(worker, _msg("Một."))

    assert synth.one_shot_calls == ["Một."]
    assert synth.opened == []


@pytest.mark.asyncio
async def test_the_sentences_of_one_turn_share_a_context() -> None:
    """The whole point. Two sentences of the same turn must be one prosodic thread, not two
    independent generations."""
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0))
    await _say(worker, _msg("Câu hai.", chunk=1))

    assert len(synth.opened) == 1, "the second sentence opened a second context"
    assert [text for text, _ in synth.turns[0].spoken] == ["Câu một.", "Câu hai."]
    assert synth.one_shot_calls == []


@pytest.mark.asyncio
async def test_the_turn_ends_where_the_speaker_stopped() -> None:
    # is_final_chunk is the only signal that carries where the SPEAKER stopped, as opposed to
    # where a chunk boundary happened to fall.
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0))
    assert synth.turns[0].closed is False

    await _say(worker, _msg("Câu hai.", chunk=1, final=True))

    assert synth.turns[0].closed is True
    assert synth.connections[0].closed is True, "the socket outlived its context"
    assert worker._turns == {}, "a finished turn must not be reused by the next one"


@pytest.mark.asyncio
async def test_a_new_turn_after_the_last_one_closed_opens_a_fresh_context() -> None:
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Lượt một.", final=True))
    await _say(worker, _msg("Lượt hai.", final=True))

    assert len(synth.opened) == 2


@pytest.mark.asyncio
async def test_a_broken_context_still_produces_audio() -> None:
    """The failure that matters. A dead socket must not become silence in the meeting — this
    sentence falls back to the one-shot path, and the dead turn is discarded rather than
    retried into."""
    worker, synth = _worker(continuity=True)
    await _say(worker, _msg("Câu một.", chunk=0))
    synth.turns[0].fail = True

    audio, _duration, _voice = await _say(worker, _msg("Câu hai.", chunk=1))

    assert synth.one_shot_calls == ["Câu hai."]
    assert len(audio) > 44
    assert worker._turns == {}, "the failed turn was kept and would fail again"


@pytest.mark.asyncio
async def test_a_voice_change_mid_meeting_does_not_continue_the_old_voice() -> None:
    # voice_clone_max_upgrades replaces a speaker's voice mid-meeting. Continuing a turn into a
    # different voice would be a worse seam than the one this removes.
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0), voice="voice-a")
    await _say(worker, _msg("Câu hai.", chunk=1), voice="voice-b")

    assert len(synth.opened) == 2


@pytest.mark.asyncio
async def test_ending_a_room_abandons_its_turns_without_waiting() -> None:
    worker, synth = _worker(continuity=True)
    # BaseWorker._cleanup_room touches state that __init__ normally creates; these workers are
    # built with __new__, so the fields it clears have to exist.
    worker._key_locks = {}
    worker._route_states = {}
    worker._room_routes = {}
    worker._translation_active = {}
    worker._paused_rooms = set()
    await _say(worker, _msg("Câu một.", chunk=0))

    worker._cleanup_room("m1")

    assert worker._turns == {}
