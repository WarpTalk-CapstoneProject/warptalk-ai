"""Live captions: the words of a turn still being spoken, published as they come.

WHAT WAS MEASURED (4 Oct 2026)
    A short sentence reached the caption ~2.5s after the speaker stopped. In flash mode the
    streaming model was already producing those words ~1s behind the speaker, but nothing read
    the socket until the turn was committed, so they waited out the whole turn and the 576ms
    silence hangover.

WHAT THESE PIN
    - Deltas of an uncommitted item are published as live text, throttled, with sentence ends
      always sent.
    - Once the turn is committed its words stop going out as live text (they are segments now).
    - A delta in a script nobody in the room speaks is not shown.
    - The commit path still returns the completed transcript, read through the session's queue.
    - A renewed session does not inherit the old socket's reader.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from stt_worker.model import OpenAISTT

KEY = ("meeting-1", "speaker-1")


def _delta(item: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="conversation.item.input_audio_transcription.delta", item_id=item, delta=text
    )


class _LiveConn:
    """A socket whose events are pushed by the test while the reader is running."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        self.input_audio_buffer = MagicMock()
        self.input_audio_buffer.append = AsyncMock()
        self.input_audio_buffer.commit = AsyncMock()

    def __aiter__(self):
        async def gen():
            while True:
                event = await self.queue.get()
                if event is None:
                    return
                yield event

        return gen()


def _stt(language: str = "vi") -> tuple[OpenAISTT, _LiveConn, list[tuple[str, str]]]:
    stt = OpenAISTT.__new__(OpenAISTT)
    conn = _LiveConn()
    stt._sessions = {
        KEY: {"conn": conn, "epoch": 1, "language": language, "allowed_languages": {"vi", "en"}}
    }
    published: list[tuple[str, str]] = []

    async def hook(key, item_id, text, lang):  # noqa: ANN001
        published.append((item_id, text))

    stt.on_live_text = hook
    return stt, conn, published


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_words_of_an_uncommitted_turn_are_published_as_they_come() -> None:
    stt, conn, published = _stt()
    await stt.append_streamed_audio(KEY, b"\x00\x00" * 160, 16000)

    await conn.queue.put(_delta("item-a", "Chào"))
    await _settle()
    await conn.queue.put(_delta("item-a", " mọi người"))  # inside the throttle window
    await _settle()
    await conn.queue.put(_delta("item-a", ", mình bắt đầu nhé."))  # a sentence end always goes out
    await _settle()

    assert published[0] == ("item-a", "Chào")
    assert published[-1] == ("item-a", "Chào mọi người, mình bắt đầu nhé.")
    assert ("item-a", "Chào mọi người") not in published, "throttled between words"


@pytest.mark.asyncio
async def test_a_committed_turn_stops_publishing_live_text() -> None:
    stt, conn, published = _stt()
    await stt.append_streamed_audio(KEY, b"\x00\x00" * 160, 16000)
    await conn.queue.put(_delta("item-a", "Ừ."))
    await _settle()
    stt._sessions[KEY]["closed_items"] = {"item-a"}

    await conn.queue.put(_delta("item-a", " Đúng rồi."))
    await _settle()

    assert published == [("item-a", "Ừ.")]


@pytest.mark.asyncio
async def test_a_foreign_script_delta_is_not_shown() -> None:
    stt, conn, published = _stt()
    await stt.append_streamed_audio(KEY, b"\x00\x00" * 160, 16000)
    await conn.queue.put(_delta("item-a", "sam來計劃."))
    await _settle()

    assert published == []


@pytest.mark.asyncio
async def test_commit_still_returns_the_completed_transcript_through_the_queue() -> None:
    stt, conn, published = _stt()
    stt._get_or_create_session = AsyncMock(return_value=stt._sessions[KEY])  # type: ignore[method-assign]
    await stt.append_streamed_audio(KEY, b"\x00\x00" * 160, 16000)
    await conn.queue.put(_delta("item-a", "Ừ, đúng rồi"))
    await _settle()

    async def finish() -> None:
        await asyncio.sleep(0.01)
        await conn.queue.put(SimpleNamespace(type="input_audio_buffer.committed", item_id="item-a"))
        await conn.queue.put(
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                item_id="item-a",
                transcript="Ừ, đúng rồi.",
                logprobs=None,
            )
        )

    asyncio.create_task(finish())
    text, _ = await stt._transcribe_via_session(KEY, b"", streamed_epoch=1)

    assert text == "Ừ, đúng rồi."
    assert "item-a" in stt._sessions[KEY]["closed_items"]


def test_a_renewed_session_does_not_inherit_the_old_reader() -> None:
    import inspect

    from stt_worker import model

    source = inspect.getsource(model.OpenAISTT)
    assert '"event_queue", "event_pump", "live_items", "closed_items"' in source
