"""A spoken turn must come back as one piece of speech, split at the right places.

The real Cartesia WebSocket cannot be reached from CI, and that is exactly why this file exists:
the two protocol mistakes recorded in `CartesiaSynthesizer.synthesize`'s comments survived review
because the HTTP path could not be run either. The sentence-boundary logic is therefore driven
through a Protocol seam with a scripted server, so the part that can be got wrong is the part
that is tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tts_worker.prosody_context import ProsodyContext, wav_header

SAMPLE_RATE = 16000
HEADER = 44


@dataclass
class _Chunk:
    audio: bytes
    flush_id: int | None
    type: str = "chunk"


@dataclass
class _FlushDone:
    flush_id: int
    type: str = "flush_done"


@dataclass
class _Done:
    type: str = "done"


@dataclass
class _Error:
    error: str
    type: str = "error"


class _ScriptedContext:
    """Replays a canned server response per push, and records what was sent."""

    def __init__(self, script: list[list[Any]]) -> None:
        self._script = script
        self.pushes: list[dict[str, Any]] = []
        self.closed = False
        self.cancelled = False

    async def push(self, transcript: str, *, continue_: bool = True, **kwargs: Any) -> None:
        self.pushes.append({"transcript": transcript, "continue_": continue_, **kwargs})

    def receive(self) -> Any:
        events = self._script.pop(0) if self._script else []

        async def _gen() -> Any:
            for event in events:
                yield event

        return _gen()

    async def no_more_inputs(self) -> None:
        self.closed = True

    async def cancel(self) -> None:
        self.cancelled = True


def _pcm(n: int) -> bytes:
    return b"\x01\x02" * n


@pytest.mark.asyncio
async def test_each_sentence_comes_back_as_its_own_audio() -> None:
    transport = _ScriptedContext(
        [
            [_Chunk(_pcm(100), 1), _FlushDone(1)],
            [_Chunk(_pcm(200), 2), _FlushDone(2)],
        ]
    )
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    first, first_ms = await ctx.speak("Câu một.")
    second, second_ms = await ctx.speak("Câu hai.")

    assert len(first) - HEADER == 200
    assert len(second) - HEADER == 400
    assert first_ms == pytest.approx(200 / 2 / SAMPLE_RATE * 1000, abs=1)
    assert second_ms == pytest.approx(400 / 2 / SAMPLE_RATE * 1000, abs=1)


@pytest.mark.asyncio
async def test_every_sentence_is_pushed_as_a_continuation() -> None:
    """The whole point. Without `continue`, the model treats each sentence as a complete
    utterance and applies a final-sentence cadence to a clause that has more coming — which is
    the disjointed reading the one-shot HTTP path produces today."""
    transport = _ScriptedContext([[_FlushDone(1)], [_FlushDone(2)]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.")
    await ctx.speak("Hai.")

    assert [p["continue_"] for p in transport.pushes] == [True, True]
    assert all(p["flush"] is True for p in transport.pushes), (
        "without a flush per sentence there is no boundary to read up to, and the first "
        "sentence would swallow the rest of the turn"
    )


@pytest.mark.asyncio
async def test_audio_from_an_earlier_sentence_is_not_replayed() -> None:
    """Chunks for a previous flush can still be in flight when the next sentence is pushed.
    Collecting them would repeat the previous sentence's audio inside this one."""
    transport = _ScriptedContext(
        [
            [_Chunk(_pcm(50), 1), _FlushDone(1)],
            [_Chunk(_pcm(999), 1), _Chunk(_pcm(50), 2), _FlushDone(2)],
        ]
    )
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.")
    second, _ = await ctx.speak("Hai.")

    assert len(second) - HEADER == 100, "a stale flush_id 1 chunk leaked into sentence 2"


@pytest.mark.asyncio
async def test_delivery_is_sent_per_sentence() -> None:
    # A speaker's tempo and loudness move within a turn as readily as between turns, so the
    # measured delivery cannot be fixed once at context creation.
    transport = _ScriptedContext([[_FlushDone(1)], [_FlushDone(2)]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.", {"speed": 1.1, "volume": 1.0})
    await ctx.speak("Hai.", {"speed": 0.9, "volume": 1.2})

    assert transport.pushes[0]["generation_config"] == {"speed": 1.1, "volume": 1.0}
    assert transport.pushes[1]["generation_config"] == {"speed": 0.9, "volume": 1.2}


@pytest.mark.asyncio
async def test_no_delivery_measured_means_no_generation_config_sent() -> None:
    # An unmeasured utterance must be synthesized exactly as it was before prosody existed —
    # the same rule synthesize() already follows.
    transport = _ScriptedContext([[_FlushDone(1)]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.")

    assert "generation_config" not in transport.pushes[0]


@pytest.mark.asyncio
async def test_a_server_error_is_raised_so_the_caller_can_fall_back() -> None:
    # Swallowing this would publish silence into a live meeting. Raising lets the worker drop
    # to the proven one-shot path for this sentence.
    transport = _ScriptedContext([[_Error("context expired")]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    with pytest.raises(RuntimeError, match="context expired"):
        await ctx.speak("Một.")


@pytest.mark.asyncio
async def test_the_server_ending_the_context_is_not_an_error() -> None:
    # `done` mid-sentence means nothing more is coming, not that something broke. Whatever
    # arrived is returned and the context marks itself finished.
    transport = _ScriptedContext([[_Chunk(_pcm(30), 1), _Done()]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    audio, _ = await ctx.speak("Một.")

    assert len(audio) - HEADER == 60
    with pytest.raises(RuntimeError, match="closed"):
        await ctx.speak("Hai.")


@pytest.mark.asyncio
async def test_closing_is_idempotent() -> None:
    # The worker closes on the final sentence AND on room teardown; a turn whose final sentence
    # never arrives is closed by the sweep. All three can land on the same context.
    transport = _ScriptedContext([])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.aclose()
    await ctx.aclose()

    assert transport.closed is True


@pytest.mark.asyncio
async def test_abandoning_cancels_rather_than_draining() -> None:
    transport = _ScriptedContext([])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.abandon()

    assert transport.cancelled is True
    assert transport.closed is False


def test_the_wav_header_matches_what_the_publish_path_strips() -> None:
    # Every consumer downstream strips a hard-coded 44 bytes. A header of any other length
    # would clip or corrupt the first samples of every dub.
    header = wav_header(1000, SAMPLE_RATE)

    assert len(header) == HEADER
    assert header[:4] == b"RIFF"
    assert header[8:12] == b"WAVE"
