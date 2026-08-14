"""A spoken turn must come back as one piece of speech, split at the right places.

The real Cartesia WebSocket cannot be reached from CI, and that is exactly why this file exists:
the two protocol mistakes recorded in `CartesiaSynthesizer.synthesize`'s comments survived review
because the HTTP path could not be run either. The sentence-boundary logic is therefore driven
through a Protocol seam with a scripted server, so the part that can be got wrong is the part
that is tested.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from tts_worker.prosody_context import (
    SENTENCE_TIMEOUT_SECONDS,
    ProsodyContext,
    wav_header,
)

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
    # 0 then 1, as the live API answers. This script said 1 then 2 until WT-400 — it was written
    # from the same wrong assumption as the code, so it agreed with the bug instead of catching it.
    transport = _ScriptedContext(
        [
            [_Chunk(_pcm(100), 0), _FlushDone(0)],
            [_Chunk(_pcm(200), 1), _FlushDone(1)],
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
    transport = _ScriptedContext([[_FlushDone(0)], [_FlushDone(1)]])
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
            [_Chunk(_pcm(50), 0), _FlushDone(0)],
            [_Chunk(_pcm(999), 0), _Chunk(_pcm(50), 1), _FlushDone(1)],
        ]
    )
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.")
    second, _ = await ctx.speak("Hai.")

    assert len(second) - HEADER == 100, "a stale flush_id 0 chunk leaked into sentence 2"


@pytest.mark.asyncio
async def test_delivery_is_sent_per_sentence() -> None:
    # A speaker's tempo and loudness move within a turn as readily as between turns, so the
    # measured delivery cannot be fixed once at context creation.
    transport = _ScriptedContext([[_FlushDone(0)], [_FlushDone(1)]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    await ctx.speak("Một.", {"speed": 1.1, "volume": 1.0})
    await ctx.speak("Hai.", {"speed": 0.9, "volume": 1.2})

    assert transport.pushes[0]["generation_config"] == {"speed": 1.1, "volume": 1.0}
    assert transport.pushes[1]["generation_config"] == {"speed": 0.9, "volume": 1.2}


@pytest.mark.asyncio
async def test_no_delivery_measured_means_no_generation_config_sent() -> None:
    # An unmeasured utterance must be synthesized exactly as it was before prosody existed —
    # the same rule synthesize() already follows.
    transport = _ScriptedContext([[_FlushDone(0)]])
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
    transport = _ScriptedContext([[_Chunk(_pcm(30), 0), _Done()]])
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


class _WedgedContext(_ScriptedContext):
    """Accepts the push, then never yields the flush_done — a socket that is up but stuck.

    This is the failure the timeout exists for, and the reason it matters more than an error:
    the worker holds a per-(speaker, language) lock while a sentence is synthesised, so a
    receive() that never returns stops that speaker's dub for the rest of the meeting, with
    nothing raised for the fallback to catch.
    """

    def receive(self) -> Any:
        async def _hang() -> Any:
            await asyncio.sleep(3600)
            yield  # pragma: no cover — unreachable, keeps this an async generator

        return _hang()


@pytest.mark.asyncio
async def test_a_wedged_socket_raises_instead_of_hanging_forever(monkeypatch) -> None:
    monkeypatch.setattr("tts_worker.prosody_context.SENTENCE_TIMEOUT_SECONDS", 0.05)
    ctx = ProsodyContext(_WedgedContext([]), SAMPLE_RATE)

    with pytest.raises(RuntimeError, match="no flush_done"):
        await asyncio.wait_for(ctx.speak("Một."), timeout=2.0)


@pytest.mark.asyncio
async def test_a_wedged_context_is_not_reused_for_the_next_sentence(monkeypatch) -> None:
    # A socket that missed one flush cannot be trusted to deliver the next; continuing to push
    # into it would stall every remaining sentence of the turn in the same way.
    monkeypatch.setattr("tts_worker.prosody_context.SENTENCE_TIMEOUT_SECONDS", 0.05)
    ctx = ProsodyContext(_WedgedContext([]), SAMPLE_RATE)

    # wait_for on BOTH calls, not just the assertion: a test that hangs is worse than a test
    # that fails — it takes the whole suite with it and reports nothing. Proven the hard way,
    # by deleting the timeout under test and watching pytest never return.
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(ctx.speak("Một."), timeout=2.0)

    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(ctx.speak("Hai."), timeout=2.0)


def test_the_timeout_is_generous_next_to_cartesias_own_latency() -> None:
    # ~90ms time-to-first-audio, sub-second for a long sentence. A bound this far above that
    # can only fire on a real failure, never on a slow success — which is what keeps the
    # fallback from stealing sentences that would have arrived.
    assert SENTENCE_TIMEOUT_SECONDS >= 3.0


# ── WT-400: which number the server puts on the first flush ──────────────────────────────────
#
# Verified against the live Cartesia API on 2026-08-14, because nothing else can verify it: two
# pushes on one context answered flush_id 0, then flush_id 1. The module said "from 1" and said
# it as a fact. Every sentence therefore waited for an id one higher than any that would arrive,
# timed out after SENTENCE_TIMEOUT_SECONDS and fell back to the one-shot path — for the whole
# life of the feature, while it was reported as ON.


@pytest.mark.asyncio
async def test_the_first_sentence_of_a_turn_expects_flush_id_zero() -> None:
    """The bug, stated as the one number it turns on.

    Under the off-by-one this returns nothing and raises: no chunk carries flush_id 1, so the
    audio is discarded, and no flush_done carries it either, so the read runs to the timeout.
    """
    transport = _ScriptedContext([[_Chunk(_pcm(100), 0), _FlushDone(0)]])
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    audio, _ms = await asyncio.wait_for(ctx.speak("Câu một."), timeout=2.0)

    assert len(audio) - HEADER == 200, "the sentence's own audio was thrown away"


@pytest.mark.asyncio
async def test_the_ids_keep_counting_from_there() -> None:
    # Sentence two is flush_id 1, not 2. Fixing only the first sentence would leave every later
    # one broken, which is the same bug with a smaller blast radius.
    transport = _ScriptedContext(
        [
            [_Chunk(_pcm(100), 0), _FlushDone(0)],
            [_Chunk(_pcm(200), 1), _FlushDone(1)],
            [_Chunk(_pcm(300), 2), _FlushDone(2)],
        ]
    )
    ctx = ProsodyContext(transport, SAMPLE_RATE)

    first, _ = await asyncio.wait_for(ctx.speak("Một."), timeout=2.0)
    second, _ = await asyncio.wait_for(ctx.speak("Hai."), timeout=2.0)
    third, _ = await asyncio.wait_for(ctx.speak("Ba."), timeout=2.0)

    assert [len(first) - HEADER, len(second) - HEADER, len(third) - HEADER] == [200, 400, 600]
    assert ctx.sentences_spoken == 3, "the count of sentences must survive the renumbering"


class _MisnumberedContext(_ScriptedContext):
    """Sends real audio under a flush_id nobody is waiting for, then keeps the socket open.

    This is what the off-by-one actually looked like from inside `_collect`: not an error, not a
    silence — a healthy stream whose every event was discarded by the flush_id filter, running
    until the timeout. Modelled with a stream that does not end, because a scripted stream that
    ends returns cleanly and never reaches the branch under test.
    """

    def receive(self) -> Any:
        async def _wrong_ids() -> Any:
            yield _Chunk(_pcm(10), 7)
            yield _Chunk(_pcm(10), 7)
            await asyncio.sleep(3600)

        return _wrong_ids()


@pytest.mark.asyncio
async def test_a_timeout_reports_the_ids_the_server_was_actually_sending(monkeypatch) -> None:
    """What turns the next mismatch into one glance instead of an afternoon.

    A wrong expected id and a wedged socket both present as "no flush_done". Naming the ids that
    DID arrive is the whole difference between them, and it is the evidence that was missing
    while this bug looked like a flaky vendor.
    """
    monkeypatch.setattr("tts_worker.prosody_context.SENTENCE_TIMEOUT_SECONDS", 0.05)
    # A server numbering from 7 — any numbering the code does not expect — on a socket that
    # stays OPEN afterwards. That is what production looked like: audio kept arriving under an
    # id nothing was waiting for, and the read ran to the timeout rather than ending.
    ctx = ProsodyContext(_MisnumberedContext([]), SAMPLE_RATE)

    with pytest.raises(RuntimeError) as caught:
        await asyncio.wait_for(ctx.speak("Một."), timeout=2.0)

    message = str(caught.value)
    assert "flush_id=0" in message, "the id that was awaited is not in the error"
    assert "[7]" in message, "the ids the server sent are not in the error"
