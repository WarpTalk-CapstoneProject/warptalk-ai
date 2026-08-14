"""One prosodic thread through Cartesia, so an utterance is spoken and not recited.

THE PROBLEM THIS SOLVES
    `CartesiaSynthesizer.synthesize` calls the one-shot HTTP endpoint, once per translated
    SENTENCE. A single spoken turn is routinely split into several sentences (see
    TranslationResultMessage.chunk_index), and each one is therefore an independent generation
    with no memory of the one before it. The model opens every sentence at its own default
    prosodic baseline, so the dub comes back as a list of separately-read sentences rather than
    as somebody talking. Cartesia's own documentation describes exactly this failure mode and
    names the fix: contexts, which "maintain prosody between their inputs".

    This is the "giọng nói liền mạch" half of Level 4, and — unlike the pitch-contour half — it
    needs no new vendor capability, no model change and no GPU. `GenerationRequestParam` in the
    already-installed cartesia-py 3.2.0 carries `context_id` and the `continue` flag. The
    capability was simply never used.

WHY THIS IS OFF BY DEFAULT
    The WebSocket path has never been exercised in this codebase, and it cannot be exercised
    locally: it needs a live Cartesia connection. `synthesize`'s own comments record two protocol
    mistakes in the HTTP path that survived review precisely because Cartesia was unreachable at
    the time and neither could be run. Shipping an unexercised protocol implementation switched
    ON, into the hot path of a production meeting, would repeat that.

    So it lands dark behind `TTSSettings.prosody_continuity`, every failure degrades to the
    proven one-shot path, and turning it on is a separate, deliberate decision to be taken after
    a real room has been listened to.

HOW A SENTENCE IS SEPARATED FROM THE NEXT
    The context streams audio continuously; `flush` puts a numbered boundary in that stream.
    Cartesia numbers flushes from 1 per context, and stamps every chunk with the `flush_id` it
    belongs to, so pushing each sentence with `flush=True` and reading until the matching
    `flush_done` yields exactly that sentence's audio — while the context, and therefore the
    prosody, carries on into the next one.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from shared.logger import get_logger

logger = get_logger(__name__)

# How long one sentence may take to come back before the turn is abandoned.
#
# A HANG IS WORSE THAN AN ERROR, AND THIS IS THE ONLY THING THAT TURNS ONE INTO THE OTHER.
# The worker holds a per-(speaker, language) lock while a sentence is synthesised, so a
# `receive()` that never yields its flush_done does not merely lose this sentence: it stops that
# speaker's dub entirely, for the rest of the meeting, without raising anything for the fallback
# to catch. Silence, indefinitely, with no error anywhere.
#
# 6 seconds because Cartesia's time-to-first-audio is ~90ms and a long sentence streams in well
# under a second; anything near this bound is already a failure, not a slow success. Generous
# enough never to fire on a healthy call, short enough that a wedged socket costs one sentence.
SENTENCE_TIMEOUT_SECONDS = 6.0


class ContextTransport(Protocol):
    """The slice of cartesia's AsyncWebSocketContext this module uses.

    Declared as a Protocol so the sentence-boundary logic below is testable without a network
    connection — which matters more than usual here, because the real connection is the one
    thing CI can never provide.
    """

    async def push(self, transcript: str, *, continue_: bool = True, **kwargs: Any) -> None: ...

    def receive(self) -> Any: ...

    async def no_more_inputs(self) -> None: ...

    async def cancel(self) -> None: ...


def wav_header(pcm_length: int, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """A 44-byte RIFF header for mono 16-bit PCM.

    The context returns raw PCM, but every consumer downstream — `_publish_livekit_only`,
    the cache, the dubbing row — was written against `synthesize`, which returns a WAV
    container and whose header they strip with a hard-coded 44. Returning the same shape keeps
    this change confined to synthesis instead of rippling through the publish path, where a
    mistake would be much harder to see.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + pcm_length)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", pcm_length)
    )


class ProsodyContext:
    """The sentences of one spoken turn, delivered as one continuous piece of speech."""

    def __init__(self, transport: ContextTransport, sample_rate: int) -> None:
        self._transport = transport
        self._sample_rate = sample_rate
        self._flushes = 0
        self._closed = False

    @property
    def sentences_spoken(self) -> int:
        return self._flushes

    async def speak(
        self,
        text: str,
        generation_config: dict[str, float | str] | None = None,
        on_pcm: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> tuple[bytes, int]:
        """Add one sentence to this turn and return its audio as (wav_bytes, duration_ms).

        `continue_=True` on every push: this sentence is never the last word of the turn as far
        as the model is concerned, which is what stops it from applying a final-sentence cadence
        to a clause that has more coming. The turn is ended by `aclose`, not by a push.

        `on_pcm` (WT-397) receives each raw chunk as it lands, so the listener can start hearing
        the sentence before it has finished generating. THE RETURN VALUE IS UNCHANGED — the
        chunks are still accumulated and still returned whole, because the TTS cache, billing
        (tts:results) and the transcript row all read that buffer and none of them can work from
        a stream. `on_pcm` is a tee, not a handover.
        """
        if self._closed:
            raise RuntimeError("ProsodyContext is closed")

        self._flushes += 1
        expected_flush = self._flushes

        push_kwargs: dict[str, Any] = {"flush": True}
        if generation_config:
            # Sent per sentence, not once per context: the delivery measured for THIS utterance
            # is what shared/prosody.py produces, and a speaker's tempo and loudness move within
            # a turn as readily as between turns.
            push_kwargs["generation_config"] = generation_config

        await self._transport.push(text, continue_=True, **push_kwargs)

        try:
            async with asyncio.timeout(SENTENCE_TIMEOUT_SECONDS):
                return await self._collect(expected_flush, on_pcm)
        except TimeoutError:
            # Deliberately raised, not returned as empty audio: the caller's fallback is what
            # turns this into a spoken sentence, and it only runs on an exception. The context
            # is marked closed because a socket that missed one flush cannot be trusted to
            # deliver the next.
            self._closed = True
            raise RuntimeError(
                f"Cartesia context produced no flush_done within {SENTENCE_TIMEOUT_SECONDS}s"
            ) from None

    async def _collect(
        self,
        expected_flush: int,
        on_pcm: Callable[[bytes], Awaitable[None]] | None = None,
    ) -> tuple[bytes, int]:
        pcm = bytearray()
        async for event in self._transport.receive():
            kind = getattr(event, "type", None)

            if kind == "error":
                raise RuntimeError(f"Cartesia context error: {getattr(event, 'error', event)}")

            if kind == "chunk":
                # Chunks arriving for an EARLIER flush are still this context's audio and
                # belong to a sentence already returned; chunks with no flush_id predate the
                # first boundary. Only the current sentence's are collected here.
                if getattr(event, "flush_id", None) not in (None, expected_flush):
                    continue
                audio = getattr(event, "audio", None)
                if audio:
                    pcm.extend(audio)
                    if on_pcm is not None:
                        # Deliberately not guarded: the sink is TrackStream.feed, which only
                        # enqueues and is documented never to raise. Swallowing an exception
                        # here would let a broken sink return a complete buffer that the
                        # caller then believes was already spoken — silence the caller cannot
                        # detect. Failing loudly drops to the one-shot fallback instead.
                        await on_pcm(bytes(audio))
                continue

            if kind == "flush_done" and getattr(event, "flush_id", None) == expected_flush:
                break

            if kind == "done":
                # The server ended the context while we were still reading a sentence. Whatever
                # arrived is what there is; it is not an error, but nothing more is coming.
                self._closed = True
                break

        duration_ms = int(len(pcm) / 2 / self._sample_rate * 1000) if self._sample_rate else 0
        return bytes(wav_header(len(pcm), self._sample_rate) + pcm), duration_ms

    async def aclose(self) -> None:
        """End the turn. Idempotent — the caller closes on the final sentence AND on teardown,
        and a turn whose final sentence never arrives is closed by the sweep instead."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._transport.no_more_inputs()
        except Exception:
            logger.debug("prosody_context_close_failed", exc_info=True)

    async def abandon(self) -> None:
        """Drop the turn without waiting for it to finish speaking — used when the room ends or
        the speaker's voice is replaced mid-turn."""
        if self._closed:
            return
        self._closed = True
        try:
            await self._transport.cancel()
        except Exception:
            logger.debug("prosody_context_cancel_failed", exc_info=True)
