"""Publishes synthesized TTS audio into the LiveKit room paired with a translation
room (MeetingRoom.ProviderRoomName == translationRoomId), as a dedicated bot
participant per target language.

The frontend's room page already renders <RoomAudioRenderer /> (from
@livekit/components-react) for the real-time Meeting/LiveKit connection every
translation room already makes — that component plays every subscribed audio track
in the room automatically. Publishing here means no new frontend playback code is
needed for basic playback; it also sidesteps the WAV-chunking problem entirely, since
WebRTC transport is what makes delivery live, not how the audio was internally
produced. (Frontend still needs to filter which bot identity to actually listen to
when multiple target languages are active in the same room — see room page.)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any

import numpy as np
from livekit import api, rtc

from shared.config import LiveKitSettings
from shared.logger import get_logger

logger = get_logger(__name__)

FRAME_MS = 20

# A short pause after publish_track() before the first capture_frame(), as a safety
# margin — isolated testing (see session notes) did not reproduce any failure with or
# without this delay, but it's cheap insurance against a slow WebRTC negotiation.
_PUBLISH_SETTLE_S = 0.2

# Mirrors stt_worker's SESSION_IDLE_TIMEOUT_S. A bot is keyed by (meeting_id, speaker_id,
# target_lang) — once a listener switches away from that language (see
# TranslationRoomHub.SetListenLanguage) or leaves, translation_worker stops producing
# that target_lang on any new utterance (_get_target_languages re-reads the room's
# listen-language hash on every message), so publish_pcm() is simply never called again
# for that key. Nothing previously told this bot to disconnect, so it — and its LiveKit
# room connection — would otherwise leak for the rest of the process's lifetime.
# Cloud counts every connected interpreter bot toward concurrent participants and
# participant minutes. One minute retains the handshake-reuse benefit without leaving
# unused speaker/language/voice variants connected for five extra minutes.
SESSION_IDLE_TIMEOUT_S = 60.0

# Sweeping only from _get_or_create_bot is not enough on its own: that is the one place a
# NEW bot is created, so as soon as synthesis stops — translation switched off, or simply
# nobody speaking — there is no next creation left to trigger the sweep and every bot stays
# connected for the rest of the process's life. That also has a user-visible cost beyond the
# leaked participant minutes: the web client treats a present ai-interpreter track as "this
# speaker already has a dub" (see FilteredRoomAudio), so a bot nobody swept keeps a real
# speaker's microphone muted for cross-language listeners. Reap on a timer as well, so idle
# bots leave whether or not anything else in the pipeline is still running.
_REAP_INTERVAL_S = 15.0

# Each STT chunk's translated text is synthesized as its own independent Cartesia call,
# so every clip starts/ends at whatever sample amplitude Cartesia happened to render —
# rarely zero. Splicing those hard-cut edges back-to-back on one continuous LiveKit
# track produces an audible click/pop at the start of every chunk. A short linear
# ramp in/out removes the discontinuity.
_FADE_MS = 8


def _fade_samples(sample_rate: int) -> int:
    return int(sample_rate * _FADE_MS / 1000)


def _fade(pcm_s16le: bytes, sample_rate: int, *, head: bool, tail: bool) -> bytes:
    """Ramp one or both edges of a buffer.

    Both edges is the one-shot case: a whole sentence arrives at once and its two hard cuts
    are both inside this buffer. Streaming splits that in two — the head ramp belongs to the
    first slice pushed and the tail ramp to the last, and no slice in between gets either, or
    the sentence would wobble in and out of audibility once per Cartesia chunk.
    """
    if not pcm_s16le:
        return pcm_s16le

    # Drop a stray trailing byte so the buffer is a whole number of 16-bit samples —
    # np.frombuffer(int16) rejects an odd-length buffer, and a lone half-sample carries
    # no usable audio anyway (the matching partial-frame drop happens in _capture_all).
    if len(pcm_s16le) % 2:
        pcm_s16le = pcm_s16le[:-1]

    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32)
    fade_len = min(len(samples) // 2, _fade_samples(sample_rate))
    if fade_len <= 0:
        return pcm_s16le

    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    if head:
        samples[:fade_len] *= ramp
    if tail:
        samples[-fade_len:] *= ramp[::-1]
    return samples.astype(np.int16).tobytes()


def _apply_fade(pcm_s16le: bytes, sample_rate: int) -> bytes:
    return _fade(pcm_s16le, sample_rate, head=True, tail=True)


# (meeting_id, speaker_id, target_lang, voice_key) — voice_key "" is the shared
# default/cloned track (backward-compatible identity, unchanged from before per-
# listener voice preferences existed); a non-empty voice_key ("voice-{id8}") is an
# extra track for listeners who explicitly picked that voice via SetVoicePreference.
_BotKey = tuple[str, str, str, str]

# How long close() may wait for queued audio to finish reaching the track before it gives up
# and cancels the pump.
#
# THIS BOUND IS THE POINT, NOT THE VALUE. capture_frame() applies back-pressure — it returns
# only once the track has room — so draining a sentence takes roughly as long as the sentence
# lasts, and a wedged AudioSource never returns at all. close() runs while this key's lock is
# held and while tts_worker holds its own per-(speaker, target_lang) lock, so an unbounded wait
# here would stop that speaker's dub for the rest of the meeting: the exact failure
# ProsodyContext.SENTENCE_TIMEOUT_SECONDS exists to prevent, reintroduced one layer down.
#
# 30s because a dub is bounded by STT's 6-second chunk and comes back well under 15s even at
# the slowest speed setting. Anything near this is already a failure, not a long sentence.
_DRAIN_TIMEOUT_S = 30.0


class TrackStream:
    """One sentence's audio, pushed onto the track while Cartesia is still generating it.

    WT-397. `publish_pcm` takes a finished sentence, so the first sample reached the listener
    only after the last one had been synthesized — Cartesia streams audio back chunk by chunk
    and all of it was held until the sentence was complete. Feeding each chunk on as it lands
    removes that wait from the stage measured at p50 1.00s.

    WHY THERE IS A QUEUE AND A PUMP RATHER THAN A DIRECT AWAIT
        `feed` is called from inside ProsodyContext._collect, which is wrapped in a 6-second
        timeout. capture_frame() back-pressures to real time, so awaiting it there would make
        a 10-second dub take 10 seconds to hand over and blow that timeout — turning every
        long sentence into a fallback, which is the double-audio hazard this ticket is mostly
        about. So `feed` only enqueues, returns immediately, and a pump task does the waiting
        outside the reader's timeout.

    WHAT `spoken_bytes` IS FOR
        The caller needs to know, after the fact, whether the listener heard any of this
        sentence — because if they did, the one-shot fallback must NOT be played on top of it.
        Zero means nothing was published and the ordinary publish path is still safe.
    """

    def __init__(self, publisher: LiveKitTTSPublisher, key: _BotKey, sample_rate: int) -> None:
        self._publisher = publisher
        self._key = key
        self._sample_rate = sample_rate
        self._frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2
        # Held back from every write so the closing ramp has real audio to ramp down.
        #
        # A WHOLE FRAME on top of the fade, and that part is not padding. `_capture_from`
        # cannot send a trailing partial frame, so the last thing pushed must have at least one
        # complete frame in it for the ramp to live in — hold back only the fade and the final
        # slice is shorter than a frame, gets dropped in its entirety, and the sentence ends on
        # the hard cut the fade exists to remove. Costs 20ms of latency, against a stage
        # measured in seconds.
        self._keep_bytes = int(sample_rate * FRAME_MS / 1000) * 2 + _fade_samples(sample_rate) * 2
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._pending = bytearray()
        self._pump: asyncio.Task[None] | None = None
        self._faded_in = False
        self._spoken_bytes = 0
        self._broken = False
        self._first_audio_at: float | None = None

    @property
    def spoken_bytes(self) -> int:
        return self._spoken_bytes

    @property
    def first_audio_at(self) -> float | None:
        """`time.monotonic()` when the first frame reached the track, or None if none did.

        This is the number WT-397 actually moves. `tts_synthesis` measures the whole call and
        now spans the hand-over as well as the generation, so it RISES with streaming on even
        though the listener starts hearing the sentence far sooner — see the caller.
        """
        return self._first_audio_at

    async def feed(self, pcm_s16le: bytes) -> None:
        """Hand one Cartesia chunk over. Never blocks, never raises — see the class docstring."""
        if pcm_s16le:
            self._queue.put_nowait(pcm_s16le)

    def start(self) -> None:
        if self._pump is None:
            self._pump = asyncio.create_task(self._pump_loop())

    async def close(self) -> None:
        if self._pump is None:
            return
        self._queue.put_nowait(None)
        try:
            await asyncio.wait_for(self._pump, timeout=_DRAIN_TIMEOUT_S)
        except TimeoutError:
            self._pump.cancel()
            with suppress(asyncio.CancelledError):
                await self._pump
            logger.warning(
                "livekit_tts_stream_drain_timeout",
                meeting_id=self._key[0],
                speaker_id=self._key[1],
                target_lang=self._key[2],
                spoken_bytes=self._spoken_bytes,
            )
        finally:
            self._pump = None

    async def _pump_loop(self) -> None:
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            self._pending.extend(chunk)
            await self._drain(final=False)
        await self._drain(final=True)

    async def _drain(self, *, final: bool) -> None:
        if final:
            out = bytes(self._pending)
            self._pending.clear()
            # Trim to whole frames BEFORE fading: what _capture_from will not send must not be
            # where the ramp went. _keep_bytes guarantees a full frame survives here for every
            # sentence that reached the non-final branch at all.
            usable = len(out) - (len(out) % self._frame_bytes)
            if usable <= 0:
                return
            # A sentence shorter than _keep_bytes never reached the non-final branch, so both
            # of its edges are still here.
            out = _fade(out[:usable], self._sample_rate, head=not self._faded_in, tail=True)
            self._faded_in = True
            await self._capture(out)
            return

        room = len(self._pending) - self._keep_bytes
        usable = room - (room % self._frame_bytes) if room > 0 else 0
        if usable <= 0:
            return
        out = bytes(self._pending[:usable])
        del self._pending[:usable]
        out = _fade(out, self._sample_rate, head=not self._faded_in, tail=False)
        self._faded_in = True
        await self._capture(out)

    async def _capture(self, pcm_s16le: bytes) -> None:
        """Push one slice, with the same evict-and-resume retry publish_pcm uses.

        Once both attempts have failed the stream stops trying for the rest of the sentence:
        re-connecting per chunk would mint a new bot identity every few hundred milliseconds,
        which costs participant minutes and makes the listener hear the dub restart repeatedly.
        """
        if self._broken or not pcm_s16le:
            return

        remaining = pcm_s16le
        for attempt in range(2):
            try:
                bot = await self._publisher._get_or_create_bot(*self._key, self._sample_rate)
            except Exception:
                logger.exception(
                    "livekit_tts_bot_connect_error",
                    meeting_id=self._key[0],
                    speaker_id=self._key[1],
                    target_lang=self._key[2],
                    voice_key=self._key[3],
                )
                self._broken = True
                return

            bot["last_used"] = time.monotonic()
            sent = await self._publisher._capture_from(bot["source"], remaining, self._sample_rate)
            if sent and self._first_audio_at is None:
                self._first_audio_at = time.monotonic()
            self._spoken_bytes += sent
            if sent >= len(remaining):
                return

            remaining = remaining[sent:]
            logger.warning(
                "livekit_tts_stream_retry",
                meeting_id=self._key[0],
                speaker_id=self._key[1],
                target_lang=self._key[2],
                voice_key=self._key[3],
                attempt=attempt,
                remaining_bytes=len(remaining),
            )
            stale = self._publisher._bots.pop(self._key, None)
            if stale is not None:
                await self._publisher._close_bot(stale)

        self._broken = True


class LiveKitTTSPublisher:
    """One bot participant + audio track per (meeting_id, speaker_id, target_lang,
    voice_key), reused across every synthesized sentence for that key — same
    session-reuse pattern as stt_worker's realtime sessions, so only the first
    sentence for a given key pays the room-join handshake.

    Keying by speaker (not just language) is what lets concurrent speakers be dubbed in
    PARALLEL: each speaker's interpreted audio is its own independent WebRTC track that
    the client mixes natively, instead of every speaker's dub being serialized onto a
    single shared "ai-interpreter-{lang}" track. It also lines up one interpreter track
    per human speaker, so a listener can attribute (and a cloned voice can match) the dub
    to the person who actually spoke.

    Keying by voice_key on top of that is what lets a listener hear a DIFFERENT voice
    for the same speaker+language than everyone else who hasn't picked one — see
    TTSWorker._resolve_voice_variants.
    """

    def __init__(self, settings: LiveKitSettings) -> None:
        self.settings = settings
        self._bots: dict[_BotKey, dict[str, Any]] = {}
        # One lock per key, held for a caller's ENTIRE publish_pcm() call — not just bot
        # creation. tts_worker now dispatches translate:results messages concurrently
        # (see TTSWorker._consume_loop), so two sentences for the SAME key can genuinely
        # run at the same time; without this, both could reach _get_or_create_bot()
        # before either finishes connecting (the original bug this lock existed to
        # prevent — LiveKit kicks the second connection with "DuplicateIdentity"), and
        # even after that, two concurrent _capture_all() calls on the SAME AudioSource
        # would interleave sentence 2's frames into the middle of sentence 1's,
        # corrupting playback order. A different key (another speaker, another target
        # language, or another voice variant) has its own independent lock and runs
        # fully in parallel — this is what actually lets concurrent speakers be dubbed
        # in parallel end-to-end.
        self._locks: dict[_BotKey, asyncio.Lock] = {}
        # Started lazily by the first bot creation (see _ensure_reaper) rather than in
        # __init__, which runs outside any event loop.
        self._reaper: asyncio.Task[None] | None = None

    async def publish_pcm(
        self,
        meeting_id: str,
        speaker_id: str,
        target_lang: str,
        pcm_s16le: bytes,
        sample_rate: int,
        voice_key: str = "",
    ) -> None:
        """Feed raw 16-bit mono PCM (no WAV header) into this speaker's interpreter track.

        `voice_key` selects WHICH track for this (speaker, target_lang): "" is the
        shared default/cloned track, anything else ("voice-{id8}") is a dedicated
        track for listeners who explicitly chose that voice.

        capture_frame() fails intermittently with "InvalidState" — a known, sporadic,
        upstream LiveKit issue (livekit/rust-sdks#497, livekit/agents-js#270), not tied
        to any particular usage pattern we could reproduce deterministically. There's
        no documented fix, so the pragmatic mitigation is: on failure, evict the bot
        and retry once on a brand-new connection before giving up for this sentence.
        """
        if not pcm_s16le:
            return

        key: _BotKey = (meeting_id, speaker_id, target_lang, voice_key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        # Trimmed to whole frames before fading, not after. _capture_from cannot send a
        # trailing partial frame, so fading the raw buffer put the closing ramp into bytes that
        # were then discarded — for any sentence whose length was not a frame multiple, which
        # is most of them, the dub ended on the hard cut the fade exists to remove. Found while
        # building the streaming path (WT-397); the same arithmetic was always wrong here.
        frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2
        if frame_bytes > 0:
            pcm_s16le = pcm_s16le[: len(pcm_s16le) - (len(pcm_s16le) % frame_bytes)]
            if not pcm_s16le:
                return
        # Faded once, here, rather than inside each attempt: a retry resumes partway through
        # this buffer, and re-fading a slice would put a fade-in in the middle of a word.
        pcm_s16le = _apply_fade(pcm_s16le, sample_rate)
        sent = 0
        async with lock:
            for attempt in range(2):
                try:
                    bot = await self._get_or_create_bot(
                        meeting_id, speaker_id, target_lang, voice_key, sample_rate
                    )
                    bot["last_used"] = time.monotonic()
                except Exception:
                    logger.exception(
                        "livekit_tts_bot_connect_error",
                        meeting_id=meeting_id,
                        speaker_id=speaker_id,
                        target_lang=target_lang,
                        voice_key=voice_key,
                    )
                    return

                sent += await self._capture_from(bot["source"], pcm_s16le[sent:], sample_rate)
                if sent >= len(pcm_s16le):
                    return

                logger.warning(
                    "livekit_tts_publish_retry",
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    target_lang=target_lang,
                    voice_key=voice_key,
                    attempt=attempt,
                    resume_byte=sent,
                    total_bytes=len(pcm_s16le),
                )
                # Drop the connection, not just our handle on it. WT-269: a bot left
                # connected here keeps holding this identity in the room, so the retry's
                # connect() below can only be resolved by LiveKit evicting the old
                # participant — an extra, invisible reconnect per failure on a project
                # that is already rate-limit sensitive.
                stale = self._bots.pop(key, None)
                if stale is not None:
                    await self._close_bot(stale)

    @asynccontextmanager
    async def stream(
        self,
        meeting_id: str,
        speaker_id: str,
        target_lang: str,
        sample_rate: int,
        voice_key: str = "",
    ) -> AsyncIterator[TrackStream]:
        """Open this key's track for one sentence that is still being generated.

        Holds the same per-key lock publish_pcm holds, for the same reason and now over a
        longer window: synthesis and publishing overlap, so the lock covers both. That is not
        extra contention — tts_worker already serialises the same (speaker, target_lang) at its
        own consume loop, and a different speaker, language or voice variant has its own lock
        and still runs fully in parallel.

        The stream is always closed on the way out, including when the body raised: whatever
        Cartesia managed to send before it failed is audio the listener should hear, and the
        caller decides what to do about the rest by reading `spoken_bytes` afterwards.
        """
        key: _BotKey = (meeting_id, speaker_id, target_lang, voice_key)
        lock = self._locks.setdefault(key, asyncio.Lock())
        track = TrackStream(self, key, sample_rate)
        async with lock:
            track.start()
            try:
                yield track
            finally:
                await track.close()

    async def _capture_from(
        self, source: rtc.AudioSource, pcm_s16le: bytes, sample_rate: int
    ) -> int:
        """Push frames; return how many bytes actually made it onto the track.

        This used to answer True/False, and the caller answered a False by replaying the
        WHOLE sentence on a fresh connection. capture_frame() fails sporadically with
        InvalidState, so a failure at nine tenths of a line meant the listener heard nine
        tenths of it and then the entire line again — which is worse than the truncation it
        was trying to repair, because a dub that repeats is a dub you have to think about.

        Returning progress lets the retry resume from the break instead. Nothing is spoken
        twice, and nothing is lost unless BOTH attempts fail at the same point.
        """
        frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2  # 16-bit mono
        if frame_bytes <= 0:
            return len(pcm_s16le)

        usable_len = len(pcm_s16le) - (len(pcm_s16le) % frame_bytes)
        sent = 0
        try:
            for i in range(0, usable_len, frame_bytes):
                chunk = pcm_s16le[i : i + frame_bytes]
                frame = rtc.AudioFrame(
                    data=chunk,
                    sample_rate=sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk) // 2,
                )
                await source.capture_frame(frame)
                sent = i + frame_bytes
        except Exception:
            logger.exception("livekit_tts_publish_error", sent_bytes=sent)
            return sent
        # The trailing partial frame is shorter than one frame (< 10ms) and cannot be
        # captured; counting it keeps "sent >= len" meaning "the line finished".
        return len(pcm_s16le)

    async def _get_or_create_bot(
        self, meeting_id: str, speaker_id: str, target_lang: str, voice_key: str, sample_rate: int
    ) -> dict[str, Any]:
        """Caller (publish_pcm) already holds this key's lock for its whole call, so no
        locking is needed here — only one task can ever be inside this method for a
        given key at a time."""
        self._ensure_reaper()
        self._sweep_idle_bots()

        key: _BotKey = (meeting_id, speaker_id, target_lang, voice_key)
        cached = self._bots.get(key)
        if cached is not None:
            return cached

        # Language first so the frontend can match by a stable prefix
        # (`ai-interpreter-{lang}-`) — speaker_id is a GUID that contains its own
        # hyphens, so putting it last keeps the language token unambiguous. voice_key
        # (when set — "voice-{id8}") sits between language and speaker; a GUID never
        # starts with "voice-", so the frontend can tell a voice-suffixed identity
        # apart from a bare default one unambiguously. The `ai-interpreter-` prefix
        # still matches livekit_ingress_worker's _is_ai_bot_identity filter, so this
        # bot's own track is never re-ingested.
        identity = (
            f"ai-interpreter-{target_lang}-{voice_key}-{speaker_id}"
            if voice_key
            else f"ai-interpreter-{target_lang}-{speaker_id}"
        )
        token = (
            api.AccessToken(self.settings.api_key, self.settings.api_secret)
            .with_identity(identity)
            .with_name(f"AI Interpreter ({target_lang})")
            .with_grants(api.VideoGrants(room_join=True, room=meeting_id))
            .to_jwt()
        )

        room = rtc.Room()
        await room.connect(self.settings.url, token)

        source = rtc.AudioSource(sample_rate=sample_rate, num_channels=1)
        track = rtc.LocalAudioTrack.create_audio_track("tts-audio", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        await asyncio.sleep(_PUBLISH_SETTLE_S)

        bot = {"room": room, "source": source, "last_used": time.monotonic()}
        self._bots[key] = bot
        logger.info(
            "livekit_tts_bot_published",
            meeting_id=meeting_id,
            speaker_id=speaker_id,
            target_lang=target_lang,
            voice_key=voice_key,
            identity=identity,
        )
        return bot

    def _ensure_reaper(self) -> None:
        """Start the idle-bot reaper once, from whichever coroutine first creates a bot."""
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL_S)
            try:
                self._sweep_idle_bots()
            except Exception:
                # A single bad sweep must not silently end the loop and quietly restore the
                # leak this exists to prevent.
                logger.exception("livekit_tts_reaper_error")

    def _sweep_idle_bots(self) -> None:
        now = time.monotonic()
        stale = [
            k
            for k, b in self._bots.items()
            if now - b["last_used"] > SESSION_IDLE_TIMEOUT_S and not self._is_publishing(k)
        ]
        for k in stale:
            bot = self._bots.pop(k)
            self._locks.pop(k, None)
            asyncio.create_task(self._close_bot(bot))
            logger.info(
                "livekit_tts_bot_idle_closed",
                meeting_id=k[0],
                speaker_id=k[1],
                target_lang=k[2],
                voice_key=k[3],
            )

    def _is_publishing(self, key: _BotKey) -> bool:
        """Whether publish_pcm currently holds this key's lock.

        The inline sweep could only ever run for a DIFFERENT key than the caller's, but the
        reaper runs on its own schedule and would otherwise be free to disconnect a bot
        mid-sentence. `last_used` is stamped before synthesis is captured, so a clip that
        somehow outran SESSION_IDLE_TIMEOUT_S would be cut off in the middle.
        """
        lock = self._locks.get(key)
        return lock is not None and lock.locked()

    @staticmethod
    async def _close_bot(bot: dict[str, Any]) -> None:
        try:
            await bot["room"].disconnect()
        except Exception:
            logger.exception("livekit_tts_bot_close_error")

    async def close_all(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
            with suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        for bot in self._bots.values():
            try:
                await bot["room"].disconnect()
            except Exception:
                logger.exception("livekit_tts_bot_close_error")
        self._bots.clear()
