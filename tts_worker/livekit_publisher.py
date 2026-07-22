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

# Each STT chunk's translated text is synthesized as its own independent Cartesia call,
# so every clip starts/ends at whatever sample amplitude Cartesia happened to render —
# rarely zero. Splicing those hard-cut edges back-to-back on one continuous LiveKit
# track produces an audible click/pop at the start of every chunk. A short linear
# ramp in/out removes the discontinuity.
_FADE_MS = 8


def _apply_fade(pcm_s16le: bytes, sample_rate: int) -> bytes:
    if not pcm_s16le:
        return pcm_s16le

    # Drop a stray trailing byte so the buffer is a whole number of 16-bit samples —
    # np.frombuffer(int16) rejects an odd-length buffer, and a lone half-sample carries
    # no usable audio anyway (the matching partial-frame drop happens in _capture_all).
    if len(pcm_s16le) % 2:
        pcm_s16le = pcm_s16le[:-1]

    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32)
    fade_len = min(len(samples) // 2, int(sample_rate * _FADE_MS / 1000))
    if fade_len <= 0:
        return pcm_s16le

    ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    samples[:fade_len] *= ramp
    samples[-fade_len:] *= ramp[::-1]
    return samples.astype(np.int16).tobytes()


class LiveKitTTSPublisher:
    """One bot participant + audio track per (meeting_id, speaker_id, target_lang),
    reused across every synthesized sentence for that triple — same session-reuse
    pattern as stt_worker's realtime sessions, so only the first sentence for a given
    speaker+language pays the room-join handshake.

    Keying by speaker (not just language) is what lets concurrent speakers be dubbed in
    PARALLEL: each speaker's interpreted audio is its own independent WebRTC track that
    the client mixes natively, instead of every speaker's dub being serialized onto a
    single shared "ai-interpreter-{lang}" track. It also lines up one interpreter track
    per human speaker, so a listener can attribute (and a cloned voice can match) the dub
    to the person who actually spoke.
    """

    def __init__(self, settings: LiveKitSettings) -> None:
        self.settings = settings
        self._bots: dict[tuple[str, str, str], dict] = {}
        # publish_pcm() is called fire-and-forget (asyncio.create_task) from
        # tts_worker, so two sentences translated close together can both reach
        # _get_or_create_bot() before either has finished connecting — without this
        # lock, both create a room connection under the SAME bot identity, and
        # LiveKit kicks one with "DuplicateIdentity" (observed live).
        self._locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    async def publish_pcm(
        self,
        meeting_id: str,
        speaker_id: str,
        target_lang: str,
        pcm_s16le: bytes,
        sample_rate: int,
    ) -> None:
        """Feed raw 16-bit mono PCM (no WAV header) into this speaker's interpreter track.

        capture_frame() fails intermittently with "InvalidState" — a known, sporadic,
        upstream LiveKit issue (livekit/rust-sdks#497, livekit/agents-js#270), not tied
        to any particular usage pattern we could reproduce deterministically. There's
        no documented fix, so the pragmatic mitigation is: on failure, evict the bot
        and retry once on a brand-new connection before giving up for this sentence.
        """
        if not pcm_s16le:
            return

        key = (meeting_id, speaker_id, target_lang)
        for attempt in range(2):
            try:
                bot = await self._get_or_create_bot(meeting_id, speaker_id, target_lang, sample_rate)
            except Exception:
                logger.exception(
                    "livekit_tts_bot_connect_error",
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    target_lang=target_lang,
                )
                return

            if await self._capture_all(bot["source"], pcm_s16le, sample_rate):
                return

            logger.warning(
                "livekit_tts_publish_retry",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                target_lang=target_lang,
                attempt=attempt,
            )
            self._bots.pop(key, None)

    async def _capture_all(self, source: rtc.AudioSource, pcm_s16le: bytes, sample_rate: int) -> bool:
        """Push all frames; returns False (and logs) on the first capture_frame failure."""
        frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2  # 16-bit mono
        if frame_bytes <= 0:
            return True

        pcm_s16le = _apply_fade(pcm_s16le, sample_rate)
        usable_len = len(pcm_s16le) - (len(pcm_s16le) % frame_bytes)
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
            return True
        except Exception:
            logger.exception("livekit_tts_publish_error")
            return False

    async def _get_or_create_bot(
        self, meeting_id: str, speaker_id: str, target_lang: str, sample_rate: int
    ) -> dict:
        key = (meeting_id, speaker_id, target_lang)
        cached = self._bots.get(key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check: another task may have created the bot while we waited for the lock.
            cached = self._bots.get(key)
            if cached is not None:
                return cached

            # Language first so the frontend can match by a stable prefix
            # (`ai-interpreter-{lang}-`) — speaker_id is a GUID that contains its own
            # hyphens, so putting it last keeps the language token unambiguous. The
            # `ai-interpreter-` prefix still matches livekit_ingress_worker's
            # _is_ai_bot_identity filter, so this bot's own track is never re-ingested.
            identity = f"ai-interpreter-{target_lang}-{speaker_id}"
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

            bot = {"room": room, "source": source}
            self._bots[key] = bot
            logger.info(
                "livekit_tts_bot_published",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                target_lang=target_lang,
                identity=identity,
            )
            return bot

    async def close_all(self) -> None:
        for bot in self._bots.values():
            try:
                await bot["room"].disconnect()
            except Exception:
                logger.exception("livekit_tts_bot_close_error")
        self._bots.clear()
