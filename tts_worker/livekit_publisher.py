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

from livekit import api, rtc

from shared.config import LiveKitSettings
from shared.logger import get_logger

logger = get_logger(__name__)

FRAME_MS = 20

# A short pause after publish_track() before the first capture_frame(), as a safety
# margin — isolated testing (see session notes) did not reproduce any failure with or
# without this delay, but it's cheap insurance against a slow WebRTC negotiation.
_PUBLISH_SETTLE_S = 0.2


class LiveKitTTSPublisher:
    """One bot participant + audio track per (meeting_id, target_lang), reused
    across every synthesized sentence for that pair — same session-reuse pattern as
    stt_worker's realtime sessions, so only the first sentence in a target language
    pays the room-join handshake.
    """

    def __init__(self, settings: LiveKitSettings) -> None:
        self.settings = settings
        self._bots: dict[tuple[str, str], dict] = {}
        # publish_pcm() is called fire-and-forget (asyncio.create_task) from
        # tts_worker, so two sentences translated close together can both reach
        # _get_or_create_bot() before either has finished connecting — without this
        # lock, both create a room connection under the SAME bot identity, and
        # LiveKit kicks one with "DuplicateIdentity" (observed live).
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def publish_pcm(
        self, meeting_id: str, target_lang: str, pcm_s16le: bytes, sample_rate: int
    ) -> None:
        """Feed raw 16-bit mono PCM (no WAV header) into the bot's audio track.

        capture_frame() fails intermittently with "InvalidState" — a known, sporadic,
        upstream LiveKit issue (livekit/rust-sdks#497, livekit/agents-js#270), not tied
        to any particular usage pattern we could reproduce deterministically. There's
        no documented fix, so the pragmatic mitigation is: on failure, evict the bot
        and retry once on a brand-new connection before giving up for this sentence.
        """
        if not pcm_s16le:
            return

        for attempt in range(2):
            try:
                bot = await self._get_or_create_bot(meeting_id, target_lang, sample_rate)
            except Exception:
                logger.exception(
                    "livekit_tts_bot_connect_error", meeting_id=meeting_id, target_lang=target_lang
                )
                return

            if await self._capture_all(bot["source"], pcm_s16le, sample_rate):
                return

            logger.warning(
                "livekit_tts_publish_retry",
                meeting_id=meeting_id,
                target_lang=target_lang,
                attempt=attempt,
            )
            self._bots.pop((meeting_id, target_lang), None)

    async def _capture_all(self, source: rtc.AudioSource, pcm_s16le: bytes, sample_rate: int) -> bool:
        """Push all frames; returns False (and logs) on the first capture_frame failure."""
        frame_bytes = int(sample_rate * FRAME_MS / 1000) * 2  # 16-bit mono
        if frame_bytes <= 0:
            return True

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
        self, meeting_id: str, target_lang: str, sample_rate: int
    ) -> dict:
        key = (meeting_id, target_lang)
        cached = self._bots.get(key)
        if cached is not None:
            return cached

        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            # Re-check: another task may have created the bot while we waited for the lock.
            cached = self._bots.get(key)
            if cached is not None:
                return cached

            identity = f"ai-interpreter-{target_lang}"
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
