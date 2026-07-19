"""TTS Worker — Consumes translated text, produces synthesized audio.

Pipeline:
    Redis Stream (translate:results:{meetingId})
    → Cartesia Sonic Turbo (default voice until clone ready, then cloned voice)
    → Redis Stream (tts:results:{meetingId})

Voice cloning:
    Background task buffers audio:chunks per speaker.
    Once voice_clone_min_seconds of audio collected → POST /voices/clone.
    voice_id cached in Redis: voice:{meeting_id}:{speaker_id}
    All synthesis calls after that use the cloned voice.
"""

from __future__ import annotations

import asyncio
import hashlib
import time

from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage
from tts_worker.livekit_publisher import LiveKitTTSPublisher
from tts_worker.synthesizer import CartesiaSynthesizer

# Standard WAV header size for the pcm_s16le format CartesiaSynthesizer requests —
# used to strip the header before feeding audio into the LiveKit track (which wants
# raw PCM frames, not a WAV container).
_WAV_HEADER_BYTES = 44

# Cartesia's voices.clone() requires a concrete `language`, but AudioChunkMessage.language
# defaults to "auto" (STT does language auto-detection, not the audio-chunk producer) — so
# fall back to "en" for anything Cartesia's SDK wouldn't accept as a real language code.
_CARTESIA_SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it", "ko", "nl", "pl", "ru", "sv",
    "tr", "tl", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms", "sk", "da", "ta", "uk",
    "hu", "no", "vi", "bn", "th", "he", "ka", "id", "te", "gu", "kn", "ml", "mr", "pa",
}


def _clone_language(hint: str) -> str:
    return hint if hint in _CARTESIA_SUPPORTED_LANGUAGES else "en"


class TTSWorker(BaseWorker):
    """Text-to-Speech worker using Cartesia Sonic Turbo."""

    worker_name = "tts"
    input_stream = "translate:results"
    consumer_group = "tts-workers"
    _audio_consumer_group = "tts-audio-workers"
    _running = True

    def __init__(
        self,
        tts_settings: TTSSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_settings = tts_settings or TTSSettings()
        self.cartesia: CartesiaSynthesizer | None = None
        self.livekit_publisher: LiveKitTTSPublisher | None = None

    async def load_model(self) -> None:
        self.cartesia = CartesiaSynthesizer(
            api_key=self.tts_settings.api_key,
            model=self.tts_settings.model,
            sample_rate=self.tts_settings.sample_rate,
        )
        await self.cartesia.load()
        self.livekit_publisher = LiveKitTTSPublisher(self.settings.livekit)
        asyncio.create_task(self._consume_audio_for_cloning())
        self.logger.info("tts_worker_ready", model=self.tts_settings.model)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Synthesize one translated text segment."""
        translation = TranslationResultMessage.from_redis(data)
        text = translation.translated_text

        route_status = self._route_states.get(translation.meeting_id, "AUDIO_ROUTING_ACTIVE")
        if route_status == "PAUSED":
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - translation.timestamp_ms
        await self.redis.publish_telemetry(translation.meeting_id, self.worker_name, e2e_latency_ms)

        if route_status == "TEXT_ONLY_MODE" or not text.strip():
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

        voice_id = await self._get_voice_id(translation.meeting_id, translation.speaker_id)
        voice_type = "cloned" if voice_id else "default"

        cache_key = self._cache_key(
            speaker_id=translation.speaker_id,
            target_lang=translation.target_lang,
            text=text,
            voice_mode=voice_type,
        )

        if self.tts_settings.cache_enabled:
            cached_audio = await self.redis.get(cache_key)
            if cached_audio:
                # synthesize() wasn't called (that's the point of the cache), so it never
                # told us which Cartesia voice a fresh call would resolve to — recompute
                # the same way it would, without an API call.
                resolved_voice_id = voice_id or CartesiaSynthesizer._default_voice_id(
                    translation.target_lang
                )
                await self._publish_result(
                    translation=translation,
                    audio_bytes=cached_audio,
                    duration_ms=0,
                    voice_type=voice_type,
                    voice_id=voice_id,
                    provider_voice_id=resolved_voice_id,
                    cache_key=cache_key,
                    cache_hit=True,
                    synthesis_latency_ms=0,
                )
                return

        t0 = time.monotonic()
        try:
            audio_bytes, duration_ms, resolved_voice_id = await self.cartesia.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
            )
        except Exception as e:
            self.logger.error("cartesia_synthesis_failed", error=str(e))
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="tts_unavailable",
                payload={"error": str(e)},
            )
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

        synthesis_latency_ms = int((time.monotonic() - t0) * 1000)

        if audio_bytes:
            await self._publish_result(
                translation=translation,
                audio_bytes=audio_bytes,
                duration_ms=duration_ms,
                voice_type=voice_type,
                voice_id=voice_id,
                provider_voice_id=resolved_voice_id,
                cache_key=cache_key,
                cache_hit=False,
                synthesis_latency_ms=synthesis_latency_ms,
            )
            if self.tts_settings.cache_enabled:
                await self.redis.set_with_ttl(
                    cache_key, audio_bytes, self.tts_settings.cache_ttl_seconds
                )

        if translation.is_final_chunk:
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="final_chunk_processed",
                payload={"segmentId": translation.segment_id},
            )

        self.logger.info(
            "audio_synthesized",
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            voice_type=voice_type,
            duration_ms=duration_ms,
            synthesis_latency_ms=synthesis_latency_ms,
            text=text[:60],
            is_final=translation.is_final_chunk,
        )

    async def _publish_result(
        self,
        translation: TranslationResultMessage,
        audio_bytes: bytes,
        duration_ms: int,
        voice_type: str,
        voice_id: str | None,
        provider_voice_id: str,
        cache_key: str,
        cache_hit: bool,
        synthesis_latency_ms: int,
    ) -> None:
        result = TTSResultMessage(
            segment_id=translation.segment_id,
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            audio_data=audio_bytes,
            duration_ms=duration_ms,
            voice_type=voice_type,
            voice_mode=voice_type,
            clone_strength=1.0 if voice_id else 0.0,
            anchor_provider="cartesia",
            clone_provider="cartesia" if voice_id else "",
            provider_voice_id=provider_voice_id,
            render_location="server",
            cache_key=cache_key,
            cache_hit=cache_hit,
            synthesis_latency_ms=synthesis_latency_ms,
            fallback_reason="" if voice_id else "voice_profile_not_ready",
            target_lang=translation.target_lang,
            is_final_chunk=translation.is_final_chunk,
            timestamp_ms=translation.timestamp_ms,
        )
        await self.publish("tts:results", translation.meeting_id, result.to_redis())

        # Also publish as a LiveKit audio track — see livekit_publisher.py for why:
        # the room page's existing RoomAudioRenderer plays it automatically. Fire-and-
        # forget: this must never block or fail the tts:results publish above, which
        # billing_worker/TranscriptRedisConsumerService depend on regardless of whether
        # the LiveKit side is reachable.
        pcm = audio_bytes[_WAV_HEADER_BYTES:] if len(audio_bytes) > _WAV_HEADER_BYTES else b""
        if pcm and self.livekit_publisher is not None:
            asyncio.create_task(
                self.livekit_publisher.publish_pcm(
                    translation.meeting_id, translation.target_lang, pcm, self.tts_settings.sample_rate
                )
            )

    async def _get_voice_id(self, meeting_id: str, speaker_id: str) -> str | None:
        """Return cached Cartesia voice_id for this speaker, or None.

        Re-checks consent on every call (not just before cloning) — if the speaker
        revokes voice clone consent mid-session, synthesis must fall back to the
        default voice immediately, even though a voice_id is still cached.
        """
        if not self.is_voice_clone_consented(meeting_id, speaker_id):
            return None
        cached = await self.redis.hget(f"voice:{meeting_id}:{speaker_id}", "voice_id")
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached
        return None

    async def _consume_audio_for_cloning(self) -> None:
        """Buffer raw audio per speaker; clone voice once enough is collected."""
        # {(meeting_id, speaker_id): accumulated_audio_bytes}
        buffers: dict[tuple[str, str], bytearray] = {}
        buffer_seconds: dict[tuple[str, str], float] = {}
        buffer_lang: dict[tuple[str, str], str] = {}

        while self._running:
            try:
                async for _msg_id, data in self.redis.consume(
                    stream="audio:chunks",
                    group=self._audio_consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=5,
                ):
                    try:
                        chunk = AudioChunkMessage.from_redis(data)
                        key = (chunk.meeting_id, chunk.speaker_id)

                        # Consent gate: never buffer/clone a speaker's voice (biometric
                        # data) unless they have at least one current outgoing route with
                        # VoiceCloneEnabled = true. See base_worker.is_voice_clone_consented.
                        if not self.is_voice_clone_consented(chunk.meeting_id, chunk.speaker_id):
                            buffers.pop(key, None)
                            buffer_seconds.pop(key, None)
                            buffer_lang.pop(key, None)
                            continue

                        # Skip if voice already cloned for this speaker
                        if await self._get_voice_id(chunk.meeting_id, chunk.speaker_id):
                            continue

                        buffers.setdefault(key, bytearray()).extend(chunk.audio_data)
                        # PCM 16-bit mono: 2 bytes per sample
                        duration_s = len(chunk.audio_data) / 2 / max(chunk.sample_rate, 1)
                        buffer_seconds[key] = buffer_seconds.get(key, 0.0) + duration_s
                        buffer_lang[key] = chunk.language

                        if buffer_seconds[key] >= self.tts_settings.voice_clone_min_seconds:
                            audio_snapshot = bytes(buffers.pop(key))
                            del buffer_seconds[key]
                            clone_lang = _clone_language(buffer_lang.pop(key, "en"))
                            asyncio.create_task(
                                self._clone_and_cache(
                                    chunk.meeting_id, chunk.speaker_id, audio_snapshot, clone_lang
                                )
                            )
                    except Exception:
                        self.logger.exception("audio_chunk_processing_error")
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("audio_consumer_error")
                await asyncio.sleep(2)

    async def _clone_and_cache(
        self, meeting_id: str, speaker_id: str, audio_bytes: bytes, language: str = "en"
    ) -> None:
        """Clone voice via Cartesia and cache voice_id in Redis."""
        label = f"speaker-{speaker_id[:8]}-{meeting_id[:8]}"
        try:
            voice_id = await self.cartesia.clone_voice(audio_bytes, label, language)
            cache_key = f"voice:{meeting_id}:{speaker_id}"
            await self.redis.hset(cache_key, "voice_id", voice_id)
            # hset has no TTL of its own — without this the key lives in Redis forever.
            await self.redis.expire(cache_key, self.tts_settings.voice_clone_key_ttl_seconds)
            self.logger.info(
                "voice_cached",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                voice_id=voice_id,
            )
            await self.redis.publish_system_event(
                room_id=meeting_id,
                event_type="voice_clone_ready",
                payload={"speakerId": speaker_id, "voiceId": voice_id},
            )
        except Exception as e:
            self.logger.error(
                "voice_clone_failed",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                error=str(e),
            )

    @staticmethod
    def _cache_key(speaker_id: str, target_lang: str, text: str, voice_mode: str) -> str:
        normalized = " ".join(text.casefold().split())
        digest = hashlib.sha256(
            f"{speaker_id}|{target_lang}|{normalized}|{voice_mode}".encode()
        ).hexdigest()
        return f"tts:cache:{digest}"
