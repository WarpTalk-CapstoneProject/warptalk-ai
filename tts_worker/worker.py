"""TTS Worker — Consumes translated text, produces synthesized audio.

Pipeline:
    Redis Stream (translate:results:{meetingId})
    → Edge-TTS (default voice, 0-5s) or XTTS v2 (cloned voice, 5s+)
    → Redis Stream (tts:results:{meetingId})

Progressive Voice Cloning:
    0-5s:    Edge-TTS with neutral default voice (~100ms)
    5s+:     XTTS v2 with speaker embedding v1 (~300ms, ~70% match)
    15s+:    XTTS v2 with refined embedding v2 (~300ms, ~90% match)
"""

from __future__ import annotations

import asyncio
<<<<<<< Updated upstream
import base64
=======
import hashlib
>>>>>>> Stashed changes
import time

from shared.audio_utils import bytes_to_numpy
from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage

from tts_worker.embedding_extractor import EmbeddingExtractor
from tts_worker.synthesizer import EdgeTTSSynthesizer, XTTSSynthesizer


class TTSWorker(BaseWorker):
    """Text-to-Speech worker with progressive voice cloning."""

    worker_name = "tts"
    input_stream = "translate:results"
    consumer_group = "tts-workers"
    _embedding_consumer_group = "embedding-workers"
    _running = True

    def __init__(
        self,
        tts_settings: TTSSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_settings = tts_settings or TTSSettings()
        self.xtts: XTTSSynthesizer | None = None
        self.edge_tts: EdgeTTSSynthesizer | None = None
        self.embedding_extractor: EmbeddingExtractor | None = None

    async def load_model(self) -> None:
        """Load both TTS engines and embedding extractor."""
        # Edge-TTS (always available, no GPU)
        self.edge_tts = EdgeTTSSynthesizer(
            default_voice=self.tts_settings.default_voice,
        )
        await self.edge_tts.load()

        # XTTS v2 (GPU voice cloning) - Try to load, but fallback if not installed
        try:
            self.xtts = XTTSSynthesizer(
                model_name=self.tts_settings.xtts_model,
                device=self.tts_settings.device,
                sample_rate=self.tts_settings.sample_rate,
            )
            await self.xtts.load()

            # Embedding extractor (shares XTTS model reference)
            self.embedding_extractor = EmbeddingExtractor(
                redis=self.redis,
                min_seconds=self.tts_settings.embedding_min_seconds,
                refine_seconds=self.tts_settings.embedding_refine_seconds,
            )
            await self.embedding_extractor.load_model()

            # Start background task to consume audio chunks for embedding extraction
            asyncio.create_task(self._consume_audio_for_embedding())
            self.logger.info("embedding_audio_consumer_started")
        except Exception as e:
            self.logger.warning(
                "xtts_load_failed",
                reason=str(e),
                message="XTTS not available. Only EdgeTTS will be used.",
            )
            self.xtts = None
            self.embedding_extractor = None

    async def _consume_audio_for_embedding(self) -> None:
        """Background task: consume audio:chunks to feed EmbeddingExtractor.

        Uses a separate consumer group ('embedding-workers') so it doesn't
        compete with the STT worker for audio chunks.
        """
        while self._running:
            try:
                # Scan for active meeting streams by checking known meetings
                # The consumer group pattern ensures we only get new chunks
                async for msg_id, data in self.redis.consume(
                    stream="audio:chunks:*",
                    group=self._embedding_consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=5,
                ):
                    try:
                        chunk = AudioChunkMessage.from_redis(data)
                        audio_np = bytes_to_numpy(
                            chunk.audio_data, chunk.sample_rate,
                        )
                        await self.embedding_extractor.add_audio(
                            meeting_id=chunk.meeting_id,
                            speaker_id=chunk.speaker_id,
                            audio=audio_np,
                            sample_rate=chunk.sample_rate,
                        )
                    except Exception:
                        self.logger.exception(
                            "embedding_audio_chunk_error",
                            message_id=str(msg_id),
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("embedding_consumer_error")
                await asyncio.sleep(2)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Synthesize text to speech with progressive voice cloning."""
        translation = TranslationResultMessage.from_redis(data)
        text = translation.translated_text

        # Check route state
        route_status = self._route_states.get(translation.meeting_id, "AUDIO_ROUTING_ACTIVE")
        if route_status == "PAUSED":
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - translation.timestamp_ms
        await self.redis.publish_telemetry(translation.meeting_id, self.worker_name, e2e_latency_ms)

        if route_status == "TEXT_ONLY_MODE" or not translation.translated_text.strip():
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id}
                )
            return

        # Check if we have a voice embedding for this speaker
<<<<<<< Updated upstream
        embedding = None
        if route_status != "VOICE_CLONE_FALLBACK":
            try:
                embedding = await self.redis.redis.hget(
                    f"speaker:{translation.meeting_id}:{translation.speaker_id}",
                    "embedding",
                )
            except AttributeError:
                # If self.redis (RedisStreamClient) has hget directly
                embedding = await getattr(self.redis, "hget")(
                    f"speaker:{translation.meeting_id}:{translation.speaker_id}",
                    "embedding",
                )

        audio_bytes = b""
        duration_ms = 0
        voice_type = "default"
=======
        embedding = await self.redis.hget(
            f"speaker:{translation.meeting_id}:{translation.speaker_id}",
            "embedding",
        )
        anchor_provider = self._provider_name(self.edge_tts, self.tts_settings.anchor_provider)
        clone_provider = self._provider_name(self.xtts, self.tts_settings.clone_provider)
        should_clone, fallback_reason = self._should_clone(text, embedding)
        selected_voice_mode = "blended" if should_clone and self.tts_settings.blend_enabled else (
            "cloned" if should_clone else "standard"
        )
        selected_provider = clone_provider if should_clone else anchor_provider
        cache_key = self._cache_key(
            speaker_id=translation.speaker_id,
            target_lang=translation.target_lang,
            text=text,
            voice_mode=selected_voice_mode,
            provider=selected_provider,
        )
>>>>>>> Stashed changes

        if self.tts_settings.cache_enabled:
            cached_audio = await self.redis.get(cache_key)
            if cached_audio:
                result = TTSResultMessage(
                    segment_id=translation.segment_id,
                    meeting_id=translation.meeting_id,
                    speaker_id=translation.speaker_id,
                    audio_data=cached_audio,
                    duration_ms=0,
                    voice_type="blended" if should_clone and self.tts_settings.blend_enabled else (
                        "cloned" if should_clone else "default"
                    ),
                    voice_mode=selected_voice_mode,
                    clone_strength=self._clone_strength(embedding) if should_clone else 0.0,
                    anchor_provider=anchor_provider,
                    clone_provider=clone_provider if should_clone else "",
                    render_location="server",
                    cache_key=cache_key,
                    cache_hit=True,
                    fallback_reason=fallback_reason,
                    target_lang=translation.target_lang,
                )
                await self.publish("tts:results", translation.meeting_id, result.to_redis())
                return

        t0 = time.monotonic()
        conversion_latency_ms = 0
        synthesis_latency_ms = 0

        if should_clone:
            # Voice cloning available → use XTTS v2
<<<<<<< Updated upstream
            try:
                audio_bytes, duration_ms = await self.xtts.synthesize(
                    text=translation.translated_text,
                    language=translation.target_lang,
                    speaker_embedding=embedding,
                )
                voice_type = "cloned"
            except Exception as e:
                self.logger.error("xtts_synthesis_failed", error=str(e))
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="voice_clone_unavailable",
                    payload={"error": str(e)}
                )
                # Fallback to Edge-TTS
                embedding = None

        if embedding is None or self.xtts is None:
            # No embedding yet or XTTS disabled/failed → use Edge-TTS default voice
            try:
                audio_bytes, duration_ms = await self.edge_tts.synthesize(
                    text=translation.translated_text,
                    language=translation.target_lang,
                )
                voice_type = "default"
            except Exception as e:
                self.logger.error("edge_tts_synthesis_failed", error=str(e))
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="edge_tts_unavailable",
                    payload={"error": str(e)}
                )

        if audio_bytes:
            result = TTSResultMessage(
                segment_id=translation.segment_id,
                meeting_id=translation.meeting_id,
                speaker_id=translation.speaker_id,
                audio_data=audio_bytes,
                duration_ms=duration_ms,
                voice_type=voice_type,
                target_lang=translation.target_lang,
                is_final_chunk=translation.is_final_chunk,
                timestamp_ms=translation.timestamp_ms,
            )

            await self.publish("tts:results", translation.meeting_id, result.to_redis())

        if translation.is_final_chunk:
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="final_chunk_processed",
                payload={"segmentId": translation.segment_id}
            )
=======
            clone_t0 = time.monotonic()
            audio_bytes, duration_ms = await self.xtts.synthesize(
                text=text,
                language=translation.target_lang,
                speaker_embedding=embedding,
            )
            conversion_latency_ms = int((time.monotonic() - clone_t0) * 1000)
            voice_type = "blended" if self.tts_settings.blend_enabled else "cloned"
            voice_mode = "blended" if self.tts_settings.blend_enabled else "cloned"
            clone_strength = self._clone_strength(embedding)
        else:
            # No embedding yet or XTTS disabled → use Edge-TTS default voice
            anchor_t0 = time.monotonic()
            audio_bytes, duration_ms = await self.edge_tts.synthesize(
                text=text,
                language=translation.target_lang,
            )
            synthesis_latency_ms = int((time.monotonic() - anchor_t0) * 1000)
            voice_type = "default"
            voice_mode = "standard"
            clone_strength = 0.0
            clone_provider = ""

        # Feed original audio to embedding extractor (background)
        # The original audio comes from the audio:chunks stream;
        # here we can accumulate from the chunk data if available
        # This is handled separately by the embedding extractor
        # listening to audio:chunks or receiving audio via the TTS worker

        result = TTSResultMessage(
            segment_id=translation.segment_id,
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            audio_data=audio_bytes,
            duration_ms=duration_ms,
            voice_type=voice_type,
            voice_mode=voice_mode,
            clone_strength=clone_strength,
            anchor_provider=anchor_provider,
            clone_provider=clone_provider,
            render_location="server",
            cache_key=cache_key,
            cache_hit=False,
            synthesis_latency_ms=synthesis_latency_ms or int((time.monotonic() - t0) * 1000),
            conversion_latency_ms=conversion_latency_ms,
            fallback_reason=fallback_reason,
            target_lang=translation.target_lang,
        )

        await self.publish("tts:results", translation.meeting_id, result.to_redis())
        if self.tts_settings.cache_enabled and audio_bytes:
            await self.redis.set_with_ttl(
                cache_key,
                audio_bytes,
                self.tts_settings.cache_ttl_seconds,
            )
>>>>>>> Stashed changes

        self.logger.info(
            "audio_synthesized",
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            voice_type=voice_type,
            voice_mode=voice_mode,
            clone_strength=clone_strength,
            duration_ms=duration_ms,
<<<<<<< Updated upstream
            text=translation.translated_text[:60],
            is_final=translation.is_final_chunk,
=======
            text=text[:60],
>>>>>>> Stashed changes
        )

    def _should_clone(self, text: str, embedding: bytes | None) -> tuple[bool, str]:
        if len(text.strip()) < self.tts_settings.min_clone_chars:
            return False, "short_utterance"
        if embedding is None:
            return False, "voice_profile_not_ready"
        if self.xtts is None:
            return False, "clone_provider_unavailable"
        return True, ""

    def _clone_strength(self, embedding: bytes | None) -> float:
        if embedding is None or self.xtts is None:
            return 0.0
        return max(0.0, min(1.0, self.tts_settings.default_clone_strength))

    def _cache_key(
        self,
        speaker_id: str,
        target_lang: str,
        text: str,
        voice_mode: str,
        provider: str,
    ) -> str:
        normalized = " ".join(text.casefold().split())
        digest = hashlib.sha256(
            f"{speaker_id}|{target_lang}|{normalized}|{voice_mode}|{provider}".encode()
        ).hexdigest()
        return f"tts:cache:{digest}"

    @staticmethod
    def _provider_name(provider: object | None, fallback: str) -> str:
        value = getattr(provider, "provider_name", None)
        if isinstance(value, str):
            return value
        return fallback
