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
import base64

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

        # Check if we have a voice embedding for this speaker
        embedding = await self.redis.hget(
            f"speaker:{translation.meeting_id}:{translation.speaker_id}",
            "embedding",
        )

        if embedding is not None and self.xtts is not None:
            # Voice cloning available → use XTTS v2
            audio_bytes, duration_ms = await self.xtts.synthesize(
                text=translation.translated_text,
                language=translation.target_lang,
                speaker_embedding=embedding,
            )
            voice_type = "cloned"
        else:
            # No embedding yet or XTTS disabled → use Edge-TTS default voice
            audio_bytes, duration_ms = await self.edge_tts.synthesize(
                text=translation.translated_text,
                language=translation.target_lang,
            )
            voice_type = "default"

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
            target_lang=translation.target_lang,
        )

        await self.publish("tts:results", translation.meeting_id, result.to_redis())

        self.logger.info(
            "audio_synthesized",
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            voice_type=voice_type,
            duration_ms=duration_ms,
            text=translation.translated_text[:60],
        )
