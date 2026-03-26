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

import base64

from shared.audio_utils import bytes_to_numpy
from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.schemas import TranslationResultMessage, TTSResultMessage

from tts_worker.embedding_extractor import EmbeddingExtractor
from tts_worker.synthesizer import EdgeTTSSynthesizer, XTTSSynthesizer


class TTSWorker(BaseWorker):
    """Text-to-Speech worker with progressive voice cloning."""

    worker_name = "tts"
    input_stream = "translate:results"
    consumer_group = "tts-workers"

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

        # XTTS v2 (GPU voice cloning)
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

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Synthesize text to speech with progressive voice cloning."""
        translation = TranslationResultMessage.from_redis(data)

        # Check if we have a voice embedding for this speaker
        embedding = await self.redis.hget(
            f"speaker:{translation.meeting_id}:{translation.speaker_id}",
            "embedding",
        )

        if embedding is not None:
            # Voice cloning available → use XTTS v2
            audio_bytes, duration_ms = await self.xtts.synthesize(
                text=translation.translated_text,
                language=translation.target_lang,
                speaker_embedding=embedding,
            )
            voice_type = "cloned"
        else:
            # No embedding yet → use Edge-TTS default voice
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
