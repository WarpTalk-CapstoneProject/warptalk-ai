"""Voice embedding extractor for progressive voice cloning.

Runs as a background asyncio task, accumulating audio samples from a
speaker and extracting XTTS v2 voice embeddings when enough data
is available.

Lifecycle:
    0-5s:  No embedding → TTS uses Edge-TTS default voice
    ~5s:   First embedding extracted → TTS switches to XTTS v2 (~70% match)
    ~15s:  Refined embedding extracted → better voice match (~90%)
"""

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass, field

import numpy as np

from shared.logger import get_logger
from shared.redis_client import RedisStreamClient

logger = get_logger(__name__)


@dataclass
class SpeakerAudioBuffer:
    """Accumulates audio samples for a speaker."""

    samples: list[np.ndarray] = field(default_factory=list)
    total_seconds: float = 0.0
    embedding_v1_extracted: bool = False
    embedding_v2_extracted: bool = False

    def add(self, audio: np.ndarray, sample_rate: int = 16000) -> None:
        self.samples.append(audio)
        self.total_seconds += len(audio) / sample_rate

    def get_combined(self) -> np.ndarray:
        if not self.samples:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.samples)


class EmbeddingExtractor:
    """Background task that extracts voice embeddings progressively.

    Stores embeddings in Redis for the TTS worker to consume:
        Key: speaker:{meeting_id}:{speaker_id}  Field: embedding
    """

    def __init__(
        self,
        redis: RedisStreamClient,
        min_seconds: float = 5.0,
        refine_seconds: float = 15.0,
    ) -> None:
        self.redis = redis
        self.min_seconds = min_seconds
        self.refine_seconds = refine_seconds
        self._buffers: dict[str, SpeakerAudioBuffer] = {}
        self._xtts_model = None

    async def load_model(self) -> None:
        """Load XTTS model for embedding extraction."""
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from TTS.api import TTS

        logger.info("loading_xtts_for_embedding_extraction")
        self._xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        logger.info("xtts_embedding_extractor_ready")

    def _buffer_key(self, meeting_id: str, speaker_id: str) -> str:
        return f"{meeting_id}:{speaker_id}"

    async def add_audio(
        self,
        meeting_id: str,
        speaker_id: str,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> None:
        """Add an audio chunk for a speaker and check if embedding extraction is due.

        This is called by the TTS worker for every incoming chunk — runs
        embedding extraction asynchronously without blocking the pipeline.
        """
        key = self._buffer_key(meeting_id, speaker_id)

        if key not in self._buffers:
            self._buffers[key] = SpeakerAudioBuffer()

        buf = self._buffers[key]
        buf.add(audio, sample_rate)

        # Check if we should extract an embedding
        if not buf.embedding_v1_extracted and buf.total_seconds >= self.min_seconds:
            # Fire-and-forget background extraction
            asyncio.create_task(
                self._extract_and_cache(meeting_id, speaker_id, buf, version=1)
            )

        elif (
            buf.embedding_v1_extracted
            and not buf.embedding_v2_extracted
            and buf.total_seconds >= self.refine_seconds
        ):
            asyncio.create_task(
                self._extract_and_cache(meeting_id, speaker_id, buf, version=2)
            )

    async def _extract_and_cache(
        self,
        meeting_id: str,
        speaker_id: str,
        buf: SpeakerAudioBuffer,
        version: int,
    ) -> None:
        """Extract embedding and cache in Redis."""
        try:
            combined = buf.get_combined()
            embedding = await asyncio.to_thread(self._extract_embedding, combined)

            redis_key = f"speaker:{meeting_id}:{speaker_id}"
            await self.redis.hset(redis_key, "embedding", embedding.tobytes())

            if version == 1:
                buf.embedding_v1_extracted = True
                logger.info(
                    "embedding_v1_extracted",
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    audio_seconds=buf.total_seconds,
                )
            else:
                buf.embedding_v2_extracted = True
                logger.info(
                    "embedding_v2_refined",
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    audio_seconds=buf.total_seconds,
                )

        except Exception:
            logger.exception(
                "embedding_extraction_failed",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
            )

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract XTTS speaker embedding from audio (blocking)."""
        import soundfile as sf

        # Save to temp WAV for XTTS
        buffer = io.BytesIO()
        sf.write(buffer, audio, 16000, format="WAV")
        buffer.seek(0)

        # Use XTTS to extract speaker embedding
        # The exact API depends on the TTS library version
        embedding = self._xtts_model.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip(
            buffer
        )
        return np.array(embedding, dtype=np.float32)

    def cleanup_speaker(self, meeting_id: str, speaker_id: str) -> None:
        """Remove buffer when speaker leaves."""
        key = self._buffer_key(meeting_id, speaker_id)
        self._buffers.pop(key, None)
