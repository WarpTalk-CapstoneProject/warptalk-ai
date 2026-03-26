"""Whisper STT model wrapper.

Provides a synchronous `transcribe()` method designed to be called
via `asyncio.to_thread()` from the async worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)


@dataclass(slots=True)
class TranscribedSegment:
    """A single transcribed text segment."""

    text: str
    language: str
    confidence: float
    start_ms: int
    end_ms: int


class WhisperSTT:
    """Faster-Whisper model wrapper for low-latency STT.

    Uses INT8 quantization and beam_size=1 (greedy) for sub-200ms
    inference on 1s audio chunks.
    """

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "int8",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    def load(self) -> None:
        """Load the Faster-Whisper model. Call via asyncio.to_thread()."""
        from faster_whisper import WhisperModel

        logger.info(
            "loading_whisper_model",
            model_size=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("whisper_model_loaded")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        chunk_offset_ms: int = 0,
    ) -> list[TranscribedSegment]:
        """Transcribe audio array to text segments.

        This is a BLOCKING call — must be run via asyncio.to_thread().

        Args:
            audio: NumPy float32 array at 16kHz
            language: Source language hint, None for auto-detect
            chunk_offset_ms: Offset to add to segment timestamps

        Returns:
            List of transcribed segments
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        segments_iter, info = self._model.transcribe(
            audio,
            language=language if language != "auto" else None,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            without_timestamps=False,
        )

        results: list[TranscribedSegment] = []
        for segment in segments_iter:
            text = segment.text.strip()
            if not text:
                continue

            results.append(
                TranscribedSegment(
                    text=text,
                    language=info.language,
                    confidence=round(segment.avg_logprob, 4) if segment.avg_logprob else 0.0,
                    start_ms=chunk_offset_ms + int(segment.start * 1000),
                    end_ms=chunk_offset_ms + int(segment.end * 1000),
                )
            )

        return results
