"""STT Worker — Consumes audio chunks, produces text segments.

Pipeline:
    Redis Stream (audio:chunks:{meetingId})
    → Whisper STT (asyncio.to_thread)
    → Redis Stream (stt:results:{meetingId})
"""

from __future__ import annotations

import asyncio
import time

from shared.audio_utils import bytes_to_numpy
from shared.base_worker import BaseWorker
from shared.config import STTSettings
from shared.schemas import AudioChunkMessage, STTResultMessage

from stt_worker.model import WhisperSTT


class STTWorker(BaseWorker):
    """Speech-to-Text worker using whisper.cpp (Metal GPU)."""

    worker_name = "stt"
    input_stream = "audio:chunks"
    consumer_group = "stt-workers"

    def __init__(self, stt_settings: STTSettings | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stt_settings = stt_settings or STTSettings()
        self.model: WhisperSTT | None = None

    async def load_model(self) -> None:
        """Load Whisper model in a thread to avoid blocking the event loop."""
        self.model = WhisperSTT(
            model_size=self.stt_settings.model,
            device=self.stt_settings.device,
            compute_type=self.stt_settings.compute_type,
            beam_size=self.stt_settings.beam_size,
            vad_filter=self.stt_settings.vad_filter,
        )
        await asyncio.to_thread(self.model.load)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Process one audio chunk: transcribe and publish results."""
        chunk = AudioChunkMessage.from_redis(data)

        self.logger.debug(
            "processing_chunk",
            meeting_id=chunk.meeting_id,
            speaker_id=chunk.speaker_id,
            chunk_index=chunk.chunk_index,
        )

        # Convert audio bytes to numpy array
        audio_array = bytes_to_numpy(chunk.audio_data, sample_rate=chunk.sample_rate)

        # Calculate time offset for this chunk
        chunk_offset_ms = chunk.chunk_index * self.settings.chunk_duration_ms

        # Auto-detect language — works well with VAD-gated audio (no silence)
        language_hint = chunk.language if chunk.language != "auto" else None

        t0 = time.monotonic()
        segments = await asyncio.to_thread(
            self.model.transcribe,
            audio_array,
            language=language_hint,
            chunk_offset_ms=chunk_offset_ms,
        )
        inference_ms = int((time.monotonic() - t0) * 1000)
        audio_ms = int(len(audio_array) / 16000 * 1000)

        self.logger.info(
            "inference_complete",
            inference_ms=inference_ms,
            audio_ms=audio_ms,
            rtf=round(inference_ms / max(audio_ms, 1), 2),
            segments=len(segments),
        )

        # Publish each transcribed segment
        for segment in segments:
            result = STTResultMessage(
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                text=segment.text,
                language=segment.language,
                confidence=segment.confidence,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                chunk_index=chunk.chunk_index,
            )

            await self.publish("stt:results", chunk.meeting_id, result.to_redis())

            self.logger.info(
                "segment_transcribed",
                meeting_id=chunk.meeting_id,
                text=segment.text[:80],
                language=segment.language,
                confidence=segment.confidence,
                inference_ms=inference_ms,
            )
