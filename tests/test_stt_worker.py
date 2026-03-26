"""Tests for STT Worker — mock Whisper model, verify output schema."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage, STTResultMessage

from stt_worker.model import TranscribedSegment, WhisperSTT
from stt_worker.worker import STTWorker


class TestWhisperSTT:
    """WhisperSTT model wrapper tests."""

    def test_transcribe_returns_segments(self) -> None:
        """transcribe() should return list of TranscribedSegment."""
        stt = WhisperSTT.__new__(WhisperSTT)
        stt.beam_size = 1
        stt.vad_filter = True

        # Mock the model
        mock_segment = MagicMock()
        mock_segment.text = " Hello, world!"
        mock_segment.avg_logprob = -0.25
        mock_segment.start = 0.0
        mock_segment.end = 1.5

        mock_info = MagicMock()
        mock_info.language = "en"

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], mock_info)

        import numpy as np
        audio = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(audio)

        assert len(result) == 1
        assert result[0].text == "Hello, world!"
        assert result[0].language == "en"
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1500

    def test_transcribe_skips_empty_text(self) -> None:
        """transcribe() should skip segments with empty text."""
        stt = WhisperSTT.__new__(WhisperSTT)
        stt.beam_size = 1
        stt.vad_filter = True

        mock_segment = MagicMock()
        mock_segment.text = "   "
        mock_segment.avg_logprob = -0.5
        mock_segment.start = 0.0
        mock_segment.end = 0.5

        stt._model = MagicMock()
        stt._model.transcribe.return_value = ([mock_segment], MagicMock(language="en"))

        import numpy as np
        result = stt.transcribe(np.zeros(16000, dtype=np.float32))
        assert len(result) == 0

    def test_transcribe_raises_if_not_loaded(self) -> None:
        """transcribe() should raise if model not loaded."""
        stt = WhisperSTT()
        import numpy as np
        with pytest.raises(RuntimeError, match="Model not loaded"):
            stt.transcribe(np.zeros(16000, dtype=np.float32))


class TestSTTWorker:
    """STT Worker process() tests."""

    async def test_process_publishes_stt_result(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """process() should publish STTResultMessage for each segment."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()

        # Mock model
        worker.model = MagicMock()
        worker.model.transcribe.return_value = [
            TranscribedSegment(
                text="Hello",
                language="en",
                confidence=0.95,
                start_ms=0,
                end_ms=1000,
            )
        ]

        # Build message
        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            language="auto",
        )

        with patch("stt_worker.worker.asyncio") as mock_asyncio:
            mock_asyncio.to_thread = AsyncMock(
                return_value=worker.model.transcribe.return_value
            )
            await worker.process(b"msg-1", chunk.to_redis())

        # Verify publish was called
        mock_redis_client._redis.xadd.assert_called_once()
        call_args = mock_redis_client._redis.xadd.call_args
        assert "stt:results:meeting-1" in str(call_args)
