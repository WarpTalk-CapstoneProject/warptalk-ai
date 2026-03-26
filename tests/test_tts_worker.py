"""Tests for TTS Worker — verify progressive voice cloning logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import TranslationResultMessage

from tts_worker.embedding_extractor import EmbeddingExtractor, SpeakerAudioBuffer
from tts_worker.worker import TTSWorker


class TestSpeakerAudioBuffer:
    """SpeakerAudioBuffer accumulation tests."""

    def test_add_tracks_seconds(self) -> None:
        buf = SpeakerAudioBuffer()
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        buf.add(audio, sample_rate=16000)

        assert buf.total_seconds == pytest.approx(1.0)
        assert len(buf.samples) == 1

    def test_multiple_adds_accumulate(self) -> None:
        buf = SpeakerAudioBuffer()
        for _ in range(5):
            buf.add(np.zeros(16000, dtype=np.float32), 16000)

        assert buf.total_seconds == pytest.approx(5.0)
        assert len(buf.samples) == 5

    def test_get_combined(self) -> None:
        buf = SpeakerAudioBuffer()
        buf.add(np.ones(8000, dtype=np.float32), 16000)
        buf.add(np.ones(8000, dtype=np.float32) * 2, 16000)

        combined = buf.get_combined()
        assert len(combined) == 16000
        assert combined[0] == 1.0
        assert combined[8000] == 2.0


class TestEmbeddingExtractor:
    """EmbeddingExtractor progressive cloning tests."""

    async def test_no_extraction_before_min_seconds(self) -> None:
        """Should not extract embedding before min_seconds reached."""
        redis = MagicMock()
        redis.hset = AsyncMock()

        extractor = EmbeddingExtractor(
            redis=redis,
            min_seconds=5.0,
            refine_seconds=15.0,
        )

        # Add 3s of audio (less than min 5s)
        audio = np.zeros(16000 * 3, dtype=np.float32)
        await extractor.add_audio("m1", "s1", audio, 16000)

        # No embedding should be cached
        redis.hset.assert_not_called()

    def test_buffer_key_format(self) -> None:
        extractor = EmbeddingExtractor.__new__(EmbeddingExtractor)
        assert extractor._buffer_key("meeting-1", "speaker-1") == "meeting-1:speaker-1"


class TestTTSWorker:
    """TTS Worker process() tests."""

    async def test_uses_edge_tts_when_no_embedding(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should use Edge-TTS when no voice embedding is cached."""
        worker = TTSWorker.__new__(TTSWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.tts_settings = TTSSettings()

        # No embedding in Redis
        mock_redis_client._redis.hget = AsyncMock(return_value=None)

        # Mock Edge-TTS
        worker.edge_tts = MagicMock()
        worker.edge_tts.synthesize = AsyncMock(return_value=(b"audio", 1000))
        worker.xtts = MagicMock()
        worker.xtts.synthesize = AsyncMock()

        msg = TranslationResultMessage(
            segment_id="seg-1",
            meeting_id="m1",
            speaker_id="s1",
            original_text="Hello",
            translated_text="Xin chào",
            source_lang="en",
            target_lang="vi",
        )

        await worker.process(b"msg-1", msg.to_redis())

        # Edge-TTS should be called, not XTTS
        worker.edge_tts.synthesize.assert_called_once()
        worker.xtts.synthesize.assert_not_called()

    async def test_uses_xtts_when_embedding_exists(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should use XTTS v2 when voice embedding is cached."""
        worker = TTSWorker.__new__(TTSWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.tts_settings = TTSSettings()

        # Embedding exists in Redis
        fake_embedding = np.zeros(256, dtype=np.float32).tobytes()
        mock_redis_client._redis.hget = AsyncMock(return_value=fake_embedding)

        # Mock synthesizers
        worker.edge_tts = MagicMock()
        worker.edge_tts.synthesize = AsyncMock()
        worker.xtts = MagicMock()
        worker.xtts.synthesize = AsyncMock(return_value=(b"cloned-audio", 1500))

        msg = TranslationResultMessage(
            segment_id="seg-1",
            meeting_id="m1",
            speaker_id="s1",
            original_text="Hello",
            translated_text="Xin chào",
            source_lang="en",
            target_lang="vi",
        )

        await worker.process(b"msg-1", msg.to_redis())

        # XTTS should be called with embedding, not Edge-TTS
        worker.xtts.synthesize.assert_called_once()
        worker.edge_tts.synthesize.assert_not_called()
        call_kwargs = worker.xtts.synthesize.call_args
        assert call_kwargs[1].get("speaker_embedding") == fake_embedding or \
               call_kwargs.kwargs.get("speaker_embedding") == fake_embedding
