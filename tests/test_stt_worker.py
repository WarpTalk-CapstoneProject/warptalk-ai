"""Tests for STT Worker — mock OpenAI STT, verify output schema."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker.model import OpenAISTT, TranscribedSegment, _filter_segments, _normalize_language
from stt_worker.worker import STTWorker


def _segment(text: str, avg_logprob: float = -0.3, no_speech_prob: float = 0.01) -> dict:
    return {
        "text": text,
        "start": 0.0,
        "end": 1.0,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }


class TestOpenAISTT:
    """OpenAISTT wrapper tests."""

    async def test_transcribe_returns_segments(self) -> None:
        """transcribe() should return filtered list of TranscribedSegment."""
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.api_key = ""
        stt.model = "gpt-4o-mini-transcribe"

        mock_segment = MagicMock()
        mock_segment.model_dump.return_value = {
            "text": " Hello, world!",
            "start": 0.0,
            "end": 1.5,
            "avg_logprob": -0.25,
            "no_speech_prob": 0.01,
        }

        mock_result = MagicMock()
        mock_result.language = "en"
        mock_result.segments = [mock_segment]

        stt._client = MagicMock()
        stt._client.audio.transcriptions.create = AsyncMock(return_value=mock_result)

        result = await stt.transcribe(b"fake_audio", sample_rate=16000)

        assert len(result) == 1
        assert result[0].text == "Hello, world!"
        assert result[0].language == "en"
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1500

    async def test_transcribe_empty_bytes_returns_empty(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt._client = MagicMock()
        result = await stt.transcribe(b"")
        assert result == []

    async def test_transcribe_api_error_raises_for_worker_degrade_signal(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.api_key = ""
        stt.model = "gpt-4o-mini-transcribe"
        stt._client = MagicMock()
        stt._client.audio.transcriptions.create = AsyncMock(
            side_effect=Exception("API error")
        )

        with pytest.raises(Exception, match="API error"):
            await stt.transcribe(b"audio_bytes", sample_rate=16000)


class TestFilterSegments:
    """_filter_segments utility tests."""

    def test_allowed_language_en(self) -> None:
        segs = [_segment("Hello")]
        result = _filter_segments(segs, "en", 0)
        assert len(result) == 1
        assert result[0].language == "en"

    def test_allowed_language_vi(self) -> None:
        segs = [_segment("Hệ thống đang hoạt động tốt")]
        result = _filter_segments(segs, "vi", 0)
        assert len(result) == 1

    def test_unknown_language_filtered(self) -> None:
        segs = [_segment("こんにちは")]
        result = _filter_segments(segs, "ja", 0)
        assert result == []

    def test_low_confidence_filtered(self) -> None:
        segs = [_segment("Some text", avg_logprob=-1.5)]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_high_no_speech_filtered(self) -> None:
        segs = [_segment("Some text", no_speech_prob=0.8)]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_hallucination_filtered(self) -> None:
        segs = [_segment("thank you")]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_chunk_offset_applied(self) -> None:
        segs = [
            {
                "text": "Hello",
                "start": 1.0,
                "end": 2.0,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.01,
            }
        ]
        result = _filter_segments(segs, "en", chunk_offset_ms=5000)
        assert result[0].start_ms == 6000
        assert result[0].end_ms == 7000

    def test_full_language_name_normalized(self) -> None:
        """OpenAI returns full language names when language=None."""
        segs = [_segment("Hello")]
        result = _filter_segments(segs, "english", 0)
        assert len(result) == 1
        assert result[0].language == "en"


class TestNormalizeLanguage:
    def test_full_name_to_code(self) -> None:
        assert _normalize_language("english") == "en"
        assert _normalize_language("vietnamese") == "vi"
        assert _normalize_language("Chinese") == "zh"

    def test_code_passthrough(self) -> None:
        assert _normalize_language("en") == "en"
        assert _normalize_language("vi") == "vi"


class TestSTTWorker:
    """STT Worker process() tests."""

    async def test_process_publishes_stt_result(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """process() should publish STTResultMessage for each transcribed segment."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello",
                    language="en",
                    confidence=-0.25,
                    start_ms=0,
                    end_ms=1000,
                )
            ]
        )

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            language="auto",
        )

        await worker.process(b"msg-1", chunk.to_redis())

        # BaseWorker.publish() calls xadd twice: room stream + global stream
        mock_redis_client._redis.xadd.assert_called()
        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("stt:results" in s for s in streams_published)

    async def test_process_skips_paused_room(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """process() should skip messages for paused rooms."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = {"meeting-1"}
        worker.model = MagicMock()

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        mock_redis_client._redis.xadd.assert_not_called()

    async def test_process_publishes_system_event_on_stt_error(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """STT provider failures should emit an explicit degrade signal."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings(model="gpt-4o-mini-transcribe")
        worker._paused_rooms = set()

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(side_effect=RuntimeError("provider down"))

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=3,
            audio_data=sample_audio_bytes,
            is_final_chunk=False,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("translationRoom:system_events" in stream for stream in streams_published)
        assert not any("stt:results" in stream for stream in streams_published)
