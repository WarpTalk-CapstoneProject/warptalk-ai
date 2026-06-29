"""Tests for TTS Worker — Cartesia voice cloning and synthesis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import TranslationResultMessage
from tts_worker.worker import TTSWorker


def _make_worker(mock_redis_client, worker_settings, tts_settings=None):
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.tts_settings = tts_settings or TTSSettings()
    worker._route_states = {}
    worker._consumer_name = "test-consumer"
    worker.worker_name = "tts"
    worker.cartesia = MagicMock()
    worker.cartesia.synthesize = AsyncMock(return_value=(b"audio_bytes", 1000))
    worker.cartesia.clone_voice = AsyncMock(return_value="test-voice-id")
    return worker


def _make_msg(text="Xin chào bạn", target_lang="vi", is_final=False):
    return TranslationResultMessage(
        segment_id="seg-1",
        meeting_id="m1",
        speaker_id="s1",
        original_text="Hello",
        translated_text=text,
        source_lang="en",
        target_lang=target_lang,
        is_final_chunk=is_final,
    )


class TestTTSWorker:
    """TTSWorker process() tests with CartesiaSynthesizer."""

    async def test_uses_default_voice_when_no_voice_id(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """synthesize() called with voice_id=None when no clone cached."""
        worker = _make_worker(mock_redis_client, worker_settings)

        # No voice_id, no cache
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_called_once()
        _, kwargs = worker.cartesia.synthesize.call_args
        assert kwargs.get("voice_id") is None

    async def test_uses_cloned_voice_when_voice_id_cached(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """synthesize() called with voice_id when clone is cached in Redis."""
        worker = _make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hget.return_value = b"cached-voice-id"
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_called_once()
        _, kwargs = worker.cartesia.synthesize.call_args
        assert kwargs.get("voice_id") == "cached-voice-id"

    async def test_publishes_cloned_voice_metadata(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Published TTSResultMessage should reflect cloned voice fields."""
        worker = _make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hget.return_value = b"voice-abc"
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        mock_redis_client._redis.xadd.assert_called()
        # Find the tts:results publish (first call is room-specific stream)
        tts_call = next(
            c for c in mock_redis_client._redis.xadd.call_args_list
            if "tts:results" in str(c.args[0])
        )
        published = tts_call.args[1]
        assert published["voice_type"] == "cloned"
        assert published["voice_mode"] == "cloned"
        assert published["clone_strength"] == "1.0"
        assert published["anchor_provider"] == "cartesia"
        assert published["clone_provider"] == "cartesia"
        assert published["cache_hit"] == "false"

    async def test_publishes_default_voice_metadata_when_no_clone(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Published result should reflect voice_type=default when no voice_id."""
        worker = _make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        tts_call = next(
            c for c in mock_redis_client._redis.xadd.call_args_list
            if "tts:results" in str(c.args[0])
        )
        published = tts_call.args[1]
        assert published["voice_type"] == "default"
        assert published["clone_strength"] == "0.0"
        assert published["fallback_reason"] == "voice_profile_not_ready"

    async def test_skips_empty_text(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """process() should not call synthesize for empty translated_text."""
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hget.return_value = None

        await worker.process(b"msg-1", _make_msg(text="   ").to_redis())

        worker.cartesia.synthesize.assert_not_called()

    async def test_cache_hit_skips_synthesis(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Cache hit should publish immediately without calling synthesize."""
        worker = _make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = b"cached-audio"

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_not_called()
        mock_redis_client._redis.xadd.assert_called()
        tts_call = next(
            c for c in mock_redis_client._redis.xadd.call_args_list
            if "tts:results" in str(c.args[0])
        )
        assert tts_call.args[1]["cache_hit"] == "true"

    async def test_synthesis_error_does_not_publish_audio(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """On synthesis failure, no TTSResultMessage should be published."""
        worker = _make_worker(mock_redis_client, worker_settings)
        worker.cartesia.synthesize = AsyncMock(side_effect=Exception("API down"))

        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        # xadd may be called for system event (via publish_system_event → xadd)
        # but the audio publish should NOT have been called with audio_data
        for call in mock_redis_client._redis.xadd.call_args_list:
            stream = call.args[0] if call.args else ""
            if "tts:results" in stream:
                pytest.fail("TTSResultMessage should not be published on synthesis error")

    async def test_paused_route_skips_synthesis(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """PAUSED route should return immediately without synthesis."""
        worker = _make_worker(mock_redis_client, worker_settings)
        worker._route_states = {"m1": "PAUSED"}
        mock_redis_client._redis.hget.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_not_called()

    async def test_final_chunk_publishes_final_chunk_processed_event(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """is_final_chunk=True should trigger final_chunk_processed system event."""
        worker = _make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg(is_final=True).to_redis())

        # final_chunk_processed is published via publish_system_event → xadd to system_events
        xadd_calls = mock_redis_client._redis.xadd.call_args_list
        system_event_calls = [
            c for c in xadd_calls if "system_events" in str(c.args[0])
        ]
        assert len(system_event_calls) > 0


class TestGetVoiceId:
    """_get_voice_id Redis lookup tests."""

    async def test_returns_none_when_not_cached(
        self, mock_redis_client, worker_settings
    ) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        mock_redis_client._redis.hget.return_value = None

        result = await worker._get_voice_id("m1", "s1")
        assert result is None

    async def test_returns_decoded_string(
        self, mock_redis_client, worker_settings
    ) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        mock_redis_client._redis.hget.return_value = b"voice-xyz"

        result = await worker._get_voice_id("m1", "s1")
        assert result == "voice-xyz"


class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = TTSWorker._cache_key("s1", "vi", "Xin chào", "cloned")
        k2 = TTSWorker._cache_key("s1", "vi", "Xin chào", "cloned")
        assert k1 == k2

    def test_different_text_different_key(self) -> None:
        k1 = TTSWorker._cache_key("s1", "vi", "Hello", "default")
        k2 = TTSWorker._cache_key("s1", "vi", "Goodbye", "default")
        assert k1 != k2

    def test_different_voice_mode_different_key(self) -> None:
        k1 = TTSWorker._cache_key("s1", "vi", "Hello", "default")
        k2 = TTSWorker._cache_key("s1", "vi", "Hello", "cloned")
        assert k1 != k2

    def test_case_insensitive(self) -> None:
        k1 = TTSWorker._cache_key("s1", "vi", "hello world", "default")
        k2 = TTSWorker._cache_key("s1", "vi", "HELLO WORLD", "default")
        assert k1 == k2

    def test_starts_with_prefix(self) -> None:
        k = TTSWorker._cache_key("s1", "vi", "Hello", "default")
        assert k.startswith("tts:cache:")
