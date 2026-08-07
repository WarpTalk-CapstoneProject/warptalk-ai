"""Tests for TTS Worker — Cartesia voice cloning and synthesis."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import TranslationResultMessage
from tts_worker.synthesizer import CartesiaSynthesizer
from tts_worker.worker import TTSWorker


def _make_worker(mock_redis_client, worker_settings, tts_settings=None, consented=True):
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.tts_settings = tts_settings or TTSSettings()
    worker._route_states = {}
    # Voice-clone consent gate reads _room_routes (populated in real usage by
    # AudioRouteCacheService's AUDIO_ROUTES_UPDATED broadcast — see
    # shared.base_worker.is_voice_clone_consented). Default the standard test
    # speaker/room pair ("s1" in "m1") to consented so existing cloned-voice
    # assertions keep testing what they say they test.
    worker._room_routes = (
        {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]} if consented else {}
    )
    worker._consumer_name = "test-consumer"
    worker.worker_name = "tts"
    worker.cartesia = MagicMock()
    worker.cartesia.synthesize = AsyncMock(return_value=(b"audio_bytes", 1000, "resolved-voice-id"))
    worker.cartesia.clone_voice = AsyncMock(return_value="test-voice-id")
    # Empty catalog by default (no Cartesia list_voices call was made) — makes
    # _hashed_default_voice_id() fall back to CartesiaSynthesizer._default_voice_id(),
    # matching the pre-multi-voice single hardcoded default most tests still assume.
    worker.cartesia.list_voices = AsyncMock(return_value=[])
    worker.livekit_publisher = MagicMock()
    worker.livekit_publisher.publish_pcm = AsyncMock()
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
        """synthesize() gets an explicit resolved voice_id when no clone is cached —
        _resolve_voice_variants() always resolves a concrete default (catalog-hashed,
        or the static fallback when the catalog is empty) before calling synthesize(),
        so downstream code never has to special-case voice_id=None itself."""
        worker = _make_worker(mock_redis_client, worker_settings)

        # No voice_id, no cache
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_called_once()
        _, kwargs = worker.cartesia.synthesize.call_args
        assert kwargs.get("voice_id") == CartesiaSynthesizer._default_voice_id("vi")

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
            c
            for c in mock_redis_client._redis.xadd.call_args_list
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
            c
            for c in mock_redis_client._redis.xadd.call_args_list
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
            c
            for c in mock_redis_client._redis.xadd.call_args_list
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
        system_event_calls = [c for c in xadd_calls if "system_events" in str(c.args[0])]
        assert len(system_event_calls) > 0

    async def test_publishes_to_livekit_with_wav_header_stripped(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Synthesized audio must reach the LiveKit publisher as raw PCM (WAV header
        stripped), targeting the room via meeting_id/target_lang — this is what lets
        the frontend's existing RoomAudioRenderer play it with no new playback code.
        """
        worker = _make_worker(mock_redis_client, worker_settings)
        header = b"R" * 44
        pcm_body = b"\x01\x02" * 100
        worker.cartesia.synthesize = AsyncMock(
            return_value=(header + pcm_body, 1000, "resolved-voice-id")
        )

        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None

        await worker.process(b"msg-1", _make_msg(target_lang="vi").to_redis())

        worker.livekit_publisher.publish_pcm.assert_awaited_once()
        args, kwargs = worker.livekit_publisher.publish_pcm.call_args
        call_args = {
            **dict(
                zip(["meeting_id", "speaker_id", "target_lang", "pcm_s16le", "sample_rate"], args)
            ),
            **kwargs,
        }
        assert call_args["meeting_id"] == "m1"
        assert call_args["speaker_id"] == "s1"
        assert call_args["target_lang"] == "vi"
        assert call_args["pcm_s16le"] == pcm_body

    async def test_cache_hit_still_publishes_to_livekit(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """A cache hit skips Cartesia entirely but must still reach LiveKit — cached
        audio is just as real/playable as a fresh synthesis.
        """
        worker = _make_worker(mock_redis_client, worker_settings)
        cached_audio = (b"R" * 44) + (b"\x03\x04" * 50)
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = cached_audio

        await worker.process(b"msg-1", _make_msg().to_redis())

        worker.cartesia.synthesize.assert_not_called()
        worker.livekit_publisher.publish_pcm.assert_awaited_once()


class TestSameLanguageIsNeverDubbed:
    """S6 — this worker is the last place the echo can be stopped.

    TTSWorker.process had no source_lang/target_lang comparison at all, so a
    same-language TranslationResultMessage was synthesized and pushed onto an
    ai-interpreter LiveKit track. The listener is already subscribed to the speaker's raw
    mic, so they heard the real voice and a synthetic copy of the same words together.

    translation_worker no longer produces these messages, but translate:results is a Redis
    stream that outlives a deploy — messages built by the previous revision are replayed
    into this one.
    """

    async def test_same_language_message_is_not_synthesized_or_published(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)

        message = _make_msg(text="Hello there", target_lang="en")  # source_lang is "en"

        await worker.process(b"msg-1", message.to_redis())

        worker.cartesia.synthesize.assert_not_called()
        worker.livekit_publisher.publish_pcm.assert_not_awaited()

    async def test_regional_variant_is_also_an_echo(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """en -> en-GB: translator.translate returns the text verbatim for matching base
        tags, so an exact-match guard would still dub the speaker's own words back."""
        worker = _make_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _make_msg(target_lang="en-GB").to_redis())

        worker.cartesia.synthesize.assert_not_called()
        worker.livekit_publisher.publish_pcm.assert_not_awaited()

    async def test_final_chunk_bookkeeping_still_fires(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """billing_worker and TranscriptRedisConsumerService key off this event — dropping
        it on a skipped segment would stall them, not just mute one dub."""
        worker = _make_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _make_msg(target_lang="en", is_final=True).to_redis())

        system_events = [
            call.args[1]
            for call in mock_redis_client._redis.xadd.call_args_list
            if "system_events" in str(call.args[0])
        ]
        assert [event["event_type"] for event in system_events] == ["final_chunk_processed"]
        assert json.loads(system_events[0]["payload"]) == {"segmentId": "seg-1"}
        worker.cartesia.synthesize.assert_not_called()

    async def test_a_real_translation_is_untouched(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.synthesize = AsyncMock(
            return_value=((b"R" * 44) + (b"\x03\x04" * 50), 1000, "resolved-voice-id")
        )

        await worker.process(b"msg-1", _make_msg(target_lang="vi").to_redis())

        worker.cartesia.synthesize.assert_called_once()
        worker.livekit_publisher.publish_pcm.assert_awaited_once()


class TestGetVoiceId:
    """_get_voice_id Redis lookup tests."""

    async def test_returns_none_when_not_cached(self, mock_redis_client, worker_settings) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
        mock_redis_client._redis.hget.return_value = None

        result = await worker._get_voice_id("m1", "s1")
        assert result is None

    async def test_returns_decoded_string(self, mock_redis_client, worker_settings) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
        mock_redis_client._redis.hget.return_value = b"voice-xyz"

        result = await worker._get_voice_id("m1", "s1")
        assert result == "voice-xyz"

    async def test_returns_none_when_not_consented_even_if_cached(
        self, mock_redis_client, worker_settings
    ) -> None:
        """Consent gate wins even when a voice_id is already cached from before
        the speaker revoked consent — see base_worker.is_voice_clone_consented."""
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": False}]}
        mock_redis_client._redis.hget.return_value = b"voice-xyz"

        result = await worker._get_voice_id("m1", "s1")
        assert result is None

    async def test_returns_none_when_room_routes_unknown(
        self, mock_redis_client, worker_settings
    ) -> None:
        """Fail closed: no route data received yet for this room means no consent."""
        worker = TTSWorker.__new__(TTSWorker)
        worker.redis = mock_redis_client
        worker._room_routes = {}
        mock_redis_client._redis.hget.return_value = b"voice-xyz"

        result = await worker._get_voice_id("m1", "s1")
        assert result is None


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


class TestConsumeLoopConcurrency:
    """_consume_loop() must dispatch DIFFERENT (speaker, target_lang) keys
    concurrently, while keeping any ONE key's own messages strictly ordered (they
    share a single LiveKit track — see LiveKitTTSPublisher — and Cartesia's per-call
    latency varies, so out-of-order dispatch could play sentence 2 before sentence 1)."""

    def _make_worker(self) -> TTSWorker:
        worker = TTSWorker.__new__(TTSWorker)
        worker.logger = MagicMock()
        worker._shutdown_event = asyncio.Event()
        worker._consumer_name = "test-consumer"
        worker.input_stream = "translate:results"
        worker.consumer_group = "tts-workers"
        worker._key_locks = {}
        return worker

    async def test_different_keys_dispatch_concurrently(self, mock_redis_client) -> None:
        worker = self._make_worker()
        worker.redis = mock_redis_client

        started: list[bytes] = []
        both_started = asyncio.Event()

        async def fake_process(message_id: bytes, data: dict) -> None:
            started.append(message_id)
            if len(started) == 2:
                both_started.set()
            # msg-1 can only reach here if msg-2 (a DIFFERENT speaker+lang) has
            # already started — impossible unless both run concurrently.
            await asyncio.wait_for(both_started.wait(), timeout=1.0)

        worker.process = fake_process

        async def fake_consume_concurrent(*, handler, **kwargs):
            await asyncio.gather(
                handler(b"msg-1", {"meeting_id": "m1", "speaker_id": "s1", "target_lang": "vi"}),
                handler(b"msg-2", {"meeting_id": "m1", "speaker_id": "s2", "target_lang": "ja"}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

        assert started == [b"msg-1", b"msg-2"]

    async def test_same_key_chunks_stay_ordered(self, mock_redis_client) -> None:
        worker = self._make_worker()
        worker.redis = mock_redis_client

        events: list[tuple[str, bytes]] = []

        async def fake_process(message_id: bytes, data: dict) -> None:
            events.append(("start", message_id))
            await asyncio.sleep(0.05)
            events.append(("end", message_id))

        worker.process = fake_process

        async def fake_consume_concurrent(*, handler, **kwargs):
            # Same speaker AND same target_lang — same LiveKit track, must stay ordered
            # even though this is sentence 1 and 2 of one utterance.
            await asyncio.gather(
                handler(b"msg-1", {"meeting_id": "m1", "speaker_id": "s1", "target_lang": "vi"}),
                handler(b"msg-2", {"meeting_id": "m1", "speaker_id": "s1", "target_lang": "vi"}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

        assert events == [
            ("start", b"msg-1"),
            ("end", b"msg-1"),
            ("start", b"msg-2"),
            ("end", b"msg-2"),
        ]

    async def test_same_speaker_different_target_lang_dispatch_concurrently(
        self, mock_redis_client
    ) -> None:
        """One speaker dubbed into two languages at once must not serialize just
        because it's the same speaker — target_lang is part of the key."""
        worker = self._make_worker()
        worker.redis = mock_redis_client

        started: list[bytes] = []
        both_started = asyncio.Event()

        async def fake_process(message_id: bytes, data: dict) -> None:
            started.append(message_id)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1.0)

        worker.process = fake_process

        async def fake_consume_concurrent(*, handler, **kwargs):
            await asyncio.gather(
                handler(b"msg-1", {"meeting_id": "m1", "speaker_id": "s1", "target_lang": "vi"}),
                handler(b"msg-2", {"meeting_id": "m1", "speaker_id": "s1", "target_lang": "ja"}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

        assert started == [b"msg-1", b"msg-2"]

    async def test_cleanup_room_purges_key_locks(self) -> None:
        worker = self._make_worker()
        worker._route_states = {}
        worker._paused_rooms = set()
        worker._room_routes = {}
        worker._key_locks = {
            ("m1", "s1", "vi"): asyncio.Lock(),
            ("m2", "s2", "en"): asyncio.Lock(),
        }

        worker._cleanup_room("m1")

        assert ("m1", "s1", "vi") not in worker._key_locks
        assert ("m2", "s2", "en") in worker._key_locks


class TestVoiceCatalog:
    """_get_voice_catalog() / _hashed_default_voice_id() — the auto-diversity pool."""

    async def test_fetches_and_caches_catalog(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.list_voices = AsyncMock(
            return_value=[{"id": "v1", "name": "A", "gender": "m"}]
        )

        catalog = await worker._get_voice_catalog("vi")

        assert catalog == [{"id": "v1", "name": "A", "gender": "m"}]
        worker.cartesia.list_voices.assert_awaited_once_with(
            "vi", limit=worker.tts_settings.voice_catalog_size
        )
        mock_redis_client._redis.setex.assert_called()

    async def test_uses_cache_without_calling_cartesia(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = json.dumps(
            [{"id": "cached-v", "name": "X", "gender": ""}]
        ).encode()

        catalog = await worker._get_voice_catalog("vi")

        assert catalog == [{"id": "cached-v", "name": "X", "gender": ""}]
        worker.cartesia.list_voices.assert_not_called()

    async def test_hashed_default_is_deterministic_per_speaker(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.list_voices = AsyncMock(
            return_value=[{"id": f"v{i}", "name": f"V{i}", "gender": ""} for i in range(3)]
        )

        first = await worker._hashed_default_voice_id("vi", "speaker-A")
        second = await worker._hashed_default_voice_id("vi", "speaker-A")

        assert first == second
        assert first in {"v0", "v1", "v2"}

    async def test_hashed_default_falls_back_to_static_when_catalog_empty(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.list_voices = AsyncMock(return_value=[])

        voice_id = await worker._hashed_default_voice_id("vi", "speaker-A")

        assert voice_id == CartesiaSynthesizer._default_voice_id("vi")

    async def test_different_speakers_can_get_different_default_voices(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """The actual fix for 'A and B sound identical when they talk over each
        other': with a multi-voice catalog, different un-cloned speakers must not
        all collapse onto the same single voice."""
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.list_voices = AsyncMock(
            return_value=[{"id": f"v{i}", "name": f"V{i}", "gender": ""} for i in range(6)]
        )

        results = {await worker._hashed_default_voice_id("vi", f"speaker-{i}") for i in range(20)}

        assert len(results) > 1


class TestExplicitVoiceChoices:
    """_get_explicit_voice_choices() — cross-referencing who's listening in a
    language against who explicitly picked a voice for it."""

    async def test_filters_to_listeners_currently_in_this_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)

        async def hgetall_side_effect(key):
            if key.endswith(":languages"):
                return {b"listener-1": b"vi", b"listener-2": b"en"}
            if key.endswith(":voice_preferences"):
                return {b"listener-1": b"chosen-voice-1", b"listener-2": b"chosen-voice-2"}
            return {}

        mock_redis_client._redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        choices = await worker._get_explicit_voice_choices("m1", "vi")

        # listener-2 is tuned to "en", not "vi" — their pick must not leak into "vi"'s set.
        assert choices == {"chosen-voice-1"}

    async def test_empty_when_nobody_listening_in_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall = AsyncMock(return_value={})

        choices = await worker._get_explicit_voice_choices("m1", "vi")

        assert choices == set()


class TestVoiceVariantFanOut:
    """process() end-to-end: an explicit listener voice preference must reach
    LiveKit as its own track, WITHOUT causing a second billing_worker charge for
    what is still just one translated utterance."""

    async def test_explicit_preference_adds_livekit_track_without_extra_billing(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _make_worker(mock_redis_client, worker_settings, consented=False)  # not cloned
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None  # no cache hits anywhere
        # Audio must exceed _WAV_HEADER_BYTES (44) or the LiveKit publish is skipped
        # as "nothing left after stripping the header" — see _publish_livekit_only.
        worker.cartesia.synthesize = AsyncMock(
            return_value=(b"R" * 44 + b"\x01\x02" * 50, 1000, "resolved-voice-id")
        )

        async def hgetall_side_effect(key):
            if key.endswith(":languages"):
                return {b"listener-1": b"vi"}
            if key.endswith(":voice_preferences"):
                return {b"listener-1": b"preferred-voice-xyz"}
            return {}

        mock_redis_client._redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        await worker.process(b"msg-1", _make_msg(target_lang="vi").to_redis())

        # Two Cartesia calls (default hashed voice + the explicit preference voice)
        # and two LiveKit publishes — one per distinct voice variant.
        assert worker.cartesia.synthesize.await_count == 2
        assert worker.livekit_publisher.publish_pcm.await_count == 2
        voice_keys = {
            c.kwargs.get("voice_key") for c in worker.livekit_publisher.publish_pcm.await_args_list
        }
        assert voice_keys == {"", f"voice-{'preferred-voice-xyz'[:8]}"}

        # Only the DEFAULT variant publishes tts:results (2 xadd calls: per-room +
        # global stream, via BaseWorker.publish()'s dual-write) — the preference
        # variant must not add a second billing event for the same utterance.
        tts_result_calls = [
            c
            for c in mock_redis_client._redis.xadd.call_args_list
            if "tts:results" in str(c.args[0])
        ]
        assert len(tts_result_calls) == 2

    async def test_preference_matching_default_voice_is_not_duplicated(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """If a listener's explicit pick happens to equal the resolved default voice,
        it must not trigger a second, redundant synthesis+track."""
        worker = _make_worker(mock_redis_client, worker_settings, consented=False)
        mock_redis_client._redis.hget.return_value = None
        mock_redis_client._redis.get.return_value = None
        worker.cartesia.list_voices = AsyncMock(return_value=[])  # -> static default id
        worker.cartesia.synthesize = AsyncMock(
            return_value=(b"R" * 44 + b"\x01\x02" * 50, 1000, "resolved-voice-id")
        )
        default_id = CartesiaSynthesizer._default_voice_id("vi")

        async def hgetall_side_effect(key):
            if key.endswith(":languages"):
                return {b"listener-1": b"vi"}
            if key.endswith(":voice_preferences"):
                return {b"listener-1": default_id.encode()}
            return {}

        mock_redis_client._redis.hgetall = AsyncMock(side_effect=hgetall_side_effect)

        await worker.process(b"msg-1", _make_msg(target_lang="vi").to_redis())

        assert worker.cartesia.synthesize.await_count == 1
        assert worker.livekit_publisher.publish_pcm.await_count == 1
