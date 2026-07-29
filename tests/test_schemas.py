"""Tests for shared.schemas — Pydantic message serialization roundtrips."""

from __future__ import annotations

from decimal import Decimal

import pytest

from shared.schemas import (
    AIUsageMessage,
    AudioChunkMessage,
    ChatRequestMessage,
    ChatResultMessage,
    ProviderUsageMessage,
    STTResultMessage,
    TranslationResultMessage,
    TTSResultMessage,
)


def test_chat_request_origin_roundtrip() -> None:
    request = ChatRequestMessage(
        request_id="request-1",
        conversation_id="conversation-1",
        workspace_id="workspace-1",
        user_id="user-1",
        origin="meeting_chat",
    )

    restored = ChatRequestMessage.from_redis(request.to_redis())

    assert restored.origin == "meeting_chat"


def test_chat_result_origin_roundtrip() -> None:
    result = ChatResultMessage(
        request_id="request-1",
        conversation_id="conversation-1",
        type="completed",
        origin="meeting_chat",
    )

    restored = ChatResultMessage.from_redis(result.to_redis())

    assert restored.origin == "meeting_chat"


class TestAudioChunkMessage:
    """AudioChunkMessage serialize/deserialize tests."""

    def test_roundtrip(self, sample_audio_bytes: bytes) -> None:
        """to_redis → from_redis produces identical data."""
        original = AudioChunkMessage(
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            chunk_index=5,
            audio_data=sample_audio_bytes,
            language="en",
            sample_rate=16000,
        )

        redis_data = original.to_redis()
        restored = AudioChunkMessage.from_redis(redis_data)

        assert restored.meeting_id == original.meeting_id
        assert restored.speaker_id == original.speaker_id
        assert restored.chunk_index == original.chunk_index
        assert restored.audio_data == original.audio_data
        assert restored.language == original.language
        assert restored.sample_rate == original.sample_rate

    def test_redis_values_are_strings(self, sample_audio_bytes: bytes) -> None:
        """All Redis field values must be strings."""
        msg = AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        )
        redis_data = msg.to_redis()
        for key, value in redis_data.items():
            assert isinstance(key, str), f"Key {key} is not str"
            assert isinstance(value, str), f"Value for {key} is not str"

    def test_from_redis_bytes_keys(self, sample_audio_bytes: bytes) -> None:
        """from_redis handles byte keys/values from Redis."""
        msg = AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        )
        # Simulate Redis returning bytes
        redis_data = {k.encode(): v.encode() for k, v in msg.to_redis().items()}
        restored = AudioChunkMessage.from_redis(redis_data)
        assert restored.meeting_id == "m1"
        assert restored.audio_data == sample_audio_bytes

    def test_from_redis_accepts_gateway_translation_room_id_alias(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Gateway uses translation_room_id for the same canonical meeting stream."""
        redis_data = AudioChunkMessage(
            meeting_id="room-123",
            speaker_id="s1",
            chunk_index=1,
            audio_data=sample_audio_bytes,
            source_runtime="desktop",
            vad_confidence=0.84,
            speech_start_ms=120,
            speech_end_ms=880,
            input_lufs=-18.5,
            noise_suppression_enabled=True,
        ).to_redis()
        redis_data["translation_room_id"] = redis_data.pop("meeting_id")

        restored = AudioChunkMessage.from_redis(redis_data)

        assert restored.meeting_id == "room-123"
        assert restored.source_runtime == "desktop"
        assert restored.vad_confidence == pytest.approx(0.84)
        assert restored.speech_start_ms == 120
        assert restored.speech_end_ms == 880
        assert restored.input_lufs == pytest.approx(-18.5)
        assert restored.noise_suppression_enabled is True


class TestSTTResultMessage:
    """STTResultMessage tests."""

    def test_roundtrip(self) -> None:
        original = STTResultMessage(
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            text="Hello, how are you?",
            language="en",
            confidence=0.95,
            start_ms=1000,
            end_ms=3000,
            chunk_index=2,
        )

        redis_data = original.to_redis()
        restored = STTResultMessage.from_redis(redis_data)

        assert restored.meeting_id == original.meeting_id
        assert restored.text == original.text
        assert restored.language == original.language
        assert restored.confidence == original.confidence
        assert restored.start_ms == original.start_ms
        assert restored.end_ms == original.end_ms

    def test_auto_generates_segment_id(self) -> None:
        msg = STTResultMessage(meeting_id="m1", speaker_id="s1", text="test", language="en")
        assert msg.segment_id  # Should be auto-generated UUID


class TestTranslationResultMessage:
    """TranslationResultMessage tests."""

    def test_roundtrip(self) -> None:
        original = TranslationResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            original_text="Hello",
            translated_text="Xin chào",
            source_lang="en",
            target_lang="vi",
            confidence=0.9,
        )

        redis_data = original.to_redis()
        restored = TranslationResultMessage.from_redis(redis_data)

        assert restored.original_text == original.original_text
        assert restored.translated_text == original.translated_text
        assert restored.source_lang == original.source_lang
        assert restored.target_lang == original.target_lang


class TestTTSResultMessage:
    """TTSResultMessage tests."""

    def test_roundtrip(self) -> None:
        audio = b"fake-audio-bytes-here"
        original = TTSResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            audio_data=audio,
            duration_ms=1500,
            voice_type="cloned",
            target_lang="vi",
        )

        redis_data = original.to_redis()
        restored = TTSResultMessage.from_redis(redis_data)

        assert restored.audio_data == audio
        assert restored.duration_ms == 1500
        assert restored.char_count == 0
        assert restored.voice_type == "cloned"

    def test_roundtrip_voice_blending_metadata(self) -> None:
        audio = b"fake-audio-bytes-here"
        original = TTSResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            audio_data=audio,
            duration_ms=1500,
            voice_type="blended",
            voice_mode="blended",
            clone_strength=0.6,
            anchor_provider="edge",
            clone_provider="xtts",
            render_location="server",
            cache_key="voice-cache-key",
            cache_hit=True,
            synthesis_latency_ms=120,
            conversion_latency_ms=240,
            fallback_reason="",
            target_lang="vi",
        )

        restored = TTSResultMessage.from_redis(original.to_redis())

        assert restored.voice_type == "blended"
        assert restored.voice_mode == "blended"
        assert restored.clone_strength == pytest.approx(0.6)
        assert restored.anchor_provider == "edge"
        assert restored.clone_provider == "xtts"
        assert restored.render_location == "server"
        assert restored.cache_key == "voice-cache-key"
        assert restored.cache_hit is True
        assert restored.synthesis_latency_ms == 120
        assert restored.conversion_latency_ms == 240
        assert restored.fallback_reason == ""

    def test_from_redis_old_tts_result_defaults_voice_metadata(self) -> None:
        audio = b"fake-audio-bytes-here"
        old_payload = TTSResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            audio_data=audio,
            duration_ms=1500,
            voice_type="default",
            target_lang="vi",
        ).to_redis()
        for field in (
            "voice_mode",
            "clone_strength",
            "anchor_provider",
            "clone_provider",
            "render_location",
            "cache_key",
            "cache_hit",
            "synthesis_latency_ms",
            "conversion_latency_ms",
            "fallback_reason",
        ):
            old_payload.pop(field, None)

        restored = TTSResultMessage.from_redis(old_payload)

        assert restored.voice_mode == "standard"
        assert restored.clone_strength == 0.0
        assert restored.anchor_provider == ""
        assert restored.clone_provider == ""
        assert restored.render_location == "server"
        assert restored.cache_hit is False


class TestAIUsageMessage:
    """AIUsageMessage tests."""

    def test_roundtrip(self) -> None:
        original = AIUsageMessage(
            workspace_id="workspace-1",
            room_id="room-1",
            user_id="user-1",
            charge_type="TRANSLATION",
            model="gpt-4.1-mini",
            prompt_tokens=120,
            cached_tokens=20,
            completion_tokens=40,
            source_lang="en",
            target_lang="vi",
            idempotency_key="TRANSLATION:room-1:1",
        )

        restored = AIUsageMessage.from_redis(original.to_redis())

        assert restored.workspace_id == original.workspace_id
        assert restored.room_id == original.room_id
        assert restored.prompt_tokens == 120
        assert restored.cached_tokens == 20
        assert restored.completion_tokens == 40
        assert restored.idempotency_key == original.idempotency_key


class TestProviderUsageMessage:
    """ProviderUsageMessage tests."""

    def test_roundtrip_voice_clone_enrollment(self) -> None:
        original = ProviderUsageMessage(
            room_id="room-1",
            user_id="speaker-1",
            charge_type="VOICE_CLONE_ENROLLMENT",
            provider="cartesia",
            model="cartesia-localizing-voice",
            quantity=Decimal(1),
            unit="profile",
            idempotency_key="VOICE_CLONE_ENROLLMENT:room-1:speaker-1",
        )

        restored = ProviderUsageMessage.from_redis(original.to_redis())

        assert restored.room_id == "room-1"
        assert restored.user_id == "speaker-1"
        assert restored.charge_type == "VOICE_CLONE_ENROLLMENT"
        assert restored.provider == "cartesia"
        assert restored.model == "cartesia-localizing-voice"
        assert restored.quantity == Decimal(1)
        assert restored.unit == "profile"


@pytest.mark.parametrize(
    "lang_pair",
    [
        ("en", "vi"),
        ("zh", "ja"),
        ("fr", "de"),
        ("vi", "en"),
    ],
)
def test_translation_language_pairs(lang_pair: tuple[str, str]) -> None:
    """TranslationResultMessage handles various language pairs."""
    src, tgt = lang_pair
    msg = TranslationResultMessage(
        segment_id="s1",
        meeting_id="m1",
        speaker_id="sp1",
        original_text="test",
        translated_text="translated",
        source_lang=src,
        target_lang=tgt,
    )
    restored = TranslationResultMessage.from_redis(msg.to_redis())
    assert restored.source_lang == src
    assert restored.target_lang == tgt
