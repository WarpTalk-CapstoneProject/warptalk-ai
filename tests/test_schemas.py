"""Tests for shared.schemas — Pydantic message serialization roundtrips."""

from __future__ import annotations

import pytest

from shared.schemas import (
    AudioChunkMessage,
    STTResultMessage,
    TranslationResultMessage,
    TTSResultMessage,
)


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
        redis_data = {
            k.encode(): v.encode() for k, v in msg.to_redis().items()
        }
        restored = AudioChunkMessage.from_redis(redis_data)
        assert restored.meeting_id == "m1"
        assert restored.audio_data == sample_audio_bytes


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
        msg = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="test", language="en"
        )
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
        assert restored.voice_type == "cloned"


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
