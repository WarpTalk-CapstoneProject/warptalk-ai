"""Tests for shared.schemas — Pydantic message serialization roundtrips."""

from __future__ import annotations

import pytest

from shared.schemas import (
    STT_UNKNOWN_CONFIDENCE,
    AudioChunkMessage,
    ChatRequestMessage,
    ChatResultMessage,
    ProsodyEnvelope,
    STTResultMessage,
    SuggestionResultMessage,
    TranslationResultMessage,
    TTSResultMessage,
    optional_confidence,
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


class TestSuggestionResultMessage:
    """SuggestionResultMessage serialize/deserialize tests."""

    def test_roundtrip(self) -> None:
        original = SuggestionResultMessage(
            meeting_id="room-1",
            segment_id="segment-1",
            category="term",
            content="RAG = Retrieval Augmented Generation",
            detail="Nobody has defined the acronym yet in this meeting.",
            confidence=0.82,
            language="vi",
            token_count=145,
        )

        restored = SuggestionResultMessage.from_redis(original.to_redis())

        assert restored.meeting_id == "room-1"
        assert restored.segment_id == "segment-1"
        assert restored.category == "term"
        assert restored.content == "RAG = Retrieval Augmented Generation"
        assert restored.detail == "Nobody has defined the acronym yet in this meeting."
        assert restored.confidence == pytest.approx(0.82)
        assert restored.language == "vi"
        assert restored.token_count == 145
        assert restored.timestamp_ms == original.timestamp_ms

    def test_type_defaults_to_suggestion(self) -> None:
        """The gateway routes on `type` — it must be on the wire without being set."""
        message = SuggestionResultMessage(
            meeting_id="room-1",
            segment_id="segment-1",
            category="action",
            content="Ai chốt deadline cho phần này?",
        )

        assert message.to_redis()["type"] == "suggestion"

    def test_roundtrip_from_redis_bytes(self) -> None:
        """Redis hands back bytes, not str — from_redis must decode both keys and values."""
        encoded = {
            key.encode(): value.encode()
            for key, value in SuggestionResultMessage(
                meeting_id="room-1",
                segment_id="segment-1",
                category="clarification",
                content="Câu hỏi này chưa được trả lời.",
                confidence=0.75,
            )
            .to_redis()
            .items()
        }

        restored = SuggestionResultMessage.from_redis(encoded)

        assert restored.meeting_id == "room-1"
        assert restored.category == "clarification"
        assert restored.confidence == pytest.approx(0.75)

    def test_optional_fields_absent_on_the_wire(self) -> None:
        """An older/partial producer must not crash the consumer."""
        restored = SuggestionResultMessage.from_redis(
            {
                "meeting_id": "room-1",
                "segment_id": "segment-1",
                "category": "fact",
                "content": "Doanh thu Q2 là 1.2 tỷ.",
            }
        )

        assert restored.type == "suggestion"
        assert restored.detail == ""
        assert restored.confidence == 0.0
        assert restored.token_count == 0


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


class TestProsodyOnTheWire:
    """How an utterance was said, carried from the audio to the synthesizer.

    The pipeline throws away delivery at the STT boundary. These pin the two properties the
    carrying depends on: an unmeasured message says nothing at all (rather than claiming
    neutral), and a malformed one costs the dub its delivery and nothing else.
    """

    def _envelope(self) -> ProsodyEnvelope:
        return ProsodyEnvelope(
            pitch_lift=1.24,
            pitch_variation=1.55,
            energy_ratio=1.4,
            rate_ratio=1.15,
            arousal="high",
            valence="positive",
        )

    def test_unmeasured_prosody_is_absent_not_neutral(self) -> None:
        payload = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en"
        ).to_redis()

        # A "neutral" placeholder would tell the TTS worker to send speed=1.0/volume=1.0 to
        # Cartesia for a speaker nobody has measured. Absence is what makes it send nothing.
        assert "prosody" not in payload
        assert STTResultMessage.from_redis(payload).prosody is None

    def test_stt_result_roundtrips_delivery(self) -> None:
        original = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en", prosody=self._envelope()
        )

        restored = STTResultMessage.from_redis(original.to_redis())

        assert restored.prosody is not None
        assert restored.prosody.pitch_lift == pytest.approx(1.24)
        assert restored.prosody.energy_ratio == pytest.approx(1.4)
        assert restored.prosody.arousal == "high"
        assert restored.prosody.valence == "positive"

    def test_translation_result_carries_it_unchanged(self) -> None:
        original = TranslationResultMessage(
            segment_id="seg-1",
            meeting_id="m1",
            speaker_id="s1",
            original_text="xin chào",
            translated_text="hello",
            source_lang="vi",
            target_lang="en",
            prosody=self._envelope(),
        )

        restored = TranslationResultMessage.from_redis(original.to_redis())

        assert restored.prosody == original.prosody

    @pytest.mark.parametrize("raw", ["", "not json", "{}", '{"pl":"abc"}', "[1,2,3]"])
    def test_a_broken_envelope_is_dropped_rather_than_raised(self, raw: str) -> None:
        # Delivery is a decoration on the audio. Losing it must never be able to take the
        # meeting's transcript or dub with it, so decoding cannot raise for any input.
        assert ProsodyEnvelope.from_wire(raw) is None

    def test_a_message_with_a_broken_envelope_still_parses(self) -> None:
        payload = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en"
        ).to_redis()
        payload["prosody"] = "{corrupt"

        restored = STTResultMessage.from_redis(payload)

        assert restored.text == "hi"
        assert restored.prosody is None


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
            source_stt_confidence=-0.3421,
        )

        redis_data = original.to_redis()
        restored = TranslationResultMessage.from_redis(redis_data)

        assert restored.original_text == original.original_text
        assert restored.translated_text == original.translated_text
        assert restored.source_lang == original.source_lang
        assert restored.target_lang == original.target_lang
        assert restored.source_stt_confidence == pytest.approx(-0.3421)

    def test_carries_no_field_named_confidence(self) -> None:
        """WT-278: the translator produces no quality score, so nothing on a translation may be
        called `confidence`. The only number available is the SOURCE segment's STT avg_logprob."""
        payload = TranslationResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            original_text="Hello",
            translated_text="Xin chào",
            source_lang="en",
            target_lang="vi",
            source_stt_confidence=-0.3421,
        ).to_redis()

        assert "confidence" not in payload
        assert payload["source_stt_confidence"] == "-0.3421"

    def test_unknown_source_confidence_is_omitted_from_the_wire(self) -> None:
        """WT-277: a Redis stream field is a string, so "unknown" cannot be a value — the field is
        left out entirely and consumers store NULL. It must not become 0.0, 1.0 or "None"."""
        payload = TranslationResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            original_text="Hello",
            translated_text="Xin chào",
            source_lang="en",
            target_lang="vi",
        ).to_redis()

        assert "source_stt_confidence" not in payload
        assert TranslationResultMessage.from_redis(payload).source_stt_confidence is None

    def test_latency_survives_the_round_trip(self) -> None:
        """The number translation_worker measured has to reach TranscriptService intact.

        transcript.translation_contents.latency_ms and TranslationContent.LatencyMs both existed
        from the start and were NULL on all 3803 rows ever written, because the worker measured
        the duration, logged it, and dropped it at the process boundary. Carrying it on the
        message is the whole of the fix, so this asserts the carrying.
        """
        payload = TranslationResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            original_text="Hello",
            translated_text="Xin chao",
            source_lang="en",
            target_lang="vi",
            latency_ms=734,
        ).to_redis()

        assert payload["latency_ms"] == "734"
        assert TranslationResultMessage.from_redis(payload).latency_ms == 734

    def test_unmeasured_latency_is_omitted_rather_than_zero(self) -> None:
        """Nothing translated means no duration to report — and 0 is not that.

        The empty-sentence flush publishes a message no translator ever touched. Sending 0 for it
        would land a real-looking measurement in the column and drag down every average computed
        over it, which is worse than the NULL this replaces.
        """
        payload = TranslationResultMessage(
            segment_id="seg-123",
            meeting_id="meeting-123",
            speaker_id="speaker-1",
            original_text="",
            translated_text="",
            source_lang="en",
            target_lang="vi",
        ).to_redis()

        assert "latency_ms" not in payload
        assert TranslationResultMessage.from_redis(payload).latency_ms is None


class TestOptionalConfidence:
    """WT-277: every flavour of "the producer told us nothing" must collapse to None."""

    def test_absent_or_blank_is_unknown(self) -> None:
        assert optional_confidence(None) is None
        assert optional_confidence("") is None
        assert optional_confidence("   ") is None

    def test_unparsable_is_unknown(self) -> None:
        assert optional_confidence("not-a-number") is None

    def test_non_finite_is_unknown(self) -> None:
        assert optional_confidence("nan") is None
        assert optional_confidence("inf") is None

    def test_stt_sentinel_is_unknown(self) -> None:
        """stt_worker/model.py uses -1.0 for "this event exposed no token logprobs"."""
        assert optional_confidence(str(STT_UNKNOWN_CONFIDENCE)) is None

    def test_genuine_measurement_survives(self) -> None:
        assert optional_confidence("-0.3421") == pytest.approx(-0.3421)


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
