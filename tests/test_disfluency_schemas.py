"""WT-716 wire contract: clean_text/clean_flags on STT results and CleanSentenceMessage."""

from __future__ import annotations

import json

from shared.schemas import TRANSCRIPT_CLEAN_STREAM, CleanSentenceMessage, STTResultMessage


def _stt(**overrides: object) -> STTResultMessage:
    fields: dict[str, object] = {
        "segment_id": "seg-1",
        "meeting_id": "room-1",
        "speaker_id": "spk-1",
        "text": "um we ship",
        "language": "en",
        "timestamp_ms": 1,
    }
    fields.update(overrides)
    return STTResultMessage(**fields)  # type: ignore[arg-type]


def test_message_without_clean_fields_is_unchanged_on_the_wire() -> None:
    payload = _stt().to_redis()
    assert "clean_text" not in payload
    assert "clean_flags" not in payload

    restored = STTResultMessage.from_redis(payload)
    assert restored.clean_text is None
    assert restored.clean_flags == ()
    assert restored.display_text == "um we ship"
    assert restored.to_redis() == payload


def test_clean_fields_roundtrip() -> None:
    message = _stt(clean_text="We ship.", clean_flags=("fillers_removed", "escalate"))
    payload = message.to_redis()
    assert payload["clean_text"] == "We ship."
    assert payload["clean_flags"] == "fillers_removed,escalate"

    restored = STTResultMessage.from_redis(payload)
    assert restored.clean_text == "We ship."
    assert restored.clean_flags == ("fillers_removed", "escalate")
    assert restored.display_text == "We ship."
    assert restored.text == "um we ship"


def test_filler_only_empty_clean_text_survives_the_wire() -> None:
    message = _stt(text="Ummm", clean_text="", clean_flags=("filler_only",))
    payload = message.to_redis()
    assert payload["clean_text"] == ""

    restored = STTResultMessage.from_redis(payload)
    assert restored.clean_text == ""
    assert restored.display_text == ""


def test_from_redis_accepts_bytes_and_blank_flags() -> None:
    payload = {k.encode(): v.encode() for k, v in _stt().to_redis().items()}
    payload[b"clean_flags"] = b""
    restored = STTResultMessage.from_redis(payload)
    assert restored.clean_flags == ()


def test_clean_sentence_roundtrip() -> None:
    message = CleanSentenceMessage(
        meeting_id="room-1",
        sentence_id="11111111-1111-1111-1111-111111111111",
        revision=2,
        speaker_id="spk-1",
        segment_ids=["a-guid", "b-guid"],
        clean_text="We need to go through the list.",
        language="en",
        flags=["self_repair", "escalate"],
        source="llm",
        timestamp_ms=42,
    )
    payload = message.to_redis()
    assert all(isinstance(v, str) for v in payload.values())
    assert json.loads(payload["segment_ids"]) == ["a-guid", "b-guid"]
    assert payload["flags"] == "self_repair,escalate"
    assert payload["revision"] == "2"

    assert CleanSentenceMessage.from_redis(payload) == message


def test_clean_sentence_defaults_and_tolerant_parse() -> None:
    message = CleanSentenceMessage(
        meeting_id="room-1", speaker_id="spk-1", clean_text="Hi.", language="en"
    )
    assert message.revision == 0
    assert message.source == "prepass"
    assert message.flags == []
    assert message.sentence_id

    payload = message.to_redis()
    assert payload["flags"] == ""
    payload["segment_ids"] = "not json"
    restored = CleanSentenceMessage.from_redis(payload)
    assert restored.segment_ids == []
    assert restored.flags == []


def test_stream_name() -> None:
    assert TRANSCRIPT_CLEAN_STREAM == "transcript:clean"
