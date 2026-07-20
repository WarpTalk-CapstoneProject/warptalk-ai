"""Tests for Billing Settlement Worker — segment-id extraction helper."""

from __future__ import annotations

import uuid

from billing_worker.worker import _extract_underlying_segment_id


class TestExtractUnderlyingSegmentId:
    """Byte-for-byte port of TranscriptRedisConsumerService.ExtractUnderlyingSegmentId (C#)."""

    def test_plain_guid_passthrough(self) -> None:
        guid = str(uuid.uuid4())
        assert _extract_underlying_segment_id(guid) == guid

    def test_composite_segment_id_single_digit_chunk(self) -> None:
        guid = str(uuid.uuid4())
        composite = f"{guid}-c0"
        assert _extract_underlying_segment_id(composite) == guid

    def test_composite_segment_id_multi_digit_chunk(self) -> None:
        guid = str(uuid.uuid4())
        composite = f"{guid}-c12"
        assert _extract_underlying_segment_id(composite) == guid

    def test_malformed_input_returns_none(self) -> None:
        assert _extract_underlying_segment_id("not-a-guid-at-all") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_underlying_segment_id("") is None

    def test_none_like_falsy_returns_none(self) -> None:
        assert _extract_underlying_segment_id(None) is None  # type: ignore[arg-type]
