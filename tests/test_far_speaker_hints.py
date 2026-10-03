"""Live far-side speaker names: caption hints -> far_speaker_* on stand-in segments.

The desktop will read Meet's captions and XADD `{name, t_ms, source}` to
`meeting:{room}:far_speaker_hints`. A stand-in segment whose time window a (lag-shifted) hint
lands in carries that name on stt:results — optional fields, absent otherwise, so every older
consumer and every non-bridge segment sees exactly what it saw before.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.config import STTSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.far_speaker import (
    CaptionHintTracker,
    FarSpeakerHint,
    FarSpeakerTracker,
    SegmentWindow,
    attribute_far_speaker,
    far_speaker_hints_key,
    parse_hint,
)
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.model import TranscribedSegment
from stt_worker.worker import STTWorker

W = SegmentWindow(start_ms=10_000, end_ms=12_000)


def _h(name: str, t_ms: int, source: str = "meet_caption") -> FarSpeakerHint:
    return FarSpeakerHint(name=name, t_ms=t_ms, source=source)


class TestAttributeFarSpeaker:
    def test_no_hints_no_answer(self) -> None:
        assert attribute_far_speaker(W, [], lag_ms=500) is None

    def test_single_hint_inside_the_window(self) -> None:
        got = attribute_far_speaker(W, [_h("Lan", 11_500)], lag_ms=500)
        assert got is not None
        assert (got.name, got.source, got.confidence) == ("Lan", "meet_caption", 1.0)

    def test_lag_shifts_the_hint_back(self) -> None:
        # Seen at 12_400: without lag it is outside, with 500ms lag it was spoken at 11_900.
        assert attribute_far_speaker(W, [_h("Lan", 12_400)], lag_ms=0, max_gap_ms=0) is None
        got = attribute_far_speaker(W, [_h("Lan", 12_400)], lag_ms=500, max_gap_ms=0)
        assert got is not None and got.name == "Lan"

    def test_majority_wins_and_confidence_is_its_share(self) -> None:
        hints = [_h("Lan", 10_200), _h("Lan", 10_800), _h("Minh", 11_600)]
        got = attribute_far_speaker(W, hints, lag_ms=0)
        assert got is not None
        assert got.name == "Lan"
        assert got.confidence == round(2 / 3, 3)

    def test_tie_goes_to_the_latest_speaker(self) -> None:
        hints = [_h("Lan", 10_200), _h("Minh", 11_600)]
        got = attribute_far_speaker(W, hints, lag_ms=0)
        assert got is not None and got.name == "Minh" and got.confidence == 0.5

    def test_names_are_compared_case_insensitively(self) -> None:
        hints = [_h("lan pham", 10_200), _h("Lan Pham", 10_800), _h("Minh", 11_000)]
        got = attribute_far_speaker(W, hints, lag_ms=0)
        assert got is not None and got.name == "Lan Pham"

    def test_nearest_hint_outside_the_window_names_it_at_reduced_confidence(self) -> None:
        hints = [_h("Lan", 9_000), _h("Minh", 12_300)]
        got = attribute_far_speaker(W, hints, lag_ms=0, max_gap_ms=1_500)
        assert got is not None
        assert got.name == "Minh"  # 300ms after vs 1000ms before
        assert 0 < got.confidence < 0.5

    def test_inside_beats_nearest(self) -> None:
        hints = [_h("Lan", 11_000), _h("Minh", 12_001)]
        got = attribute_far_speaker(W, hints, lag_ms=0)
        assert got is not None and got.name == "Lan" and got.confidence == 1.0

    def test_hint_too_far_away_is_no_answer(self) -> None:
        assert attribute_far_speaker(W, [_h("Lan", 20_000)], lag_ms=0, max_gap_ms=1_500) is None

    def test_b3_sole_name_around_the_window_clears_the_live_threshold(self) -> None:
        # A short line with no hint inside it yet: only Lan's caption, just before AND just after.
        # Nobody else was named near it, so the gateway (>= 0.6) must show "Lan", not the fallback.
        hints = [_h("Lan", 9_400), _h("Lan", 12_500)]
        got = attribute_far_speaker(W, hints, lag_ms=0, max_gap_ms=1_500)
        assert got is not None and got.name == "Lan"
        assert 0.6 <= got.confidence <= 0.85

    def test_b3_hand_over_with_only_the_previous_speaker_before_stays_low(self) -> None:
        # Lan stopped, Minh's first line has no hint of its own yet: only Lan's hints BEFORE it.
        # That must not be a confident "Lan".
        hints = [_h("Lan", 9_000), _h("Lan", 9_600)]
        got = attribute_far_speaker(W, hints, lag_ms=0, max_gap_ms=1_500)
        assert got is not None and got.name == "Lan" and got.confidence < 0.5

    def test_b3_two_names_near_the_window_stay_low(self) -> None:
        hints = [_h("Lan", 9_500), _h("Minh", 12_400)]
        got = attribute_far_speaker(W, hints, lag_ms=0, max_gap_ms=1_500)
        assert got is not None and got.confidence < 0.5

    def test_b3_default_lag_matches_meet_caption_delay(self) -> None:
        # Meet's captions trail the words by ~0.5-1.5 s; a caption seen 1.4 s after the window
        # ended still lands inside it with the default lag.
        lag = STTSettings().far_speaker_hint_lag_ms
        assert lag == 1_000
        got = attribute_far_speaker(W, [_h("Lan", 12_900)], lag_ms=lag, max_gap_ms=0)
        assert got is not None and got.name == "Lan" and got.confidence == 1.0

    def test_source_is_carried_from_the_winning_hint(self) -> None:
        got = attribute_far_speaker(W, [_h("Lan", 11_000, source="diarizer")], lag_ms=0)
        assert got is not None and got.source == "diarizer"


class TestParseHint:
    def test_bytes_fields(self) -> None:
        hint = parse_hint({b"name": b" Lan  Pham ", b"t_ms": b"123", b"source": b"meet_caption"})
        assert hint == FarSpeakerHint("Lan Pham", 123, "meet_caption")

    def test_defaults_source(self) -> None:
        assert parse_hint({"name": "Lan", "t_ms": "5"}) == FarSpeakerHint("Lan", 5, "meet_caption")

    def test_rejects_unusable_entries(self) -> None:
        assert parse_hint({"name": "", "t_ms": "5"}) is None
        assert parse_hint({"name": "Lan", "t_ms": "soon"}) is None
        assert parse_hint({"name": "Lan"}) is None
        assert parse_hint({"name": "Lan", "t_ms": "0"}) is None

    def test_key(self) -> None:
        assert far_speaker_hints_key("r1") == "meeting:r1:far_speaker_hints"


class TestCaptionHintTracker:
    async def test_reads_the_room_key_and_attributes(self) -> None:
        seen: list[str] = []

        async def read(key: str, count: int) -> list[Any]:
            seen.append(key)
            return [(b"1-0", {b"name": b"Lan", b"t_ms": b"11000"}), (b"0-1", {})]

        tracker: FarSpeakerTracker = CaptionHintTracker(read, lag_ms=0, max_gap_ms=0)
        got = await tracker.attribute("r1", W)
        assert got is not None and got.name == "Lan"
        assert seen == ["meeting:r1:far_speaker_hints"]

    async def test_fails_open(self) -> None:
        async def read(key: str, count: int) -> list[Any]:
            raise RuntimeError("down")

        tracker = CaptionHintTracker(read, lag_ms=0, max_gap_ms=0)
        assert await tracker.attribute("r1", W) is None

    async def test_caches_briefly(self) -> None:
        calls = 0

        async def read(key: str, count: int) -> list[Any]:
            nonlocal calls
            calls += 1
            return []

        now = [100.0]
        tracker = CaptionHintTracker(read, lag_ms=0, max_gap_ms=0, clock=lambda: now[0])
        await tracker.attribute("r1", W)
        await tracker.attribute("r1", W)
        assert calls == 1
        now[0] += 1.0
        await tracker.attribute("r1", W)
        assert calls == 2


class TestSchemaBackwardCompatibility:
    def test_absent_fields_are_absent_on_the_wire(self) -> None:
        msg = STTResultMessage(meeting_id="m", speaker_id="s", text="hi", language="en")
        wire = msg.to_redis()
        assert not any(k.startswith("far_speaker") for k in wire)

    def test_old_payload_parses_to_none(self) -> None:
        old = {
            "meeting_id": "m",
            "speaker_id": "s",
            "text": "hi",
            "language": "en",
        }
        msg = STTResultMessage.from_redis(old)
        assert msg.far_speaker_name is None
        assert msg.far_speaker_source is None
        assert msg.far_speaker_confidence is None

    def test_round_trip(self) -> None:
        msg = STTResultMessage(
            meeting_id="m",
            speaker_id=EXTERNAL_BRIDGE_SPEAKER_ID,
            text="hi",
            language="en",
            far_speaker_name="Lan Pham",
            far_speaker_source="meet_caption",
            far_speaker_confidence=0.667,
        )
        wire = msg.to_redis()
        assert wire["far_speaker_name"] == "Lan Pham"
        assert wire["far_speaker_source"] == "meet_caption"
        assert wire["far_speaker_confidence"] == "0.667"
        back = STTResultMessage.from_redis({k.encode(): v.encode() for k, v in wire.items()})
        assert (back.far_speaker_name, back.far_speaker_source, back.far_speaker_confidence) == (
            "Lan Pham",
            "meet_caption",
            0.667,
        )

    def test_source_without_name_is_not_sent(self) -> None:
        msg = STTResultMessage(
            meeting_id="m", speaker_id="s", text="hi", language="en", far_speaker_source="x"
        )
        assert "far_speaker_source" not in msg.to_redis()

    def test_audio_chunk_suppressed_overlap_is_optional(self) -> None:
        chunk = AudioChunkMessage(meeting_id="m", speaker_id="s", chunk_index=0, audio_data=b"")
        assert "suppressed_overlap_ms" not in chunk.to_redis()
        assert AudioChunkMessage.from_redis(chunk.to_redis()).suppressed_overlap_ms == 0
        marked = chunk.model_copy(update={"suppressed_overlap_ms": 96})
        assert marked.to_redis()["suppressed_overlap_ms"] == "96"
        assert AudioChunkMessage.from_redis(marked.to_redis()).suppressed_overlap_ms == 96


class TestWorkerAttachesHints:
    def _worker(self, mock_redis_client: Any, hints: list[Any], **stt: Any) -> STTWorker:
        worker = STTWorker.__new__(STTWorker)
        worker.settings = WorkerSettings()
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings(**stt)
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}

        async def xrevrange(key: str, count: int = 0) -> list[Any]:
            return hints if key == "meeting:room-1:far_speaker_hints" else []

        mock_redis_client._redis.xrevrange = AsyncMock(side_effect=xrevrange)
        published: list[STTResultMessage] = []

        async def capture(result: STTResultMessage, _mid: bytes | None = None) -> STTResultMessage:
            published.append(result)
            return result

        worker._publish_stt_result = capture  # type: ignore[method-assign]
        worker.published = published  # type: ignore[attr-defined]
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello from the far side.",
                    language="en",
                    confidence=-0.2,
                    start_ms=0,
                    end_ms=1000,
                )
            ]
        )
        return worker

    async def test_stand_in_segment_gets_the_caption_name(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        now_ms = int(time.time() * 1000)
        hints = [(b"1-0", {b"name": b"Lan Pham", b"t_ms": str(now_ms).encode()})]
        worker = self._worker(mock_redis_client, hints, far_speaker_hint_max_gap_ms=10_000)
        chunk = AudioChunkMessage(
            meeting_id="room-1",
            speaker_id=EXTERNAL_BRIDGE_SPEAKER_ID,
            chunk_index=0,
            audio_data=sample_audio_bytes,
            language="en",
            timestamp_ms=now_ms,
        )
        await worker.process(b"msg-1", chunk.to_redis())
        [result] = worker.published  # type: ignore[attr-defined]
        assert result.far_speaker_name == "Lan Pham"
        assert result.far_speaker_source == "meet_caption"
        assert result.far_speaker_confidence is not None and result.far_speaker_confidence > 0

    async def test_named_speaker_segment_is_never_labelled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        now_ms = int(time.time() * 1000)
        hints = [(b"1-0", {b"name": b"Lan Pham", b"t_ms": str(now_ms).encode()})]
        worker = self._worker(mock_redis_client, hints)
        chunk = AudioChunkMessage(
            meeting_id="room-1",
            speaker_id="alice",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            timestamp_ms=now_ms,
        )
        await worker.process(b"msg-1", chunk.to_redis())
        [result] = worker.published  # type: ignore[attr-defined]
        assert result.far_speaker_name is None

    async def test_hints_disabled(self, mock_redis_client: Any, sample_audio_bytes: bytes) -> None:
        now_ms = int(time.time() * 1000)
        hints = [(b"1-0", {b"name": b"Lan Pham", b"t_ms": str(now_ms).encode()})]
        worker = self._worker(mock_redis_client, hints, far_speaker_hints_enabled=False)
        chunk = AudioChunkMessage(
            meeting_id="room-1",
            speaker_id=EXTERNAL_BRIDGE_SPEAKER_ID,
            chunk_index=0,
            audio_data=sample_audio_bytes,
            timestamp_ms=now_ms,
        )
        await worker.process(b"msg-1", chunk.to_redis())
        [result] = worker.published  # type: ignore[attr-defined]
        assert result.far_speaker_name is None
