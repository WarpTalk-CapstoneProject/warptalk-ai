"""Late far-side speaker names: the hand-over line gets its name about a second after it went out.

PO 2026-10-03. When Lan stops and Minh starts on the Google Meet side, Minh's first line is
finalized before any caption naming Minh arrives, so it is published as "Google Meet participants"
(only Lan's hints are near, all before it: the hand-over guess, <= 0.5). The line is NOT held back;
stt_worker asks again at ~+1 s and ~+2.5 s and, on the first confident answer, publishes one
`far_speaker_late` entry on `stt:far_speaker_late:{room}` keyed by the line's segment_id.

What must never happen: a late answer replacing a name that was shown, or LAN's name reaching
Minh's line through the hand-over guess - whatever threshold is configured.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.far_speaker import (
    BASIS_INSIDE,
    BASIS_NEAREST,
    BASIS_SOLE_NEAR,
    FarSpeakerAttribution,
    FarSpeakerHint,
    SegmentWindow,
    attribute_far_speaker,
)
from shared.far_speaker_late import (
    FAR_SPEAKER_LATE_STREAM,
    MAX_LATE_ATTEMPTS,
    LateFarSpeakerName,
    LateFarSpeakerNamer,
    is_late_answer,
    late_attempt_delays_ms,
    name_hash,
    needs_late_attribution,
    was_shown,
)
from shared.schemas import AudioChunkMessage
from stt_worker.model import TranscribedSegment
from stt_worker.worker import STTWorker

W = SegmentWindow(start_ms=10_000, end_ms=12_000)
MIN = 0.6


def _h(name: str, t_ms: int) -> FarSpeakerHint:
    return FarSpeakerHint(name=name, t_ms=t_ms)


def _a(name: str, confidence: float, basis: str = BASIS_INSIDE) -> FarSpeakerAttribution:
    return FarSpeakerAttribution(
        name=name, source="meet_caption", confidence=confidence, basis=basis
    )


class TestAttributionBasis:
    """attribute_far_speaker says WHICH rule answered, so the late path can refuse the guess."""

    def test_inside(self) -> None:
        got = attribute_far_speaker(W, [_h("Lan", 11_000)], lag_ms=0)
        assert got is not None and got.basis == BASIS_INSIDE

    def test_sole_near_with_a_hint_after(self) -> None:
        got = attribute_far_speaker(W, [_h("Lan", 12_200)], lag_ms=0, max_gap_ms=1_500)
        assert got is not None and got.basis == BASIS_SOLE_NEAR

    def test_hand_over_guess_is_nearest(self) -> None:
        # Lan's hints, all BEFORE Minh's first line: the hand-over case.
        got = attribute_far_speaker(
            W, [_h("Lan", 9_000), _h("Lan", 9_700)], lag_ms=0, max_gap_ms=1_500
        )
        assert got is not None and got.basis == BASIS_NEAREST
        assert got.confidence <= 0.5


class TestDecisions:
    def test_unnamed_or_low_lines_need_a_late_name(self) -> None:
        assert needs_late_attribution(None, MIN)
        assert needs_late_attribution(_a("Lan", 0.5, BASIS_NEAREST), MIN)
        assert needs_late_attribution(_a("Lan", 0.59), MIN)
        assert needs_late_attribution(_a("   ", 1.0), MIN)

    def test_a_shown_name_is_never_renamed(self) -> None:
        assert was_shown(_a("Lan", 0.6), MIN)
        assert not needs_late_attribution(_a("Lan", 0.6), MIN)
        assert not needs_late_attribution(_a("Lan", 1.0, BASIS_NEAREST), MIN)

    def test_inside_and_sole_near_answers_are_accepted_at_the_threshold(self) -> None:
        assert is_late_answer(_a("Minh", 0.667, BASIS_INSIDE), MIN)
        assert is_late_answer(_a("Minh", 0.6, BASIS_SOLE_NEAR), MIN)

    def test_below_the_threshold_is_refused(self) -> None:
        assert not is_late_answer(None, MIN)
        assert not is_late_answer(_a("Minh", 0.5, BASIS_INSIDE), MIN)
        assert not is_late_answer(_a("Minh", 0.55, BASIS_SOLE_NEAR), MIN)

    def test_the_hand_over_guess_is_refused_whatever_its_score(self) -> None:
        # A threshold configured below the guess's 0.5 cap must not let it through late.
        assert not is_late_answer(_a("Lan", 0.5, BASIS_NEAREST), 0.3)
        assert not is_late_answer(_a("Lan", 0.9, BASIS_NEAREST), MIN)

    def test_an_unknown_basis_is_refused(self) -> None:
        # A tracker that does not say how it decided (WT-677, until it does) cannot rename late.
        assert not is_late_answer(_a("Minh", 1.0, ""), MIN)

    def test_schedule_is_bounded_and_sorted(self) -> None:
        assert late_attempt_delays_ms((2500, 1000)) == (1000, 2500)
        assert late_attempt_delays_ms((1000, 1000, 0, -5, 60_000)) == (1000,)
        assert late_attempt_delays_ms(()) == ()
        assert len(late_attempt_delays_ms(range(1, 100))) == MAX_LATE_ATTEMPTS


def test_wire_shape_is_the_fixed_contract() -> None:
    msg = LateFarSpeakerName(
        meeting_id="room-1",
        segment_id="seg-1",
        name="  Minh   Tran ",
        source="meet_caption",
        confidence=0.82,
        t_ms=1_759_300_000_123,
    )
    assert msg.to_redis() == {
        "type": "far_speaker_late",
        "meeting_id": "room-1",
        "segment_id": "seg-1",
        "far_speaker_name": "Minh Tran",
        "far_speaker_source": "meet_caption",
        "far_speaker_confidence": "0.82",
        "t_ms": "1759300000123",
    }
    assert FAR_SPEAKER_LATE_STREAM == "stt:far_speaker_late"


class HintsTracker:
    """The real rule over a hint list the test grows between attempts."""

    def __init__(self, hints: list[FarSpeakerHint], lag_ms: int = 0) -> None:
        self.hints = hints
        self.lag_ms = lag_ms
        self.calls = 0

    async def attribute(
        self, meeting_id: str, window: SegmentWindow
    ) -> FarSpeakerAttribution | None:
        self.calls += 1
        return attribute_far_speaker(window, list(self.hints), self.lag_ms, 1_500)


class _Harness:
    def __init__(
        self,
        tracker: Any,
        *,
        arrivals: dict[int, list[FarSpeakerHint]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.published: list[LateFarSpeakerName] = []
        self.slept: list[float] = []
        self.logger = MagicMock()
        self._arrivals = arrivals or {}

        async def publish(message: LateFarSpeakerName) -> None:
            self.published.append(message)

        async def sleep(seconds: float) -> None:
            self.slept.append(seconds)
            # Hints that "arrive" during this sleep (keyed by attempt number, 1-based).
            for hint in self._arrivals.get(len(self.slept), []):
                tracker.hints.append(hint)

        kwargs.setdefault("min_confidence", MIN)
        kwargs.setdefault("delays_ms", (1000, 2500))
        self.namer = LateFarSpeakerNamer(
            tracker, publish, self.logger, sleep=sleep, clock_ms=lambda: 42, **kwargs
        )

    async def drain(self) -> None:
        await asyncio.gather(*self.namer.pending_tasks(), return_exceptions=True)

    def events(self) -> list[str]:
        return [c.args[0] for c in self.logger.info.call_args_list]


class TestLateNamer:
    async def test_hand_over_line_gets_the_new_speakers_name_late(self) -> None:
        # Minh's first line at 10_000-12_000. At finalization only Lan's hints, all before it.
        tracker = HintsTracker([_h("Lan", 9_000), _h("Lan", 9_700)])
        original = await tracker.attribute("room-1", W)
        assert original is not None and original.name == "Lan"
        assert needs_late_attribution(original, MIN)  # published as the fallback label

        # Minh's captions arrive during the second wait.
        h = _Harness(tracker, arrivals={2: [_h("Minh", 11_000), _h("Minh", 11_800)]})
        assert h.namer.schedule("room-1", "seg-minh-1", W)
        await h.drain()

        [late] = h.published
        assert (late.meeting_id, late.segment_id, late.name) == ("room-1", "seg-minh-1", "Minh")
        assert late.confidence >= MIN
        assert late.t_ms == 42
        assert h.slept == [1.0, 1.5]  # +1 s, then +2.5 s after finalization
        assert "far_speaker_late_attributed" in h.events()

    async def test_stops_at_the_first_confident_answer(self) -> None:
        tracker = HintsTracker([_h("Lan", 9_000)])
        h = _Harness(tracker, arrivals={1: [_h("Minh", 11_000)]})
        h.namer.schedule("room-1", "seg-1", W)
        await h.drain()
        assert [m.name for m in h.published] == ["Minh"]
        assert tracker.calls == 1
        assert h.slept == [1.0]

    async def test_gives_up_without_ever_sending_the_hand_over_guess(self) -> None:
        # Nothing new arrives: Lan's hints stay the only ones, before the window. Even with the
        # threshold configured below the guess's 0.5 cap, LAN must not reach Minh's line.
        tracker = HintsTracker([_h("Lan", 9_000), _h("Lan", 9_900)])
        h = _Harness(tracker, min_confidence=0.3)
        h.namer.schedule("room-1", "seg-1", W)
        await h.drain()
        assert h.published == []
        assert tracker.calls == 2
        assert h.events()[-1] == "far_speaker_late_gave_up"
        assert not h.namer.pending_tasks()

    async def test_logs_never_carry_the_raw_name(self) -> None:
        tracker = HintsTracker([_h("Minh Tran", 11_000)])
        h = _Harness(tracker)
        h.namer.schedule("room-1", "seg-1", W)
        await h.drain()
        logged = repr(h.logger.mock_calls)
        assert "Minh" not in logged
        assert name_hash("Minh Tran") in logged

    async def test_at_most_one_task_per_segment_and_bounded_overall(self) -> None:
        tracker = HintsTracker([])
        h = _Harness(tracker, max_pending=1)
        assert h.namer.schedule("room-1", "seg-1", W)
        assert not h.namer.schedule("room-1", "seg-1", W)  # same line again
        assert not h.namer.schedule("room-1", "seg-2", W)  # over capacity
        assert "far_speaker_late_skipped" in h.events()
        await h.drain()
        assert not h.namer.pending_tasks()
        assert h.namer.schedule("room-1", "seg-2", W)  # room again once the first finished
        await h.drain()

    async def test_off_when_no_schedule_or_no_capacity(self) -> None:
        assert not _Harness(HintsTracker([]), delays_ms=()).namer.schedule("r", "s", W)
        assert not _Harness(HintsTracker([]), max_pending=0).namer.schedule("r", "s", W)

    async def test_meeting_end_cancels_only_that_meetings_lines(self) -> None:
        gate = asyncio.Event()

        async def publish(message: LateFarSpeakerName) -> None:
            published.append(message)

        async def sleep(_s: float) -> None:
            await gate.wait()

        published: list[LateFarSpeakerName] = []
        tracker = HintsTracker([_h("Minh", 11_000)])
        namer = LateFarSpeakerNamer(
            tracker, publish, MagicMock(), min_confidence=MIN, delays_ms=(1000,), sleep=sleep
        )
        namer.schedule("room-1", "seg-1", W)
        namer.schedule("room-2", "seg-2", W)
        await asyncio.sleep(0)
        assert namer.cancel_meeting("room-1") == 1
        gate.set()
        await asyncio.gather(*namer.pending_tasks(), return_exceptions=True)
        assert [m.segment_id for m in published] == ["seg-2"]
        assert namer.cancel_all() == 0

    async def test_a_failing_tracker_or_publish_is_contained(self) -> None:
        class Flaky:
            calls = 0

            async def attribute(self, meeting_id: str, window: SegmentWindow) -> Any:
                Flaky.calls += 1
                if Flaky.calls == 1:
                    raise RuntimeError("redis down")
                return _a("Minh", 1.0)

        logger = MagicMock()

        async def publish(_m: LateFarSpeakerName) -> None:
            raise RuntimeError("xadd failed")

        async def sleep(_s: float) -> None:
            return None

        namer = LateFarSpeakerNamer(
            Flaky(), publish, logger, min_confidence=MIN, delays_ms=(1000, 2500), sleep=sleep
        )
        namer.schedule("room-1", "seg-1", W)
        await asyncio.gather(*namer.pending_tasks())
        warnings = [c.args[0] for c in logger.warning.call_args_list]
        assert warnings == [
            "far_speaker_late_attribution_failed",
            "far_speaker_late_publish_failed",
        ]


class ScriptedTracker:
    """Answers in order and records the windows it was asked about."""

    def __init__(self, answers: list[FarSpeakerAttribution | None]) -> None:
        self.answers = answers
        self.windows: list[SegmentWindow] = []

    async def attribute(
        self, meeting_id: str, window: SegmentWindow
    ) -> FarSpeakerAttribution | None:
        self.windows.append(window)
        return self.answers.pop(0) if self.answers else None


class TestWorkerSchedulesLateNames:
    def _worker(self, mock_redis_client: Any, tracker: ScriptedTracker, **stt: Any) -> STTWorker:
        worker = STTWorker.__new__(STTWorker)
        worker.settings = WorkerSettings()
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        stt.setdefault("far_speaker_late_delays_ms", (1, 2))
        worker.stt_settings = STTSettings(**stt)
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}
        worker._far_speaker_tracker_impl = tracker
        mock_redis_client._redis.xrevrange = AsyncMock(return_value=[])
        order: list[str] = []
        published: list[Any] = []

        async def capture(result: Any, _mid: bytes | None = None) -> Any:
            order.append("line")
            published.append(result)
            return result

        async def publish(prefix: str, meeting_id: str, data: dict[str, str]) -> str:
            order.append(prefix)
            late.append((prefix, meeting_id, data))
            return "1-0"

        late: list[tuple[str, str, dict[str, str]]] = []
        worker._publish_stt_result = capture  # type: ignore[method-assign]
        worker.publish = publish  # type: ignore[method-assign]
        worker.published = published  # type: ignore[attr-defined]
        worker.late = late  # type: ignore[attr-defined]
        worker.order = order  # type: ignore[attr-defined]
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

    def _chunk(self, audio: bytes, speaker_id: str = EXTERNAL_BRIDGE_SPEAKER_ID) -> dict[str, Any]:
        return AudioChunkMessage(
            meeting_id="room-1",
            speaker_id=speaker_id,
            chunk_index=0,
            audio_data=audio,
            language="en",
            timestamp_ms=int(time.time() * 1000),
        ).to_redis()

    async def _drain(self, worker: STTWorker) -> None:
        namer = getattr(worker, "_late_far_speaker_namer_impl", None)
        if namer is not None:
            await asyncio.gather(*namer.pending_tasks())

    async def test_unnamed_line_is_published_first_then_named_late(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        tracker = ScriptedTracker([_a("Lan", 0.4, BASIS_NEAREST), None, _a("Minh", 1.0)])
        worker = self._worker(mock_redis_client, tracker)
        await worker.process(b"msg-1", self._chunk(sample_audio_bytes))
        [line] = worker.published  # type: ignore[attr-defined]
        # Published as before: the low guess rides along, and the gateway shows the fallback.
        assert line.far_speaker_name == "Lan" and line.far_speaker_confidence == 0.4

        await self._drain(worker)
        [(prefix, meeting_id, data)] = worker.late  # type: ignore[attr-defined]
        assert (prefix, meeting_id) == ("stt:far_speaker_late", "room-1")
        assert data["segment_id"] == line.segment_id
        assert data["far_speaker_name"] == "Minh"
        assert data["type"] == "far_speaker_late"
        assert worker.order == ["line", "stt:far_speaker_late"]  # type: ignore[attr-defined]
        # The late attempts asked about exactly the window the line was attributed on.
        assert len(set(tracker.windows)) == 1 and len(tracker.windows) == 3

    async def test_a_shown_name_is_never_scheduled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        tracker = ScriptedTracker([_a("Lan", 0.9), _a("Minh", 1.0)])
        worker = self._worker(mock_redis_client, tracker)
        await worker.process(b"msg-1", self._chunk(sample_audio_bytes))
        await self._drain(worker)
        assert worker.late == []  # type: ignore[attr-defined]
        assert len(tracker.windows) == 1

    async def test_a_named_participant_is_never_scheduled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        tracker = ScriptedTracker([_a("Minh", 1.0)])
        worker = self._worker(mock_redis_client, tracker)
        await worker.process(b"msg-1", self._chunk(sample_audio_bytes, speaker_id="alice"))
        await self._drain(worker)
        assert worker.late == [] and tracker.windows == []  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "stt", [{"far_speaker_late_delays_ms": ()}, {"far_speaker_hints_enabled": False}]
    )
    async def test_off_switches(
        self, mock_redis_client: Any, sample_audio_bytes: bytes, stt: dict[str, Any]
    ) -> None:
        tracker = ScriptedTracker([None, _a("Minh", 1.0)])
        worker = self._worker(mock_redis_client, tracker, **stt)
        await worker.process(b"msg-1", self._chunk(sample_audio_bytes))
        await self._drain(worker)
        assert worker.late == []  # type: ignore[attr-defined]
        assert len(worker.published) == 1  # type: ignore[attr-defined]

    async def test_room_end_cancels_the_pending_late_name(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        tracker = ScriptedTracker([None, _a("Minh", 1.0)])
        worker = self._worker(mock_redis_client, tracker, far_speaker_late_delays_ms=(5000,))
        await worker.process(b"msg-1", self._chunk(sample_audio_bytes))
        namer = worker._late_far_speaker_namer_impl
        tasks = namer.pending_tasks()
        assert len(tasks) == 1
        # The base-class state _cleanup_room also clears; __new__ skipped BaseWorker.__init__.
        worker._route_states = {}
        worker._translation_active = {}
        worker._room_routes = {}
        worker._speaker_locks = {}
        worker._cleanup_room("room-1")
        await asyncio.gather(*tasks, return_exceptions=True)
        assert tasks[0].cancelled()
        assert worker.late == []  # type: ignore[attr-defined]
