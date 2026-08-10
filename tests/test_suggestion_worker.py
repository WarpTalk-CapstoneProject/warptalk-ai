"""Tests for SuggestionWorker — gating, budgets and publish shape.

Everything here runs without a model or a Redis server: the suggester is a recording
stub and Redis is a small in-memory double that reproduces the two semantics the worker
actually depends on (SET NX returning False for a key that exists, INCR counting up).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from shared.config import SuggestionSettings, WorkerSettings
from shared.schemas import STTResultMessage
from suggestion_worker.suggester import (
    GeneratedSuggestion,
    NullSuggester,
    SuggestionDecision,
    TranscriptTurn,
)
from suggestion_worker.worker import SuggestionWorker


class FakeRedis:
    """Only the handful of calls SuggestionWorker makes.

    `policy` stands in for the key the gateway projects into Redis
    (translationRoom:{id}:ai_policy). Most tests here are about gating and rate limiting,
    so consent defaults to granted; set it to None to simulate a room whose policy the
    gateway has not published yet.
    """

    def __init__(self, policy: bool | None = True) -> None:
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}
        self.published: list[tuple[str, dict[str, str]]] = []
        self.policy = policy

    async def get(self, key: str) -> str | None:
        if key.endswith(":ai_policy") and key not in self.values:
            if self.policy is None:
                return None
            return json.dumps({"allow_external_llm": self.policy})
        return self.values.get(key)

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def publish(self, stream: str, data: dict[str, str]) -> bytes:
        self.published.append((stream, data))
        return b"1-0"


class RecordingSuggester:
    """Returns a canned verdict and records how often each stage ran."""

    def __init__(
        self,
        decision: SuggestionDecision | None = None,
        suggestion: GeneratedSuggestion | None = None,
    ) -> None:
        self.decision = decision or SuggestionDecision.decline()
        self.suggestion = suggestion
        self.decide_calls: list[tuple[list[TranscriptTurn], TranscriptTurn]] = []
        self.generate_calls: int = 0
        self.loaded: bool = False

    async def load(self) -> None:
        self.loaded = True

    async def decide(
        self,
        window: Sequence[TranscriptTurn],
        segment: TranscriptTurn,
    ) -> SuggestionDecision:
        self.decide_calls.append((list(window), segment))
        return self.decision

    async def generate(
        self,
        window: Sequence[TranscriptTurn],
        segment: TranscriptTurn,
        decision: SuggestionDecision,
        context_snapshot: str = "",
    ) -> GeneratedSuggestion | None:
        self.generate_calls += 1
        return self.suggestion


def build_worker(
    suggester: Any = None,
    policy: bool | None = True,
    **overrides: Any,
) -> tuple[SuggestionWorker, FakeRedis, RecordingSuggester]:
    settings = SuggestionSettings(enabled=True, **overrides)
    recorder = suggester or RecordingSuggester()
    worker = SuggestionWorker(
        suggestion_settings=settings,
        suggester=recorder,
        settings=WorkerSettings(),
    )
    fake_redis = FakeRedis(policy=policy)
    worker.redis = fake_redis  # type: ignore[assignment]
    return worker, fake_redis, recorder


def stt_message(
    text: str = "chúng ta cần chốt deadline cho phần tích hợp này",
    *,
    # AVG LOGPROB (always <= 0), not a 0-1 score — this mirrors what stt_worker actually
    # publishes. Production segments span -0.699 to 0.000, so -0.1 stands in for clean
    # speech. Do not confuse this with SuggestionDecision.confidence below, which really
    # is 0-1: the two live on different scales and are gated by different settings.
    confidence: float = -0.1,
    is_final_chunk: bool = False,
    meeting_id: str = "room-1",
    speaker_id: str = "speaker-1",
) -> dict[bytes, bytes]:
    payload = STTResultMessage(
        segment_id="segment-1",
        meeting_id=meeting_id,
        speaker_id=speaker_id,
        text=text,
        language="vi",
        confidence=confidence,
        is_final_chunk=is_final_chunk,
    ).to_redis()
    return {key.encode(): value.encode() for key, value in payload.items()}


def approving_suggester() -> RecordingSuggester:
    return RecordingSuggester(
        decision=SuggestionDecision(
            should_suggest=True,
            category="action",
            confidence=0.9,
            token_count=40,
        ),
        suggestion=GeneratedSuggestion(
            content="Chưa có ai nhận phần tích hợp — cần chốt người phụ trách.",
            detail="Deadline được nhắc tới nhưng không có owner.",
            category="action",
            token_count=110,
        ),
    )


class TestStageZeroGating:
    """Rejections that must happen before a single token is spent."""

    @pytest.mark.asyncio
    async def test_disabled_worker_never_calls_the_model(self) -> None:
        worker, _, suggester = build_worker()
        worker.suggestion_settings = SuggestionSettings(enabled=False)

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == []

    @pytest.mark.asyncio
    async def test_empty_end_of_chunk_marker_is_ignored(self) -> None:
        """stt_worker's trailing marker carries text="" — it is not a segment."""
        worker, _, suggester = build_worker()

        await worker.process(b"1-0", stt_message(text="", is_final_chunk=True))

        assert suggester.decide_calls == []
        assert worker._windows.get("room-1") is None or not worker._windows["room-1"]

    @pytest.mark.asyncio
    async def test_real_segments_are_considered_despite_is_final_chunk_false(self) -> None:
        """Regression guard: on this stream, is_final_chunk=False is the NORMAL case for a
        segment that has content. Gating it out would disable the feature entirely."""
        worker, _, suggester = build_worker()

        await worker.process(b"1-0", stt_message(is_final_chunk=False))

        assert len(suggester.decide_calls) == 1

    @pytest.mark.asyncio
    async def test_short_utterance_is_dropped(self) -> None:
        worker, _, suggester = build_worker()

        await worker.process(b"1-0", stt_message(text="ừ đúng rồi"))

        assert suggester.decide_calls == []

    @pytest.mark.asyncio
    async def test_low_stt_confidence_is_dropped(self) -> None:
        worker, _, suggester = build_worker()

        # -0.6: poor but still above stt_worker's own -0.7 discard floor, so this is a
        # segment that really does reach this worker and really should be declined.
        await worker.process(b"1-0", stt_message(confidence=-0.6))

        assert suggester.decide_calls == []

    @pytest.mark.asyncio
    async def test_paused_room_is_dropped(self) -> None:
        worker, _, suggester = build_worker()
        worker._paused_rooms.add("room-1")

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == []

    @pytest.mark.asyncio
    async def test_ended_room_is_dropped(self) -> None:
        worker, _, suggester = build_worker()
        worker._route_states["room-1"] = "ENDED"

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == []


class TestContextWindow:
    @pytest.mark.asyncio
    async def test_short_segments_still_build_context(self) -> None:
        """Too short to suggest on, but it may be what makes the next segment meaningful."""
        worker, _, suggester = build_worker()

        await worker.process(b"1-0", stt_message(text="ừ đúng rồi"))
        await worker.process(b"2-0", stt_message())

        assert [turn.text for turn in worker._windows["room-1"]][0] == "ừ đúng rồi"
        window, subject = suggester.decide_calls[0]
        assert [turn.text for turn in window] == ["ừ đúng rồi"]
        assert subject.text.startswith("chúng ta cần chốt")

    @pytest.mark.asyncio
    async def test_window_is_bounded(self) -> None:
        worker, _, _ = build_worker(window_size=3)

        for index in range(6):
            await worker.process(b"1-0", stt_message(text=f"câu nói số {index} trong cuộc họp"))

        assert len(worker._windows["room-1"]) == 3

    @pytest.mark.asyncio
    async def test_cleanup_room_drops_the_window(self) -> None:
        worker, _, _ = build_worker()
        await worker.process(b"1-0", stt_message())
        assert "room-1" in worker._windows

        worker._cleanup_room("room-1")

        assert "room-1" not in worker._windows


class TestDecisionGate:
    @pytest.mark.asyncio
    async def test_declined_segment_publishes_nothing(self) -> None:
        worker, redis, suggester = build_worker()

        await worker.process(b"1-0", stt_message())

        assert suggester.generate_calls == 0
        assert redis.published == []

    @pytest.mark.asyncio
    async def test_declining_does_not_burn_the_cooldown(self) -> None:
        """A declined segment must leave the room free to suggest on the very next one."""
        worker, redis, _ = build_worker()

        await worker.process(b"1-0", stt_message())

        assert redis.values == {}

    @pytest.mark.asyncio
    async def test_low_confidence_decision_is_dropped(self) -> None:
        suggester = approving_suggester()
        suggester.decision = SuggestionDecision(
            should_suggest=True, category="action", confidence=0.5
        )
        worker, redis, _ = build_worker(suggester=suggester)

        await worker.process(b"1-0", stt_message())

        assert suggester.generate_calls == 0
        assert redis.published == []

    @pytest.mark.asyncio
    async def test_unknown_category_is_dropped(self) -> None:
        suggester = approving_suggester()
        suggester.decision = SuggestionDecision(
            should_suggest=True, category="banter", confidence=0.95
        )
        worker, redis, _ = build_worker(suggester=suggester)

        await worker.process(b"1-0", stt_message())

        assert suggester.generate_calls == 0
        assert redis.published == []

    @pytest.mark.asyncio
    async def test_empty_generation_publishes_nothing(self) -> None:
        suggester = approving_suggester()
        suggester.suggestion = GeneratedSuggestion(content="   ", category="action")
        worker, redis, _ = build_worker(suggester=suggester)

        await worker.process(b"1-0", stt_message())

        assert redis.published == []


class TestRateLimiting:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_the_next_segment(self) -> None:
        worker, redis, suggester = build_worker(suggester=approving_suggester())

        await worker.process(b"1-0", stt_message())
        assert suggester.generate_calls == 1

        await worker.process(b"2-0", stt_message())

        assert suggester.generate_calls == 1, "second segment should be inside the cooldown"
        assert len(suggester.decide_calls) == 1, "cooldown must be checked before deciding"

    @pytest.mark.asyncio
    async def test_slot_lost_to_another_replica_suppresses_the_suggestion(self) -> None:
        """The read-only probe can pass while a peer claims the slot a moment later."""
        worker, redis, suggester = build_worker(suggester=approving_suggester())
        original_probe = worker._cooldown_active

        async def probe_then_let_a_peer_win(room_id: str) -> bool:
            result = await original_probe(room_id)
            redis.values[SuggestionWorker._cooldown_key(room_id)] = "1"
            return result

        worker._cooldown_active = probe_then_let_a_peer_win  # type: ignore[method-assign]

        await worker.process(b"1-0", stt_message())

        assert len(suggester.decide_calls) == 1
        assert suggester.generate_calls == 0
        assert redis.published == []

    @pytest.mark.asyncio
    async def test_budget_exhaustion_stops_further_suggestions(self) -> None:
        suggester = approving_suggester()
        worker, redis, _ = build_worker(suggester=suggester, max_per_meeting=2)
        cooldown_key = SuggestionWorker._cooldown_key("room-1")

        for _ in range(4):
            redis.values.pop(cooldown_key, None)  # simulate the cooldown lapsing
            await worker.process(b"1-0", stt_message())

        assert suggester.generate_calls == 2
        assert len(redis.published) == 2 * 2  # room stream + global stream per publish

    @pytest.mark.asyncio
    async def test_budget_is_tracked_per_room(self) -> None:
        suggester = approving_suggester()
        worker, redis, _ = build_worker(suggester=suggester, max_per_meeting=1)

        await worker.process(b"1-0", stt_message(meeting_id="room-1"))
        await worker.process(b"2-0", stt_message(meeting_id="room-2"))

        assert suggester.generate_calls == 2


class TestPublish:
    @pytest.mark.asyncio
    async def test_publishes_to_both_streams_with_suggestion_type(self) -> None:
        worker, redis, _ = build_worker(suggester=approving_suggester())

        await worker.process(b"1-0", stt_message())

        streams = [stream for stream, _ in redis.published]
        assert streams == ["ai_assistant:results:room-1", "ai_assistant:results"]

        _, payload = redis.published[0]
        assert payload["type"] == "suggestion"
        assert payload["segment_id"] == "segment-1"
        assert payload["category"] == "action"
        assert payload["language"] == "vi"
        assert payload["detail"] == "Deadline được nhắc tới nhưng không có owner."
        assert payload["token_count"] == "150", "decide + generate tokens are billed together"

    @pytest.mark.asyncio
    async def test_content_is_truncated_to_the_strip_width(self) -> None:
        suggester = approving_suggester()
        suggester.suggestion = GeneratedSuggestion(content="a" * 400, category="term")
        worker, redis, _ = build_worker(suggester=suggester, max_suggestion_chars=40)

        await worker.process(b"1-0", stt_message())

        _, payload = redis.published[0]
        assert len(payload["content"]) == 40


class TestNullSuggester:
    @pytest.mark.asyncio
    async def test_default_worker_is_silent(self) -> None:
        """The shipped default declines everything, so an unconfigured deployment is inert."""
        worker = SuggestionWorker(
            suggestion_settings=SuggestionSettings(enabled=True),
            settings=WorkerSettings(),
        )
        redis = FakeRedis()
        worker.redis = redis  # type: ignore[assignment]

        assert isinstance(worker.suggester, NullSuggester)

        await worker.process(b"1-0", stt_message())

        assert redis.published == []


class TestWorkspaceConsent:
    """Transcript text must not reach an external LLM without the workspace's consent."""

    @pytest.mark.asyncio
    async def test_no_published_policy_means_no_suggestion(self) -> None:
        """Fail closed: an unpublished policy is 'unknown', not 'allowed'."""
        worker, _, suggester = build_worker(suggester=approving_suggester(), policy=None)

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == [], "no transcript may leave the process yet"

    @pytest.mark.asyncio
    async def test_workspace_opted_out_of_external_llm(self) -> None:
        worker, redis, suggester = build_worker(suggester=approving_suggester(), policy=False)

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == []
        assert redis.published == []

    @pytest.mark.asyncio
    async def test_consent_is_checked_before_the_cooldown_is_probed(self) -> None:
        """A room that never gets to suggest must not have its budget touched either."""
        worker, redis, _ = build_worker(suggester=approving_suggester(), policy=False)

        await worker.process(b"1-0", stt_message())

        assert redis.values == {}
        assert redis.counters == {}

    @pytest.mark.asyncio
    async def test_malformed_policy_payload_fails_closed(self) -> None:
        worker, redis, suggester = build_worker(suggester=approving_suggester())
        redis.values["translationRoom:room-1:ai_policy"] = "{not json"

        await worker.process(b"1-0", stt_message())

        assert suggester.decide_calls == []

    @pytest.mark.asyncio
    async def test_policy_is_cached_rather_than_read_per_segment(self) -> None:
        worker, redis, _ = build_worker()
        reads: list[str] = []
        original_get = redis.get

        async def counting_get(key: str) -> str | None:
            reads.append(key)
            return await original_get(key)

        redis.get = counting_get  # type: ignore[method-assign]

        for _ in range(3):
            await worker.process(b"1-0", stt_message())

        assert sum(1 for key in reads if key.endswith(":ai_policy")) == 1

    @pytest.mark.asyncio
    async def test_cleanup_room_drops_the_cached_policy(self) -> None:
        worker, _, _ = build_worker()
        await worker.process(b"1-0", stt_message())
        assert "room-1" in worker._policies

        worker._cleanup_room("room-1")

        assert "room-1" not in worker._policies
