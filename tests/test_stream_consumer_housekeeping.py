"""Consumer groups keep only live consumers, and nothing stays pending on a dead one.

Every pod name is a new consumer, so each rollout left one more corpse in every group
(production 2026-09-30: 62 on stt-frame-workers, 59 on translate-workers, one live pod each).
The side groups (STT frames, TTS clone/preview/delete/clone-sampling) had no reclaim at all, so
an entry left by a pod killed mid-handler stayed pending until the stream expired.
And the live summary had no single owner, so two replicas each acted on a duplicated
end-of-meeting marker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from redis.asyncio import ResponseError

from ai_assistant_worker.worker import AIAssistantWorker
from shared.base_worker import BaseWorker
from shared.redis_client import RedisStreamClient
from shared.schemas import STTResultMessage

HOUR_MS = 60 * 60 * 1000


def _client(consumers: list[dict[str, Any]], held: int = 0) -> RedisStreamClient:
    client = RedisStreamClient.__new__(RedisStreamClient)
    client._settings = MagicMock(retry_max_attempts=1, retry_base_delay=0.0)
    client._pool = None
    client._redis = AsyncMock()
    client._redis.xinfo_consumers = AsyncMock(return_value=consumers)
    client._redis.xgroup_delconsumer = AsyncMock(return_value=held)
    return client


class TestPruneIdleConsumers:
    async def test_only_silent_consumers_with_nothing_pending_are_deleted(self) -> None:
        client = _client(
            [
                {"name": b"translation-pod-dead", "pending": 0, "idle": 5 * HOUR_MS},
                {"name": b"translation-pod-holding", "pending": 3, "idle": 5 * HOUR_MS},
                {"name": b"translation-pod-live", "pending": 0, "idle": 1_500},
                {"name": b"translation-pod-self", "pending": 0, "idle": 5 * HOUR_MS},
            ]
        )

        deleted = await client.prune_idle_consumers(
            "stt:results", "translate-workers", keep="translation-pod-self", min_idle_ms=HOUR_MS
        )

        assert deleted == ["translation-pod-dead"]
        client._redis.xgroup_delconsumer.assert_awaited_once_with(
            "stt:results", "translate-workers", "translation-pod-dead"
        )

    async def test_a_missing_group_is_nothing_to_prune(self) -> None:
        client = _client([])
        client._redis.xinfo_consumers = AsyncMock(side_effect=ResponseError("NOGROUP no group"))

        assert await client.prune_idle_consumers("s", "g", keep="me", min_idle_ms=HOUR_MS) == []


class _Worker(BaseWorker):
    worker_name = "probe"
    input_stream = "input"
    consumer_group = "probe-workers"

    async def load_model(self) -> None:
        return None

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        return None


def _worker() -> _Worker:
    worker = _Worker.__new__(_Worker)
    worker.logger = MagicMock()
    worker._consumer_name = "probe-host-1"
    worker.redis = MagicMock()
    worker.redis.redis = MagicMock()
    worker.redis.redis.xack = AsyncMock()
    worker.redis.reclaim_stale = AsyncMock(return_value=[(b"7-0", {b"k": b"v"})])
    worker.redis.prune_idle_consumers = AsyncMock(return_value=["probe-host-0"])
    return worker


class TestSideGroupHousekeeping:
    async def test_stale_live_audio_is_acknowledged_not_replayed(self) -> None:
        worker = _worker()

        await worker._housekeep_side_group("audio:frames", "stt-frame-workers")

        worker.redis.redis.xack.assert_awaited_once_with(
            "audio:frames", "stt-frame-workers", b"7-0"
        )
        assert worker.redis.reclaim_stale.await_args.kwargs["min_idle_ms"] > 120_000
        worker.redis.prune_idle_consumers.assert_awaited_once()

    async def test_a_request_gets_one_more_attempt_and_is_then_acknowledged(self) -> None:
        worker = _worker()
        redeliver = AsyncMock(side_effect=RuntimeError("still failing"))

        await worker._housekeep_side_group("voice:delete_requests", "g", redeliver=redeliver)

        redeliver.assert_awaited_once_with({b"k": b"v"})
        worker.redis.redis.xack.assert_awaited_once()

    async def test_it_runs_at_most_once_per_interval(self) -> None:
        worker = _worker()

        await worker._housekeep_side_group("audio:frames", "stt-frame-workers")
        await worker._housekeep_side_group("audio:frames", "stt-frame-workers")

        assert worker.redis.reclaim_stale.await_count == 1

    async def test_a_redis_failure_never_reaches_the_loop(self) -> None:
        worker = _worker()
        worker.redis.reclaim_stale = AsyncMock(side_effect=ConnectionError("down"))
        worker.redis.prune_idle_consumers = AsyncMock(side_effect=ConnectionError("down"))

        await worker._housekeep_side_group("audio:frames", "stt-frame-workers")

    async def test_the_main_group_is_pruned_after_its_reclaim(self) -> None:
        worker = _worker()
        worker.redis.reclaim_stale = AsyncMock(return_value=[])

        await worker._recover_stale_messages()

        worker.redis.prune_idle_consumers.assert_awaited_once_with(
            "input", "probe-workers", keep="probe-host-1", min_idle_ms=HOUR_MS
        )


def _assistant(claimed: bool) -> AIAssistantWorker:
    worker = AIAssistantWorker.__new__(AIAssistantWorker)
    worker.logger = MagicMock()
    worker._consumer_name = "assistant-host-2"
    worker._transcripts = {"m1": [("s1", "hello", 1)]}
    worker._pause_gaps = {}
    worker._gap_open = set()
    worker._filler_only_ms = {}
    worker._paused_rooms = set()
    worker.redis = MagicMock()
    worker.redis.set_if_absent = AsyncMock(return_value=claimed)
    worker.redis.delete_if_value = AsyncMock(return_value=True)
    worker.redis.rpush_capped = AsyncMock()
    worker.is_transcript_paused = AsyncMock(return_value=False)  # type: ignore[method-assign]
    worker._generate_summary = AsyncMock()  # type: ignore[method-assign]
    return worker


_END = STTResultMessage(
    meeting_id="m1", speaker_id="system", text="__MEETING_END__", language="en", confidence=1.0
).to_redis()


class TestOneSummaryPerMeetingEnd:
    async def test_the_replica_that_claims_the_meeting_summarises_it(self) -> None:
        worker = _assistant(claimed=True)

        await worker.process(b"1-0", _END)

        worker._generate_summary.assert_awaited_once_with("m1")
        key, owner, _ttl = worker.redis.set_if_absent.await_args.args
        assert key == "assistant:summary:claim:m1"
        assert owner == "assistant-host-2"

    async def test_a_duplicate_marker_elsewhere_does_not_summarise_again(self) -> None:
        worker = _assistant(claimed=False)

        await worker.process(b"2-0", _END)

        worker._generate_summary.assert_not_awaited()
        assert "m1" not in worker._transcripts

    async def test_a_failed_summary_gives_the_claim_back_for_its_retry(self) -> None:
        worker = _assistant(claimed=True)
        worker._generate_summary = AsyncMock(side_effect=RuntimeError("model 500"))  # type: ignore[method-assign]

        try:
            await worker.process(b"1-0", _END)
        except RuntimeError:
            pass

        worker.redis.delete_if_value.assert_awaited_once_with(
            "assistant:summary:claim:m1", "assistant-host-2"
        )
