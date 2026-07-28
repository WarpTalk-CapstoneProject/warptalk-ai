from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.base_worker import BaseWorker


class FailingWorker(BaseWorker):
    worker_name = "failing"
    input_stream = "input"
    consumer_group = "failing-workers"

    async def load_model(self) -> None:
        return None

    async def process(
        self,
        message_id: bytes,
        data: dict[bytes, bytes],
    ) -> None:
        raise RuntimeError("provider unavailable")


async def test_processing_error_propagates_so_stream_message_is_not_acked() -> None:
    """BaseWorker must let Redis retain a failed message in the pending list."""
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await worker._process_and_log_errors(b"1-0", {b"text": b"hello"})


async def test_stale_pending_message_is_reprocessed_and_acked() -> None:
    """A message left pending by a dead worker must be recoverable."""
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()
    worker._consumer_name = "failing-host-2"
    worker.redis = MagicMock()
    worker.redis.reclaim_stale = AsyncMock(return_value=[(b"2-0", {b"text": b"retry me"})])
    worker.redis.redis = MagicMock()
    worker.redis.redis.xack = AsyncMock()
    worker._process_and_log_errors = AsyncMock()

    await worker._recover_stale_messages()

    worker._process_and_log_errors.assert_awaited_once_with(
        b"2-0",
        {b"text": b"retry me"},
    )
    worker.redis.redis.xack.assert_awaited_once_with(
        "input",
        "failing-workers",
        b"2-0",
    )


async def test_poison_message_moves_to_dlq_after_delivery_limit() -> None:
    """A repeatedly failing message must be inspectable instead of retrying forever."""
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()
    worker._consumer_name = "failing-host-2"
    worker.redis = MagicMock()
    worker.redis.reclaim_stale = AsyncMock(return_value=[(b"2-0", {b"text": b"retry me"})])
    worker.redis.pending_delivery_count = AsyncMock(return_value=5)
    worker.redis.publish = AsyncMock(return_value="3-0")
    worker.redis.redis = MagicMock()
    worker.redis.redis.xack = AsyncMock()

    await worker._recover_stale_messages()

    worker.redis.publish.assert_awaited_once()
    dlq_stream, payload = worker.redis.publish.await_args.args
    assert dlq_stream == "input:dead-letter"
    assert payload["original_message_id"] == "2-0"
    assert payload["delivery_attempts"] == 5
    worker.redis.redis.xack.assert_awaited_once_with(
        "input",
        "failing-workers",
        b"2-0",
    )


async def test_worker_heartbeat_is_written_with_ttl() -> None:
    """Operators must be able to distinguish an idle worker from a dead worker."""
    worker = FailingWorker.__new__(FailingWorker)
    worker._consumer_name = "failing-host-2"
    worker.redis = MagicMock()
    worker.redis.set_with_ttl = AsyncMock()

    await worker._publish_heartbeat()

    key, value, ttl = worker.redis.set_with_ttl.await_args.args
    assert key == "warptalk:worker:heartbeat:failing:host-2"
    payload = json.loads(value)
    assert payload["worker"] == "failing"
    assert isinstance(payload["last_progress_unix_ms"], int)
    assert ttl == 30


async def test_processing_timeout_propagates_so_hung_message_remains_pending() -> None:
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()
    worker.processing_timeout_seconds = 0.01

    async def hang_forever(*_: object) -> None:
        await asyncio.sleep(60)

    worker.process = AsyncMock(side_effect=hang_forever)

    with pytest.raises(asyncio.TimeoutError):
        await worker._process_and_log_errors(b"1-0", {b"text": b"hello"})


async def test_heartbeat_loop_recovers_after_transient_redis_failure() -> None:
    """One Redis restart must not permanently disable worker liveness evidence."""
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()
    worker.heartbeat_interval_seconds = 0
    worker._shutdown_event = asyncio.Event()

    attempts = 0

    async def publish_heartbeat() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Redis is loading")
        worker._shutdown_event.set()

    worker._publish_heartbeat = AsyncMock(side_effect=publish_heartbeat)

    await asyncio.wait_for(worker._heartbeat_loop(), timeout=1)

    assert worker._publish_heartbeat.await_count == 2
    worker.logger.exception.assert_called_once_with("worker_heartbeat_failed")


async def test_route_listener_resubscribes_after_transient_redis_failure() -> None:
    """Redis restart must not permanently detach route-control subscriptions."""
    worker = FailingWorker.__new__(FailingWorker)
    worker.logger = MagicMock()
    worker._shutdown_event = asyncio.Event()
    worker.redis = MagicMock()
    worker._pubsub = None

    failed_pubsub = MagicMock()
    failed_pubsub.psubscribe = AsyncMock(side_effect=ConnectionError("Redis restarted"))
    failed_pubsub.close = AsyncMock()

    recovered_pubsub = MagicMock()
    recovered_pubsub.psubscribe = AsyncMock()
    recovered_pubsub.close = AsyncMock()

    async def recovered_messages():
        worker._shutdown_event.set()
        if False:
            yield {}

    recovered_pubsub.listen = recovered_messages
    worker.redis.redis.pubsub.side_effect = [failed_pubsub, recovered_pubsub]

    await asyncio.wait_for(worker._listen_route_updates(), timeout=3)

    assert worker.redis.redis.pubsub.call_count == 2
    recovered_pubsub.psubscribe.assert_awaited_once_with("translationRoom:*:events")
    failed_pubsub.close.assert_awaited_once()
    recovered_pubsub.close.assert_awaited_once()
