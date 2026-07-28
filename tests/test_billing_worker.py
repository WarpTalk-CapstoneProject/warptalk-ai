"""Tests for Billing Settlement Worker — segment-id extraction helper."""

from __future__ import annotations

import asyncio
import json
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from billing_worker import db as billing_db
from billing_worker.worker import BillingSettlementWorker, _extract_underlying_segment_id
from shared.health_probe import check_worker


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


async def test_subscription_resolution_uses_redis_room_projection_not_foreign_database() -> None:
    room_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    subscription_id = uuid.uuid4()
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(
        return_value=(f'{{"WorkspaceId":"{workspace_id}","Status":"IN_PROGRESS"}}').encode()
    )
    worker.db = MagicMock()
    worker.db.resolve_subscription = AsyncMock(
        return_value=(subscription_id, uuid.UUID(workspace_id))
    )

    resolved = await worker._resolve_subscription(room_id)

    assert resolved == (subscription_id, uuid.UUID(workspace_id))
    worker.db.resolve_subscription.assert_awaited_once_with(workspace_id)


async def test_subscription_resolution_fails_when_room_projection_is_missing() -> None:
    room_id = str(uuid.uuid4())
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(return_value=None)
    worker.db = MagicMock()
    worker.logger = MagicMock()

    with pytest.raises(RuntimeError, match="Room projection is unavailable"):
        await worker._resolve_subscription(room_id)


def test_credit_charge_rounds_rate_card_cost_up_to_whole_credit() -> None:
    calculate_credit_charge = getattr(billing_db, "calculate_credit_charge", None)
    assert callable(calculate_credit_charge)
    assert calculate_credit_charge(1.0, Decimal("0.25")) == 1
    assert calculate_credit_charge(61.0, Decimal("0.25")) == 16


async def test_settlement_error_propagates_so_message_remains_pending() -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    handler = AsyncMock(side_effect=RuntimeError("database unavailable"))
    process_message = getattr(worker, "_process_settlement_message", None)
    assert callable(process_message)

    try:
        await process_message(
            "stt:results",
            b"1-0",
            {b"text": b"hello"},
            handler,
        )
    except RuntimeError as error:
        assert str(error) == "database unavailable"
    else:
        raise AssertionError("settlement failure was swallowed")


async def test_billing_heartbeat_satisfies_shared_health_probe(monkeypatch) -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._consumer_name = "billing-host-1"
    worker.redis = MagicMock()
    worker.redis.set_with_ttl = AsyncMock()

    await worker._publish_heartbeat()

    _, payload, _ = worker.redis.set_with_ttl.await_args.args
    redis = AsyncMock()
    redis.mget = AsyncMock(return_value=[payload.encode()])
    client = MagicMock()
    client.redis = redis
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    monkeypatch.setenv("WORKER_HEALTH_NAME", "billing")

    with (
        patch("shared.health_probe.RedisStreamClient", return_value=client),
        patch("shared.health_probe.socket.gethostname", return_value="host-1"),
    ):
        assert await check_worker() is True

    assert isinstance(json.loads(payload)["last_progress_unix_ms"], int)


async def test_billing_heartbeat_loop_recovers_after_transient_redis_failure() -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    worker.heartbeat_interval_seconds = 0
    worker._shutdown_event = asyncio.Event()
    attempts = 0

    async def publish_heartbeat() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Redis restarted")
        worker._shutdown_event.set()

    worker._publish_heartbeat = AsyncMock(side_effect=publish_heartbeat)

    await asyncio.wait_for(worker._heartbeat_loop(), timeout=1)

    assert worker._publish_heartbeat.await_count == 2
    worker.logger.exception.assert_called_once_with("billing_heartbeat_failed")
