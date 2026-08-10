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


class TestBillableSurface:
    """WT-344 — what a meeting pays for, pinned.

    Transcription and the inline assistant are FREE on the owner's call; translation and
    dubbing are what a workspace spends credits on. This is a product decision that lives
    in exactly one place — which streams the settlement worker subscribes to — so it is
    worth asserting directly rather than inferring from behaviour.

    These would both have passed while the STT and suggestion handlers still existed, so
    they assert the ABSENCE of the handler as well as the absence of the stream: a future
    change that re-adds a handler without re-subscribing (or vice versa) fails here.
    """

    @staticmethod
    def _subscribed_streams() -> set[str]:
        import inspect

        source = inspect.getsource(BillingSettlementWorker.start)
        return {
            stream
            for stream in (
                "stt:results",
                "translate:results",
                "tts:results",
                "ai_assistant:results",
            )
            if f'"{stream}"' in source
        }

    def test_only_translation_and_dubbing_are_billed(self) -> None:
        assert self._subscribed_streams() == {"translate:results", "tts:results"}

    def test_free_pipelines_have_no_settlement_handler_left_behind(self) -> None:
        # A handler with no subscription is dead code that reads as a live feature — the
        # exact shape of defect this codebase has hit repeatedly.
        assert not hasattr(BillingSettlementWorker, "_handle_stt")
        assert not hasattr(BillingSettlementWorker, "_handle_suggestion")

    def test_the_billable_handlers_are_still_wired(self) -> None:
        assert hasattr(BillingSettlementWorker, "_handle_translation")
        assert hasattr(BillingSettlementWorker, "_handle_tts")


class _FakeConnection:
    """Records what was executed and replays canned rows, so settlement can be tested
    without a database. Only fetchrow is used by record_usage_and_charge."""

    def __init__(self, rate_row, settle_row) -> None:
        self._rate_row = rate_row
        self._settle_row = settle_row
        self.queries: list[str] = []
        self.settle_args: tuple = ()

    async def fetchrow(self, query, *args):
        self.queries.append(query)
        if "usage_rate_card" in query:
            return self._rate_row
        if "settle_usage_charge" in query:
            self.settle_args = args
            return self._settle_row
        raise AssertionError(f"unexpected query: {query}")

    async def execute(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("settlement must not write its own statements")

    async def fetchval(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("settlement must not write its own statements")


class _FakePool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _repository(settle_row):
    repo = billing_db.BillingRepository.__new__(billing_db.BillingRepository)
    conn = _FakeConnection(
        rate_row={"id": uuid.uuid4(), "unit_price": Decimal("0.25"), "currency": "CRD"},
        settle_row=settle_row,
    )
    repo._pool = _FakePool(conn)
    return repo, conn


async def _settle(repo, **overrides):
    kwargs = dict(
        subscription_id=uuid.uuid4(),
        user_id=str(uuid.uuid4()),
        workspace_id=uuid.uuid4(),
        translation_room_id=str(uuid.uuid4()),
        usage_type="TRANSLATION",
        charge_type="TRANSLATION",
        reference_id=str(uuid.uuid4()),
        reference_type="translation_content",
        quantity=4.0,
        unit="second",
        idempotency_key="TRANSLATION:seg:vi",
    )
    kwargs.update(overrides)
    return await repo.record_usage_and_charge(**kwargs)


class TestSettlementGoesThroughTheDatabaseFunction:
    """The worker must not reimplement settlement.

    It used to write the usage record, the balance UPDATE and the credit transaction itself.
    That version had no overage, never wrote service_state, and raised when a workspace ran
    out of credits — turning an expected business state into a crash loop on redelivery.
    """

    async def test_it_calls_settle_usage_charge_and_writes_nothing_itself(self) -> None:
        repo, conn = _repository(
            {
                "applied": True,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": 900,
                "service_state": "healthy",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is True
        assert outcome.service_state == "healthy"
        assert outcome.balance_after == 900
        # _FakeConnection.execute/fetchval raise, so reaching here proves no hand-written
        # INSERT or UPDATE survived.
        assert any("settle_usage_charge" in query for query in conn.queries)

    async def test_running_out_of_credits_is_a_state_not_an_exception(self) -> None:
        # The whole point of the consolidation. The old code raised
        # "Insufficient credits for subscription ...", which crashed the handler and left the
        # Redis message pending forever.
        repo, _ = _repository(
            {
                "applied": True,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": -12,
                "service_state": "in_overage",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is True
        assert outcome.service_state == "in_overage"
        assert outcome.balance_after == -12

    async def test_a_suspended_subscription_is_refused_without_raising(self) -> None:
        repo, _ = _repository(
            {
                "applied": False,
                "transaction_id": None,
                "usage_record_id": None,
                "balance_after": 0,
                "service_state": "suspended",
                "suspended_reason": "overage_cap",
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is False
        assert outcome.replayed is False
        assert outcome.suspended_reason == "overage_cap"

    async def test_a_replay_is_distinguishable_from_a_refusal(self) -> None:
        # Both have applied=False. Only the replay carries the original transaction, and the
        # two must not be logged or reacted to the same way.
        repo, _ = _repository(
            {
                "applied": False,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": 900,
                "service_state": "healthy",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is False
        assert outcome.replayed is True

    async def test_a_missing_rate_card_still_raises(self) -> None:
        # A misconfiguration, not a business state — it must not be settled silently at zero.
        repo, conn = _repository({"applied": True})
        conn._rate_row = None

        with pytest.raises(RuntimeError, match="No active usage rate card"):
            await _settle(repo)
