"""Tests for Billing Settlement Worker — segment-id extraction helper."""

from __future__ import annotations

import asyncio
import json
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from billing_worker.db import BillingRepository, UsageRate
from billing_worker.worker import BillingSettlementWorker, _extract_underlying_segment_id
from shared.config import DatabaseSettings, RedisSettings
from shared.health_probe import check_worker
from shared.schemas import AIUsageMessage, ProviderUsageMessage


class FakeBillingRepository(BillingRepository):
    def __init__(self, rows: dict[tuple[str | None, str | None], dict[str, Any] | None]) -> None:
        super().__init__()
        self._pool = object()  # type: ignore[assignment]
        self.rows = rows
        self.fetch_count = 0

    async def _fetch_usage_rate_row(
        self,
        *,
        charge_type: str,
        unit: str,
        currency: str,
        provider: str,
        model: str,
        source_language_code: str | None,
        target_language_code: str | None,
    ) -> dict[str, Any] | None:
        _ = (charge_type, unit, currency, provider, model)
        self.fetch_count += 1
        exact_key = (source_language_code, target_language_code)
        return self.rows.get(exact_key) or self.rows.get((None, None))


class FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))


class FakeAcquire:
    def __init__(self, conn: FakeConnection) -> None:
        self.conn = conn

    async def __aenter__(self) -> FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakePool:
    def __init__(self) -> None:
        self.conn = FakeConnection()

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.conn)


class FakeRedis:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, str]] = []

    async def rpush(self, key: str, value: str) -> None:
        self.pushed.append((key, value))


def _rate_row(unit_price: str = "0.006575") -> dict[str, Any]:
    return {
        "id": uuid.uuid4(),
        "unit_price": Decimal(unit_price),
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "provider_unit_cost": Decimal("0.00000040"),
        "markup_multiplier": Decimal("2.5"),
    }


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


class TestResolveUsageRate:
    async def test_exact_language_rate_wins(self) -> None:
        exact = _rate_row("0.010000")
        generic = _rate_row("0.020000")
        repo = FakeBillingRepository({("vi", "en"): exact, (None, None): generic})

        rate = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
            source_language_code="vi",
            target_language_code="en",
        )

        assert rate is not None
        assert rate.id == exact["id"]
        assert rate.unit_price == Decimal("0.010000")

    async def test_generic_rate_fallback_when_language_specific_missing(self) -> None:
        generic = _rate_row("0.020000")
        repo = FakeBillingRepository({(None, None): generic})

        rate = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
            source_language_code="ja",
            target_language_code="en",
        )

        assert rate is not None
        assert rate.id == generic["id"]
        assert rate.unit_price == Decimal("0.020000")

    async def test_missing_rate_returns_none(self) -> None:
        repo = FakeBillingRepository({})

        rate = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
            source_language_code="vi",
            target_language_code="en",
        )

        assert rate is None

    async def test_invalid_rate_row_missing_provider_or_model_returns_none(self) -> None:
        invalid_row = _rate_row("0.020000")
        invalid_row["provider"] = ""
        repo = FakeBillingRepository({(None, None): invalid_row})

        rate = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
        )

        assert rate is None

    async def test_cache_reuses_rate_for_same_lookup(self) -> None:
        repo = FakeBillingRepository({(None, None): _rate_row("0.020000")})

        first = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
        )
        second = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit="token_in",
            provider="openai",
            model="gpt-4.1-mini",
        )

        assert first == second
        assert repo.fetch_count == 1

    @pytest.mark.parametrize(
        ("unit", "provider", "model"),
        [
            ("", "openai", "gpt-4.1-mini"),
            ("token_in", "", "gpt-4.1-mini"),
            ("token_in", "openai", ""),
        ],
    )
    async def test_missing_provider_model_or_unit_is_guarded(
        self,
        unit: str,
        provider: str,
        model: str,
    ) -> None:
        repo = FakeBillingRepository({(None, None): _rate_row("0.020000")})

        rate = await repo.resolve_usage_rate(
            charge_type="TRANSLATION",
            unit=unit,
            provider=provider,
            model=model,
        )

        assert rate is None
        assert repo.fetch_count == 0


class TestBillingAccumulatorFormula:
    def test_accumulator_key_uses_language_pricing_scope(self) -> None:
        subscription_id = uuid.uuid4()
        key = BillingSettlementWorker._accumulator_key(
            subscription_id,
            "room-1",
            "TRANSLATION",
            "vi",
            "en",
        )

        assert key == f"billing:acc:{subscription_id}:room-1:TRANSLATION:vi:en"

    def test_accumulator_key_uses_placeholder_for_base_rate_scope(self) -> None:
        subscription_id = uuid.uuid4()
        key = BillingSettlementWorker._accumulator_key(
            subscription_id,
            "room-1",
            "STT",
            None,
            None,
        )

        assert key == f"billing:acc:{subscription_id}:room-1:STT:_:_"

    def test_micro_credit_conversion_rounds_to_integer_micro_units(self) -> None:
        assert BillingSettlementWorker._to_micro(Decimal("1.2345674")) == 1234567
        assert BillingSettlementWorker._to_micro(Decimal("1.2345675")) == 1234568
        assert BillingSettlementWorker._from_micro("1234568") == Decimal("1.234568")

    def test_unit_breakdown_reads_micro_quantities_and_rate_snapshots(self) -> None:
        rate_id = uuid.uuid4()
        worker = BillingSettlementWorker()

        breakdown = worker._unit_breakdown(
            {
                "quantity_micro_token_in": "120000000",
                "rate_token_in_id": str(rate_id),
                "rate_token_in_price_micro": "6575",
                "provider_token_in": "openai",
                "model_token_in": "gpt-4.1-mini",
            }
        )

        assert breakdown == [
            {
                "unit": "token_in",
                "quantity": "120",
                "pricing_rate_card_id": str(rate_id),
                "unit_price_snapshot": "0.006575",
                "provider": "openai",
                "model": "gpt-4.1-mini",
            }
        ]


class TestRecordUsageAndCharge:
    async def test_temp_usage_log_includes_rate_snapshot_metadata(self) -> None:
        repo = BillingRepository()
        fake_pool = FakePool()
        fake_redis = FakeRedis()
        repo._pool = fake_pool  # type: ignore[assignment]
        repo._redis = fake_redis  # type: ignore[assignment]
        pricing_rate_card_id = uuid.uuid4()
        subscription_id = uuid.uuid4()
        workspace_id = uuid.uuid4()
        room_id = uuid.uuid4()

        recorded = await repo.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=None,
            workspace_id=workspace_id,
            translation_room_id=str(room_id),
            usage_type="AI_ASSISTANT",
            charge_type="AI_ASSISTANT",
            reference_id=None,
            reference_type="billing_accumulator",
            quantity=100,
            unit="token_out",
            credits_consumed=14,
            idempotency_key="AI_ASSISTANT:token_out:openai:gpt-4.1:room:1",
            pricing_rate_card_id=pricing_rate_card_id,
            unit_price_snapshot=Decimal("0.131500"),
            provider="openai",
            model="gpt-4.1",
        )

        assert recorded is True
        assert fake_redis.pushed
        key, payload = fake_redis.pushed[0]
        temp_log = json.loads(payload)
        assert key == "warptalk:billing:temp_usage_logs"
        assert temp_log["PricingRateCardId"] == str(pricing_rate_card_id)
        assert temp_log["UnitPriceSnapshot"] == 0.1315
        assert temp_log["Provider"] == "openai"
        assert temp_log["Model"] == "gpt-4.1"


class TestBillingRepositoryConnection:
    async def test_connect_passes_redis_password_setting(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        class FakePool:
            async def close(self) -> None:
                return None

        class FakeRedisClient:
            async def aclose(self) -> None:
                return None

        async def fake_create_pool(**kwargs: Any) -> FakePool:
            captured["pool_kwargs"] = kwargs
            return FakePool()

        def fake_from_url(url: str, **kwargs: Any) -> FakeRedisClient:
            captured["redis_url"] = url
            captured["redis_kwargs"] = kwargs
            return FakeRedisClient()

        monkeypatch.setattr("billing_worker.db.asyncpg.create_pool", fake_create_pool)
        monkeypatch.setattr("billing_worker.db.aioredis.from_url", fake_from_url)

        repo = BillingRepository(
            settings=DatabaseSettings(dsn="postgresql://example"),
            redis_settings=RedisSettings(
                url="redis://localhost:6379",
                password="secret",
            ),
        )

        await repo.connect()

        assert captured["redis_url"] == "redis://localhost:6379"
        assert captured["redis_kwargs"]["password"] == "secret"


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


async def test_subscription_resolution_uses_event_workspace_without_room_projection() -> None:
    room_id = "assistant-conversation-1"
    workspace_id = str(uuid.uuid4())
    subscription_id = uuid.uuid4()
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock()
    worker.db = MagicMock()
    worker.db.resolve_subscription = AsyncMock(
        return_value=(subscription_id, uuid.UUID(workspace_id))
    )

    resolved = await worker._resolve_subscription(room_id, workspace_id)

    assert resolved == (subscription_id, uuid.UUID(workspace_id))
    worker.redis.get.assert_not_awaited()
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


async def test_voice_clone_enrollment_provider_usage_records_immediate_charge() -> None:
    subscription_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    rate_id = uuid.uuid4()
    user_id = str(uuid.uuid4())
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._resolve_subscription = AsyncMock(return_value=(subscription_id, workspace_id))
    worker._accumulate_and_maybe_flush = AsyncMock()
    worker.db = MagicMock()
    worker.db.resolve_usage_rate = AsyncMock(
        return_value=UsageRate(
            id=rate_id,
            unit_price=Decimal("2.4"),
            provider="cartesia",
            model="cartesia-localizing-voice",
        )
    )
    worker.db.record_usage_and_charge = AsyncMock(return_value=True)
    message = ProviderUsageMessage(
        room_id=str(uuid.uuid4()),
        user_id=user_id,
        charge_type="VOICE_CLONE_ENROLLMENT",
        provider="cartesia",
        model="cartesia-localizing-voice",
        quantity=Decimal("1"),
        unit="profile",
        idempotency_key="VOICE_CLONE_ENROLLMENT:room-1:user-1",
    )

    await worker._handle_provider_usage(message.to_redis())

    worker._resolve_subscription.assert_awaited_once_with(message.room_id, None)
    worker.db.resolve_usage_rate.assert_awaited_once_with(
        charge_type="VOICE_CLONE_ENROLLMENT",
        unit="profile",
        provider="cartesia",
        model="cartesia-localizing-voice",
    )
    worker._accumulate_and_maybe_flush.assert_not_awaited()
    worker.db.record_usage_and_charge.assert_awaited_once()
    kwargs = worker.db.record_usage_and_charge.await_args.kwargs
    assert kwargs["subscription_id"] == subscription_id
    assert kwargs["user_id"] == user_id
    assert kwargs["workspace_id"] == workspace_id
    assert kwargs["translation_room_id"] == message.room_id
    assert kwargs["reference_id"] is None
    assert kwargs["reference_type"] == "voice_clone"
    assert kwargs["credits_consumed"] == 3
    assert kwargs["idempotency_key"] == message.idempotency_key
    assert kwargs["pricing_rate_card_id"] == rate_id
    assert kwargs["unit_price_snapshot"] == Decimal("2.4")


async def test_ai_usage_with_workspace_id_does_not_require_room_projection() -> None:
    subscription_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    rate_id = uuid.uuid4()
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._resolve_subscription = AsyncMock(return_value=(subscription_id, workspace_id))
    worker._accumulate_and_maybe_flush = AsyncMock()
    worker.db = MagicMock()
    worker.db.resolve_usage_rate = AsyncMock(
        return_value=UsageRate(
            id=rate_id,
            unit_price=Decimal("0.01"),
            provider="openai",
            model="gpt-4.1",
        )
    )
    message = AIUsageMessage(
        workspace_id=str(workspace_id),
        room_id="assistant-conversation-1",
        user_id=str(uuid.uuid4()),
        charge_type="AI_ASSISTANT",
        model="gpt-4.1",
        prompt_tokens=0,
        cached_tokens=0,
        completion_tokens=10,
        idempotency_key="AI_ASSISTANT:req-1",
    )

    await worker._handle_ai_usage(message.to_redis())

    worker._resolve_subscription.assert_awaited_once_with(
        message.room_id,
        message.workspace_id,
    )
    worker._accumulate_and_maybe_flush.assert_awaited_once()


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
