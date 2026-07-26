"""Tests for Billing Settlement Worker — segment-id extraction helper."""

from __future__ import annotations

import json
import uuid
from decimal import Decimal
from typing import Any

import pytest

from billing_worker.db import BillingRepository
from billing_worker.worker import _extract_underlying_segment_id
from shared.config import DatabaseSettings, RedisSettings


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
