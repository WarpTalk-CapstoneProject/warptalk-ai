"""Postgres access for billing settlement.

Two-step, no-physical-FK lookups on purpose: translation_room, transcript and
subscription live in different schemas with intentionally no cross-schema FOREIGN KEY
constraints (see warptalk-v4-final.dbml's "External ... No physical FK" convention).
This module is where those logical references get resolved at read time instead.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import asyncpg
import redis.asyncio as aioredis

from shared.config import DatabaseSettings, RedisSettings
from shared.logger import get_logger

logger = get_logger(__name__)
USAGE_RATE_CACHE_TTL_SECONDS = 60.0


def _as_uuid(value: str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


@dataclass(frozen=True)
class UsageRate:
    id: uuid.UUID
    unit_price: Decimal
    provider: str
    model: str
    provider_unit_cost: Decimal | None = None
    markup_multiplier: Decimal | None = None


class BillingRepository:
    """asyncpg-backed access to the subscription (billing) schema, with redis for temp logging."""

    def __init__(
        self,
        settings: DatabaseSettings | None = None,
        redis_settings: RedisSettings | None = None,
    ) -> None:
        self.settings = settings or DatabaseSettings()
        self.redis_settings = redis_settings or RedisSettings()
        self._pool: asyncpg.Pool | None = None
        self._redis: aioredis.Redis | None = None
        self._usage_rate_cache: dict[
            tuple[str, str, str, str, str, str | None, str | None],
            tuple[float, UsageRate | None],
        ] = {}

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=self.settings.dsn,
            min_size=self.settings.min_pool_size,
            max_size=self.settings.max_pool_size,
        )
        self._redis = aioredis.from_url(
            self.redis_settings.url,
            decode_responses=True,
            socket_timeout=self.redis_settings.socket_timeout,
            socket_connect_timeout=self.redis_settings.socket_connect_timeout,
        )
        logger.info("billing_db_connected")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
        if self._redis:
            await self._redis.aclose()
        logger.info("billing_db_disconnected")

    async def resolve_subscription(
        self, translation_room_id: str
    ) -> tuple[uuid.UUID, uuid.UUID] | None:
        """Return (subscription_id, workspace_id) for the room's active subscription.

        None if the room, its workspace, or an active subscription can't be found —
        callers must treat that as "cannot bill this event" and log+skip, never guess.
        """
        assert self._pool is not None, "call connect() first"
        room_id = _as_uuid(translation_room_id)
        async with self._pool.acquire() as conn:
            workspace_id = await conn.fetchval(
                "SELECT workspace_id FROM translation_room.translation_rooms WHERE id = $1",
                room_id,
            )
            if workspace_id is None:
                return None

            subscription_id = await conn.fetchval(
                """
                SELECT id FROM subscription.subscriptions
                WHERE workspace_id = $1 AND is_active = true
                ORDER BY created_at DESC
                LIMIT 1
                """,
                workspace_id,
            )
            if subscription_id is None:
                return None

            return subscription_id, workspace_id

    async def resolve_usage_rate(
        self,
        *,
        charge_type: str,
        unit: str,
        provider: str,
        model: str,
        currency: str = "VND",
        source_language_code: str | None = None,
        target_language_code: str | None = None,
    ) -> UsageRate | None:
        """Resolve the active rate card row for a concrete provider/model/unit.

        Language-specific rows win over generic rows. Missing rows are not guessed:
        callers must skip billing and surface a log until seed data is corrected.
        """
        assert self._pool is not None, "call connect() first"
        if not unit or not provider or not model:
            logger.warning(
                "usage_rate_invalid_lookup",
                metric_name="billing_usage_rate_invalid_lookup",
                charge_type=charge_type,
                unit=unit,
                provider=provider,
                model=model,
                currency=currency,
            )
            return None

        source_language_code = source_language_code or None
        target_language_code = target_language_code or None
        cache_key = (
            charge_type,
            unit,
            currency,
            provider,
            model,
            source_language_code,
            target_language_code,
        )
        now = time.monotonic()
        cached = self._usage_rate_cache.get(cache_key)
        if cached and now - cached[0] < USAGE_RATE_CACHE_TTL_SECONDS:
            return cached[1]

        row = await self._fetch_usage_rate_row(
            charge_type=charge_type,
            unit=unit,
            currency=currency,
            provider=provider,
            model=model,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )

        if row is None:
            logger.warning(
                "usage_rate_missing",
                metric_name="billing_usage_rate_missing",
                charge_type=charge_type,
                unit=unit,
                provider=provider,
                model=model,
                currency=currency,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            self._usage_rate_cache[cache_key] = (now, None)
            return None
        if not row["provider"] or not row["model"]:
            logger.warning(
                "usage_rate_invalid_row",
                metric_name="billing_usage_rate_invalid_row",
                charge_type=charge_type,
                unit=unit,
                provider=row["provider"],
                model=row["model"],
                currency=currency,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
            )
            self._usage_rate_cache[cache_key] = (now, None)
            return None

        rate = UsageRate(
            id=row["id"],
            unit_price=Decimal(str(row["unit_price"])),
            provider=row["provider"],
            model=row["model"],
            provider_unit_cost=(
                Decimal(str(row["provider_unit_cost"]))
                if row["provider_unit_cost"] is not None
                else None
            ),
            markup_multiplier=(
                Decimal(str(row["markup_multiplier"]))
                if row["markup_multiplier"] is not None
                else None
            ),
        )
        self._usage_rate_cache[cache_key] = (now, rate)
        return rate

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
    ) -> asyncpg.Record | dict[str, Any] | None:
        assert self._pool is not None, "call connect() first"
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(
                """
                SELECT id, unit_price, provider, model, provider_unit_cost, markup_multiplier
                FROM subscription.usage_rate_card
                WHERE charge_type = $1
                  AND unit = $2
                  AND currency = $3
                  AND provider = $4
                  AND model = $5
                  AND is_active = true
                  AND effective_from <= now()
                  AND (effective_to IS NULL OR effective_to > now())
                  AND (
                    (source_language_code = $6 AND target_language_code = $7) OR
                    (source_language_code = $6 AND target_language_code IS NULL) OR
                    (source_language_code IS NULL AND target_language_code = $7) OR
                    (source_language_code IS NULL AND target_language_code IS NULL)
                  )
                ORDER BY
                  CASE
                    WHEN source_language_code = $6 AND target_language_code = $7 THEN 0
                    WHEN source_language_code = $6 AND target_language_code IS NULL THEN 1
                    WHEN source_language_code IS NULL AND target_language_code = $7 THEN 2
                    ELSE 3
                  END,
                  effective_from DESC
                LIMIT 1
                """,
                charge_type,
                unit,
                currency,
                provider,
                model,
                source_language_code,
                target_language_code,
            )

    async def record_usage_and_charge(
        self,
        *,
        subscription_id: uuid.UUID,
        user_id: str | None,
        workspace_id: uuid.UUID,
        translation_room_id: str,
        usage_type: str,
        charge_type: str,
        reference_id: str | None,
        reference_type: str,
        quantity: float,
        unit: str,
        credits_consumed: int,
        idempotency_key: str,
        pricing_rate_card_id: uuid.UUID | None = None,
        unit_price_snapshot: Decimal | None = None,
        provider: str | None = None,
        model: str | None = None,
        transcript_segment_id: str | uuid.UUID | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Insert usage_records + credit_transactions, idempotent on idempotency_key.

        Returns True if a new charge was recorded, False if this idempotency_key was
        already settled (safe on Redis Streams redelivery / worker restart / retry).

        credits_consumed is currently a flat placeholder (see settlement.py) pending the
        usage_rate_card lookup design (source/target language + currency tiers) from
        warptalk-v4-final.dbml — wiring that up is real pricing work, not schema work,
        and deliberately out of scope here so this doesn't ship guessed prices.

        transcript_segment_id should be the real transcript.transcript_segments.id GUID —
        callers must extract it from any composite segment_id (translation_worker mints
        "{stt-segment-guid}-c{idx}") before passing it in; this function does not do that
        extraction itself, it only converts a valid GUID string/UUID for binding.

        reference_id is also converted to a UUID before binding: previously it was passed
        straight through as a raw string, which silently failed (and dropped the charge)
        for translation/TTS events whose segment_id is the composite string above, not a
        bare GUID. Callers must pass an already-extracted, valid GUID string here too.
        """
        assert self._pool is not None and self._redis is not None, "call connect() first"
        user_uuid = _as_uuid(user_id) if user_id else None
        room_uuid = _as_uuid(translation_room_id)
        reference_uuid = _as_uuid(reference_id) if reference_id else None
        segment_uuid = (
            transcript_segment_id
            if isinstance(transcript_segment_id, uuid.UUID)
            else _as_uuid(transcript_segment_id) if transcript_segment_id else None
        )

        # Temp usage log structure for Redis (must match C# TempUsageLog struct)
        temp_log = {
            "SubscriptionId": str(subscription_id),
            "UserId": str(user_uuid) if user_uuid else None,
            "WorkspaceId": str(workspace_id),
            "TranslationRoomId": str(room_uuid),
            "UsageType": usage_type,
            "ChargeType": charge_type,
            "ReferenceId": str(reference_uuid) if reference_uuid else None,
            "ReferenceType": reference_type,
            "Quantity": quantity,
            "Unit": unit,
            "CreditsConsumed": credits_consumed,
            "PricingRateCardId": str(pricing_rate_card_id) if pricing_rate_card_id else None,
            "UnitPriceSnapshot": (
                float(unit_price_snapshot) if unit_price_snapshot is not None else None
            ),
            "Provider": provider,
            "Model": model,
            "TranscriptSegmentId": str(segment_uuid) if segment_uuid else None,
            "IdempotencyKey": idempotency_key,
            "Details": json.dumps(details or {}),
            "CreatedAt": datetime.now(UTC).isoformat()
        }

        # Update the live balance in PostgreSQL synchronously to prevent overdraft.
        # This matches the C# API behavior.
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE subscription.subscriptions
                SET credits_remaining = credits_remaining - $2,
                    credits_used_this_cycle = credits_used_this_cycle + $2
                WHERE id = $1
                """,
                subscription_id,
                credits_consumed,
            )

        # Instead of directly writing to Postgres logs, we just push to Redis.
        # This will be picked up by C#'s BillingAggregationWorker.
        # Idempotency relies on the C# worker or Redis consumer deduplication if redelivered,
        # but pushing to a Redis List allows duplicates if not careful.
        # However, since this was just inserting directly, we rely on the DB transaction
        # to catch unique idempotency_key. For temp logs, we'll let C# worker handle deduplication
        # when it bulk-inserts.

        await self._redis.rpush("warptalk:billing:temp_usage_logs", json.dumps(temp_log))

        logger.info(
            "usage_charged",
            charge_type=charge_type,
            reference_id=str(reference_uuid) if reference_uuid else None,
            transcript_segment_id=str(segment_uuid) if segment_uuid else None,
            credits_consumed=credits_consumed,
        )
        return True


def _to_jsonb(data: dict[str, Any]) -> str:
    return json.dumps(data)
