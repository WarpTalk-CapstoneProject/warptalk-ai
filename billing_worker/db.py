"""Postgres access for billing settlement.

Two-step, no-physical-FK lookups on purpose: translation_room, transcript and
subscription live in different schemas with intentionally no cross-schema FOREIGN KEY
constraints (see warptalk-v4-final.dbml's "External ... No physical FK" convention).
This module is where those logical references get resolved at read time instead.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Any

import asyncpg

from shared.config import DatabaseSettings
from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SettlementOutcome:
    """What subscription.settle_usage_charge decided.

    `applied` is False for three different reasons, and the caller must be able to tell them
    apart: a replay of an already-settled key (`replayed`), a subscription that is suspended
    or gone, or an overage cap that this charge would cross. Collapsing them into one boolean
    is how the previous version ended up raising on a normal, expected business state.
    """

    applied: bool
    replayed: bool
    balance_after: int | None
    service_state: str | None
    suspended_reason: str | None
    credits_consumed: int


def _as_uuid(value: str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def calculate_credit_charge(quantity: float, unit_price: Decimal) -> int:
    """Apply an immutable rate-card snapshot and bill whole credits."""
    raw_cost = Decimal(str(quantity)) * unit_price
    return max(1, int(raw_cost.to_integral_value(rounding=ROUND_CEILING)))


class BillingRepository:
    """asyncpg-backed access to the subscription (billing) schema."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings()
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=self.settings.dsn,
            min_size=self.settings.min_pool_size,
            max_size=self.settings.max_pool_size,
        )
        logger.info("billing_db_connected")

    async def disconnect(self) -> None:
        if self._pool:
            await self._pool.close()
        logger.info("billing_db_disconnected")

    async def resolve_subscription(self, workspace_id: str) -> tuple[uuid.UUID, uuid.UUID] | None:
        """Return the active Billing-owned subscription for a workspace projection."""
        assert self._pool is not None, "call connect() first"
        workspace_uuid = _as_uuid(workspace_id)
        async with self._pool.acquire() as conn:
            subscription_id = await conn.fetchval(
                """
                SELECT id FROM subscription.subscriptions
                WHERE workspace_id = $1 AND is_active = true
                ORDER BY created_at DESC
                LIMIT 1
                """,
                workspace_uuid,
            )
            if subscription_id is None:
                return None

            return subscription_id, workspace_uuid

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
        idempotency_key: str,
        transcript_segment_id: str | uuid.UUID | None = None,
        source_language_code: str | None = None,
        target_language_code: str | None = None,
        currency: str = "CRD",
        details: dict[str, Any] | None = None,
    ) -> SettlementOutcome:
        """Settle one billable event through subscription.settle_usage_charge.

        This worker used to write the usage record, the balance UPDATE and the credit
        transaction itself, in three statements it owned. That reimplemented — badly — a
        function the billing schema already provides, and the gap was not cosmetic:

        * Running out of credits raised `Insufficient credits`, which crashed the handler and
          left the event to be redelivered forever. Overage exists precisely so that running
          out is a *state*, not an error.
        * `service_state` and `suspended_reason` were never written, so nothing downstream
          could tell a healthy workspace from one in overage or suspended.
        * `overage_credits_this_cycle` and `overage_started_at` were never touched, so the
          overage cap could not be enforced and top-ups had nothing to reconcile against.
        * The idempotency probe ran *before* taking a row lock. Two concurrent deliveries of
          the same key would both miss it. The function checks after `FOR UPDATE`, which is
          the only place the check is authoritative.

        Rate resolution stays here because the function takes the rate card id, the unit
        price snapshot and the credit amount as inputs — it prices nothing itself, by design,
        so the snapshot that was actually charged is the one recorded.

        Returns a SettlementOutcome. It never raises for a business outcome; it still raises
        if there is no rate card, because that is a misconfiguration rather than a state.
        """
        assert self._pool is not None, "call connect() first"
        user_uuid = _as_uuid(user_id) if user_id else None
        room_uuid = _as_uuid(translation_room_id)
        reference_uuid = _as_uuid(reference_id) if reference_id else None
        segment_uuid = (
            transcript_segment_id
            if isinstance(transcript_segment_id, uuid.UUID)
            else _as_uuid(transcript_segment_id)
            if transcript_segment_id
            else None
        )

        async with self._pool.acquire() as conn:
            rate = await conn.fetchrow(
                """
                SELECT id, unit_price, currency
                FROM subscription.usage_rate_card
                WHERE charge_type = $1
                  AND currency = $2
                  AND effective_from <= now()
                  AND (effective_to IS NULL OR effective_to > now())
                  AND (source_language_code = $3 OR source_language_code IS NULL)
                  AND (target_language_code = $4 OR target_language_code IS NULL)
                ORDER BY
                  (source_language_code IS NOT NULL)::int DESC,
                  (target_language_code IS NOT NULL)::int DESC,
                  effective_from DESC
                LIMIT 1
                """,
                charge_type,
                currency,
                source_language_code,
                target_language_code,
            )
            if rate is None:
                raise RuntimeError(
                    "No active usage rate card for "
                    f"{charge_type}/{source_language_code}/{target_language_code}/{currency}"
                )
            rate_card_id = rate["id"]
            unit_price = Decimal(str(rate["unit_price"]))
            applied_currency = rate["currency"]
            credits_consumed = calculate_credit_charge(quantity, unit_price)

            row = await conn.fetchrow(
                """
                SELECT applied, transaction_id, usage_record_id,
                       balance_after, service_state, suspended_reason
                FROM subscription.settle_usage_charge(
                    $1, $2, $3, $4, $5, $6, $7, $8, $9,
                    $10, $11, $12, $13, $14, $15, $16, $17::jsonb
                )
                """,
                subscription_id,
                user_uuid,
                workspace_id,
                usage_type,
                charge_type,
                reference_uuid,
                reference_type,
                room_uuid,
                segment_uuid,
                quantity,
                unit,
                credits_consumed,
                idempotency_key,
                rate_card_id,
                unit_price,
                applied_currency,
                _to_jsonb(details or {}),
            )

        if row is None:
            # settle_usage_charge always RETURN QUERYs exactly one row on every path. Getting
            # none back means the deployed function is not the one this code was written
            # against — a broken migration, not a business outcome, so it must not be
            # mistaken for "not applied" and quietly dropped.
            raise RuntimeError(
                "subscription.settle_usage_charge returned no row for "
                f"{charge_type}/{idempotency_key}"
            )

        applied = bool(row["applied"])
        # A replay is the one not-applied case that still carries the original transaction —
        # that is how it is told apart from a refusal, which carries none.
        replayed = not applied and row["transaction_id"] is not None

        outcome = SettlementOutcome(
            applied=applied,
            replayed=replayed,
            balance_after=row["balance_after"],
            service_state=row["service_state"],
            suspended_reason=row["suspended_reason"],
            credits_consumed=credits_consumed,
        )

        if applied:
            logger.info(
                "usage_charged",
                charge_type=charge_type,
                reference_id=str(reference_uuid) if reference_uuid else None,
                transcript_segment_id=str(segment_uuid) if segment_uuid else None,
                credits_consumed=credits_consumed,
                balance_after=outcome.balance_after,
                service_state=outcome.service_state,
            )
        elif replayed:
            logger.info(
                "usage_charge_replayed",
                charge_type=charge_type,
                idempotency_key=idempotency_key,
            )
        else:
            logger.warning(
                "usage_charge_refused",
                charge_type=charge_type,
                service_state=outcome.service_state,
                suspended_reason=outcome.suspended_reason,
                credits_consumed=credits_consumed,
            )

        return outcome


def _to_jsonb(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data)
