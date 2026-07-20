"""Postgres access for billing settlement.

Two-step, no-physical-FK lookups on purpose: translation_room, transcript and
subscription live in different schemas with intentionally no cross-schema FOREIGN KEY
constraints (see warptalk-v4-final.dbml's "External ... No physical FK" convention).
This module is where those logical references get resolved at read time instead.
"""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg

from shared.config import DatabaseSettings
from shared.logger import get_logger

logger = get_logger(__name__)


def _as_uuid(value: str) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


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
        assert self._pool is not None, "call connect() first"
        user_uuid = _as_uuid(user_id) if user_id else None
        room_uuid = _as_uuid(translation_room_id)
        reference_uuid = _as_uuid(reference_id) if reference_id else None
        segment_uuid = (
            transcript_segment_id
            if isinstance(transcript_segment_id, uuid.UUID)
            else _as_uuid(transcript_segment_id) if transcript_segment_id else None
        )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchval(
                    "SELECT id FROM subscription.credit_transactions WHERE idempotency_key = $1",
                    idempotency_key,
                )
                if existing is not None:
                    return False

                usage_record_id = await conn.fetchval(
                    """
                    INSERT INTO subscription.usage_records
                        (subscription_id, user_id, workspace_id, translation_room_id,
                         usage_type, unit, quantity, credits_consumed, segment_id, details)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb)
                    RETURNING id
                    """,
                    subscription_id,
                    user_uuid,
                    workspace_id,
                    room_uuid,
                    usage_type,
                    unit,
                    quantity,
                    credits_consumed,
                    segment_uuid,
                    _to_jsonb(details or {}),
                )

                new_balance = await conn.fetchval(
                    """
                    UPDATE subscription.subscriptions
                    SET credits_remaining = credits_remaining - $2,
                        credits_used_this_cycle = credits_used_this_cycle + $2
                    WHERE id = $1
                    RETURNING credits_remaining
                    """,
                    subscription_id,
                    credits_consumed,
                )

                await conn.execute(
                    """
                    INSERT INTO subscription.credit_transactions
                        (subscription_id, user_id, amount, type, reference_id, reference_type,
                         balance_after, charge_type, usage_record_id, idempotency_key,
                         transcript_segment_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """,
                    subscription_id,
                    user_uuid,
                    -credits_consumed,
                    charge_type,
                    reference_uuid,
                    reference_type,
                    new_balance if new_balance is not None else 0,
                    charge_type,
                    usage_record_id,
                    idempotency_key,
                    segment_uuid,
                )

        logger.info(
            "usage_charged",
            charge_type=charge_type,
            reference_id=str(reference_uuid) if reference_uuid else None,
            transcript_segment_id=str(segment_uuid) if segment_uuid else None,
            credits_consumed=credits_consumed,
        )
        return True


def _to_jsonb(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data)
