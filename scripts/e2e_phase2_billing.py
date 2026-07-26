"""Run a local/staging Phase 2 billing E2E smoke test.

Required environment:
  BILLING_DB_DSN=postgresql://user:password@host:5432/warptalk
  REDIS_URL=redis://host:6379
  REDIS_PASSWORD=optional-password

The script creates an isolated test workspace/room/subscription, sends billable
messages through the AI billing worker handlers, and leaves temp logs in Redis
for the backend BillingAggregationWorker to aggregate. Use --cleanup-only to
remove prior E2E data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from decimal import Decimal
from typing import Any

import asyncpg
import redis.asyncio as redis

from billing_worker.worker import BillingSettlementWorker
from shared.config import BillingSettings, RedisSettings
from shared.schemas import (
    AIUsageMessage,
    AudioChunkMessage,
    ProviderUsageMessage,
    TTSResultMessage,
)

TEMP_USAGE_LOG_LIST = "warptalk:billing:temp_usage_logs"
E2E_WORKSPACE_NAME = "Billing Phase2 E2E"
DEFAULT_USER_ID = uuid.UUID("019ea677-6c84-7d7b-9f48-738b3cde41a9")


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def _cleanup(conn: asyncpg.Connection, redis_client: redis.Redis) -> None:
    await redis_client.delete(TEMP_USAGE_LOG_LIST)
    for pattern in ("billing:acc:*", "billing:charge:*"):
        async for key in redis_client.scan_iter(pattern):
            await redis_client.delete(key)

    await conn.execute(
        """
        WITH e2e_ws AS (
            SELECT id FROM workspace.workspaces WHERE name = $1
        ),
        del_tx AS (
            DELETE FROM subscription.credit_transactions
            WHERE workspace_id IN (SELECT id FROM e2e_ws)
        ),
        del_usage AS (
            DELETE FROM subscription.usage_records
            WHERE workspace_id IN (SELECT id FROM e2e_ws)
        ),
        del_sub AS (
            DELETE FROM subscription.subscriptions
            WHERE workspace_id IN (SELECT id FROM e2e_ws)
        ),
        del_rooms AS (
            DELETE FROM translation_room.translation_rooms
            WHERE workspace_id IN (SELECT id FROM e2e_ws)
        )
        DELETE FROM workspace.workspaces WHERE id IN (SELECT id FROM e2e_ws)
        """,
        E2E_WORKSPACE_NAME,
    )


async def _setup_data(conn: asyncpg.Connection, user_id: uuid.UUID) -> dict[str, uuid.UUID]:
    plan_id = await conn.fetchval(
        """
        SELECT id
        FROM subscription.plans
        WHERE slug = 'enterprise'
          AND is_active = true
        LIMIT 1
        """
    )
    if plan_id is None:
        raise RuntimeError("Missing active subscription.plans row with slug='enterprise'.")

    ids = {
        "user_id": user_id,
        "workspace_id": uuid.uuid4(),
        "room_id": uuid.uuid4(),
        "subscription_id": uuid.uuid4(),
        "plan_id": plan_id,
    }
    room_code = ("E2E" + ids["room_id"].hex[:9]).upper()[:12]

    await conn.execute(
        """
        INSERT INTO workspace.workspaces (id, name, slug, owner_id, created_by, updated_by)
        VALUES ($1, $2, $3, $4, $4, $4)
        """,
        ids["workspace_id"],
        E2E_WORKSPACE_NAME,
        f"billing-e2e-{ids['workspace_id'].hex[:12]}",
        ids["user_id"],
    )
    await conn.execute(
        """
        INSERT INTO translation_room.translation_rooms (
            id, workspace_id, host_id, title, translation_room_code,
            source_language, target_languages, status, started_at, created_by, updated_by
        )
        VALUES ($1, $2, $3, 'Billing Phase2 E2E Room', $4,
                'vi', '["en"]'::jsonb, 'ACTIVE', now(), $3, $3)
        """,
        ids["room_id"],
        ids["workspace_id"],
        ids["user_id"],
        room_code,
    )
    await conn.execute(
        """
        INSERT INTO subscription.subscriptions (
            id, user_id, workspace_id, plan_id, status, credits_remaining,
            credits_used_this_cycle, current_period_start, current_period_end, is_active
        )
        VALUES ($1, $2, $3, $4, 'active', 1000000, 0,
                now(), now() + interval '30 days', true)
        """,
        ids["subscription_id"],
        ids["user_id"],
        ids["workspace_id"],
        ids["plan_id"],
    )
    return ids


async def _send_billing_events(
    ids: dict[str, uuid.UUID],
    redis_url: str,
    redis_password: str,
) -> None:
    worker = BillingSettlementWorker(
        billing_settings=BillingSettings(),
        redis_settings=RedisSettings(url=redis_url, password=redis_password),
    )
    await worker.redis.connect()
    await worker.db.connect()

    room_id = str(ids["room_id"])
    user_id = str(ids["user_id"])
    workspace_id = str(ids["workspace_id"])

    try:
        await worker._handle_stt(
            AudioChunkMessage(
                meeting_id=room_id,
                speaker_id=user_id,
                chunk_index=1,
                audio_data=b"\x01\x02" * 16000,
                sample_rate=16000,
                is_final_chunk=True,
            ).to_redis()
        )
        await worker._handle_ai_usage(
            AIUsageMessage(
                workspace_id=workspace_id,
                room_id=room_id,
                user_id=user_id,
                charge_type="TRANSLATION",
                model="gpt-4.1-mini",
                prompt_tokens=100,
                cached_tokens=20,
                completion_tokens=50,
                source_lang="vi",
                target_lang="en",
                idempotency_key=f"E2E:TRANSLATION:{room_id}",
            ).to_redis()
        )
        await worker._handle_ai_usage(
            AIUsageMessage(
                workspace_id=workspace_id,
                room_id=room_id,
                user_id=user_id,
                charge_type="AI_ASSISTANT",
                model="gpt-4.1",
                prompt_tokens=1000,
                cached_tokens=200,
                completion_tokens=300,
                idempotency_key=f"E2E:AI_ASSISTANT:{room_id}",
            ).to_redis()
        )
        await worker._handle_ai_usage(
            AIUsageMessage(
                workspace_id=workspace_id,
                room_id=room_id,
                user_id=user_id,
                charge_type="AI_SUMMARY",
                model="gpt-4o-mini",
                prompt_tokens=800,
                cached_tokens=100,
                completion_tokens=200,
                idempotency_key=f"E2E:AI_SUMMARY:{room_id}",
            ).to_redis()
        )
        await worker._handle_tts(
            TTSResultMessage(
                segment_id=f"{uuid.uuid4()}-en-c0",
                meeting_id=room_id,
                speaker_id=user_id,
                audio_data=b"RIFF-e2e-standard",
                duration_ms=2500,
                char_count=120,
                voice_type="default",
                voice_mode="standard",
                target_lang="en",
                is_final_chunk=True,
            ).to_redis()
        )
        await worker._handle_tts(
            TTSResultMessage(
                segment_id=f"{uuid.uuid4()}-en-c0",
                meeting_id=room_id,
                speaker_id=user_id,
                audio_data=b"RIFF-e2e-clone",
                duration_ms=3000,
                char_count=80,
                voice_type="cloned",
                voice_mode="cloned",
                clone_provider="cartesia",
                target_lang="en",
                is_final_chunk=True,
            ).to_redis()
        )
        await worker._handle_provider_usage(
            ProviderUsageMessage(
                workspace_id=workspace_id,
                room_id=room_id,
                user_id=user_id,
                charge_type="VOICE_CLONE_ENROLLMENT",
                provider="cartesia",
                model="cartesia-localizing-voice",
                quantity=Decimal("1"),
                unit="profile",
                idempotency_key=f"E2E:VOICE_CLONE_ENROLLMENT:{room_id}",
            ).to_redis()
        )
    finally:
        await worker.db.disconnect()
        await worker.redis.disconnect()


async def _summarize(
    conn: asyncpg.Connection,
    redis_client: redis.Redis,
    subscription_id: uuid.UUID,
) -> dict[str, Any]:
    temp_logs = await redis_client.lrange(TEMP_USAGE_LOG_LIST, 0, -1)
    subscription = await conn.fetchrow(
        """
        SELECT credits_remaining, credits_used_this_cycle
        FROM subscription.subscriptions
        WHERE id = $1
        """,
        subscription_id,
    )
    return {
        "temp_log_count": len(temp_logs),
        "subscription_after_ai_worker": dict(subscription) if subscription else None,
        "temp_log_charge_types": [
            f"{json.loads(item)['ChargeType']}/{json.loads(item)['Unit']}/"
            f"{json.loads(item).get('Model')}"
            for item in temp_logs
        ],
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-only", action="store_true")
    parser.add_argument("--keep-data", action="store_true")
    parser.add_argument("--user-id", default=str(DEFAULT_USER_ID))
    args = parser.parse_args()

    pg_dsn = _env("BILLING_DB_DSN")
    redis_url = _env("REDIS_URL", "redis://localhost:6379")
    redis_password = os.getenv("REDIS_PASSWORD", "")

    redis_client = redis.from_url(
        redis_url,
        password=redis_password or None,
        decode_responses=True,
    )
    conn = await asyncpg.connect(pg_dsn)
    try:
        await _cleanup(conn, redis_client)
        if args.cleanup_only:
            print(json.dumps({"cleanup": "done"}, indent=2))
            return

        ids = await _setup_data(conn, uuid.UUID(args.user_id))
        await _send_billing_events(ids, redis_url, redis_password)
        summary = await _summarize(conn, redis_client, ids["subscription_id"])
        summary["ids"] = {key: str(value) for key, value in ids.items()}
        print(json.dumps(summary, indent=2))

        if not args.keep_data:
            await _cleanup(conn, redis_client)
    finally:
        await conn.close()
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
