"""Run Phase 5 billing calibration queries against a real billing database."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from decimal import Decimal
from typing import Any

import asyncpg
from dotenv import load_dotenv

BASELINE_FX_RATE_USD_VND = Decimal("26300")

AUDIT_SQL_TEMPLATE = """
WITH room_sessions AS (
    SELECT
        id,
        translation_room_id,
        GREATEST(
            EXTRACT(
                EPOCH FROM (
                    LEAST(COALESCE(ended_at, NOW()), $2::timestamptz)
                    - GREATEST(started_at, $1::timestamptz)
                )
            ),
            0
        )::numeric AS wall_clock_seconds
    FROM translation_room.translation_room_sessions
    WHERE started_at IS NOT NULL
      AND started_at < $2::timestamptz
      AND COALESCE(ended_at, NOW()) >= $1::timestamptz
),
stt_usage AS (
    SELECT
        translation_room_id,
        SUM(quantity)::numeric AS stt_billed_seconds,
        SUM(credits_consumed)::numeric AS stt_credits
    FROM subscription.usage_records
    WHERE recorded_at >= $1::timestamptz
      AND recorded_at < $2::timestamptz
      AND {charge_type_column} = 'STT'
      AND unit = 'second'
    GROUP BY translation_room_id
),
overall AS (
    SELECT
        COUNT(DISTINCT rs.translation_room_id) AS room_count,
        COUNT(*) AS session_count,
        COALESCE(SUM(rs.wall_clock_seconds), 0)::numeric AS wall_clock_seconds,
        COALESCE(SUM(su.stt_billed_seconds), 0)::numeric AS stt_billed_seconds,
        COALESCE(SUM(su.stt_credits), 0)::numeric AS stt_credits
    FROM room_sessions rs
    LEFT JOIN stt_usage su ON su.translation_room_id = rs.translation_room_id
),
charged AS (
    SELECT
        COALESCE(SUM(credits_consumed), 0)::numeric AS credits_consumed,
        COALESCE(SUM(credits_consumed), 0)::numeric * 4 AS retail_vnd
    FROM subscription.usage_records
    WHERE recorded_at >= $1::timestamptz
      AND recorded_at < $2::timestamptz
),
provider_cost AS (
    {provider_cost_sql}
)
SELECT
    $6::text AS provider_cost_mode,
    o.room_count,
    o.session_count,
    ROUND(o.wall_clock_seconds, 2) AS wall_clock_seconds,
    ROUND(o.stt_billed_seconds, 2) AS stt_billed_seconds,
    ROUND(o.stt_billed_seconds / NULLIF(o.wall_clock_seconds, 0), 4)
        AS observed_talk_density_times_vad_padding,
    ROUND((o.stt_billed_seconds / NULLIF(o.wall_clock_seconds, 0)) / 1.15, 4)
        AS implied_talk_density_if_vad_padding_1_15,
    ROUND(c.retail_vnd, 2) AS retail_vnd,
    ROUND(pc.provider_vnd, 2) AS provider_vnd,
    ROUND(c.retail_vnd / NULLIF(pc.provider_vnd, 0), 4) AS markup
FROM overall o
CROSS JOIN charged c
CROSS JOIN provider_cost pc;
"""

ACTUAL_PROVIDER_COST_SQL = "SELECT ($3::numeric + $4::numeric) AS provider_vnd"

ESTIMATED_PROVIDER_COST_SQL = """
WITH breakdown_cost AS (
    SELECT COALESCE(
        SUM((item->>'quantity')::numeric * urc.provider_unit_cost * $5::numeric),
        0
    )::numeric AS provider_vnd
    FROM subscription.usage_records ur
    CROSS JOIN LATERAL jsonb_array_elements(
        CASE
            WHEN jsonb_typeof(ur.details->'unit_breakdown') = 'array'
                THEN ur.details->'unit_breakdown'
            ELSE '[]'::jsonb
        END
    ) item
    JOIN subscription.usage_rate_card urc
      ON urc.id = NULLIF(item->>'pricing_rate_card_id', '')::uuid
    WHERE ur.recorded_at >= $1::timestamptz
      AND ur.recorded_at < $2::timestamptz
      AND jsonb_typeof(ur.details->'unit_breakdown') = 'array'
      AND urc.provider_unit_cost IS NOT NULL
),
fallback_cost AS (
    SELECT COALESCE(SUM(ur.quantity * urc.provider_unit_cost * $5::numeric), 0)::numeric
        AS provider_vnd
    FROM subscription.usage_records ur
    JOIN subscription.credit_transactions ct ON ct.usage_record_id = ur.id
    JOIN subscription.usage_rate_card urc ON urc.id = ct.pricing_rate_card_id
    WHERE ur.recorded_at >= $1::timestamptz
      AND ur.recorded_at < $2::timestamptz
      AND (
          jsonb_typeof(ur.details->'unit_breakdown') IS DISTINCT FROM 'array'
          OR jsonb_array_length(
              CASE
                  WHEN jsonb_typeof(ur.details->'unit_breakdown') = 'array'
                      THEN ur.details->'unit_breakdown'
                  ELSE '[]'::jsonb
              END
          ) = 0
      )
      AND urc.provider_unit_cost IS NOT NULL
)
SELECT (breakdown_cost.provider_vnd + fallback_cost.provider_vnd)::numeric AS provider_vnd
FROM breakdown_cost
CROSS JOIN fallback_cost
"""


async def usage_charge_type_column(conn: asyncpg.Connection) -> str:
    has_charge_type = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'subscription'
              AND table_name = 'usage_records'
              AND column_name = 'charge_type'
        )
        """
    )
    return "charge_type" if has_charge_type else "usage_type"


def decimal_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def classify_markup(markup: Decimal | None) -> str:
    if markup is None:
        return "missing_provider_dashboard_cost"
    if markup < Decimal("1.0"):
        return "loss"
    if markup < Decimal("1.5") or markup > Decimal("4.0"):
        return "outside_phase5_target"
    return "within_phase5_target"


async def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    conn = await asyncpg.connect(args.db_dsn)
    try:
        charge_type_column = await usage_charge_type_column(conn)
        provider_cost_sql = (
            ESTIMATED_PROVIDER_COST_SQL
            if args.provider_cost_mode == "estimated"
            else ACTUAL_PROVIDER_COST_SQL
        )
        row = await conn.fetchrow(
            AUDIT_SQL_TEMPLATE.format(
                charge_type_column=charge_type_column,
                provider_cost_sql=provider_cost_sql,
            ),
            args.from_ts,
            args.to_ts,
            Decimal(args.openai_provider_vnd),
            Decimal(args.cartesia_provider_vnd),
            BASELINE_FX_RATE_USD_VND,
            args.provider_cost_mode,
        )
    finally:
        await conn.close()

    if row is None:
        raise RuntimeError("Phase 5 billing audit query returned no rows.")

    result = dict(row)
    result["status"] = classify_markup(result.get("markup"))
    return result


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dsn", default=os.getenv("BILLING_DB_DSN", ""))
    parser.add_argument("--from-ts", required=True, help="Inclusive UTC timestamp.")
    parser.add_argument("--to-ts", required=True, help="Exclusive UTC timestamp.")
    parser.add_argument("--openai-provider-vnd", default="0")
    parser.add_argument("--cartesia-provider-vnd", default="0")
    parser.add_argument(
        "--provider-cost-mode",
        choices=("actual", "estimated"),
        default="actual",
        help=(
            "actual: compare against provider dashboard inputs. "
            "estimated: temporary baseline from usage_rate_card.provider_unit_cost."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if not args.db_dsn:
        raise RuntimeError("Missing --db-dsn or BILLING_DB_DSN.")
    result = await run_audit(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=decimal_default))


if __name__ == "__main__":
    asyncio.run(main())
