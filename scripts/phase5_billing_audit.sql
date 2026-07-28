-- Phase 5 billing audit queries.
--
-- Usage with psql:
--   psql "$BILLING_DB_DSN" \
--     -v from_ts="'2026-07-01T00:00:00Z'" \
--     -v to_ts="'2026-08-01T00:00:00Z'" \
--     -v openai_provider_vnd="0" \
--     -v cartesia_provider_vnd="0" \
--     -f scripts/phase5_billing_audit.sql
--
-- Fill openai_provider_vnd and cartesia_provider_vnd from the provider dashboards for
-- the exact same [from_ts, to_ts) window. A markup below 1.0 means the workload is
-- losing money; Phase 5 target is 1.5x..4.0x.

\if :{?from_ts}
\else
\set from_ts '''1970-01-01T00:00:00Z'''
\endif

\if :{?to_ts}
\else
\set to_ts '''2999-01-01T00:00:00Z'''
\endif

\if :{?openai_provider_vnd}
\else
\set openai_provider_vnd 0
\endif

\if :{?cartesia_provider_vnd}
\else
\set cartesia_provider_vnd 0
\endif

SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'subscription'
          AND table_name = 'usage_records'
          AND column_name = 'charge_type'
    )
    THEN 'charge_type'
    ELSE 'usage_type'
END AS usage_charge_type_column
\gset

WITH room_sessions AS (
    SELECT
        id,
        translation_room_id,
        GREATEST(
            EXTRACT(
                EPOCH FROM (
                    LEAST(COALESCE(ended_at, NOW()), :to_ts::timestamptz)
                    - GREATEST(started_at, :from_ts::timestamptz)
                )
            ),
            0
        )::numeric AS wall_clock_seconds
    FROM translation_room.translation_room_sessions
    WHERE started_at IS NOT NULL
      AND started_at < :to_ts::timestamptz
      AND COALESCE(ended_at, NOW()) >= :from_ts::timestamptz
),
stt_usage AS (
    SELECT
        translation_room_id,
        SUM(quantity)::numeric AS stt_billed_seconds,
        SUM(credits_consumed)::numeric AS stt_credits
    FROM subscription.usage_records
    WHERE recorded_at >= :from_ts::timestamptz
      AND recorded_at < :to_ts::timestamptz
      AND :usage_charge_type_column = 'STT'
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
    WHERE recorded_at >= :from_ts::timestamptz
      AND recorded_at < :to_ts::timestamptz
),
provider_cost AS (
    SELECT (:openai_provider_vnd::numeric + :cartesia_provider_vnd::numeric) AS provider_vnd
)
SELECT
    'P5.1 observed STT ratio' AS audit,
    room_count,
    session_count,
    ROUND(wall_clock_seconds, 2) AS wall_clock_seconds,
    ROUND(stt_billed_seconds, 2) AS stt_billed_seconds,
    ROUND(stt_billed_seconds / NULLIF(wall_clock_seconds, 0), 4)
        AS observed_talk_density_times_vad_padding,
    ROUND((stt_billed_seconds / NULLIF(wall_clock_seconds, 0)) / 1.15, 4)
        AS implied_talk_density_if_vad_padding_1_15,
    NULL::numeric AS retail_vnd,
    NULL::numeric AS provider_vnd,
    NULL::numeric AS markup
FROM overall

UNION ALL

SELECT
    'P5.2 charged revenue vs provider dashboards' AS audit,
    NULL::bigint AS room_count,
    NULL::bigint AS session_count,
    NULL::numeric AS wall_clock_seconds,
    NULL::numeric AS stt_billed_seconds,
    NULL::numeric AS observed_talk_density_times_vad_padding,
    NULL::numeric AS implied_talk_density_if_vad_padding_1_15,
    ROUND(c.retail_vnd, 2) AS retail_vnd,
    ROUND(pc.provider_vnd, 2) AS provider_vnd,
    ROUND(c.retail_vnd / NULLIF(pc.provider_vnd, 0), 4) AS markup
FROM charged c
CROSS JOIN provider_cost pc;
