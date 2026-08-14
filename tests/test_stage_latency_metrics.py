"""Making pipeline latency answerable without reconstructing it by hand.

WHY THIS EXISTS
    A tester reported the dubbed voice arriving 5-10 seconds late. There was no latency metric
    anywhere in Prometheus, so the answer had to be rebuilt by hand out of Redis stream entry ids
    — and it turned out to be worth having: on one sample of 43 segments the pipeline split as

        speech -> STT        p50 0.77s   p95 1.15s
        STT -> translate     p50 0.80s   p95 1.40s
        translate -> TTS     p50 1.00s   p95 8.54s     <- the whole tail lives here
        TOTAL                p50 3.07s   p95 11.44s

    Every stage worker was already computing that number and handing it to `publish_telemetry`,
    which publishes to a pub/sub channel with no subscriber. It was measured and thrown away,
    every segment, for the life of the pipeline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from metrics_exporter.metrics import collect_metrics
from shared.config import RedisSettings
from shared.redis_client import LATENCY_KEY_PREFIX, RedisStreamClient

pytestmark = pytest.mark.asyncio


def _client() -> RedisStreamClient:
    client = RedisStreamClient(RedisSettings())
    redis = MagicMock()
    pipeline = MagicMock()
    pipeline.hincrby = MagicMock()
    pipeline.expire = MagicMock()
    pipeline.execute = AsyncMock(return_value=[1, 1, 1, True])
    redis.pipeline = MagicMock(return_value=pipeline)
    redis.publish = AsyncMock()
    client._redis = redis
    client._pipeline = pipeline  # type: ignore[attr-defined]
    return client


def _buckets(client: RedisStreamClient) -> dict[str, int]:
    return {c.args[1]: c.args[2] for c in client._pipeline.hincrby.call_args_list}


# ── Recording ────────────────────────────────────────────────────────────────────────────────


async def test_an_observation_lands_in_the_bucket_that_contains_it() -> None:
    client = _client()

    await client.record_latency("tts", 1500)

    assert _buckets(client)["le:2000"] == 1


async def test_it_lands_in_the_smallest_containing_bucket_not_the_widest() -> None:
    # A histogram whose observations all landed in the widest bucket that fits would report
    # every latency as slower than it was, which is worse than no metric.
    client = _client()

    await client.record_latency("tts", 300)

    assert _buckets(client) == {"le:500": 1, "sum": 300, "count": 1}


async def test_a_latency_past_the_last_edge_lands_in_the_overflow_bucket() -> None:
    client = _client()

    await client.record_latency("tts", 45_000)

    assert _buckets(client)["le:+Inf"] == 1


async def test_the_boundary_is_inclusive() -> None:
    client = _client()

    await client.record_latency("tts", 2000)

    assert _buckets(client)["le:2000"] == 1


async def test_sum_and_count_travel_with_every_observation() -> None:
    # histogram_quantile needs the buckets; _sum and _count are what make an average and a rate
    # possible at all. Incrementing the bucket alone would give a histogram nothing can average.
    client = _client()

    await client.record_latency("stt", 900)

    assert _buckets(client)["sum"] == 900
    assert _buckets(client)["count"] == 1


async def test_the_key_carries_a_refreshed_ttl() -> None:
    # An unbounded, never-expiring Redis key is exactly what filled production to 93% this
    # morning. A metrics key must not become the next one.
    client = _client()

    await client.record_latency("tts", 100)

    key, ttl = client._pipeline.expire.call_args.args
    assert key == f"{LATENCY_KEY_PREFIX}tts"
    assert ttl > 0


async def test_a_negative_latency_is_dropped_rather_than_recorded() -> None:
    # Clock skew between the producing and consuming worker can make one. Recording it would
    # drag the sum below the truth and no bucket describes it.
    client = _client()

    await client.record_latency("tts", -50)

    client._pipeline.execute.assert_not_awaited()


async def test_recording_can_never_fail_the_pipeline_it_measures() -> None:
    client = _client()
    client._pipeline.execute = AsyncMock(side_effect=ConnectionError("redis gone"))

    await client.record_latency("tts", 100)  # must not raise


async def test_publish_telemetry_records_as_well_as_publishes() -> None:
    """The wiring. Everything above passes on a `record_latency` nothing calls.

    All three stage workers reach this method and none of them will be changed to call the new
    one, so if this link is missing the histograms stay empty forever and look exactly like a
    healthy, idle pipeline.
    """
    client = _client()

    await client.publish_telemetry("room-1", "tts", 4200)

    client._redis.publish.assert_awaited_once()
    assert _buckets(client)["le:5000"] == 1


# ── Exporting ────────────────────────────────────────────────────────────────────────────────


class _FakeRedis:
    def __init__(self, latency: dict[str, dict[str, str]]) -> None:
        self._latency = latency

    async def xinfo_groups(self, stream: str) -> list[dict[str, object]]:
        return []

    async def xlen(self, stream: object) -> int:
        return 0

    async def hgetall(self, key: object) -> dict[str, str]:
        return self._latency[str(key)]

    async def scan_iter(self, match: str, count: int = 100):  # type: ignore[no-untyped-def]
        if match.startswith(LATENCY_KEY_PREFIX):
            for key in self._latency:
                yield key
        return


async def test_exported_buckets_are_cumulative() -> None:
    """Prometheus histogram buckets count everything at or below their edge.

    Emitting the raw per-bucket counts instead would not fail — `histogram_quantile` would return
    a confident wrong number, which is the worse outcome.
    """
    body = await collect_metrics(
        _FakeRedis(
            {
                f"{LATENCY_KEY_PREFIX}tts": {
                    "le:500": "2",
                    "le:1000": "3",
                    "le:+Inf": "1",
                    "sum": "12000",
                    "count": "6",
                }
            }
        )
    )

    assert 'warptalk_stage_latency_ms_bucket{stage="tts",le="500"} 2' in body
    assert 'warptalk_stage_latency_ms_bucket{stage="tts",le="1000"} 5' in body
    assert 'warptalk_stage_latency_ms_bucket{stage="tts",le="2000"} 5' in body
    assert 'warptalk_stage_latency_ms_bucket{stage="tts",le="+Inf"} 6' in body


async def test_count_agrees_with_the_overflow_bucket() -> None:
    # Prometheus rejects a histogram whose _count disagrees with its +Inf bucket.
    body = await collect_metrics(
        _FakeRedis({f"{LATENCY_KEY_PREFIX}stt": {"le:250": "4", "sum": "800", "count": "4"}})
    )

    assert 'warptalk_stage_latency_ms_bucket{stage="stt",le="+Inf"} 4' in body
    assert 'warptalk_stage_latency_ms_count{stage="stt"} 4' in body


async def test_the_stage_label_comes_from_the_key() -> None:
    body = await collect_metrics(
        _FakeRedis({f"{LATENCY_KEY_PREFIX}tts_synthesis": {"le:8000": "1", "sum": "7000"}})
    )

    assert 'stage="tts_synthesis"' in body


async def test_a_pipeline_with_no_observations_still_emits_valid_output() -> None:
    body = await collect_metrics(_FakeRedis({}))

    assert "warptalk_stage_latency_ms" in body
    assert body.endswith("\n")


async def test_the_tts_worker_records_its_own_synthesis_time() -> None:
    """`synthesis_latency_ms` was computed and only ever attached to a published message.

    It is the stage that owns the tail — p95 8.54s against 1.4s for translation — and it is kept
    apart from the cumulative pipeline number so a slow Cartesia call and a queue building behind
    the per-key lock read as two different things.
    """
    import inspect

    from tts_worker.worker import TTSWorker

    source = inspect.getsource(TTSWorker._synthesize_and_publish)

    assert 'record_latency("tts_synthesis", synthesis_latency_ms)' in source
