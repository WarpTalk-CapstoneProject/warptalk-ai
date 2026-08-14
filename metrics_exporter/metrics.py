from collections.abc import AsyncIterator, Awaitable
from typing import Any, Protocol

from redis.exceptions import ResponseError

from shared.redis_client import LATENCY_BUCKETS_MS, LATENCY_KEY_PREFIX

STREAM_GROUPS = (
    ("audio:chunks", "stt-workers"),
    ("stt:results", "translate-workers"),
    ("translate:results", "tts-workers"),
)
WORKER_HEARTBEATS = (
    "stt",
    "translation",
    "tts",
    "assistant",
    "assistant-chat",
    "embedding",
    "embedding-search",
    "billing",
    "livekit_ingress",
    "security",
)


class RedisMetricsClient(Protocol):
    def xinfo_groups(self, stream: str) -> Awaitable[list[dict[str, Any]]]: ...

    def scan_iter(self, match: str, count: int = 100) -> AsyncIterator[Any]: ...

    def xlen(self, stream: Any) -> Awaitable[int]: ...

    def hgetall(self, key: Any) -> Awaitable[dict[Any, Any]]: ...


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _label(value: Any) -> str:
    return _decode(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _field(group: dict[Any, Any], name: str, default: Any = 0) -> Any:
    return group.get(name, group.get(name.encode(), default))


async def _groups(redis: RedisMetricsClient, stream: str) -> list[dict[Any, Any]]:
    try:
        return await redis.xinfo_groups(stream)
    except ResponseError as error:
        if "no such key" in str(error).lower():
            return []
        raise


async def collect_metrics(redis: RedisMetricsClient) -> str:
    lines = [
        "# HELP redis_stream_group_lag Undelivered entries for a Redis Stream consumer group.",
        "# TYPE redis_stream_group_lag gauge",
        "# HELP redis_stream_group_messages_pending Entries pending acknowledgement.",
        "# TYPE redis_stream_group_messages_pending gauge",
    ]

    for stream, expected_group in STREAM_GROUPS:
        matching = None
        for group in await _groups(redis, stream):
            if _decode(_field(group, "name", "")) == expected_group:
                matching = group
                break
        lag = int(_field(matching or {}, "lag", 0) or 0)
        pending = int(_field(matching or {}, "pending", 0) or 0)
        labels = f'stream="{_label(stream)}",group="{_label(expected_group)}"'
        lines.append(f"redis_stream_group_lag{{{labels}}} {lag}")
        lines.append(f"redis_stream_group_messages_pending{{{labels}}} {pending}")

    lines.extend(
        [
            "# HELP redis_keys_count Matching live WarpTalk worker heartbeat keys.",
            "# TYPE redis_keys_count gauge",
        ]
    )
    heartbeat_counts = {worker: 0 for worker in WORKER_HEARTBEATS}
    heartbeat_prefix = "warptalk:worker:heartbeat:"
    async for heartbeat_key in redis.scan_iter(
        match=f"{heartbeat_prefix}*",
        count=100,
    ):
        decoded_key = _decode(heartbeat_key)
        if not decoded_key.startswith(heartbeat_prefix):
            continue
        worker = decoded_key.removeprefix(heartbeat_prefix).rsplit(":", 1)[0]
        if worker in heartbeat_counts:
            heartbeat_counts[worker] += 1

    for worker in WORKER_HEARTBEATS:
        pattern = f"warptalk:worker:heartbeat:{worker}:*"
        lines.append(f'redis_keys_count{{key="{_label(pattern)}"}} {heartbeat_counts[worker]}')

    lines.extend(
        [
            "# HELP redis_stream_length Current length of a WarpTalk dead-letter stream.",
            "# TYPE redis_stream_length gauge",
        ]
    )
    async for stream_key in redis.scan_iter(match="*:dead-letter", count=100):
        length = await redis.xlen(stream_key)
        lines.append(f'redis_stream_length{{stream="{_label(stream_key)}"}} {int(length)}')

    lines.extend(await _latency_histograms(redis))

    return "\n".join(lines) + "\n"


async def _latency_histograms(redis: RedisMetricsClient) -> list[str]:
    """Turn the workers' raw bucket counts into Prometheus histograms.

    The workers have no HTTP server for Prometheus to scrape, so each observation is an HINCRBY
    into `warptalk:latency:{stage}` and this — the one process that already answers /metrics —
    reads them back. Accumulating the buckets happens here so the hot path stays three
    increments and knows nothing about Prometheus's text format.

    Every stage worker already computed its latency and published it to a pub/sub channel with
    no subscriber, so when a tester reported a 5-10s dub there was no metric anywhere to say
    which stage it was. These are those same numbers, kept.
    """
    lines = [
        "# HELP warptalk_stage_latency_ms Pipeline latency observed at each stage.",
        "# TYPE warptalk_stage_latency_ms histogram",
    ]
    async for key in redis.scan_iter(match=f"{LATENCY_KEY_PREFIX}*", count=100):
        stage = _decode(key).removeprefix(LATENCY_KEY_PREFIX)
        raw = await redis.hgetall(key)
        fields = {_decode(k): _decode(v) for k, v in raw.items()}

        # Cumulative, as the format requires: each bucket counts everything at or below its
        # edge. Emitting the raw per-bucket counts would make histogram_quantile return
        # nonsense rather than fail, which is the worse kind of wrong.
        running = 0
        for edge in LATENCY_BUCKETS_MS:
            running += int(fields.get(f"le:{edge}", 0) or 0)
            lines.append(
                f'warptalk_stage_latency_ms_bucket{{stage="{_label(stage)}",le="{edge}"}} {running}'
            )
        running += int(fields.get("le:+Inf", 0) or 0)
        lines.append(
            f'warptalk_stage_latency_ms_bucket{{stage="{_label(stage)}",le="+Inf"}} {running}'
        )
        lines.append(
            f'warptalk_stage_latency_ms_sum{{stage="{_label(stage)}"}} '
            f"{int(fields.get('sum', 0) or 0)}"
        )
        # _count must equal the +Inf bucket. Reading it from its own field rather than reusing
        # `running` would let the two disagree if a write landed between the increments.
        lines.append(f'warptalk_stage_latency_ms_count{{stage="{_label(stage)}"}} {running}')
    return lines
