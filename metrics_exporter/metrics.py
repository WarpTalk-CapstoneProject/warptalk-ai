from collections.abc import AsyncIterator, Awaitable
from typing import Any, Protocol

from redis.exceptions import ResponseError

from shared.redis_client import LATENCY_BUCKETS_MS, LATENCY_KEY_PREFIX, is_per_room_stream

# The three hops of the live pipeline. These are reported even when the stream is absent, so the
# spine of the system always has a series; everything else is discovered.
CORE_STREAM_GROUPS = (
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
    # Running in production and absent from this list, so a dead suggestion worker was invisible
    # to WarpTalkAiWorkerMissing — the alert only ever asks about workers it was told to expect.
    "suggestion",
)

# BOTH spellings of a parked-message stream.
#
# The Python workers publish to `<stream>:dead-letter`; the .NET side uses `<stream>:dlq`, and
# only the first was matched — so `translationRoom:system_events:dlq`, which exists in production,
# had no series and could not raise WarpTalkDeadLetterPresent. Half the platform's parked
# messages were outside the one alert built to find them.
DEAD_LETTER_SUFFIXES = (":dead-letter", ":dlq")


class RedisMetricsClient(Protocol):
    def xinfo_groups(self, stream: str) -> Awaitable[list[dict[str, Any]]]: ...

    def scan_iter(
        self,
        match: str,
        count: int = 100,
        _type: str | None = None,
    ) -> AsyncIterator[Any]: ...

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


async def _global_streams(redis: RedisMetricsClient) -> list[str]:
    """Every permanent stream in Redis, per-room ones excluded.

    Discovered rather than listed. The hardcoded list this replaces named three of the roughly
    thirteen consumer groups the platform runs, so `redis_stream_group_lag` had no series at all
    for the other ten — and the lag alert that reads it was loaded, correct, and incapable of
    firing for any of them. A list in this file drifts every time a worker is added; asking Redis
    does not.

    Per-room streams are skipped because there is one set per meeting: including them would put a
    room id in a label and make the series count grow without bound. Their groups are the same
    groups, on the same workers, already counted here from the global stream.
    """
    streams: list[str] = []
    async for key in redis.scan_iter(match="*", count=200, _type="stream"):
        decoded = _decode(key)
        if is_per_room_stream(decoded):
            continue
        streams.append(decoded)
    return sorted(streams)


async def collect_metrics(redis: RedisMetricsClient) -> str:
    lag_lines: list[str] = []
    pending_lines: list[str] = []
    consumer_lines: list[str] = []
    group_count_lines: list[str] = []
    dead_letter_lines: list[str] = []

    streams = await _global_streams(redis)
    # Absent from Redis is not absent from the report. A core stream that has been deleted still
    # gets a 0 lag line, because a vanished series reads as "no data" on a dashboard and as
    # nothing at all in an alert expression.
    seen: set[tuple[str, str]] = set()

    for stream in streams:
        groups = await _groups(redis, stream)
        group_count_lines.append(f'redis_stream_groups{{stream="{_label(stream)}"}} {len(groups)}')
        for group in groups:
            name = _decode(_field(group, "name", ""))
            labels = f'stream="{_label(stream)}",group="{_label(name)}"'
            lag = int(_field(group, "lag", 0) or 0)
            pending = int(_field(group, "pending", 0) or 0)
            # Consumers is NOT a liveness signal: Redis keeps a consumer registered after its
            # process dies, so this counts names ever seen, not readers currently attached. What
            # it does catch is zero — a group created by a producer that nothing was ever wired
            # to read, which is how the WarpBot consumer sat at pending 0 with lag climbing and
            # looked merely idle.
            consumers = int(_field(group, "consumers", 0) or 0)
            lag_lines.append(f"redis_stream_group_lag{{{labels}}} {lag}")
            pending_lines.append(f"redis_stream_group_messages_pending{{{labels}}} {pending}")
            consumer_lines.append(f"redis_stream_group_consumers{{{labels}}} {consumers}")
            seen.add((stream, name))

        if stream.endswith(DEAD_LETTER_SUFFIXES):
            dead_letter_lines.append(
                f'redis_stream_length{{stream="{_label(stream)}"}} {int(await redis.xlen(stream))}'
            )

    for core_stream, core_group in CORE_STREAM_GROUPS:
        if (core_stream, core_group) in seen:
            continue
        labels = f'stream="{_label(core_stream)}",group="{_label(core_group)}"'
        lag_lines.append(f"redis_stream_group_lag{{{labels}}} 0")
        pending_lines.append(f"redis_stream_group_messages_pending{{{labels}}} 0")
        consumer_lines.append(f"redis_stream_group_consumers{{{labels}}} 0")

    lines = [
        "# HELP redis_stream_group_lag Undelivered entries for a Redis Stream consumer group.",
        "# TYPE redis_stream_group_lag gauge",
        *lag_lines,
        "# HELP redis_stream_group_messages_pending Entries pending acknowledgement.",
        "# TYPE redis_stream_group_messages_pending gauge",
        *pending_lines,
        "# HELP redis_stream_group_consumers Consumers currently registered in the group.",
        "# TYPE redis_stream_group_consumers gauge",
        *consumer_lines,
        "# HELP redis_stream_groups Consumer groups present on a permanent WarpTalk stream.",
        "# TYPE redis_stream_groups gauge",
        *group_count_lines,
    ]

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
            *dead_letter_lines,
        ]
    )

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
