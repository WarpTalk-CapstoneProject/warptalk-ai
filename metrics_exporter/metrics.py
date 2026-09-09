"""Prometheus exporter for the Redis Streams the WarpTalk pipeline runs on.

WHAT IS EXPORTED
    For every consumer group on every reported stream: `redis_stream_group_lag`,
    `redis_stream_group_messages_pending` and `redis_stream_group_consumers`; per stream,
    `redis_stream_groups`; per dead-letter stream, `redis_stream_length`. Plus the worker
    heartbeat counts and the stage-latency histograms the workers write into Redis.

    Groups are DISCOVERED, with XINFO GROUPS, on every scrape — never listed. A group that
    appears in Redis is in the next scrape without a code change, and a group that is destroyed
    is gone from it: the exporter keeps no state between scrapes, so there is no stale series
    to keep an alert firing, or to keep one quiet.

    WT-391. This module used to carry three (stream, group) pairs. Production runs about
    eighteen groups on the global streams — `stt:results` alone fans out to six — so
    `redis_stream_group_lag` had no series for most of them and WarpTalkAiStreamLag, a loaded
    and correct alert, could not fire for any of them. `billing-stt-workers` sat dead at lag 949
    on `stt:results` for four days, pinned the stream's trim floor, and nothing said so.

WHICH STREAMS ARE ASKED — the cardinality bound
    Global streams, always. `METRICS_GLOBAL_STREAMS` (comma-separated; default
    `shared.config.DEFAULT_GLOBAL_STREAMS`, every global stream this repo and the backend
    publish to) names them, and a named stream is reported whether or not its key exists: an
    absent one gets `redis_stream_groups{stream="..."} 0`, and the three core pipeline pairs get
    a 0 lag line, because an absent series reads as "no data" on a dashboard and as nothing at
    all in an alert expression. Any further non-room stream present in Redis is reported too,
    so a stream nobody listed — the backend's `:dlq` streams were the example — is not invisible.

    Per-room streams, only with `METRICS_EXPORT_PER_ROOM_STREAMS=true`. `BaseWorker.publish`
    writes every message to `<stream>:<roomId>` as well as `<stream>`, so there is one set of
    streams per meeting ever held; labelling them puts a room id in a label and grows the series
    count without bound. Their groups are the same groups, on the same workers, already counted
    on the global stream. The flag is for a local look at one room, not for production.
"""

import uuid
from collections.abc import AsyncIterator, Awaitable
from typing import Any, Protocol

from redis.exceptions import ResponseError

from shared.config import MetricsSettings
from shared.redis_client import LATENCY_BUCKETS_MS, LATENCY_KEY_PREFIX

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


def is_room_scoped_stream(stream: str) -> bool:
    """Whether this stream belongs to one room (or one conversation, one workspace).

    ANY segment that parses as a UUID, not only the last one. `shared.redis_client.
    is_per_room_stream` — the predicate that decides which streams may EXPIRE — looks at the
    last segment only, and defaults to "permanent" on purpose: being wrong that way costs disk,
    being wrong the other way deleted `translate:results` and its consumer groups (WT-402).

    Here the safe direction is the opposite. This predicate decides what gets a LABEL, and
    over-including a stream costs one missing series while under-including one puts an
    unbounded id into the label set. So `meeting:<uuid>:events` is room-scoped to the exporter
    even though the expiry rule would (correctly) leave it alone.
    """
    for segment in stream.split(":"):
        try:
            uuid.UUID(segment)
        except ValueError:
            continue
        return True
    return False


async def _streams_to_report(redis: RedisMetricsClient, settings: MetricsSettings) -> list[str]:
    """The configured global streams, plus whatever else Redis holds that is not room-scoped.

    Configured first so a global stream is reported while its key is absent; discovered second
    so a stream nobody configured is reported while it exists. Per-room streams are dropped
    from the discovered set unless explicitly enabled — see the module docstring.
    """
    streams: set[str] = set(settings.global_stream_names())
    async for key in redis.scan_iter(match="*", count=200, _type="stream"):
        decoded = _decode(key)
        if is_room_scoped_stream(decoded) and not settings.export_per_room_streams:
            continue
        streams.add(decoded)
    return sorted(streams)


async def collect_metrics(
    redis: RedisMetricsClient,
    settings: MetricsSettings | None = None,
) -> str:
    """One scrape. Stateless: every line is computed from what Redis holds right now."""
    settings = settings or MetricsSettings()
    lag_lines: list[str] = []
    pending_lines: list[str] = []
    consumer_lines: list[str] = []
    group_count_lines: list[str] = []
    dead_letter_lines: list[str] = []

    streams = await _streams_to_report(redis, settings)
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
        "# HELP redis_stream_groups Consumer groups present on a reported WarpTalk stream.",
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
