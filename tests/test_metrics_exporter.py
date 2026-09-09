"""The Redis Streams exporter: what it reports, and — WT-391 — how it decides what to ask.

Groups are discovered with XINFO GROUPS on every scrape, so the exporter has no list to fall
behind. Every test here drives `collect_metrics` against an in-memory Redis double whose
groups and keys can be changed BETWEEN scrapes, which is the shape of the two failures the
ticket describes: a group created after the exporter started (invisible) and a group deleted
after it started (a stale series that keeps an alert firing, or keeps one quiet).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from redis.exceptions import ResponseError

from metrics_exporter.metrics import collect_metrics, is_room_scoped_stream
from shared.config import DEFAULT_GLOBAL_STREAMS, MetricsSettings

ROOM_ID = "5f2b1c8e-9a4d-4f77-8c31-2b0e6d9a1f43"

# Every permanent stream Redis would report, plus one per-room stream and one dead-letter stream.
STREAM_KEYS = [
    b"audio:chunks",
    b"stt:results",
    b"translate:results",
    b"tts:results",
    b"voice:clone_requests",
    f"stt:results:{ROOM_ID}".encode(),
    b"stt:dead-letter",
    b"tts:dead-letter",
    # The .NET spelling. Only ":dead-letter" was matched, so every backend DLQ — this one
    # exists in production — was outside WarpTalkDeadLetterPresent.
    b"translationRoom:system_events:dlq",
]

GROUPS: dict[str, list[dict[str, Any]]] = {
    "audio:chunks": [{"name": b"stt-workers", "pending": 3, "lag": 12, "consumers": 2}],
    "stt:results": [
        {"name": "translate-workers", "pending": 0, "lag": 4, "consumers": 1},
        # A second group on the same stream. The hardcoded list this replaces could not see it.
        {"name": "billing-stt-workers", "pending": 9, "lag": 41, "consumers": 0},
    ],
    "translate:results": [{"name": "tts-workers", "pending": 1, "lag": None, "consumers": 1}],
    "tts:results": [{"name": "gateway-consumers", "pending": 0, "lag": 0, "consumers": 3}],
    "voice:clone_requests": [{"name": "tts-audio-workers", "pending": 0, "lag": 0, "consumers": 1}],
    # The per-room copy carries the same group as its global stream — that is why it is not
    # labelled by default: its series would say nothing the global one does not.
    f"stt:results:{ROOM_ID}": [
        {"name": "translate-workers", "pending": 0, "lag": 2, "consumers": 1}
    ],
    "stt:dead-letter": [],
    "tts:dead-letter": [],
    "translationRoom:system_events:dlq": [],
}

# The exporter is configured by environment in production. Tests pin the configuration so an
# operator's shell — or a future default — cannot change what a test is asserting about.
DEFAULT_SETTINGS = MetricsSettings(
    global_streams="audio:chunks,stt:results,translate:results,tts:results",
    export_per_room_streams=False,
)


class FakeRedis:
    """The four calls `collect_metrics` makes, over state a test can change between scrapes."""

    def __init__(self) -> None:
        self.groups: dict[str, list[dict[str, Any]]] = {
            stream: [dict(group) for group in groups] for stream, groups in GROUPS.items()
        }
        self.stream_keys: list[bytes] = list(STREAM_KEYS)
        self.scan_calls: list[tuple[str, str | None]] = []

    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        if stream not in self.groups:
            # What Redis answers for XINFO GROUPS on a key that does not exist. The exporter
            # reports configured streams whether or not they exist, so it has to survive this.
            raise ResponseError("ERR no such key")
        return self.groups[stream]

    async def scan_iter(
        self,
        match: str,
        count: int = 100,
        _type: str | None = None,
    ) -> AsyncIterator[bytes]:
        del count
        self.scan_calls.append((match, _type))
        if _type == "stream":
            for key in self.stream_keys:
                yield key
            return
        fixtures = {
            "warptalk:worker:heartbeat:*": [
                b"warptalk:worker:heartbeat:stt:a",
                b"warptalk:worker:heartbeat:stt:b",
                b"warptalk:worker:heartbeat:translation:a",
            ],
        }
        for key in fixtures.get(match, []):
            yield key

    async def xlen(self, stream: Any) -> int:
        return {
            "stt:dead-letter": 2,
            "tts:dead-letter": 0,
            "translationRoom:system_events:dlq": 7,
        }[stream]

    async def hgetall(self, key: Any) -> dict[Any, Any]:
        del key
        return {}


class EmptyRedis(FakeRedis):
    """A Redis with no keys at all — what the exporter sees after a flush, or on a fresh box."""

    def __init__(self) -> None:
        super().__init__()
        self.groups = {}
        self.stream_keys = []


async def test_collect_metrics_reports_lag_pending_heartbeats_and_dead_letters() -> None:
    redis = FakeRedis()
    output = await collect_metrics(redis, DEFAULT_SETTINGS)

    assert 'redis_stream_group_lag{stream="audio:chunks",group="stt-workers"} 12' in output
    assert (
        'redis_stream_group_messages_pending{stream="audio:chunks",group="stt-workers"} 3' in output
    )
    assert 'redis_stream_group_lag{stream="translate:results",group="tts-workers"} 0' in output
    assert 'redis_keys_count{key="warptalk:worker:heartbeat:stt:*"} 2' in output
    assert 'redis_stream_length{stream="stt:dead-letter"} 2' in output
    assert 'redis_stream_length{stream="tts:dead-letter"} 0' in output
    assert 'redis_stream_length{stream="translationRoom:system_events:dlq"} 7' in output
    assert redis.scan_calls == [
        ("*", "stream"),
        ("warptalk:worker:heartbeat:*", None),
        # Stage-latency histograms. Pinned like the other two: an exporter that
        # silently stops scanning a pattern reports a healthy, idle pipeline.
        ("warptalk:latency:*", None),
    ]


async def test_collect_metrics_covers_groups_no_hardcoded_list_named() -> None:
    """The point of the change: groups outside the three-hop spine get series too.

    `gateway-consumers` is the group whose disappearance took every translation, dub and
    assistant reply off the platform while the gateway reported healthy (WT-402). It was not in
    the list the exporter used to carry, so nothing about it reached Prometheus at all.
    """
    output = await collect_metrics(FakeRedis(), DEFAULT_SETTINGS)

    assert 'redis_stream_group_lag{stream="tts:results",group="gateway-consumers"} 0' in output
    assert 'redis_stream_group_lag{stream="stt:results",group="billing-stt-workers"} 41' in output
    assert (
        'redis_stream_group_lag{stream="voice:clone_requests",group="tts-audio-workers"} 0'
        in output
    )


async def test_zero_consumers_is_reported_for_a_group_nobody_ever_read() -> None:
    """Zero registered consumers means the group was never read, not that its reader died.

    Redis keeps a consumer registered after the process behind it exits, so this cannot be a
    liveness check. Zero is still worth a series: it is the exact shape of a stream that has a
    producer and no wiring on the other end, which reads as "idle" on every other metric.
    """
    output = await collect_metrics(FakeRedis(), DEFAULT_SETTINGS)

    assert (
        'redis_stream_group_consumers{stream="stt:results",group="billing-stt-workers"} 0' in output
    )
    assert 'redis_stream_group_consumers{stream="audio:chunks",group="stt-workers"} 2' in output


# --- WT-391: discovery, not a list ------------------------------------------------------------


async def test_a_group_created_after_startup_is_in_the_next_scrape() -> None:
    """A worker deployed after the exporter started needs no exporter change, and no restart.

    `suggestion-workers` is real: it ran in production on `stt:results` and had no series,
    because it was not in the three-pair list this file used to carry.
    """
    redis = FakeRedis()
    before = await collect_metrics(redis, DEFAULT_SETTINGS)
    assert 'group="suggestion-workers"' not in before

    redis.groups["stt:results"].append(
        {"name": "suggestion-workers", "pending": 1, "lag": 7, "consumers": 1}
    )
    after = await collect_metrics(redis, DEFAULT_SETTINGS)

    assert 'redis_stream_group_lag{stream="stt:results",group="suggestion-workers"} 7' in after
    assert (
        'redis_stream_group_messages_pending{stream="stt:results",group="suggestion-workers"} 1'
        in after
    )
    assert 'redis_stream_groups{stream="stt:results"} 3' in after


async def test_a_destroyed_group_leaves_no_stale_series() -> None:
    """XGROUP DESTROY must take the series with it.

    `billing-stt-workers` is the production case: a group nothing reads any more, sitting at
    lag 949 for four days and pinning the stream's trim floor. Once an operator destroys it, a
    series that kept reporting 949 would keep WarpTalkAiStreamLag firing on a group that no
    longer exists — and a series stuck at its last value would hide the next group that dies
    the same way.
    """
    redis = FakeRedis()
    before = await collect_metrics(redis, DEFAULT_SETTINGS)
    assert 'redis_stream_group_lag{stream="stt:results",group="billing-stt-workers"} 41' in before

    redis.groups["stt:results"] = [
        group for group in redis.groups["stt:results"] if group["name"] != "billing-stt-workers"
    ]
    after = await collect_metrics(redis, DEFAULT_SETTINGS)

    assert 'group="billing-stt-workers"' not in after
    assert 'redis_stream_groups{stream="stt:results"} 1' in after
    # The neighbour on the same stream is unaffected.
    assert 'redis_stream_group_lag{stream="stt:results",group="translate-workers"} 4' in after


async def test_a_deleted_stream_takes_every_one_of_its_groups_out_of_the_report() -> None:
    """The whole key going is the same rule as one group going: nothing is remembered."""
    redis = FakeRedis()
    before = await collect_metrics(redis, DEFAULT_SETTINGS)
    assert 'group="tts-audio-workers"' in before

    redis.stream_keys.remove(b"voice:clone_requests")
    del redis.groups["voice:clone_requests"]
    after = await collect_metrics(redis, DEFAULT_SETTINGS)

    assert "voice:clone_requests" not in after


async def test_per_room_streams_are_not_labelled() -> None:
    """One set of streams per meeting — labelling them would grow the series count per room."""
    output = await collect_metrics(FakeRedis(), DEFAULT_SETTINGS)

    assert ROOM_ID not in output


async def test_per_room_streams_are_reported_only_behind_the_explicit_flag() -> None:
    """`METRICS_EXPORT_PER_ROOM_STREAMS=true` is the one way a room id reaches a label.

    The default is the cardinality bound. The flag exists for a local look at one room, and
    what it exposes is the per-room copy's own groups — the same names as the global stream,
    which is exactly why they add nothing on a dashboard.
    """
    off = await collect_metrics(FakeRedis(), MetricsSettings(export_per_room_streams=False))
    on = await collect_metrics(FakeRedis(), MetricsSettings(export_per_room_streams=True))

    assert ROOM_ID not in off
    assert (
        f'redis_stream_group_lag{{stream="stt:results:{ROOM_ID}",group="translate-workers"}} 2'
        in on
    )
    assert f'redis_stream_groups{{stream="stt:results:{ROOM_ID}"}} 1' in on


async def test_the_flag_defaults_off() -> None:
    # Documented in the module docstring, README and .env.example; pinned here so a default
    # flipped by accident fails a test rather than growing production's series count per meeting.
    assert MetricsSettings().export_per_room_streams is False


def test_room_scoped_is_any_uuid_segment_not_only_the_last() -> None:
    """The expiry predicate looks at the LAST segment and defaults to "permanent" — the safe
    direction when being wrong deletes a stream (WT-402). Here the safe direction is the
    opposite: being wrong puts an unbounded id in a label, so ANY uuid segment is room-scoped.
    """
    assert is_room_scoped_stream(f"stt:results:{ROOM_ID}")
    assert is_room_scoped_stream(f"meeting:{ROOM_ID}:events")
    assert is_room_scoped_stream(f"voice:clone:state:{ROOM_ID}")
    assert not is_room_scoped_stream("stt:results")
    assert not is_room_scoped_stream("translationRoom:system_events:dlq")
    assert not is_room_scoped_stream("stt:results:latest")
    assert not is_room_scoped_stream("")


# --- WT-391: the configured stream list ---------------------------------------------------------


async def test_a_configured_global_stream_is_reported_while_its_key_is_absent() -> None:
    """Configured means "always has a series", including before the first message.

    An absent series reads as "no data" on a dashboard and as nothing at all in an alert
    expression; a `redis_stream_groups ... 0` line is a fact an alert can act on.
    """
    settings = MetricsSettings(global_streams="stt:results,voice:auto_clone_ready")
    output = await collect_metrics(FakeRedis(), settings)

    assert 'redis_stream_groups{stream="voice:auto_clone_ready"} 0' in output
    assert 'redis_stream_groups{stream="stt:results"} 2' in output


async def test_a_stream_nobody_configured_is_still_reported_while_it_exists() -> None:
    """The configured list is a floor, not a ceiling: discovery still covers the rest."""
    settings = MetricsSettings(global_streams="stt:results")
    output = await collect_metrics(FakeRedis(), settings)

    assert (
        'redis_stream_group_lag{stream="voice:clone_requests",group="tts-audio-workers"} 0'
        in output
    )
    assert 'redis_stream_length{stream="translationRoom:system_events:dlq"} 7' in output


def test_the_configured_list_is_trimmed_and_deduplicated_in_order() -> None:
    settings = MetricsSettings(global_streams=" stt:results , tts:results,,stt:results, ")

    assert settings.global_stream_names() == ("stt:results", "tts:results")


def test_the_default_list_names_the_four_pipeline_streams_and_nothing_room_scoped() -> None:
    names = MetricsSettings().global_stream_names()

    assert names == DEFAULT_GLOBAL_STREAMS
    for stream in ("audio:chunks", "stt:results", "translate:results", "tts:results"):
        assert stream in names
    # A room id in the default list would put it in a label on every scrape, everywhere.
    assert not any(is_room_scoped_stream(name) for name in names)
    assert len(set(names)) == len(names)


async def test_every_default_global_stream_has_a_series_on_an_empty_redis() -> None:
    """Fresh box, or a Redis that was flushed: the spine is still there to alert on."""
    output = await collect_metrics(EmptyRedis(), MetricsSettings())

    for stream in DEFAULT_GLOBAL_STREAMS:
        assert f'redis_stream_groups{{stream="{stream}"}} 0' in output


async def test_core_pipeline_groups_report_zero_when_the_stream_is_gone() -> None:
    """A deleted stream must not delete the series. An absent series cannot trip an alert."""
    output = await collect_metrics(EmptyRedis(), DEFAULT_SETTINGS)

    assert 'redis_stream_group_lag{stream="audio:chunks",group="stt-workers"} 0' in output
    assert 'redis_stream_group_lag{stream="stt:results",group="translate-workers"} 0' in output
    assert 'redis_stream_group_lag{stream="translate:results",group="tts-workers"} 0' in output
    assert 'redis_stream_group_consumers{stream="audio:chunks",group="stt-workers"} 0' in output


async def test_collect_metrics_escapes_prometheus_label_values() -> None:
    class EscapingRedis(FakeRedis):
        async def scan_iter(
            self,
            match: str,
            count: int = 100,
            _type: str | None = None,
        ) -> AsyncIterator[bytes]:
            del count, match
            if _type == "stream":
                yield b'unsafe"stream\\name:dead-letter'

        async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
            del stream
            return []

        async def xlen(self, stream: Any) -> int:
            del stream
            return 1

    output = await collect_metrics(EscapingRedis(), DEFAULT_SETTINGS)

    assert 'stream="unsafe\\"stream\\\\name:dead-letter"' in output


def test_uuid_segments_are_the_ones_the_pipeline_actually_mints() -> None:
    # BaseWorker.publish suffixes the stream with a room id from the room service; uuid4 and
    # uuid7 both parse, so a future id version does not silently start growing labels.
    assert is_room_scoped_stream(f"stt:results:{uuid.uuid4()}")
    assert is_room_scoped_stream("stt:results:01a00547-367f-7deb-88c0-c097396e3a62")
