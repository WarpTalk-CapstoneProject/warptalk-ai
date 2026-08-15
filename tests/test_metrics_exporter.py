from collections.abc import AsyncIterator
from typing import Any

from metrics_exporter.metrics import collect_metrics

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
    "stt:dead-letter": [],
    "tts:dead-letter": [],
    "translationRoom:system_events:dlq": [],
}


class FakeRedis:
    def __init__(self) -> None:
        self.scan_calls: list[tuple[str, str | None]] = []

    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        return GROUPS[stream]

    async def scan_iter(
        self,
        match: str,
        count: int = 100,
        _type: str | None = None,
    ) -> AsyncIterator[bytes]:
        del count
        self.scan_calls.append((match, _type))
        if _type == "stream":
            for key in STREAM_KEYS:
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


async def test_collect_metrics_reports_lag_pending_heartbeats_and_dead_letters() -> None:
    redis = FakeRedis()
    output = await collect_metrics(redis)

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
    output = await collect_metrics(FakeRedis())

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
    output = await collect_metrics(FakeRedis())

    assert (
        'redis_stream_group_consumers{stream="stt:results",group="billing-stt-workers"} 0' in output
    )
    assert 'redis_stream_group_consumers{stream="audio:chunks",group="stt-workers"} 2' in output


async def test_per_room_streams_are_not_labelled() -> None:
    """One set of streams per meeting — labelling them would grow the series count per room."""
    output = await collect_metrics(FakeRedis())

    assert ROOM_ID not in output


async def test_core_pipeline_groups_report_zero_when_the_stream_is_gone() -> None:
    """A deleted stream must not delete the series. An absent series cannot trip an alert."""

    class EmptyRedis(FakeRedis):
        async def scan_iter(
            self,
            match: str,
            count: int = 100,
            _type: str | None = None,
        ) -> AsyncIterator[bytes]:
            del count, match, _type
            return
            yield  # pragma: no cover - makes this an async generator

    output = await collect_metrics(EmptyRedis())

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

    output = await collect_metrics(EscapingRedis())

    assert 'stream="unsafe\\"stream\\\\name:dead-letter"' in output
