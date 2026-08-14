from collections.abc import AsyncIterator
from typing import Any

from metrics_exporter.metrics import collect_metrics


class FakeRedis:
    def __init__(self) -> None:
        self.scan_patterns: list[str] = []

    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        groups = {
            "audio:chunks": [
                {"name": b"stt-workers", "pending": 3, "lag": 12},
            ],
            "stt:results": [
                {"name": "translate-workers", "pending": 0, "lag": 4},
            ],
            "translate:results": [
                {"name": "tts-workers", "pending": 1, "lag": None},
            ],
        }
        return groups[stream]

    async def scan_iter(self, match: str, count: int = 100) -> AsyncIterator[bytes]:
        del count
        self.scan_patterns.append(match)
        fixtures = {
            "warptalk:worker:heartbeat:*": [
                b"warptalk:worker:heartbeat:stt:a",
                b"warptalk:worker:heartbeat:stt:b",
                b"warptalk:worker:heartbeat:translation:a",
            ],
            "*:dead-letter": [b"stt:dead-letter", b"tts:dead-letter"],
        }
        for key in fixtures.get(match, []):
            yield key

    async def xlen(self, stream: bytes) -> int:
        return {
            b"stt:dead-letter": 2,
            b"tts:dead-letter": 0,
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
    assert redis.scan_patterns == [
        "warptalk:worker:heartbeat:*",
        "*:dead-letter",
        # Stage-latency histograms. Pinned like the other two: an exporter that
        # silently stops scanning a pattern reports a healthy, idle pipeline.
        "warptalk:latency:*",
    ]


async def test_collect_metrics_escapes_prometheus_label_values() -> None:
    class EscapingRedis(FakeRedis):
        async def scan_iter(self, match: str, count: int = 100) -> AsyncIterator[bytes]:
            del count
            if match == "*:dead-letter":
                yield b'unsafe"stream\\name:dead-letter'

        async def xlen(self, stream: bytes) -> int:
            del stream
            return 1

    output = await collect_metrics(EscapingRedis())

    assert 'stream="unsafe\\"stream\\\\name:dead-letter"' in output
