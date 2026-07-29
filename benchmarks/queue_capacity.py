"""Provider-free Redis Streams backpressure benchmark for WarpTalk AI workers."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

from shared.config import RedisSettings
from shared.redis_client import RedisStreamClient

LANGUAGES = ("vi", "en", "ja", "ko", "zh", "fr", "de", "es")


@dataclass
class Measurements:
    active: int = 0
    peak_active: int = 0
    peak_pending: int = 0
    peak_lag: int = 0
    processed: int = 0


async def run(message_count: int, concurrency: int, handler_delay_ms: int) -> dict[str, Any]:
    if message_count < len(LANGUAGES):
        raise ValueError(f"message_count must be at least {len(LANGUAGES)}")
    if concurrency <= 0 or handler_delay_ms <= 0:
        raise ValueError("concurrency and handler_delay_ms must be positive")

    client = RedisStreamClient(RedisSettings())
    stream = f"warptalk:benchmark:queue:{uuid.uuid4()}"
    group = "capacity-benchmark"
    measurements = Measurements()
    observed_languages: set[str] = set()
    monitoring_done = asyncio.Event()

    await client.connect()
    try:
        for index in range(message_count):
            await client.publish(
                stream,
                {
                    "sequence": index,
                    "target_language": LANGUAGES[index % len(LANGUAGES)],
                },
            )

        async def handler(_message_id: bytes, data: dict[bytes, bytes]) -> None:
            measurements.active += 1
            measurements.peak_active = max(measurements.peak_active, measurements.active)
            observed_languages.add(data[b"target_language"].decode())
            try:
                await asyncio.sleep(handler_delay_ms / 1000)
                measurements.processed += 1
            finally:
                measurements.active -= 1

        async def monitor_queue() -> None:
            while not monitoring_done.is_set():
                groups = await client.redis.xinfo_groups(stream)
                if groups:
                    group_state = cast(dict[Any, Any], groups[0])
                    pending_raw = group_state.get("pending", group_state.get(b"pending", 0))
                    lag_raw = group_state.get("lag", group_state.get(b"lag", 0))
                    pending = int(cast(int | str | bytes, pending_raw or 0))
                    lag = int(cast(int | str | bytes, lag_raw or 0))
                    measurements.peak_pending = max(measurements.peak_pending, pending)
                    measurements.peak_lag = max(measurements.peak_lag, lag)
                await asyncio.sleep(0.002)

        started = time.perf_counter()
        monitor_task = asyncio.create_task(monitor_queue())
        try:
            await client.consume_concurrent(
                stream,
                group,
                handler,
                consumer="capacity-benchmark-1",
                block_ms=50,
                count=concurrency,
                concurrency=concurrency,
            )
        finally:
            monitoring_done.set()
            await monitor_task
        elapsed_seconds = time.perf_counter() - started

        pending = await client.redis.xpending(stream, group)
        pending_count = int(pending.get("pending", 0))
        if measurements.processed != message_count:
            raise RuntimeError(
                f"processed {measurements.processed} of {message_count} benchmark messages"
            )
        if pending_count != 0:
            raise RuntimeError(f"{pending_count} messages remained pending")
        if measurements.peak_active > concurrency:
            raise RuntimeError("worker concurrency exceeded its configured capacity")
        if measurements.peak_lag <= 0:
            raise RuntimeError("excess work was not observed queued in Redis")
        if observed_languages != set(LANGUAGES):
            raise RuntimeError("not every benchmark language was processed")

        return {
            "messages": message_count,
            "languages": len(observed_languages),
            "configured_concurrency": concurrency,
            "peak_active_handlers": measurements.peak_active,
            "peak_redis_pending": measurements.peak_pending,
            "peak_redis_lag": measurements.peak_lag,
            "pending_after_completion": pending_count,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "throughput_messages_per_second": round(message_count / elapsed_seconds, 2),
            "result": "PASS",
        }
    finally:
        await client.redis.delete(stream)
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--handler-delay-ms", type=int, default=20)
    args = parser.parse_args()
    result = asyncio.run(run(args.messages, args.concurrency, args.handler_delay_ms))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
