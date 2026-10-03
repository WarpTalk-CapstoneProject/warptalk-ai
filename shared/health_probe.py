"""Container health probe for Redis-backed WarpTalk workers.

TWO WAYS TO READ THE SAME HEARTBEAT, AND WHY THE CHEAP ONE IS THE PROBE
    Every worker writes its heartbeat to Redis every 10s (`heartbeat_key`) and, once that write
    has succeeded, stamps the mtime of a file in its own /tmp (`heartbeat_file`). The two carry
    the same fact: the event loop is turning and Redis took the write.

    `python -m shared.health_probe` reads the Redis copy. That is a fresh interpreter importing
    pydantic settings and the Sentinel client: measured on prod 3 Oct 2026 at ~1.1 CPU-seconds a
    run, billed to the worker's own CPU limit. Every 30s that was ~35m per worker, which was the
    whole idle CPU of every Python worker and kept them 40-66% CFS-throttled at rest. Worse, under
    load the probe competed for the quota of the very worker it was judging: tts-worker sat at
    its 500m limit, the probe outlived its 15s timeout three times, and the kubelet killed a
    healthy worker mid-meeting, twice in 20 minutes.

    The kubelet probe therefore reads the file's age from `sh` (milliseconds of CPU), and falls
    back to this module only when the file does not exist yet — an image from before the file,
    or the first seconds of a start. This module stays the authority for anything that is not a
    kubelet probe.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path

from shared.config import RedisSettings
from shared.redis_client import RedisStreamClient


def heartbeat_key(worker_name: str, hostname: str) -> str:
    # The worker/host delimiter must be unambiguous: names such as
    # "assistant" and "assistant-chat" coexist in one container.
    return f"warptalk:worker:heartbeat:{worker_name}:{hostname}"


def heartbeat_file(worker_name: str) -> Path:
    """The local twin of `heartbeat_key`, whose mtime the kubelet probe reads.

    The probe command in the infrastructure chart builds this same path from WORKER_HEALTH_DIR
    and WORKER_HEALTH_NAME; change both together.
    """
    return Path(os.environ.get("WORKER_HEALTH_DIR", "/tmp")) / f"warptalk-heartbeat-{worker_name}"


def touch_heartbeat_file(worker_name: str) -> None:
    """Stamp the local heartbeat. Best effort: a read-only /tmp must not stop the Redis one.

    Call it only AFTER the Redis heartbeat succeeded, so the file means exactly what the Redis
    key means and the cheap probe is not a weaker check than the one it replaces.
    """
    try:
        heartbeat_file(worker_name).touch(exist_ok=True)
    except OSError:
        pass


def heartbeat_keys(worker_names: str, hostname: str) -> list[str]:
    return [
        heartbeat_key(name.strip(), hostname) for name in worker_names.split(",") if name.strip()
    ]


async def check_worker() -> bool:
    keys = heartbeat_keys(
        os.environ.get("WORKER_HEALTH_NAME", ""),
        socket.gethostname(),
    )
    if not keys:
        return False

    settings = RedisSettings()
    client = RedisStreamClient(settings)
    try:
        await client.connect()
        heartbeat_values = await client.redis.mget(keys)
        if len(heartbeat_values) != len(keys) or any(value is None for value in heartbeat_values):
            return False

        now_unix_ms = int(time.time() * 1000)
        max_heartbeat_age_ms = (
            int(os.environ.get("WORKER_HEALTH_MAX_HEARTBEAT_AGE_SECONDS", "30")) * 1000
        )
        for raw_value in heartbeat_values:
            if raw_value is None:
                return False
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")
            payload = json.loads(raw_value)
            heartbeat_unix_ms = int(payload.get("timestamp_unix_ms", 0))
            if heartbeat_unix_ms <= 0 or now_unix_ms - heartbeat_unix_ms > max_heartbeat_age_ms:
                return False

        return True
    except Exception:
        return False
    finally:
        await client.disconnect()


def main() -> int:
    return 0 if asyncio.run(check_worker()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
