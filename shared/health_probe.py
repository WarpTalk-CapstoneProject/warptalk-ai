"""Container health probe for Redis-backed WarpTalk workers."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time

from shared.config import RedisSettings
from shared.redis_client import RedisStreamClient


def heartbeat_key(worker_name: str, hostname: str) -> str:
    # The worker/host delimiter must be unambiguous: names such as
    # "assistant" and "assistant-chat" coexist in one container.
    return f"warptalk:worker:heartbeat:{worker_name}:{hostname}"


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
