"""Shared test fixtures for warptalk-ai workers.

Uses mocked Redis for unit tests and pytest-asyncio for async tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from shared.config import RedisSettings, WorkerSettings
from shared.redis_client import RedisStreamClient


class FakeSharedRedis:
    """A minimal stand-in for the ONE Redis every worker replica shares.

    Enough of RedisStreamClient to exercise the cross-replica primitives: the lease
    (set_if_absent / extend_if_value / delete_if_value) and the fleet-wide counter
    (incr_with_ttl). Deliberately has no TTL clock — expiry is simulated by calling
    expire_lease(), which is what a replica dying looks like from the other replica's
    side. Two workers constructed with the SAME instance are two replicas of one
    deployment; that is the whole point.
    """

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def extend_if_value(self, key: str, expected: str, ttl_seconds: int) -> bool:
        return self.values.get(key) == expected

    async def delete_if_value(self, key: str, expected: str) -> bool:
        if self.values.get(key) == expected:
            del self.values[key]
            return True
        return False

    async def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def expire_lease(self, key: str) -> None:
        """Simulate the TTL elapsing — i.e. the holder stopped renewing it."""
        self.values.pop(key, None)


@pytest.fixture
def redis_settings() -> RedisSettings:
    """Redis settings for tests."""
    return RedisSettings(
        url="redis://localhost:6379",
        password="",
        max_connections=5,
        stream_maxlen=100,
        retry_max_attempts=1,
        retry_base_delay=0.01,
    )


@pytest.fixture
def worker_settings(redis_settings: RedisSettings) -> WorkerSettings:
    """Worker settings for tests."""
    return WorkerSettings(
        log_level="DEBUG",
        chunk_duration_ms=1000,
        redis=redis_settings,
    )


@pytest.fixture
def mock_redis_client(redis_settings: RedisSettings) -> RedisStreamClient:
    """A RedisStreamClient with mocked internal Redis connection.

    publish() and hset()/hget() work directly since _retry wraps
    the async call with await. The _redis mock must be an AsyncMock
    so attribute access returns coroutines automatically.
    """
    client = RedisStreamClient.__new__(RedisStreamClient)
    client._settings = redis_settings
    client._pool = None
    client._redis = AsyncMock()

    # Configure specific return values
    client._redis.xadd.return_value = b"1234567890-0"
    client._redis.xlen.return_value = 1
    client._redis.xinfo_groups.return_value = []
    client._redis.xpending.return_value = {"pending": 0}
    client._redis.xreadgroup.return_value = []
    client._redis.hget.return_value = None
    client._redis.hgetall.return_value = {}
    client._redis.get.return_value = None

    return client


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """Generate minimal valid WAV audio bytes for testing."""
    import io

    import numpy as np
    import soundfile as sf

    # 1 second of silence at 16kHz
    audio = np.zeros(16000, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format="WAV")
    buffer.seek(0)
    return buffer.read()
