"""Tests for shared.redis_client — publish, consume, retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import asyncio

import pytest

from shared.redis_client import RedisStreamClient


@pytest.fixture
def client() -> RedisStreamClient:
    """A RedisStreamClient with mocked internal Redis connection."""
    c = RedisStreamClient.__new__(RedisStreamClient)
    c._settings = MagicMock()
    c._settings.stream_maxlen = 1000
    c._settings.retry_max_attempts = 1
    c._settings.retry_base_delay = 0.01
    c._pool = None

    # Mock Redis instance
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock()
    mock_redis.xadd = AsyncMock(return_value=b"1234567890-0")
    mock_redis.xreadgroup = AsyncMock(return_value=[])
    mock_redis.xgroup_create = AsyncMock()
    mock_redis.xack = AsyncMock()
    mock_redis.hset = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)
    mock_redis.close = AsyncMock()
    c._redis = mock_redis

    return c


class TestPublish:
    """RedisStreamClient.publish tests."""

    async def test_publish_calls_xadd_with_maxlen(self, client: RedisStreamClient) -> None:
        """publish() should call XADD with MAXLEN ~."""
        await client.publish("test:stream", {"key": "value"})

        client._redis.xadd.assert_called_once_with(
            "test:stream",
            {"key": "value"},
            maxlen=1000,
            approximate=True,
        )

    async def test_publish_returns_message_id(self, client: RedisStreamClient) -> None:
        """publish() should return the Redis message ID."""
        client._redis.xadd = AsyncMock(return_value=b"9999-0")
        result = await client.publish("s", {"k": "v"})
        assert result == b"9999-0"


class TestConsume:
    """RedisStreamClient.consume tests."""

    async def test_consume_creates_group(self, client: RedisStreamClient) -> None:
        """consume() should try to create consumer group."""
        # xreadgroup returns empty → consume returns immediately
        client._redis.xreadgroup = AsyncMock(return_value=[])

        messages = []
        async for msg_id, data in client.consume("test:stream", "test-group", "w1"):
            messages.append((msg_id, data))

        client._redis.xgroup_create.assert_called_once()
        assert len(messages) == 0

    async def test_consume_yields_messages(self, client: RedisStreamClient) -> None:
        """consume() should yield (message_id, data) tuples, then return on empty."""
        # First call returns messages, second call returns empty (stops iteration)
        client._redis.xreadgroup = AsyncMock(
            side_effect=[
                [(b"test:stream", [(b"1234-0", {b"text": b"hello"})])],
                [],  # empty → break the while True loop
            ]
        )

        messages = []
        async for msg_id, data in client.consume("test:stream", "test-group", "w1"):
            messages.append((msg_id, data))

        assert len(messages) == 1
        assert messages[0][0] == b"1234-0"
        assert messages[0][1] == {b"text": b"hello"}

    async def test_consume_acks_messages(self, client: RedisStreamClient) -> None:
        """consume() should XACK each message after yielding."""
        client._redis.xreadgroup = AsyncMock(
            side_effect=[
                [(b"s", [(b"1-0", {b"k": b"v"})])],
                [],
            ]
        )

        async for _ in client.consume("s", "g", "w1"):
            pass

        client._redis.xack.assert_called_once_with("s", "g", b"1-0")


class TestHashHelpers:
    """Key-value helper tests."""

    async def test_hset(self, client: RedisStreamClient) -> None:
        """hset should delegate to Redis."""
        await client.hset("key", "field", b"value")
        client._redis.hset.assert_called_once_with("key", "field", b"value")

    async def test_hget(self, client: RedisStreamClient) -> None:
        """hget should delegate to Redis."""
        client._redis.hget = AsyncMock(return_value=b"result")
        result = await client.hget("key", "field")
        assert result == b"result"
