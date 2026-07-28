"""Tests for shared.redis_client — publish, consume, retry logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import ReadOnlyError, ResponseError

from shared.config import RedisSettings
from shared.redis_client import RedisStreamClient, _redact_redis_url


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
    mock_redis.xlen = AsyncMock(return_value=1)
    mock_redis.xinfo_groups = AsyncMock(return_value=[])
    mock_redis.xpending = AsyncMock(return_value={"pending": 0})
    mock_redis.xtrim = AsyncMock()
    mock_redis.xreadgroup = AsyncMock(return_value=[])
    mock_redis.xgroup_create = AsyncMock()
    mock_redis.xack = AsyncMock()
    mock_redis.xautoclaim = AsyncMock(return_value=[b"0-0", [], []])
    mock_redis.xpending_range = AsyncMock(return_value=[])
    mock_redis.hset = AsyncMock()
    mock_redis.hget = AsyncMock(return_value=None)
    mock_redis.close = AsyncMock()
    c._redis = mock_redis

    return c


class TestConnectionLogging:
    def test_redact_redis_url_removes_all_credentials_and_query_parameters(self) -> None:
        safe_url = _redact_redis_url(
            "rediss://worker:super-secret@redis.internal:6380/4"
            "?ssl_keyfile=/run/secrets/client.key&token=also-secret"
        )

        assert safe_url == "rediss://redis.internal:6380/4"
        assert "worker" not in safe_url
        assert "super-secret" not in safe_url
        assert "token" not in safe_url

    async def test_connect_uses_sentinel_discovered_master(self) -> None:
        settings = RedisSettings(
            password="sentinel-secret",
            sentinel_urls="redis://sentinel-a:26379,redis://sentinel-b:26379",
            sentinel_service_name="warptalk-master",
        )
        master = AsyncMock()
        master.connection_pool = MagicMock()
        master.ping = AsyncMock()
        sentinel = MagicMock()
        sentinel.master_for.return_value = master

        with patch("shared.redis_client.Sentinel", return_value=sentinel) as sentinel_type:
            redis_client = RedisStreamClient(settings)
            await redis_client.connect()

        sentinel_type.assert_called_once()
        assert sentinel_type.call_args.args[0] == [
            ("sentinel-a", 26379),
            ("sentinel-b", 26379),
        ]
        sentinel.master_for.assert_called_once()
        assert sentinel.master_for.call_args.args[0] == "warptalk-master"
        assert redis_client.redis is master


class TestPublish:
    """RedisStreamClient.publish tests."""

    async def test_publish_does_not_trim_inside_xadd(self, client: RedisStreamClient) -> None:
        """XADD MAXLEN can delete pending entries, so publishing must add first."""
        await client.publish("test:stream", {"key": "value"})

        client._redis.xadd.assert_called_once_with(
            "test:stream",
            {"key": "value"},
        )
        client._redis.xtrim.assert_not_awaited()

    async def test_publish_trims_only_before_earliest_pending_entry(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xlen = AsyncMock(return_value=1_001)
        client._redis.xinfo_groups = AsyncMock(
            return_value=[
                {"name": b"workers", "last-delivered-id": b"900-0"},
            ]
        )
        client._redis.xpending = AsyncMock(
            return_value={"pending": 1, "min": b"850-0", "max": b"850-0"}
        )

        await client.publish("test:stream", {"key": "value"})

        client._redis.xtrim.assert_awaited_once_with(
            "test:stream",
            minid=b"850-0",
            approximate=False,
        )

    async def test_publish_allows_growth_when_group_has_not_consumed(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xlen = AsyncMock(return_value=1_001)
        client._redis.xinfo_groups = AsyncMock(
            return_value=[
                {"name": b"workers", "last-delivered-id": b"0-0"},
            ]
        )

        await client.publish("test:stream", {"key": "value"})

        client._redis.xtrim.assert_not_awaited()

    async def test_publish_uses_maxlen_when_stream_has_no_consumer_groups(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xlen = AsyncMock(return_value=1_001)
        client._redis.xinfo_groups = AsyncMock(return_value=[])

        await client.publish("test:stream", {"key": "value"})

        client._redis.xtrim.assert_awaited_once_with(
            "test:stream",
            maxlen=1_000,
            approximate=True,
        )

    async def test_publish_returns_message_id(self, client: RedisStreamClient) -> None:
        """publish() should return the Redis message ID."""
        client._redis.xadd = AsyncMock(return_value=b"9999-0")
        result = await client.publish("s", {"k": "v"})
        assert result == b"9999-0"


async def test_retry_recovers_after_sentinel_read_only_failover(
    client: RedisStreamClient,
) -> None:
    client._settings.retry_max_attempts = 2
    operation = AsyncMock(
        side_effect=[
            ReadOnlyError("replica is read only"),
            "recovered",
        ]
    )

    result = await client._retry(operation)

    assert result == "recovered"
    assert operation.await_count == 2


class TestConsume:
    """RedisStreamClient.consume tests."""

    @pytest.mark.parametrize(
        ("count", "concurrency"),
        [(0, 1), (1, 0)],
    )
    async def test_concurrent_consumer_rejects_non_positive_capacity(
        self,
        client: RedisStreamClient,
        count: int,
        concurrency: int,
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            await client.consume_concurrent(
                "test:stream",
                "workers",
                AsyncMock(),
                count=count,
                concurrency=concurrency,
            )

    async def test_consume_creates_group(self, client: RedisStreamClient) -> None:
        """consume() should try to create consumer group."""
        # xreadgroup returns empty → consume returns immediately
        client._redis.xreadgroup = AsyncMock(return_value=[])

        messages = []
        async for msg_id, data in client.consume("test:stream", "test-group", "w1"):
            messages.append((msg_id, data))

        client._redis.xgroup_create.assert_called_once()
        assert len(messages) == 0

    async def test_ensure_consumer_group_accepts_existing_group(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xgroup_create = AsyncMock(
            side_effect=ResponseError("BUSYGROUP Consumer Group name already exists")
        )

        await client.ensure_consumer_group("test:stream", "test-group")

        client._redis.xgroup_create.assert_awaited_once_with(
            "test:stream",
            "test-group",
            id="0",
            mkstream=True,
        )

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

    async def test_concurrent_consumer_does_not_ack_failed_handler(
        self,
        client: RedisStreamClient,
    ) -> None:
        """A failed handler must stay pending so it can be reclaimed and retried."""
        client._redis.xreadgroup = AsyncMock(
            side_effect=[
                [(b"s", [(b"1-0", {b"k": b"v"})])],
                [],
            ]
        )

        async def failing_handler(
            message_id: bytes,
            data: dict[bytes, bytes],
        ) -> None:
            raise RuntimeError("provider unavailable")

        await client.consume_concurrent(
            "s",
            "g",
            failing_handler,
            consumer="w1",
            count=1,
            concurrency=1,
        )

        client._redis.xack.assert_not_awaited()


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


class TestPendingRecovery:
    async def test_reclaim_recreates_group_before_claiming_stale_messages(
        self,
        client: RedisStreamClient,
    ) -> None:
        await client.reclaim_stale("input", "workers", "worker-2")

        client._redis.xgroup_create.assert_awaited_once_with(
            "input",
            "workers",
            id="0",
            mkstream=True,
        )
        client._redis.xautoclaim.assert_awaited_once()

    async def test_reclaim_stale_claims_idle_pending_messages(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xautoclaim = AsyncMock(
            return_value=[
                b"0-0",
                [(b"2-0", {b"text": b"retry me"})],
                [],
            ]
        )

        messages = await client.reclaim_stale(
            "input",
            "workers",
            "worker-2",
            min_idle_ms=60_000,
            count=10,
        )

        assert messages == [(b"2-0", {b"text": b"retry me"})]
        client._redis.xautoclaim.assert_awaited_once_with(
            "input",
            "workers",
            "worker-2",
            60_000,
            start_id="0-0",
            count=10,
        )

    async def test_pending_delivery_count_reads_redis_attempt_count(
        self,
        client: RedisStreamClient,
    ) -> None:
        client._redis.xpending_range = AsyncMock(
            return_value=[
                {
                    "message_id": b"2-0",
                    "consumer": b"worker-2",
                    "time_since_delivered": 100,
                    "times_delivered": 5,
                }
            ]
        )

        attempts = await client.pending_delivery_count(
            "input",
            "workers",
            b"2-0",
        )

        assert attempts == 5
