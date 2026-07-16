"""Redis Streams consumer/producer for inter-worker communication.

Improvements over v1:
- Connection pooling with configurable max_connections
- Socket timeouts to prevent hanging
- XADD with MAXLEN ~ trimming to bound memory
- Retry with exponential backoff on connection errors
- Dynamic consumer name from hostname for Docker scaling
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from shared.config import RedisSettings
from shared.logger import get_logger

logger = get_logger(__name__)


class RedisStreamClient:
    """Async Redis Streams client for AI pipeline communication.

    Usage::

        client = RedisStreamClient(redis_settings)
        await client.connect()

        # Produce
        await client.publish("stt:results:meeting123", {"text": "hello"})

        # Consume with consumer groups
        async for msg_id, data in client.consume(
            "audio:chunks:meeting123", group="stt-workers"
        ):
            process(data)
    """

    def __init__(self, settings: RedisSettings | None = None) -> None:
        self._settings = settings or RedisSettings()
        self._pool: aioredis.ConnectionPool | None = None
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection with connection pooling."""
        self._pool = aioredis.ConnectionPool.from_url(
            self._settings.url,
            password=self._settings.password or None,
            max_connections=self._settings.max_connections,
            socket_timeout=self._settings.socket_timeout,
            socket_connect_timeout=self._settings.socket_connect_timeout,
            decode_responses=False,
        )
        self._redis = aioredis.Redis(connection_pool=self._pool)

        # Verify connection
        await self._redis.ping()
        logger.info(
            "redis_connected",
            url=self._settings.url,
            max_connections=self._settings.max_connections,
        )

    async def disconnect(self) -> None:
        """Close Redis connection and pool."""
        if self._redis:
            await self._redis.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("redis_disconnected")

    @property
    def redis(self) -> aioredis.Redis:
        if not self._redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def publish(self, stream: str, data: dict[str, Any]) -> str:
        """Publish a message to a Redis Stream with MAXLEN trimming.

        Args:
            stream: Stream key (e.g. "stt:results:meeting123")
            data: Message fields dict

        Returns:
            Redis message ID
        """
        message_id = await self._retry(
            self.redis.xadd,
            stream,
            data,
            maxlen=self._settings.stream_maxlen,
            approximate=True,
        )
        return message_id

    async def publish_telemetry(self, room_id: str, worker_type: str, latency_ms: int) -> None:
        """Publish raw telemetry data to the translationRoom:telemetry Pub/Sub channel."""
        import time
        payload = {
            "roomId": room_id,
            "routeId": "00000000-0000-0000-0000-000000000000",
            "workerType": worker_type,
            "latencyMs": latency_ms,
            "timestamp": int(time.time() * 1000)
        }
        await self._retry(self.redis.publish, "translationRoom:telemetry", json.dumps(payload))

    async def publish_system_event(self, room_id: str, event_type: str, payload: dict[str, Any]) -> str:
        """Publish an event to the translationRoom:system_events Redis Stream."""
        data = {
            "event_type": event_type,
            "route_id": "00000000-0000-0000-0000-000000000000",
            "room_id": room_id,
            "payload": json.dumps(payload)
        }
        return await self.publish("translationRoom:system_events", data)

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str | None = None,
        block_ms: int = 2000,
        count: int = 1,
    ) -> AsyncIterator[tuple[bytes, dict[bytes, bytes]]]:
        """Consume messages from a Redis Stream using consumer groups.

        Creates the consumer group if it doesn't exist.
        Yields (message_id, data) tuples.

        Args:
            stream: Stream key to consume from
            group: Consumer group name
            consumer: Consumer name (defaults to hostname)
            block_ms: Block timeout in ms (also serves as shutdown check interval)
            count: Max messages to read per batch
        """
        consumer = consumer or f"worker-{socket.gethostname()}"

        # Ensure consumer group exists
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
            logger.info("consumer_group_created", stream=stream, group=group)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
            # Group already exists — this is fine

        while True:
            messages = await self._retry(
                self.redis.xreadgroup,
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )

            if not messages:
                return  # block timeout, yield control back to caller

            for _stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    yield message_id, data
                    await self.redis.xack(stream, group, message_id)

    # ------------------------------------------------------------------
    # Key-value helpers (for voice embeddings, speaker cache, etc.)
    # ------------------------------------------------------------------

    async def hset(self, key: str, field: str, value: bytes | str) -> None:
        """Set a hash field value."""
        await self.redis.hset(key, field, value)

    async def expire(self, key: str, ttl_seconds: int) -> None:
        """Set/refresh a TTL on an existing key (e.g. after hset, which has no TTL param)."""
        await self.redis.expire(key, ttl_seconds)

    async def hget(self, key: str, field: str) -> bytes | None:
        """Get a hash field value."""
        return await self.redis.hget(key, field)

    async def set_with_ttl(self, key: str, value: bytes | str, ttl_seconds: int) -> None:
        """Set a key with expiration."""
        await self.redis.setex(key, ttl_seconds, value)

    async def get(self, key: str) -> bytes | None:
        """Get a key value."""
        return await self.redis.get(key)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _retry(self, func, *args, **kwargs):
        """Execute a Redis command with exponential backoff retry."""
        last_err = None
        for attempt in range(self._settings.retry_max_attempts):
            try:
                return await func(*args, **kwargs)
            except (aioredis.ConnectionError, aioredis.TimeoutError, OSError) as e:
                last_err = e
                delay = self._settings.retry_base_delay * (2**attempt)
                logger.warning(
                    "redis_retry",
                    attempt=attempt + 1,
                    max_attempts=self._settings.retry_max_attempts,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

        raise ConnectionError(
            f"Redis operation failed after {self._settings.retry_max_attempts} attempts"
        ) from last_err
