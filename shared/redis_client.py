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
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast
from urllib.parse import urlsplit

import redis.asyncio as aioredis
from redis.asyncio.sentinel import Sentinel

from shared.config import RedisSettings
from shared.logger import get_logger

logger = get_logger(__name__)
P = ParamSpec("P")
R = TypeVar("R")


def _redact_redis_url(url: str) -> str:
    """Return a log-safe endpoint without credentials or query parameters."""
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{hostname}{port}{parsed.path}"
    except ValueError:
        return "redis://<invalid-endpoint>"


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
        self._sentinel: Sentinel | None = None

    async def connect(self) -> None:
        """Establish Redis connection with connection pooling."""
        if self._settings.sentinel_urls:
            sentinel_endpoints: list[tuple[str, int]] = []
            sentinel_uses_tls = False
            for raw_url in self._settings.sentinel_urls.split(","):
                parsed = urlsplit(raw_url.strip())
                if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
                    raise ValueError(f"Invalid Redis Sentinel URL: {_redact_redis_url(raw_url)}")
                sentinel_endpoints.append((parsed.hostname, parsed.port or 26379))
                sentinel_uses_tls = sentinel_uses_tls or parsed.scheme == "rediss"

            connection_options: dict[str, Any] = {
                "password": self._settings.password or None,
                "max_connections": self._settings.max_connections,
                "socket_timeout": self._settings.socket_timeout,
                "socket_connect_timeout": self._settings.socket_connect_timeout,
                "decode_responses": False,
                "ssl": sentinel_uses_tls,
            }
            self._sentinel = Sentinel(  # type: ignore[no-untyped-call]
                sentinel_endpoints,
                sentinel_kwargs={
                    "password": self._settings.password or None,
                    "socket_timeout": self._settings.socket_timeout,
                    "socket_connect_timeout": self._settings.socket_connect_timeout,
                    "ssl": sentinel_uses_tls,
                },
                **connection_options,
            )
            self._redis = self._sentinel.master_for(self._settings.sentinel_service_name)
            self._pool = self._redis.connection_pool
        else:
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
            url=_redact_redis_url(self._settings.url),
            sentinel_service=(
                self._settings.sentinel_service_name if self._settings.sentinel_urls else None
            ),
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

    async def publish(self, stream: str, data: dict[str, Any]) -> bytes | str:
        """Publish a message to a Redis Stream with MAXLEN trimming.

        Args:
            stream: Stream key (e.g. "stt:results:meeting123")
            data: Message fields dict

        Returns:
            Redis message ID
        """
        redis_data: dict[str, bytes | str | int | float] = {
            key: value if isinstance(value, (bytes, str, int, float)) else json.dumps(value)
            for key, value in data.items()
        }
        message_id = await self._retry(self.redis.xadd, stream, cast(Any, redis_data))
        await self._trim_stream_without_losing_unconsumed_entries(stream)
        return message_id

    async def _trim_stream_without_losing_unconsumed_entries(self, stream: str) -> None:
        """Bound a stream only when trimming cannot remove pending/unread messages.

        Redis 7's ``XADD MAXLEN`` trims entries without considering consumer-group
        state. A slow or crashed worker can therefore retain a PEL reference whose
        payload has already disappeared. We prefer bounded growth when consumers
        lag, then reclaim space as soon as every group advances.
        """
        maxlen = self._settings.stream_maxlen
        if maxlen <= 0 or await self._retry(self.redis.xlen, stream) <= maxlen:
            return

        groups = cast(list[dict[Any, Any]], await self._retry(self.redis.xinfo_groups, stream))
        if not groups:
            await self._retry(
                self.redis.xtrim,
                stream,
                maxlen=maxlen,
                approximate=True,
            )
            return

        required_ids: list[bytes | str] = []
        for group in groups:
            group_name = group.get("name") or group.get(b"name")
            last_delivered = group.get("last-delivered-id") or group.get(b"last-delivered-id")
            if group_name is None or last_delivered is None:
                return
            if self._stream_id_tuple(last_delivered) == (0, 0):
                return

            pending = cast(
                dict[Any, Any],
                await self._retry(self.redis.xpending, stream, group_name),
            )
            pending_count = pending.get("pending", pending.get(b"pending", 0))
            pending_min = pending.get("min") or pending.get(b"min")
            required = last_delivered
            if pending_count and pending_min is not None:
                required = min(
                    (last_delivered, pending_min),
                    key=self._stream_id_tuple,
                )
            required_ids.append(required)

        earliest_required = min(required_ids, key=self._stream_id_tuple)
        await self._retry(
            self.redis.xtrim,
            stream,
            minid=earliest_required,
            approximate=False,
        )

    @staticmethod
    def _stream_id_tuple(message_id: bytes | str) -> tuple[int, int]:
        raw = message_id.decode() if isinstance(message_id, bytes) else message_id
        milliseconds, sequence = raw.split("-", 1)
        return int(milliseconds), int(sequence)

    async def publish_telemetry(self, room_id: str, worker_type: str, latency_ms: int) -> None:
        """Publish raw telemetry data to the translationRoom:telemetry Pub/Sub channel."""
        import time

        payload = {
            "roomId": room_id,
            "routeId": "00000000-0000-0000-0000-000000000000",
            "workerType": worker_type,
            "latencyMs": latency_ms,
            "timestamp": int(time.time() * 1000),
        }
        await self._retry(self.redis.publish, "translationRoom:telemetry", json.dumps(payload))

    async def publish_system_event(
        self, room_id: str, event_type: str, payload: dict[str, Any]
    ) -> bytes | str:
        """Publish an event to the translationRoom:system_events Redis Stream."""
        data = {
            "event_type": event_type,
            "route_id": "00000000-0000-0000-0000-000000000000",
            "room_id": room_id,
            "payload": json.dumps(payload),
        }
        return await self.publish("translationRoom:system_events", data)

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    async def ensure_consumer_group(self, stream: str, group: str) -> None:
        """Create a consumer group if Redis lost it during restart/failover.

        Starting at ``0`` deliberately replays every retained entry when a group
        has disappeared. Reprocessing through the existing idempotent consumers
        is safer than silently skipping work that was appended before recovery.
        """
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
            logger.info("consumer_group_created", stream=stream, group=group)
        except aioredis.ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

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

        await self.ensure_consumer_group(stream, group)

        while True:
            messages_raw = await self._retry(
                self.redis.xreadgroup,
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )
            messages = cast(
                list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]],
                messages_raw,
            )

            if not messages:
                return  # block timeout, yield control back to caller

            for _stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    yield message_id, data
                    await self.redis.xack(stream, group, message_id)

    async def consume_concurrent(
        self,
        stream: str,
        group: str,
        handler: Callable[[bytes, dict[bytes, bytes]], Awaitable[None]],
        consumer: str | None = None,
        block_ms: int = 2000,
        count: int = 4,
        concurrency: int = 4,
    ) -> None:
        """Like consume(), but runs up to `concurrency` handler calls at once instead of
        one message at a time.

        Deliberately NOT implemented as "wrap consume() and fire tasks without awaiting" —
        consume() is a generator whose XACK line runs when the caller's `async for` asks it
        for the next item, not when the current item's processing actually finishes. Racing
        ahead to the next item before the current handler completes would XACK a message
        before we know its handler even ran, let alone succeeded — a crash mid-handler would
        then lose that message silently (already acked, never redelivered). This method owns
        the whole read-dispatch-ack cycle itself instead, so each message's XACK is tied to
        its own handler's successful completion. Failed work remains pending for the
        reclaim/retry/dead-letter path.

        Only used when a worker opts in via BaseWorker.concurrency > 1 (see base_worker.py);
        consume() is untouched and remains the default path for every other worker.
        """
        if count <= 0 or concurrency <= 0:
            raise ValueError("count and concurrency must be positive")

        consumer = consumer or f"worker-{socket.gethostname()}"

        await self.ensure_consumer_group(stream, group)

        semaphore = asyncio.Semaphore(concurrency)

        async def _run_one(message_id: bytes, data: dict[bytes, bytes]) -> None:
            async with semaphore:
                try:
                    await handler(message_id, data)
                except Exception:
                    logger.exception(
                        "consume_concurrent_handler_error", message_id=message_id, stream=stream
                    )
                else:
                    await self.redis.xack(stream, group, message_id)

        while True:
            messages_raw = await self._retry(
                self.redis.xreadgroup,
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=count,
                block=block_ms,
            )
            messages = cast(
                list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]],
                messages_raw,
            )

            if not messages:
                return  # block timeout, yield control back to caller (same as consume())

            tasks = [
                asyncio.create_task(_run_one(message_id, data))
                for _stream_name, stream_messages in messages
                for message_id, data in stream_messages
            ]
            await asyncio.gather(*tasks)

    async def reclaim_stale(
        self,
        stream: str,
        group: str,
        consumer: str,
        min_idle_ms: int = 60_000,
        count: int = 10,
    ) -> list[tuple[bytes, dict[bytes, bytes]]]:
        """Claim messages abandoned by a crashed or unhealthy consumer."""
        # This runs before the normal XREADGROUP path in BaseWorker. Without
        # recreating the group here, a Redis data loss/restart produces NOGROUP
        # forever and the worker never reaches consume(), which previously made
        # a healthy-looking container completely inert.
        await self.ensure_consumer_group(stream, group)
        result = await self._retry(
            self.redis.xautoclaim,
            stream,
            group,
            consumer,
            min_idle_ms,
            start_id="0-0",
            count=count,
        )
        return list(result[1]) if len(result) > 1 else []

    async def pending_delivery_count(
        self,
        stream: str,
        group: str,
        message_id: bytes,
    ) -> int:
        """Return how many times Redis has delivered one pending message."""
        pending = await self._retry(
            self.redis.xpending_range,
            stream,
            group,
            min=message_id,
            max=message_id,
            count=1,
        )
        if not pending:
            return 0
        return int(pending[0].get("times_delivered", 0))

    # ------------------------------------------------------------------
    # Key-value helpers (for voice embeddings, speaker cache, etc.)
    # ------------------------------------------------------------------

    async def hset(self, key: str, field: str, value: bytes | str) -> None:
        """Set a hash field value."""
        await self.redis.hset(key, field, value)

    async def expire(self, key: str, ttl_seconds: int) -> None:
        """Set/refresh a TTL on an existing key (e.g. after hset, which has no TTL param)."""
        await self.redis.expire(key, ttl_seconds)

    async def hget(self, key: str, field: str) -> bytes | str | None:
        """Get a hash field value."""
        return await self.redis.hget(key, field)

    async def hgetall(self, key: str) -> dict[bytes | str, bytes | str]:
        """Get all fields/values of a hash."""
        return await self.redis.hgetall(key)

    async def set_with_ttl(self, key: str, value: bytes | str, ttl_seconds: int) -> None:
        """Set a key with expiration."""
        await self.redis.setex(key, ttl_seconds, value)

    async def get(self, key: str) -> bytes | str | None:
        """Get a key value."""
        return await self.redis.get(key)

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    async def _retry(
        self,
        func: Callable[P, Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute a Redis command with exponential backoff retry."""
        last_err: BaseException | None = None
        for attempt in range(self._settings.retry_max_attempts):
            try:
                return await func(*args, **kwargs)
            except (
                aioredis.ConnectionError,
                aioredis.TimeoutError,
                aioredis.ReadOnlyError,
                OSError,
            ) as e:
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
