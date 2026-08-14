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
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, ParamSpec, TypeVar, cast
from urllib.parse import urlsplit

import redis.asyncio as aioredis
from redis.asyncio.sentinel import Sentinel

from shared.config import RedisSettings
from shared.logger import get_logger

logger = get_logger(__name__)

# Where the stage-latency histograms live, and how they are bucketed.
#
# Chosen against what this pipeline actually does rather than a default ladder: a live dub is
# already uncomfortable at 2s and unusable past 8s, so the interesting resolution is between
# them. The wide tail exists because production p95 was measured at 11.4s — a ladder that
# topped out at 5s would have reported "everything is over 5s" and said nothing more.
LATENCY_BUCKETS_MS = (250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000)
LATENCY_KEY_PREFIX = "warptalk:latency:"
# Refreshed on every observation. Long enough to survive a quiet weekend, bounded so this can
# never become the next thing that fills Redis.
LATENCY_KEY_TTL_SECONDS = 7 * 24 * 60 * 60
P = ParamSpec("P")
R = TypeVar("R")

# Compare-and-act lease scripts. Both take the key as KEYS[1] and the expected holder as
# ARGV[1], so a replica can only ever extend or drop a claim that is still its own.
_EXTEND_IF_VALUE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

_DELETE_IF_VALUE_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""


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
        message_id = await self._append_and_refresh_ttl(stream, redis_data)
        await self._trim_stream_without_losing_unconsumed_entries(stream)
        return message_id

    async def _append_and_refresh_ttl(
        self,
        stream: str,
        redis_data: dict[str, bytes | str | int | float],
    ) -> bytes | str:
        """XADD, and push the stream's expiry back out to now + stream_ttl_seconds.

        Nothing else in this system ever deletes a stream. Rooms end, their four streams stay,
        and production reached 70 abandoned room streams holding 284 MB — the oldest untouched
        for ten days — inside a 768 MB Redis. Since the policy there is `allkeys-lru`, filling it
        does not fail a write: Redis silently deletes whichever keys look least recently used,
        which during a meeting includes that meeting's own streams and consumer groups. The
        transcript simply stops, with nothing in any log.

        The EXPIRE rides in the same pipeline as the XADD deliberately. This is the hot path —
        one publish per speech chunk — and a second round trip here would add latency to the very
        pipeline the leak was already slowing down.
        """
        ttl = self._settings.stream_ttl_seconds
        if ttl <= 0:
            return cast(
                "bytes | str", await self._retry(self.redis.xadd, stream, cast(Any, redis_data))
            )

        async def _append() -> bytes | str:
            pipeline = self.redis.pipeline(transaction=False)
            pipeline.xadd(stream, cast(Any, redis_data))
            pipeline.expire(stream, ttl)
            results = await pipeline.execute()
            return cast("bytes | str", results[0])

        return await self._retry(_append)

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

            # A group that has stopped advancing is not a slow consumer, and treating it as one
            # is what turned this safety check into the leak it was meant to prevent: on
            # 2026-08-14 `billing-stt-workers` still held the floor of `stt:results` at its
            # 2026-08-10 position, so four days of entries could not be trimmed by anyone.
            if self._group_is_stale(last_delivered):
                logger.warning(
                    "stream_group_stale_ignored_for_trim",
                    stream=stream,
                    group=group_name.decode() if isinstance(group_name, bytes) else group_name,
                    last_delivered_id=(
                        last_delivered.decode()
                        if isinstance(last_delivered, bytes)
                        else last_delivered
                    ),
                    stale_after_seconds=self._settings.stream_group_stale_after_seconds,
                )
                continue

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

        # Every group on this stream is stale. Bounded growth is no longer the safer option —
        # nobody is reading — so fall back to the plain count bound rather than growing forever.
        if not required_ids:
            await self._retry(self.redis.xtrim, stream, maxlen=maxlen, approximate=True)
            return

        earliest_required = min(required_ids, key=self._stream_id_tuple)
        await self._retry(
            self.redis.xtrim,
            stream,
            minid=earliest_required,
            approximate=False,
        )

    def _group_is_stale(self, last_delivered: bytes | str) -> bool:
        """Whether a group has gone quiet long enough to stop holding the trim floor.

        Read off the stream ID itself, whose millisecond component is the wall-clock time Redis
        stamped on the entry. That needs no extra call and no bookkeeping of our own.
        """
        stale_after = self._settings.stream_group_stale_after_seconds
        if stale_after <= 0:
            return False
        delivered_ms, _ = self._stream_id_tuple(last_delivered)
        return (time.time() * 1000) - delivered_ms > stale_after * 1000

    @staticmethod
    def _stream_id_tuple(message_id: bytes | str) -> tuple[int, int]:
        raw = message_id.decode() if isinstance(message_id, bytes) else message_id
        milliseconds, sequence = raw.split("-", 1)
        return int(milliseconds), int(sequence)

    async def publish_telemetry(self, room_id: str, worker_type: str, latency_ms: int) -> None:
        """Publish raw telemetry data to the translationRoom:telemetry Pub/Sub channel.

        The pub/sub half is live-only and has no subscriber that records anything, so until
        `record_latency` was added below, every one of these numbers was computed and thrown
        away. When a tester reported the dub arriving 5-10s late there was no metric anywhere in
        Prometheus to say which stage it was — the answer had to be reconstructed by hand from
        Redis stream entry ids.
        """
        import time

        payload = {
            "roomId": room_id,
            "routeId": "00000000-0000-0000-0000-000000000000",
            "workerType": worker_type,
            "latencyMs": latency_ms,
            "timestamp": int(time.time() * 1000),
        }
        await self._retry(self.redis.publish, "translationRoom:telemetry", json.dumps(payload))
        await self.record_latency(worker_type, latency_ms)

    async def record_latency(self, stage: str, latency_ms: int) -> None:
        """Add one observation to a durable histogram the metrics exporter can scrape.

        WHY A REDIS HASH AND NOT A PROMETHEUS CLIENT IN THE WORKER
            The workers are consumers with no HTTP server, so there is nothing for Prometheus to
            scrape them on. The exporter is already the one process that answers /metrics and it
            is deliberately stateless — it derives everything from Redis on each scrape. Keeping
            that shape means this needs no new port, no new scrape target, and no new deployment.

        Buckets are stored raw (not cumulative); the exporter accumulates them, because that is
        the only place that has to care what Prometheus's text format wants.

        The key carries a TTL, refreshed on write. A counter that resets is something
        `rate()` handles; an unbounded Redis key is what filled production to 93% this morning.
        Best effort throughout — a metric must never be able to fail the pipeline it measures.
        """
        if latency_ms < 0:
            return
        bucket = next(
            (str(edge) for edge in LATENCY_BUCKETS_MS if latency_ms <= edge),
            "+Inf",
        )
        key = f"{LATENCY_KEY_PREFIX}{stage}"
        try:
            pipeline = self.redis.pipeline(transaction=False)
            pipeline.hincrby(key, f"le:{bucket}", 1)
            pipeline.hincrby(key, "sum", latency_ms)
            pipeline.hincrby(key, "count", 1)
            pipeline.expire(key, LATENCY_KEY_TTL_SECONDS)
            await pipeline.execute()
        except Exception:
            logger.debug("latency_record_failed", stage=stage, exc_info=True)

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

    async def set_if_absent(self, key: str, value: bytes | str, ttl_seconds: int) -> bool:
        """SET NX EX — returns True only for the caller that created the key.

        The atomic building block for rate limits that must hold across replicas: AI
        workers run with replicas >= 2 and a consumer group spreads one room's messages
        over all of them, so a per-process timestamp would let every replica grant
        itself the same slot independently.
        """
        created = await self._retry(self.redis.set, key, value, nx=True, ex=ttl_seconds)
        return bool(created)

    async def extend_if_value(self, key: str, expected: str, ttl_seconds: int) -> bool:
        """Refresh a key's TTL only while it still holds `expected`.

        The renew half of a lease. A bare EXPIRE would let a replica that lost its lease
        (Redis blip, long GC pause, its own TTL elapsing) keep pushing the deadline out on
        a key that now names somebody else — two owners, which is the exact state a lease
        exists to make impossible. Compare-and-extend in one Lua step so no other replica
        can slip between the read and the write.
        """
        result = await self._retry(
            self.redis.eval,
            _EXTEND_IF_VALUE_LUA,
            1,
            key,
            expected,
            str(int(ttl_seconds)),
        )
        return bool(result)

    async def delete_if_value(self, key: str, expected: str) -> bool:
        """Delete a key only while it still holds `expected`.

        The release half of a lease. An unconditional DEL would let a replica whose lease
        already expired and was taken over delete the *new* owner's claim on its way out.
        """
        result = await self._retry(
            self.redis.eval,
            _DELETE_IF_VALUE_LUA,
            1,
            key,
            expected,
        )
        return bool(result)

    async def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        """INCR a counter, bounding its lifetime the first time it is created.

        The TTL is applied only on the 1 -> counter's first increment, so a long-running
        budget is not silently extended by later activity. Two replicas racing the very
        first INCR is harmless: exactly one of them observes 1 and sets the TTL.
        """
        value = int(await self._retry(self.redis.incr, key))
        if value == 1:
            await self.expire(key, ttl_seconds)
        return value

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
