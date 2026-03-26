"""Redis Streams consumer/producer for inter-worker communication."""

from __future__ import annotations

import redis.asyncio as aioredis

from shared.config import settings


class RedisStreamClient:
    """Async Redis Streams client for AI pipeline communication.

    Usage:
        client = RedisStreamClient()
        await client.connect()

        # Produce
        await client.publish("audio:chunks:meeting123", {"data": audio_bytes})

        # Consume
        async for message in client.consume("stt:results:meeting123", group="stt-workers"):
            process(message)
    """

    def __init__(self) -> None:
        self._redis: aioredis.Redis | None = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        self._redis = aioredis.from_url(
            settings.redis.url,
            password=settings.redis.password or None,
            decode_responses=False,
        )

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    @property
    def redis(self) -> aioredis.Redis:
        if not self._redis:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._redis

    async def publish(self, stream: str, data: dict) -> str:
        """Publish a message to a Redis Stream."""
        message_id = await self.redis.xadd(stream, data)
        return message_id

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str = "worker-1",
        block_ms: int = 5000,
    ):
        """Consume messages from a Redis Stream using consumer groups.

        Creates the consumer group if it doesn't exist.
        """
        try:
            await self.redis.xgroup_create(stream, group, id="0", mkstream=True)
        except aioredis.ResponseError:
            pass  # Group already exists

        while True:
            messages = await self.redis.xreadgroup(
                groupname=group,
                consumername=consumer,
                streams={stream: ">"},
                count=1,
                block=block_ms,
            )

            for _stream_name, stream_messages in messages:
                for message_id, data in stream_messages:
                    yield message_id, data
                    await self.redis.xack(stream, group, message_id)

    async def ack(self, stream: str, group: str, message_id: str) -> None:
        """Acknowledge a processed message."""
        await self.redis.xack(stream, group, message_id)
