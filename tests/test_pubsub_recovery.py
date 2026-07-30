from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from livekit_ingress_worker.worker import LiveKitIngressWorker


async def test_livekit_ingress_resubscribes_after_transient_redis_failure() -> None:
    """Track publication intake must recover after Redis is recreated."""
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker._shutdown_event = asyncio.Event()
    worker.redis = MagicMock()

    failed_pubsub = MagicMock()
    failed_pubsub.subscribe = AsyncMock()
    failed_pubsub.close = AsyncMock()

    failed_attempts = 0

    async def failed_message(**_kwargs):
        nonlocal failed_attempts
        failed_attempts += 1
        if failed_attempts == 1:
            raise ConnectionError("Redis restarted")
        worker._shutdown_event.set()
        return None

    failed_pubsub.get_message = AsyncMock(side_effect=failed_message)

    recovered_pubsub = MagicMock()
    recovered_pubsub.subscribe = AsyncMock()
    recovered_pubsub.close = AsyncMock()

    async def recovered_message(**_kwargs):
        worker._shutdown_event.set()
        return None

    recovered_pubsub.get_message = AsyncMock(side_effect=recovered_message)
    worker.redis.redis.pubsub.side_effect = [failed_pubsub, recovered_pubsub]

    await asyncio.wait_for(worker._consume_loop(), timeout=3)

    assert worker.redis.redis.pubsub.call_count == 2
    recovered_pubsub.subscribe.assert_awaited_once_with("meeting.track_published")
    failed_pubsub.close.assert_awaited_once()
    recovered_pubsub.close.assert_awaited_once()
