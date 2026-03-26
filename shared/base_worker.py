"""Abstract base worker with lifecycle management.

Provides:
- Redis connection with pooling and retry
- Model loading via asyncio.to_thread (non-blocking)
- Graceful shutdown with SIGTERM/SIGINT handling
- Structured logging with worker context
"""

from __future__ import annotations

import asyncio
import signal
import socket
from abc import ABC, abstractmethod
from typing import Any

from shared.config import RedisSettings, WorkerSettings
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient


class BaseWorker(ABC):
    """Abstract base for all AI pipeline workers.

    Subclasses must implement:
        - `worker_name` property
        - `input_stream` property  (e.g. "audio:chunks")
        - `consumer_group` property (e.g. "stt-workers")
        - `load_model()` — load ML model (called in thread)
        - `process(message_id, data)` — handle one message

    Example::

        class STTWorker(BaseWorker):
            worker_name = "stt"
            input_stream = "audio:chunks"
            consumer_group = "stt-workers"

            async def load_model(self):
                self.model = await asyncio.to_thread(WhisperModel, "medium")

            async def process(self, message_id, data):
                chunk = AudioChunkMessage.from_redis(data)
                result = await asyncio.to_thread(self.model.transcribe, ...)
                await self.publish("stt:results", meeting_id, result.to_redis())
    """

    # Subclasses must set these
    worker_name: str = "base"
    input_stream: str = ""
    consumer_group: str = ""

    def __init__(
        self,
        settings: WorkerSettings | None = None,
        redis_settings: RedisSettings | None = None,
    ) -> None:
        self.settings = settings or WorkerSettings()
        self.redis_settings = redis_settings or self.settings.redis
        self.redis = RedisStreamClient(self.redis_settings)
        self.logger = get_logger(f"worker.{self.worker_name}")
        self._shutdown_event = asyncio.Event()
        self._consumer_name = f"{self.worker_name}-{socket.gethostname()}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Full worker lifecycle: connect → load → consume → shutdown."""
        self._register_signal_handlers()
        self.logger.info(
            "worker_starting",
            worker=self.worker_name,
            consumer=self._consumer_name,
            stream=self.input_stream,
            group=self.consumer_group,
        )

        try:
            # 1. Connect to Redis
            await self.redis.connect()
            self.logger.info("redis_connected")

            # 2. Load model (offload blocking init to thread pool)
            await self.load_model()
            self.logger.info("model_loaded")

            # 3. Enter consume loop
            await self._consume_loop()

        except asyncio.CancelledError:
            self.logger.info("worker_cancelled")
        except Exception:
            self.logger.exception("worker_fatal_error")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self.logger.info("worker_shutting_down")
        await self._cleanup()
        await self.redis.disconnect()
        self.logger.info("worker_stopped")

    async def _cleanup(self) -> None:
        """Override for worker-specific cleanup (e.g. release GPU memory)."""

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    async def load_model(self) -> None:
        """Load the ML model. Use asyncio.to_thread for blocking loads."""

    @abstractmethod
    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Process one message from the input stream."""

    # ------------------------------------------------------------------
    # Publishing helper
    # ------------------------------------------------------------------

    async def publish(
        self,
        stream_prefix: str,
        meeting_id: str,
        data: dict[str, Any],
    ) -> str:
        """Publish result to a meeting-specific output stream.

        Args:
            stream_prefix: e.g. "stt:results"
            meeting_id: meeting identifier
            data: serialized message dict from schema.to_redis()

        Returns:
            Redis message ID
        """
        stream_key = f"{stream_prefix}:{meeting_id}"
        msg_id = await self.redis.publish(stream_key, data)
        self.logger.debug(
            "message_published",
            stream=stream_key,
            message_id=msg_id,
        )
        return msg_id

    # ------------------------------------------------------------------
    # Consume loop
    # ------------------------------------------------------------------

    async def _consume_loop(self) -> None:
        """Main consume loop with graceful shutdown support."""
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )

        while not self._shutdown_event.is_set():
            try:
                async for message_id, data in self.redis.consume(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,  # Check shutdown every 2s
                ):
                    if self._shutdown_event.is_set():
                        break

                    try:
                        await self.process(message_id, data)
                    except Exception:
                        self.logger.exception(
                            "process_error",
                            message_id=message_id,
                            stream=self.input_stream,
                        )
                        # Message is already acked in consume(); in production,
                        # we may want to push to a dead-letter stream instead.

            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                # Exponential backoff before retry
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        self.logger.info("signal_received", signal=sig.name)
        self._shutdown_event.set()
