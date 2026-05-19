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
        self._route_states: dict[str, str] = {}
        self._paused_rooms: set[str] = set()
        self._pubsub = None
        self._listener_task = None

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

            # 3. Start route updates listener
            self._listener_task = asyncio.create_task(self._listen_route_updates())

            # 4. Enter consume loop
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
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        await self._cleanup()
        await self.redis.disconnect()
        self.logger.info("worker_stopped")

    async def _cleanup(self) -> None:
        """Override for worker-specific cleanup (e.g. release GPU memory)."""

    async def _listen_route_updates(self) -> None:
        import json
        self._pubsub = self.redis.redis.pubsub()
        await self._pubsub.psubscribe("translationRoom:*:events")
        self.logger.info("route_listener_started")
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "pmessage":
                    try:
                        data = json.loads(message["data"])
                        if data.get("type") == "AUDIO_ROUTES_UPDATED":
                            # Extract room_id from channel: "translationRoom:{roomId}:events"
                            channel = message.get("channel", "")
                            if isinstance(channel, bytes):
                                channel = channel.decode("utf-8")
                            parts = channel.split(":")
                            room_id = parts[1] if len(parts) > 1 else data.get("roomId")
                            
                            # Extract status from room_status nested inside data payload, or top-level status
                            status = data.get("status")
                            if not status and "data" in data and isinstance(data["data"], dict):
                                status = data["data"].get("room_status")
                            if not status:
                                status = data.get("room_status")

                            if room_id and status:
                                self._route_states[room_id] = status
                                if status == "PAUSED":
                                    self._paused_rooms.add(room_id)
                                else:
                                    self._paused_rooms.discard(room_id)
                                
                                await self._on_route_status_changed(room_id, status)
                                
                                if status in ["FAILED", "ENDED", "CANCELLED", "TIMEOUT"]:
                                    self._cleanup_room(room_id)
                    except Exception as e:
                        self.logger.warning("failed_to_parse_route_event", error=str(e))
        except asyncio.CancelledError:
            self.logger.info("route_listener_stopped")
            if self._pubsub:
                await self._pubsub.close()

    async def _on_route_status_changed(self, room_id: str, new_status: str) -> None:
        """Override in subclasses to react to route status changes."""
        pass

    def _cleanup_room(self, room_id: str) -> None:
        """Override in subclasses to perform room-specific cleanup."""
        self._route_states.pop(room_id, None)
        self._paused_rooms.discard(room_id)

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
