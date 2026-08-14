"""Abstract base worker with lifecycle management.

Provides:
- Redis connection with pooling and retry
- Model loading via asyncio.to_thread (non-blocking)
- Graceful shutdown with SIGTERM/SIGINT handling
- Structured logging with worker context
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import time
from abc import ABC, abstractmethod
from typing import Any

from redis.asyncio.client import PubSub

from shared.config import RedisSettings, WorkerSettings
from shared.health_probe import heartbeat_key
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient

# WT-314. The room lifecycle states after which a worker must release everything it holds
# for that room — for livekit_ingress_worker that is the bot's LiveKit connection, and
# therefore billed connection minutes.
#
# This must stay in step with the backend's RoomStatus enum
# (translation-room/.../Domain/Enums/RoomStatus.cs), which is what
# AudioRouteCacheService.PublishRoutesUpdateAsync puts on the wire as room_status. It had
# drifted in both directions: "EXPIRED" is a real terminal status (set by
# TranslationRoomService.ExpireTranslationRoomAsync and by IdleRoomMonitoringWorker) and
# was missing here, so an expired room's bot was never released; "TIMEOUT" is not a status
# the backend has ever published, so that entry never matched anything.
TERMINAL_ROOM_STATUSES = frozenset({"FAILED", "ENDED", "CANCELLED", "EXPIRED"})


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

    # Opt-in: how many messages this worker processes at once. Default 1 preserves the
    # exact one-at-a-time behavior every worker has always had — audio/text workers with
    # per-room ordering assumptions (stt/translation/tts/assistant/livekit-ingress) must NOT
    # change this. Set > 1 only for workers whose messages are independent of each other and
    # whose per-message work is I/O-bound (e.g. embedding_worker: an OpenAI embed call + a
    # Qdrant upsert per message) — see consume_concurrent() in shared/redis_client.py for why
    # a single message's ack is still tied to its own handler completion under concurrency.
    concurrency: int = 1
    max_delivery_attempts: int = 5
    heartbeat_interval_seconds: int = 10
    heartbeat_ttl_seconds: int = 30
    processing_timeout_seconds: float = 120

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
        # Whether TRANSLATION is running, per room, as last reported by the backend. Separate
        # from _route_states because the room's status cannot answer it — see
        # _is_translation_active. Absent means "the backend has not said", not "no".
        self._translation_active: dict[str, bool] = {}
        self._paused_rooms: set[str] = set()
        self._room_routes: dict[str, list[dict[str, Any]]] = {}
        self._pubsub: PubSub | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._last_progress_unix_ms = int(time.time() * 1000)

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
            await self._publish_heartbeat()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

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
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        await self._cleanup()
        await self.redis.disconnect()
        self.logger.info("worker_stopped")

    async def _cleanup(self) -> None:
        """Override for worker-specific cleanup (e.g. release GPU memory)."""

    async def _publish_heartbeat(self) -> None:
        hostname = self._consumer_name.removeprefix(f"{self.worker_name}-")
        now_unix_ms = int(time.time() * 1000)
        await self.redis.set_with_ttl(
            heartbeat_key(self.worker_name, hostname),
            json.dumps(
                {
                    "worker": self.worker_name,
                    "consumer": self._consumer_name,
                    "stream": self.input_stream,
                    "group": self.consumer_group,
                    "timestamp_unix_ms": now_unix_ms,
                    "last_progress_unix_ms": getattr(
                        self,
                        "_last_progress_unix_ms",
                        now_unix_ms,
                    ),
                }
            ),
            self.heartbeat_ttl_seconds,
        )

    async def _heartbeat_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self.heartbeat_interval_seconds)
            try:
                await self._publish_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Redis restarts and failovers are expected transient faults.
                # Keep the heartbeat supervisor alive so monitoring evidence
                # returns automatically when Redis becomes available again.
                self.logger.exception("worker_heartbeat_failed")

    async def _listen_route_updates(self) -> None:
        retry_delay_seconds = 1.0
        while not self._shutdown_event.is_set():
            pubsub = self.redis.redis.pubsub()
            self._pubsub = pubsub
            connected_at: float | None = None
            try:
                await pubsub.psubscribe("translationRoom:*:events")
                connected_at = asyncio.get_running_loop().time()
                self.logger.info("route_listener_started")
                async for message in pubsub.listen():
                    if self._shutdown_event.is_set():
                        break
                    await self._handle_route_update_message(message)
            except asyncio.CancelledError:
                self.logger.info("route_listener_stopped")
                raise
            except Exception:
                self.logger.exception("route_listener_failed")
                if not self._shutdown_event.is_set():
                    if (
                        connected_at is not None
                        and asyncio.get_running_loop().time() - connected_at >= 30
                    ):
                        retry_delay_seconds = 1.0
                    await asyncio.sleep(retry_delay_seconds)
                    retry_delay_seconds = min(retry_delay_seconds * 2, 30.0)
            finally:
                try:
                    await pubsub.close()
                except Exception:
                    self.logger.warning("route_listener_close_failed")
                if self._pubsub is pubsub:
                    self._pubsub = None

    async def _handle_route_update_message(self, message: dict[str, Any]) -> None:
        if message.get("type") != "pmessage":
            return

        try:
            data = json.loads(message["data"])
            if data.get("type") != "AUDIO_ROUTES_UPDATED":
                return

            # Extract room_id from channel: "translationRoom:{roomId}:events"
            channel = message.get("channel", "")
            if isinstance(channel, bytes):
                channel = channel.decode("utf-8")
            parts = channel.split(":")
            room_id = parts[1] if len(parts) > 1 else data.get("roomId")

            inner = data.get("data") if isinstance(data.get("data"), dict) else {}
            status = data.get("status") or inner.get("room_status") or data.get("room_status")

            if room_id and isinstance(inner.get("routes"), list):
                self._room_routes[room_id] = inner["routes"]

            # The backend's own answer to "is translation running", computed from the room's
            # active TranslationRoomSession. Recorded separately from the room status below, and
            # deliberately only when present: an older backend does not send it, and treating a
            # missing field as False would silently stop translating for the whole fleet during a
            # rolling deploy.
            translation_active = inner.get("translation_active")
            if room_id and isinstance(translation_active, bool):
                self._translation_active[room_id] = translation_active

            if room_id and status:
                self._route_states[room_id] = status
                if status == "PAUSED":
                    self._paused_rooms.add(room_id)
                else:
                    self._paused_rooms.discard(room_id)

                await self._on_route_status_changed(room_id, status)

                if status in TERMINAL_ROOM_STATUSES:
                    self._cleanup_room(room_id)
        except Exception as error:
            self.logger.warning("failed_to_parse_route_event", error=str(error))

    async def _load_route_snapshot(self, room_id: str) -> bool:
        """Re-read a room's route state from Redis, into the in-memory caches.

        WHY THIS EXISTS
            `_translation_active`, `_route_states` and `_room_routes` were populated ONLY by the
            AUDIO_ROUTES_UPDATED pub/sub event. Pub/sub has no replay: a worker that was not
            subscribed at the instant the backend published — because it was restarting, which is
            what every deploy does to every worker — never learns the state and has no way to ask.

            For `_is_translation_active` that is not a degraded answer, it is a WRONG one. An
            unknown room falls through to a status lookup that is also empty, and the gate in
            translation_worker then reads False and drops every STT result for a live meeting,
            permanently, behind a `logger.debug` that INFO-level production never prints. The dub
            simply stops and nothing anywhere says so.

            The comment on `_handle_route_update_message` claims a missing field must not be read
            as False because it "would silently stop translating for the whole fleet during a
            rolling deploy". That reasoning is right and the protection was incomplete: it guards
            the field being absent from a message, not the message being missed entirely.

            Nothing new has to be published for this. The backend already writes the identical
            payload to `translationRoom:{id}:audio_routes` as a durable key — same `routes`, same
            `room_status`, same `translation_active` — and `_get_target_languages` has always read
            its own state from Redis rather than waiting to be told. This closes the asymmetry.

        Returns True when a snapshot was found and applied.
        """
        try:
            raw = await self.redis.get(f"translationRoom:{room_id}:audio_routes")
        except Exception as error:
            self.logger.warning("route_snapshot_read_failed", room_id=room_id, error=str(error))
            return False

        if not raw:
            return False

        try:
            snapshot = json.loads(raw)
        except (ValueError, TypeError) as error:
            self.logger.warning("route_snapshot_parse_failed", room_id=room_id, error=str(error))
            return False

        if not isinstance(snapshot, dict):
            return False

        if isinstance(snapshot.get("routes"), list):
            self._room_routes[room_id] = snapshot["routes"]

        status = snapshot.get("room_status")
        if isinstance(status, str) and status:
            self._route_states[room_id] = status
            if status == "PAUSED":
                self._paused_rooms.add(room_id)

        translation_active = snapshot.get("translation_active")
        if isinstance(translation_active, bool):
            self._translation_active[room_id] = translation_active

        self.logger.info(
            "route_snapshot_recovered",
            room_id=room_id,
            room_status=status,
            translation_active=translation_active,
        )
        return True

    async def _translation_active_for(self, room_id: str) -> bool:
        """`_is_translation_active`, but allowed to go and find out.

        The sync version can only answer from what a broadcast happened to deliver. This one
        recovers the snapshot from Redis on a miss, so a worker that restarted mid-meeting picks
        the meeting back up instead of staying deaf to it until the room ends.
        """
        if room_id in getattr(self, "_translation_active", {}):
            return self._translation_active[room_id]

        if await self._load_route_snapshot(room_id):
            return self._is_translation_active(room_id)

        # No snapshot at all. Fall back to the sync answer, which errs towards translating when a
        # status is known — the pre-existing rolling-deploy behaviour, deliberately unchanged.
        return self._is_translation_active(room_id)

    def _is_translation_active(self, room_id: str) -> bool:
        """Whether someone has actually started translation for this room.

        Lives here, not in one worker, because the answer decides different things in
        different stages and getting that split wrong took the transcript down: while this
        was livekit_ingress_worker's private helper it gated AUDIO, so nothing was
        transcribed until translation started and the two features could not be used
        apart. Ingress now transcribes any live meeting and only translation_worker asks
        this question, which is the stage that actually spends a translation.

        The answer is `translation_active` from AUDIO_ROUTES_UPDATED, which the backend
        computes from the room's active TranslationRoomSession — the row Start Translation
        opens and Stop closes.

        It used to be read off the room STATUS instead, and that was wrong in a way that
        could not be worked around here: a room is IN_PROGRESS from the moment somebody
        opens it, and since WT-339 opening a room deliberately does not start translation.
        So "the meeting is live" and "translation is running" were the same value, and no
        gate built on it could tell transcript-only from transcript-plus-translation.

        The status fallback remains for a backend that predates the flag — during a rolling
        deploy the old meaning is the safe one, since it errs towards translating rather
        than towards a silent pipeline.
        """
        explicit: dict[str, bool] = getattr(self, "_translation_active", {})
        if room_id in explicit:
            return explicit[room_id]

        states: dict[str, str] = getattr(self, "_route_states", {})
        return states.get(room_id) in {
            "IN_PROGRESS",
            "AUDIO_ROUTING_ACTIVE",
        }

    async def _on_route_status_changed(self, room_id: str, new_status: str) -> None:
        """Override in subclasses to react to route status changes."""
        pass

    def _cleanup_room(self, room_id: str) -> None:
        """Override in subclasses to perform room-specific cleanup."""
        self._route_states.pop(room_id, None)
        self._translation_active.pop(room_id, None)
        self._paused_rooms.discard(room_id)
        self._room_routes.pop(room_id, None)

    def is_voice_clone_consented(self, room_id: str, speaker_user_id: str) -> bool:
        """True if `speaker_user_id` has at least one current outgoing route (they are the
        source/speaker) with VoiceCloneEnabled = true.

        Voice cloning captures biometric data — this is the consent gate. Routes are keyed
        by translation_room_participants.id in Postgres, but the AI pipeline identifies
        speakers by auth user id, so this matches on the denormalized SourceUserId field
        (see TranslationRoomAudioRouteMapper.ToDto). Returns False (no consent) if routes
        haven't been received yet for this room — fail closed, never clone without a
        confirmed opt-in.
        """
        routes = self._room_routes.get(room_id, [])
        return any(
            str(route.get("SourceUserId") or "").lower() == speaker_user_id.lower()
            and bool(route.get("VoiceCloneEnabled"))
            for route in routes
        )

    def chosen_dub_voice(self, room_id: str, speaker_user_id: str) -> str | None:
        """The voice this speaker asked to be dubbed in, or None to clone them live (WT-396).

        Read from the same route snapshot as the consent gate above, and matched the same way —
        on SourceUserId, because routes are keyed by participant id in Postgres while the AI
        pipeline knows people by auth user id.

        None on an unknown room, exactly like is_voice_clone_consented, but for the opposite
        reason: there the unknown answer must fail closed because it guards biometric processing,
        here it simply means nobody has told us a preference yet and the pipeline should do what
        it always did. Missing the field entirely is also None — an older backend does not send
        it, and during a rolling deploy half the fleet is talking to one that does not.
        """
        for route in self._room_routes.get(room_id, []):
            if str(route.get("SourceUserId") or "").lower() != speaker_user_id.lower():
                continue
            voice_id = route.get("SourceDubVoiceId")
            if voice_id:
                return str(voice_id)
        return None

    async def voice_clone_consent_state(
        self, room_id: str, speaker_user_id: str
    ) -> tuple[bool, str]:
        """`is_voice_clone_consented`, allowed to go and find out — and to say which answer it gave.

        THE ASYMMETRY THIS CLOSES
            The sync version fails closed on an unknown room, which is right: never clone a voice
            without a confirmed opt-in. But it cannot tell "this speaker did not opt in" apart from
            "this worker has never been told anything about this room", and those are different
            facts with different fixes.

            `_room_routes` is populated only by the AUDIO_ROUTES_UPDATED pub/sub broadcast, and
            pub/sub has no replay. Every deploy restarts every worker, so a worker that comes up
            mid-meeting never learns that room's routes and answers "no consent" for the rest of
            the meeting — silently, for every speaker in it. That is exactly the failure
            `_translation_active_for` was added to close for the translation gate; the consent gate
            reads the same cache and never got the same treatment.

            The snapshot needs nothing new: the backend already writes the identical payload to
            `translationRoom:{id}:audio_routes` as a durable key.

        Returns (consented, reason) where reason is one of:
            "consented"      — a current outgoing route for this speaker has VoiceCloneEnabled.
            "not_opted_in"   — routes are known and none of this speaker's has it enabled.
            "routes_unknown" — no routes for this room, and no snapshot to recover them from.
                               Still fails closed, but now says so instead of looking identical
                               to a deliberate opt-out.
        """
        if self.is_voice_clone_consented(room_id, speaker_user_id):
            return True, "consented"

        if room_id not in self._room_routes and await self._load_route_snapshot(room_id):
            if self.is_voice_clone_consented(room_id, speaker_user_id):
                return True, "consented"

        if self._room_routes.get(room_id):
            return False, "not_opted_in"
        return False, "routes_unknown"

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
    ) -> bytes | str:
        """Publish result to a meeting-specific output stream.

        Args:
            stream_prefix: e.g. "stt:results"
            meeting_id: meeting identifier
            data: serialized message dict from schema.to_redis()

        Returns:
            Redis message ID
        """
        # Publish to per-meeting stream (backend expects e.g. stt:results:{roomId})
        room_stream = f"{stream_prefix}:{meeting_id}"
        msg_id = await self.redis.publish(room_stream, data)

        # Also publish to global stream (AI workers consume from here)
        await self.redis.publish(stream_prefix, data)

        self.logger.debug(
            "message_published",
            stream=room_stream,
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
            concurrency=self.concurrency,
        )

        while not self._shutdown_event.is_set():
            try:
                await self._recover_stale_messages()
                if self.concurrency > 1:
                    # Owns its own read-dispatch-ack cycle — see consume_concurrent()'s
                    # docstring for why this can't be built by just not awaiting process()
                    # in the loop below.
                    await self.redis.consume_concurrent(
                        stream=self.input_stream,
                        group=self.consumer_group,
                        handler=self._process_and_log_errors,
                        consumer=self._consumer_name,
                        block_ms=2000,
                        count=self.concurrency,
                        concurrency=self.concurrency,
                    )
                else:
                    async for message_id, data in self.redis.consume(
                        stream=self.input_stream,
                        group=self.consumer_group,
                        consumer=self._consumer_name,
                        block_ms=2000,  # Check shutdown every 2s
                    ):
                        if self._shutdown_event.is_set():
                            break

                        await self._process_and_log_errors(message_id, data)

                self._last_progress_unix_ms = int(time.time() * 1000)

            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                # Exponential backoff before retry
                await asyncio.sleep(1.0)

    async def _recover_stale_messages(self) -> None:
        """Reprocess messages abandoned in the consumer group's pending list."""
        messages = await self.redis.reclaim_stale(
            self.input_stream,
            self.consumer_group,
            self._consumer_name,
        )
        for message_id, data in messages:
            try:
                await self._process_and_log_errors(message_id, data)
            except Exception:
                attempts = await self.redis.pending_delivery_count(
                    self.input_stream,
                    self.consumer_group,
                    message_id,
                )
                if attempts >= self.max_delivery_attempts:
                    original_payload = {
                        (
                            key.decode("utf-8", errors="replace")
                            if isinstance(key, bytes)
                            else str(key)
                        ): (
                            value.decode("utf-8", errors="replace")
                            if isinstance(value, bytes)
                            else str(value)
                        )
                        for key, value in data.items()
                    }
                    await self.redis.publish(
                        f"{self.input_stream}:dead-letter",
                        {
                            "original_message_id": message_id.decode(
                                "utf-8",
                                errors="replace",
                            ),
                            "consumer_group": self.consumer_group,
                            "worker": self.worker_name,
                            "delivery_attempts": attempts,
                            "failed_at_unix_ms": int(time.time() * 1000),
                            "payload": json.dumps(original_payload),
                        },
                    )
                    await self.redis.redis.xack(
                        self.input_stream,
                        self.consumer_group,
                        message_id,
                    )
                    self.logger.error(
                        "message_dead_lettered",
                        message_id=message_id,
                        stream=self.input_stream,
                        attempts=attempts,
                    )
                # Under the limit, keep it pending. XAUTOCLAIM resets idle time,
                # preventing a hot loop while scheduling another bounded retry.
                continue
            await self.redis.redis.xack(
                self.input_stream,
                self.consumer_group,
                message_id,
            )

    async def _process_and_log_errors(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        try:
            await asyncio.wait_for(
                self.process(message_id, data),
                timeout=self.processing_timeout_seconds,
            )
            self._last_progress_unix_ms = int(time.time() * 1000)
        except Exception:
            self.logger.exception(
                "process_error",
                message_id=message_id,
                stream=self.input_stream,
            )
            # Propagate so the Redis consumer does not XACK the message. It remains
            # pending and can be reclaimed/retried by this or another worker.
            raise

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
