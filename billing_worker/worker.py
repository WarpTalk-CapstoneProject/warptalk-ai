"""Billing settlement worker.

Consumes the BILLABLE AI pipeline result streams (translate:results, tts:results)
*after the fact* — via its own consumer groups, alongside whatever else already reads
those streams (e.g. TranscriptService's Redis consumer persists segment/translation
content; this worker only settles credits, it does not duplicate that job) — and turns
each billable event into subscription.usage_records + subscription.credit_transactions
rows.

WT-344: transcription (stt:results) and the inline assistant (ai_assistant:results)
are FREE and are deliberately not consumed here at all. A meeting gets its transcript
and its assistant without spending anything; it pays for translation and for dubbing.
Those two streams still exist and are still read by the transcript pipeline — this
worker simply has no business with them.

Does not subclass shared.base_worker.BaseWorker: that class is built around one
input_stream per instance plus a route-status pub/sub listener for the real-time
pipeline. This worker needs two streams and has nothing to react to in real time —
it settles after the work is already done, so it gets its own small run loop instead.

Scope note: this worker does NOT write transcript.audio_dubbings rows. Doing that
correctly requires translation_content_id, which in turn requires
transcript.translation_contents to already be populated (that belongs in
TranscriptService's own STT/translation Redis consumer, matching how transcript_segments
and transcript_translations already get persisted there) — inserting into
audio_dubbings from here without a real translation_content_id to point at would just
violate the FK. Charging credits does not require that row to exist.
"""

from __future__ import annotations

import asyncio
import json
import signal
import socket
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from billing_worker.db import BillingRepository
from shared.config import BillingSettings, RedisSettings, WorkerSettings
from shared.health_probe import heartbeat_key
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient
from shared.schemas import (
    TranslationResultMessage,
    TTSResultMessage,
)

logger = get_logger("worker.billing")

TRANSLATION_CHARGE_TYPE = "TRANSLATION"
SettlementHandler = Callable[[Mapping[Any, Any]], Awaitable[None]]


def _extract_underlying_segment_id(raw_segment_id: str) -> str | None:
    """Port of TranscriptRedisConsumerService.ExtractUnderlyingSegmentId (C#), byte-for-byte:
    translation_worker mints segment_id as f"{stt_segment_guid}-{target_lang}-c{idx}" (the
    target_lang keeps concurrent per-listener-language translations of the same STT segment
    from colliding on the same chunk id); tts_worker carries that composite string through
    unchanged. Slicing on the first 36 chars recovers the real TranscriptSegment.Id GUID
    regardless of what follows it.

    Returns None if the string isn't a valid GUID even after stripping the suffix — callers
    must treat that as "cannot attribute this charge to a segment", not crash.
    """
    if not raw_segment_id:
        return None
    guid_part = (
        raw_segment_id[:36]
        if len(raw_segment_id) > 36 and raw_segment_id[36] == "-"
        else raw_segment_id
    )
    try:
        return str(uuid.UUID(guid_part))
    except ValueError:
        return None


class BillingSettlementWorker:
    max_delivery_attempts = 5
    heartbeat_interval_seconds = 10
    heartbeat_ttl_seconds = 30

    def __init__(
        self,
        billing_settings: BillingSettings | None = None,
        redis_settings: RedisSettings | None = None,
        worker_settings: WorkerSettings | None = None,
    ) -> None:
        self.settings = billing_settings or BillingSettings()
        self.worker_settings = worker_settings or WorkerSettings()
        self.redis = RedisStreamClient(redis_settings or self.worker_settings.redis)
        self.db = BillingRepository(self.settings.database)
        self.logger = logger
        self._consumer_name = f"billing-{socket.gethostname()}"
        self._shutdown_event = asyncio.Event()
        self._last_progress_unix_ms = int(time.time() * 1000)
        # translation_room_id -> (subscription_id, workspace_id, cached_at_monotonic)
        self._subscription_cache: dict[str, tuple[uuid.UUID, uuid.UUID, float]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._register_signal_handlers()
        await self.redis.connect()
        await self.db.connect()
        await self._publish_heartbeat()
        self.logger.info("billing_worker_started")
        try:
            # WT-344: only TRANSLATION and TTS are billable.
            #
            # STT and the inline assistant were dropped from the billable set on the owner's
            # call, and the streams are no longer consumed AT ALL rather than consumed and
            # skipped. A consumer group that reads a stream only to discard it still costs a
            # Redis round trip per utterance, still holds a pending-entry list, and still
            # shows up in lag dashboards as a worker falling behind — which is exactly the
            # signal that hid a genuinely broken consumer once already.
            #
            # The product rule this encodes: transcription is what the meeting gets for
            # free, and translation and dubbing are what it pays for.
            await asyncio.gather(
                self._heartbeat_loop(),
                self._consume_loop(
                    "translate:results", "billing-translation-workers", self._handle_translation
                ),
                self._consume_loop("tts:results", "billing-tts-workers", self._handle_tts),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self.db.disconnect()
            await self.redis.disconnect()
            self.logger.info("billing_worker_stopped")

    async def _consume_loop(
        self,
        stream: str,
        group: str,
        handler: SettlementHandler,
    ) -> None:
        while not self._shutdown_event.is_set():
            try:
                await self._recover_stale(stream, group, handler)
                async for message_id, data in self.redis.consume(
                    stream=stream,
                    group=group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                ):
                    if self._shutdown_event.is_set():
                        break
                    await self._process_settlement_message(
                        stream,
                        message_id,
                        data,
                        handler,
                    )
                self._last_progress_unix_ms = int(time.time() * 1000)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error", stream=stream)
                await asyncio.sleep(1.0)

    async def _process_settlement_message(
        self,
        stream: str,
        message_id: bytes,
        data: dict[bytes, bytes],
        handler: SettlementHandler,
    ) -> None:
        try:
            await handler(data)
            self._last_progress_unix_ms = int(time.time() * 1000)
        except Exception:
            self.logger.exception(
                "settlement_error",
                stream=stream,
                message_id=message_id,
            )
            raise

    async def _recover_stale(
        self,
        stream: str,
        group: str,
        handler: SettlementHandler,
    ) -> None:
        messages = await self.redis.reclaim_stale(
            stream,
            group,
            self._consumer_name,
        )
        for message_id, data in messages:
            try:
                await self._process_settlement_message(
                    stream,
                    message_id,
                    data,
                    handler,
                )
            except Exception:
                attempts = await self.redis.pending_delivery_count(
                    stream,
                    group,
                    message_id,
                )
                if attempts >= self.max_delivery_attempts:
                    payload = {
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
                        f"{stream}:dead-letter",
                        {
                            "original_message_id": message_id.decode(
                                "utf-8",
                                errors="replace",
                            ),
                            "consumer_group": group,
                            "worker": "billing",
                            "delivery_attempts": attempts,
                            "failed_at_unix_ms": int(time.time() * 1000),
                            "payload": json.dumps(payload),
                        },
                    )
                    await self.redis.redis.xack(stream, group, message_id)
                    self.logger.error(
                        "settlement_dead_lettered",
                        stream=stream,
                        message_id=message_id,
                        attempts=attempts,
                    )
                continue
            await self.redis.redis.xack(stream, group, message_id)

    async def _publish_heartbeat(self) -> None:
        hostname = self._consumer_name.removeprefix("billing-")
        now_unix_ms = int(time.time() * 1000)
        await self.redis.set_with_ttl(
            heartbeat_key("billing", hostname),
            json.dumps(
                {
                    "worker": "billing",
                    "consumer": self._consumer_name,
                    "streams": [
                        "translate:results",
                        "tts:results",
                    ],
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
                self.logger.exception("billing_heartbeat_failed")

    # ------------------------------------------------------------------
    # Subscription resolution (cached per translation_room_id)
    # ------------------------------------------------------------------

    async def _resolve_subscription(
        self,
        translation_room_id: str,
    ) -> tuple[uuid.UUID, uuid.UUID] | None:
        cached = self._subscription_cache.get(translation_room_id)
        now = time.monotonic()
        if cached and now - cached[2] < self.settings.subscription_cache_ttl_seconds:
            return cached[0], cached[1]

        raw_room = await self.redis.get(f"meeting:room:{translation_room_id}")
        if raw_room is None:
            self.logger.warning(
                "room_projection_missing",
                translation_room_id=translation_room_id,
            )
            raise RuntimeError(
                f"Room projection is unavailable for translation room {translation_room_id}"
            )
        if isinstance(raw_room, bytes):
            raw_room = raw_room.decode("utf-8")
        room_projection = json.loads(raw_room)
        workspace_id = room_projection.get("WorkspaceId") or room_projection.get("workspaceId")
        if not workspace_id:
            self.logger.warning(
                "room_projection_missing_workspace",
                translation_room_id=translation_room_id,
            )
            raise RuntimeError(
                f"Room projection is unavailable for translation room {translation_room_id}: "
                "workspace id is missing"
            )

        resolved = await self.db.resolve_subscription(workspace_id)
        if resolved is None:
            return None

        subscription_id, workspace_id = resolved
        self._subscription_cache[translation_room_id] = (subscription_id, workspace_id, now)
        return subscription_id, workspace_id

    # ------------------------------------------------------------------
    # Per-stream handlers
    # ------------------------------------------------------------------

    async def _handle_translation(self, data: Mapping[Any, Any]) -> None:
        msg = TranslationResultMessage.from_redis(data)
        if not msg.translated_text.strip():
            return

        resolved = await self._resolve_subscription(msg.meeting_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        # msg.segment_id here is a composite string "{stt_segment_guid}-{target_lang}-c{idx}"
        # (minted in translation_worker/worker.py), not a valid GUID on its own — recover the real
        # TranscriptSegment.Id before using it as a UUID column value. Previously the raw
        # composite string was passed straight into reference_id (a UUID column), which
        # silently failed to bind and dropped the charge.
        underlying_segment_id = _extract_underlying_segment_id(msg.segment_id)
        if underlying_segment_id is None:
            self.logger.warning("segment_id_extraction_failed", raw_segment_id=msg.segment_id)

        quantity_s = (
            max((msg.end_ms - msg.start_ms) / 1000.0, 0.1) if msg.end_ms > msg.start_ms else 1.0
        )
        await self.db.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=msg.speaker_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=TRANSLATION_CHARGE_TYPE,
            charge_type=TRANSLATION_CHARGE_TYPE,
            reference_id=underlying_segment_id,
            reference_type="translation_content",
            quantity=quantity_s,
            unit="second",
            source_language_code=msg.source_lang,
            target_language_code=msg.target_lang,
            transcript_segment_id=underlying_segment_id,
            # msg.segment_id here IS deterministic (translation_worker builds it as
            # f"{stt_result.segment_id}-{target_lang}-c{idx}"), so this key is redelivery-safe. The
            # idempotency key keeps using the raw composite msg.segment_id, unaffected by
            # the extraction above (which only changes what's stored in reference_id /
            # transcript_segment_id).
            idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{msg.segment_id}:{msg.target_lang}",
        )

    async def _handle_tts(self, data: Mapping[Any, Any]) -> None:
        msg = TTSResultMessage.from_redis(data)
        if msg.cache_hit:
            return  # reused a previously synthesized clip — no new provider cost incurred
        if not msg.audio_data:
            return

        resolved = await self._resolve_subscription(msg.meeting_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        charge_type = (
            "AUDIO_DUBBING_VOICE_CLONE" if msg.voice_type == "cloned" else "AUDIO_DUBBING_STANDARD"
        )

        # Same composite-segment-id situation as _handle_translation above — extract the
        # real TranscriptSegment.Id before using it as a UUID column value.
        underlying_segment_id = _extract_underlying_segment_id(msg.segment_id)
        if underlying_segment_id is None:
            self.logger.warning("segment_id_extraction_failed", raw_segment_id=msg.segment_id)

        quantity_s = max(msg.duration_ms / 1000.0, 0.1)
        await self.db.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=msg.speaker_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=charge_type,
            charge_type=charge_type,
            reference_id=underlying_segment_id,
            reference_type="audio_dubbing",
            quantity=quantity_s,
            unit="second",
            target_language_code=msg.target_lang,
            transcript_segment_id=underlying_segment_id,
            idempotency_key=f"{charge_type}:{msg.segment_id}:{msg.target_lang}",
            details={"clone_provider": msg.clone_provider, "voice_mode": msg.voice_mode},
        )

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)
