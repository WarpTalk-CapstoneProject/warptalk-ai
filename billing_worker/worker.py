"""Billing settlement worker.

Consumes the three AI pipeline result streams (stt:results, translate:results,
tts:results) *after the fact* — via its own consumer groups, alongside whatever else
already reads those streams (e.g. TranscriptService's Redis consumer persists segment/
translation content; this worker only settles credits, it does not duplicate that job)
— and turns each billable event into subscription.usage_records +
subscription.credit_transactions rows.

Does not subclass shared.base_worker.BaseWorker: that class is built around one
input_stream per instance plus a route-status pub/sub listener for the real-time
pipeline. This worker needs three streams and has nothing to react to in real time —
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
    STTResultMessage,
    SuggestionResultMessage,
    TranslationResultMessage,
    TTSResultMessage,
)

logger = get_logger("worker.billing")

STT_CHARGE_TYPE = "STT"
TRANSLATION_CHARGE_TYPE = "TRANSLATION"
# Inline transcript suggestions settle against the existing AI_ASSISTANT charge type
# rather than a new one — it is already declared in migration 017 and already has rate
# card rows from migration 039, so suggestions draw down the same assistant budget a
# workspace has always had instead of introducing a second one to reason about.
SUGGESTION_CHARGE_TYPE = "AI_ASSISTANT"
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
        self._pending_charges = []

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
            await asyncio.gather(
                self._heartbeat_loop(),
                self._consume_loop("stt:results", "billing-stt-workers", self._handle_stt),
                self._consume_loop(
                    "translate:results", "billing-translation-workers", self._handle_translation
                ),
                self._consume_loop("tts:results", "billing-tts-workers", self._handle_tts),
                self._consume_loop(
                    "ai_assistant:results",
                    "billing-suggestion-workers",
                    self._handle_suggestion,
                ),
                self._flush_charges_loop(),
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
                        "stt:results",
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

    async def _flush_charges_loop(self) -> None:
        """Batch process charges every 10 seconds to avoid DB spam (Demo Requirement)."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(10)
            if not self._pending_charges:
                continue

            batch = self._pending_charges
            self._pending_charges = []
            
            # Aggregate tokens deducted per room
            tokens_per_room: dict[str, int] = {}
            workspace_per_room: dict[str, str] = {}
            
            for kwargs in batch:
                room_id = kwargs.get("translation_room_id")
                ws_id = kwargs.get("workspace_id")
                try:
                    credits = await self.db.record_usage_and_charge(**kwargs)
                    if room_id and credits > 0:
                        tokens_per_room[room_id] = tokens_per_room.get(room_id, 0) + credits
                        workspace_per_room[room_id] = str(ws_id)
                except RuntimeError as e:
                    if "Insufficient credits" in str(e):
                        self.logger.warning("meeting_credit_exhausted", room_id=room_id)
                        if room_id:
                            await self.redis.publish("warptalk:translation-room:commands", json.dumps({
                                "Command": "MeetingCreditExhausted",
                                "RoomId": str(room_id),
                                "WorkspaceId": str(ws_id) if ws_id else None
                            }))
                except Exception:
                    self.logger.exception("batch_charge_failed")

            # Publish TokenUsageUpdated for each room
            for room_id, tokens in tokens_per_room.items():
                if tokens > 0:
                    try:
                        await self.redis.publish("warptalk:translation-room:commands", json.dumps({
                            "Command": "TokenUsageUpdated",
                            "RoomId": str(room_id),
                            "WorkspaceId": workspace_per_room.get(room_id),
                            "TokensDeducted": tokens
                        }))
                    except Exception:
                        self.logger.exception("failed_to_publish_token_usage", room_id=room_id)

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

    async def _handle_stt(self, data: Mapping[Any, Any]) -> None:
        msg = STTResultMessage.from_redis(data)
        if not msg.text.strip():
            return  # empty/flush messages carry no billable transcription

        resolved = await self._resolve_subscription(msg.meeting_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        quantity_s = (
            max((msg.end_ms - msg.start_ms) / 1000.0, 0.1) if msg.end_ms > msg.start_ms else 1.0
        )
        self._pending_charges.append(dict(
            subscription_id=subscription_id,
            user_id=msg.speaker_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=STT_CHARGE_TYPE,
            charge_type=STT_CHARGE_TYPE,
            reference_id=msg.segment_id,
            reference_type="transcript_segment",
            quantity=quantity_s,
            unit="second",
            source_language_code=msg.language,
            transcript_segment_id=msg.segment_id,
            idempotency_key=f"{STT_CHARGE_TYPE}:{msg.segment_id}:NA",
        ))

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
        self._pending_charges.append(dict(
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
            idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{msg.segment_id}:{msg.target_lang}",
        ))

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
        self._pending_charges.append(dict(
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
        ))

    async def _handle_suggestion(self, data: Mapping[Any, Any]) -> None:
        """Settle one inline transcript suggestion against the AI_ASSISTANT budget.

        Shares the ai_assistant:results stream with meeting summaries and action items, so
        anything that is not a suggestion is skipped here — the summary path predates this
        worker and is not metered per message.

        Unlike the three real-time handlers above, this one NEVER re-raises. A settlement
        failure there is worth retrying because the user already received a caption they
        must be charged for; here the alternative is a suggestion — an optional aside —
        wedging this consumer group into an endless redelivery loop over one un-priced
        message. The charge is dropped with a loud log instead.
        """
        if data.get(b"type", data.get("type")) not in (b"suggestion", "suggestion"):
            return

        msg = SuggestionResultMessage.from_redis(data)
        if msg.token_count <= 0:
            # Nothing was actually spent (or the producer did not report it) — recording a
            # zero-quantity charge would only add noise to the usage ledger.
            return

        try:
            resolved = await self._resolve_subscription(msg.meeting_id)
        except Exception:
            self.logger.exception("suggestion_subscription_lookup_failed", room=msg.meeting_id)
            return

        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        # Unlike translation/TTS, this segment id is the STT segment's own GUID carried
        # through untouched by suggestion_worker — no composite suffix to strip.
        segment_id = _extract_underlying_segment_id(msg.segment_id)

        try:
            self._pending_charges.append(dict(
                subscription_id=subscription_id,
                user_id=None,
                workspace_id=workspace_id,
                translation_room_id=msg.meeting_id,
                usage_type=SUGGESTION_CHARGE_TYPE,
                charge_type=SUGGESTION_CHARGE_TYPE,
                reference_id=segment_id,
                reference_type="transcript_segment",
                quantity=float(msg.token_count),
                unit="token",
                source_language_code=msg.language or None,
                transcript_segment_id=segment_id,
                idempotency_key=f"{SUGGESTION_CHARGE_TYPE}:{msg.segment_id}",
                details={"category": msg.category, "confidence": msg.confidence},
            ))
        except Exception:
            self.logger.exception(
                "suggestion_settlement_failed",
                room=msg.meeting_id,
                segment_id=msg.segment_id,
                tokens=msg.token_count,
            )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)
