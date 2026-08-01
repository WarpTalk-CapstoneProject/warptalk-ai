"""Billing settlement worker.

Consumes billable AI pipeline streams (audio:chunks, ai:usage, billing:usage,
tts:results) *after the fact* — via its own consumer groups, alongside whatever else
already reads those streams (e.g. TranscriptService's Redis consumer persists segment/
translation content; this worker only settles credits, it does not duplicate that job)
— and turns each billable event into subscription.usage_records +
subscription.credit_transactions rows.

Does not subclass shared.base_worker.BaseWorker: that class is built around one
input_stream per instance plus a route-status pub/sub listener for the real-time
pipeline. This worker needs four streams and has nothing to react to in real time —
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
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from billing_worker.db import BillingRepository
from shared.config import BillingSettings, RedisSettings, STTSettings, TTSSettings, WorkerSettings
from shared.health_probe import heartbeat_key
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient
from shared.schemas import AIUsageMessage, AudioChunkMessage, ProviderUsageMessage, TTSResultMessage

logger = get_logger("worker.billing")

STT_CHARGE_TYPE = "STT"
OPENAI_PROVIDER = "openai"
TTS_STANDARD_CHARGE_TYPE = "AUDIO_DUBBING_STANDARD"
TTS_VOICE_CLONE_CHARGE_TYPE = "AUDIO_DUBBING_VOICE_CLONE"
DEFAULT_TTS_VOICE_CLONE_MODEL = "sonic-3.5-clone"
ACCUMULATOR_KEY_PREFIX = "billing:acc"
ACCUMULATOR_SEQUENCE_PREFIX = "billing:accseq"
CHARGE_DEDUPE_PREFIX = "billing:charge"
EVENT_DEDUPE_PREFIX = "billing:event"
MICRO_SCALE = Decimal("1000000")
SettlementHandler = Callable[[Mapping[Any, Any]], Awaitable[None]]


@dataclass(frozen=True)
class BillingEvent:
    subscription_id: uuid.UUID
    workspace_id: uuid.UUID
    translation_room_id: str
    usage_type: str
    charge_type: str
    quantity: Decimal
    unit: str
    credits_event: Decimal
    event_idempotency_key: str
    provider: str
    model: str
    pricing_rate_card_id: uuid.UUID
    unit_price_snapshot: Decimal
    source_language_code: str | None = None
    target_language_code: str | None = None
    force_flush: bool = False
    details: dict | None = None


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
        self.db = BillingRepository(
            self.settings.database,
            redis_settings or self.worker_settings.redis,
        )
        self.stt_settings = STTSettings()
        self.tts_settings = TTSSettings()
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
            await asyncio.gather(
                self._heartbeat_loop(),
                self._consume_loop("audio:chunks", "billing-stt-workers", self._handle_stt),
                self._consume_loop("ai:usage", "billing-ai-usage-workers", self._handle_ai_usage),
                self._consume_loop(
                    "billing:usage", "billing-provider-usage-workers", self._handle_provider_usage
                ),
                self._consume_loop("tts:results", "billing-tts-workers", self._handle_tts),
                self._flush_loop(),
            )
        except asyncio.CancelledError:
            pass
        finally:
            await self._flush_all_accumulators()
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
                        "audio:chunks",
                        "ai:usage",
                        "billing:usage",
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

    async def _handle_stt(self, data: Mapping[Any, Any]) -> None:
        msg = AudioChunkMessage.from_redis(data)
        if not msg.audio_data or msg.sample_rate <= 0:
            return

        resolved = await self._resolve_subscription(msg.meeting_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        audio_seconds = Decimal(len(msg.audio_data)) / Decimal(2) / Decimal(msg.sample_rate)
        if audio_seconds <= 0:
            return
        rate = await self.db.resolve_usage_rate(
            charge_type=STT_CHARGE_TYPE,
            unit="second",
            provider=self.stt_settings.provider,
            model=self.stt_settings.model,
        )
        if rate is None:
            self._log_missing_rate_skip(
                event_source="audio_chunks",
                charge_type=STT_CHARGE_TYPE,
                unit="second",
                provider=self.stt_settings.provider,
                model=self.stt_settings.model,
                translation_room_id=msg.meeting_id,
            )
            return

        await self._accumulate_and_maybe_flush(BillingEvent(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=STT_CHARGE_TYPE,
            charge_type=STT_CHARGE_TYPE,
            quantity=audio_seconds,
            unit="second",
            credits_event=audio_seconds * rate.unit_price,
            event_idempotency_key=f"{STT_CHARGE_TYPE}:{msg.meeting_id}:{msg.speaker_id}:{msg.chunk_index}",
            provider=rate.provider,
            model=rate.model,
            pricing_rate_card_id=rate.id,
            unit_price_snapshot=rate.unit_price,
            source_language_code=None,
            target_language_code=None,
            force_flush=msg.is_final_chunk,
            details={"event_source": "audio_chunks", "sample_rate": msg.sample_rate},
        ))

    async def _handle_ai_usage(self, data: Mapping[Any, Any]) -> None:
        msg = AIUsageMessage.from_redis(data)
        if msg.prompt_tokens <= 0 and msg.cached_tokens <= 0 and msg.completion_tokens <= 0:
            return

        resolved = await self._resolve_subscription(msg.room_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.room_id)
            return
        subscription_id, workspace_id = resolved

        uncached_prompt_tokens = max(msg.prompt_tokens - msg.cached_tokens, 0)
        token_components = (
            ("token_in", uncached_prompt_tokens),
            ("token_in_cached", msg.cached_tokens),
            ("token_out", msg.completion_tokens),
        )

        for unit, quantity in token_components:
            if quantity <= 0:
                continue

            rate = await self.db.resolve_usage_rate(
                charge_type=msg.charge_type,
                unit=unit,
                provider=OPENAI_PROVIDER,
                model=msg.model,
                source_language_code=msg.source_lang or None,
                target_language_code=msg.target_lang or None,
            )
            if rate is None:
                self._log_missing_rate_skip(
                    event_source="ai_usage",
                    charge_type=msg.charge_type,
                    unit=unit,
                    provider=OPENAI_PROVIDER,
                    model=msg.model,
                    translation_room_id=msg.room_id,
                    source_language_code=msg.source_lang or None,
                    target_language_code=msg.target_lang or None,
                )
                continue

            quantity_decimal = Decimal(quantity)
            await self._accumulate_and_maybe_flush(BillingEvent(
                subscription_id=subscription_id,
                workspace_id=workspace_id,
                translation_room_id=msg.room_id,
                usage_type=msg.charge_type,
                charge_type=msg.charge_type,
                quantity=quantity_decimal,
                unit=unit,
                credits_event=quantity_decimal * rate.unit_price,
                event_idempotency_key=f"{msg.idempotency_key}:{unit}",
                provider=rate.provider,
                model=rate.model,
                pricing_rate_card_id=rate.id,
                unit_price_snapshot=rate.unit_price,
                source_language_code=msg.source_lang or None,
                target_language_code=msg.target_lang or None,
                force_flush=True,
                details={
                    "event_source": "ai_usage",
                    "source_lang": msg.source_lang,
                    "target_lang": msg.target_lang,
                    "prompt_tokens": msg.prompt_tokens,
                    "cached_tokens": msg.cached_tokens,
                    "completion_tokens": msg.completion_tokens,
                    "token_component": unit,
                },
            ))

    async def _handle_provider_usage(self, data: Mapping[Any, Any]) -> None:
        msg = ProviderUsageMessage.from_redis(data)
        if msg.quantity <= 0:
            return

        resolved = await self._resolve_subscription(msg.room_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.room_id)
            return
        subscription_id, workspace_id = resolved

        rate = await self.db.resolve_usage_rate(
            charge_type=msg.charge_type,
            unit=msg.unit,
            provider=msg.provider,
            model=msg.model,
        )
        if rate is None:
            self._log_missing_rate_skip(
                event_source="provider_usage",
                charge_type=msg.charge_type,
                unit=msg.unit,
                provider=msg.provider,
                model=msg.model,
                translation_room_id=msg.room_id,
            )
            return

        if msg.charge_type == "VOICE_CLONE_ENROLLMENT":
            credits_consumed = int((msg.quantity * rate.unit_price).to_integral_value(rounding=ROUND_CEILING))
            await self.db.record_usage_and_charge(
                subscription_id=subscription_id,
                user_id=msg.user_id,
                workspace_id=workspace_id,
                translation_room_id=msg.room_id,
                usage_type=msg.charge_type,
                charge_type=msg.charge_type,
                reference_id=None,
                reference_type="voice_clone",
                quantity=float(msg.quantity),
                unit=msg.unit,
                credits_consumed=credits_consumed,
                transcript_segment_id=None,
                idempotency_key=msg.idempotency_key,
                pricing_rate_card_id=rate.id,
                unit_price_snapshot=rate.unit_price,
                provider=rate.provider,
                model=rate.model,
                details={
                    "event_source": "provider_usage",
                    "user_id": msg.user_id,
                },
            )
        else:
            await self._accumulate_and_maybe_flush(BillingEvent(
                subscription_id=subscription_id,
                workspace_id=workspace_id,
                translation_room_id=msg.room_id,
                usage_type=msg.charge_type,
                charge_type=msg.charge_type,
                quantity=msg.quantity,
                unit=msg.unit,
                credits_event=msg.quantity * rate.unit_price,
                event_idempotency_key=msg.idempotency_key,
                provider=rate.provider,
                model=rate.model,
                pricing_rate_card_id=rate.id,
                unit_price_snapshot=rate.unit_price,
                source_language_code=None,
                target_language_code=None,
                force_flush=True,
                details={
                    "event_source": "provider_usage",
                    "user_id": msg.user_id,
                },
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
            TTS_VOICE_CLONE_CHARGE_TYPE if msg.voice_type == "cloned" else TTS_STANDARD_CHARGE_TYPE
        )
        model = (
            DEFAULT_TTS_VOICE_CLONE_MODEL
            if msg.voice_type == "cloned"
            else self.tts_settings.model
        )

        # Same composite-segment-id situation as _handle_translation above — extract the
        # real TranscriptSegment.Id before using it as a UUID column value.
        underlying_segment_id = _extract_underlying_segment_id(msg.segment_id)
        if underlying_segment_id is None:
            self.logger.warning(
                "segment_id_extraction_failed", raw_segment_id=msg.segment_id
            )

        quantity_chars = Decimal(msg.char_count)
        if quantity_chars <= 0:
            self.logger.warning(
                "tts_charge_skipped_missing_char_count",
                translation_room_id=msg.meeting_id,
                segment_id=msg.segment_id,
            )
            return

        rate = await self.db.resolve_usage_rate(
            charge_type=charge_type,
            unit="character",
            provider=self.tts_settings.provider,
            model=model,
        )
        if rate is None:
            self._log_missing_rate_skip(
                event_source="tts_results",
                charge_type=charge_type,
                unit="character",
                provider=self.tts_settings.provider,
                model=model,
                translation_room_id=msg.meeting_id,
            )
            return

        await self._accumulate_and_maybe_flush(BillingEvent(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=charge_type,
            charge_type=charge_type,
            quantity=quantity_chars,
            unit="character",
            credits_event=quantity_chars * rate.unit_price,
            event_idempotency_key=f"{charge_type}:{msg.segment_id}:{msg.target_lang}",
            provider=rate.provider,
            model=rate.model,
            pricing_rate_card_id=rate.id,
            unit_price_snapshot=rate.unit_price,
            source_language_code=None,
            target_language_code=msg.target_lang or None,
            force_flush=msg.is_final_chunk,
            details={
                "event_source": "tts_results",
                "clone_provider": msg.clone_provider,
                "voice_mode": msg.voice_mode,
                "char_count": msg.char_count,
                "duration_ms": msg.duration_ms,
                "transcript_segment_id": underlying_segment_id,
            },
        ))

    # ------------------------------------------------------------------
    # Redis accumulator + flush
    # ------------------------------------------------------------------

    async def _accumulate_and_maybe_flush(self, event: BillingEvent) -> None:
        dedupe_key = f"{EVENT_DEDUPE_PREFIX}:{event.event_idempotency_key}"
        accepted = await self.redis.redis.set(
            dedupe_key,
            "1",
            ex=self.settings.billing_event_dedupe_ttl_seconds,
            nx=True,
        )
        if not accepted:
            self.logger.info(
                "billing_event_duplicate_skipped",
                idempotency_key=event.event_idempotency_key,
            )
            return

        key = self._accumulator_key(
            event.subscription_id,
            event.translation_room_id,
            event.charge_type,
            event.source_language_code,
            event.target_language_code,
        )
        now_ms = int(time.time() * 1000)
        pricing_scope = self._pricing_scope(
            event.source_language_code,
            event.target_language_code,
        )

        if not await self.redis.redis.hexists(key, "window_seq"):
            window_seq = await self.redis.redis.incr(
                f"{ACCUMULATOR_SEQUENCE_PREFIX}:{event.subscription_id}:"
                f"{event.translation_room_id}:{event.charge_type}:{pricing_scope}"
            )
            await self.redis.redis.hset(
                key,
                mapping={
                    "subscription_id": str(event.subscription_id),
                    "workspace_id": str(event.workspace_id),
                    "translation_room_id": event.translation_room_id,
                    "usage_type": event.usage_type,
                    "charge_type": event.charge_type,
                    "pricing_scope": pricing_scope,
                    "source_language_code": event.source_language_code or "",
                    "target_language_code": event.target_language_code or "",
                    "window_seq": str(window_seq),
                    "window_start_ms": str(now_ms),
                },
            )

        micro_credits_event = self._to_micro(event.credits_event)
        quantity_micro = self._to_micro(event.quantity)
        price_micro = self._to_micro(event.unit_price_snapshot)
        pipe = self.redis.redis.pipeline(transaction=True)
        pipe.hincrby(key, "micro_credits", micro_credits_event)
        pipe.hincrby(key, f"quantity_micro_{event.unit}", quantity_micro)
        pipe.hincrby(key, "event_count", 1)
        pipe.hset(key, f"rate_{event.unit}_id", str(event.pricing_rate_card_id))
        pipe.hset(key, f"rate_{event.unit}_price_micro", price_micro)
        pipe.hset(key, f"provider_{event.unit}", event.provider)
        pipe.hset(key, f"model_{event.unit}", event.model)
        await pipe.execute()

        if event.details:
            await self.redis.redis.hset(
                key,
                "last_event_details",
                self._json_details(event.details),
            )

        snapshot = self._decode_hash(await self.redis.redis.hgetall(key))
        if self._should_flush(snapshot, now_ms, force=event.force_flush):
            await self._flush_accumulator(key)

    async def _flush_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(1.0)
            await self._flush_due_accumulators()

    async def _flush_due_accumulators(self) -> None:
        now_ms = int(time.time() * 1000)
        async for raw_key in self.redis.redis.scan_iter(f"{ACCUMULATOR_KEY_PREFIX}:*"):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            snapshot = self._decode_hash(await self.redis.redis.hgetall(key))
            if snapshot and self._should_flush(snapshot, now_ms):
                await self._flush_accumulator(key)

    async def _flush_all_accumulators(self) -> None:
        async for raw_key in self.redis.redis.scan_iter(f"{ACCUMULATOR_KEY_PREFIX}:*"):
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            await self._flush_accumulator(key)

    async def _flush_accumulator(self, key: str) -> None:
        pipe = self.redis.redis.pipeline(transaction=True)
        pipe.hgetall(key)
        pipe.delete(key)
        raw_snapshot, _ = await pipe.execute()
        snapshot = self._decode_hash(raw_snapshot)
        if not snapshot:
            return

        credits_raw = self._micro_decimal_from_snapshot(snapshot, "micro_credits")
        if credits_raw <= 0:
            return

        credits_consumed = int(credits_raw.to_integral_value(rounding=ROUND_CEILING))
        if credits_consumed <= 0:
            return
        if credits_consumed > self.settings.max_credits_per_flush:
            self.logger.error(
                "billing_flush_rejected_over_cap",
                accumulator_key=key,
                credits_consumed=credits_consumed,
                max_credits_per_flush=self.settings.max_credits_per_flush,
            )
            return

        idempotency_key = (
            f"{snapshot['charge_type']}:{snapshot.get('pricing_scope', '_:_')}:"
            f"{snapshot['translation_room_id']}:{snapshot['window_seq']}"
        )
        charge_dedupe_key = f"{CHARGE_DEDUPE_PREFIX}:{idempotency_key}"
        accepted = await self.redis.redis.set(
            charge_dedupe_key,
            "1",
            ex=self.settings.billing_event_dedupe_ttl_seconds,
            nx=True,
        )
        if not accepted:
            self.logger.info("billing_flush_duplicate_skipped", idempotency_key=idempotency_key)
            return

        try:
            breakdown = self._unit_breakdown(snapshot)
            primary = self._primary_breakdown_unit(breakdown)
            unit = primary["unit"]
            quantity = primary["quantity"]
            pricing_rate_card_id = primary.get("pricing_rate_card_id")
            unit_price_snapshot = primary.get("unit_price_snapshot")
            await self.db.record_usage_and_charge(
                subscription_id=uuid.UUID(snapshot["subscription_id"]),
                user_id=None,
                workspace_id=uuid.UUID(snapshot["workspace_id"]),
                translation_room_id=snapshot["translation_room_id"],
                usage_type=snapshot["usage_type"],
                charge_type=snapshot["charge_type"],
                reference_id=None,
                reference_type="billing_accumulator",
                quantity=float(quantity),
                unit=unit,
                credits_consumed=credits_consumed,
                transcript_segment_id=None,
                idempotency_key=idempotency_key,
                pricing_rate_card_id=pricing_rate_card_id,
                unit_price_snapshot=unit_price_snapshot,
                provider=primary.get("provider"),
                model=primary.get("model"),
                details={
                    "event_count": int(float(snapshot.get("event_count", "0"))),
                    "raw_credits": str(credits_raw),
                    "rounding_delta": str(Decimal(credits_consumed) - credits_raw),
                    "rounding_rate": str(
                        Decimal("0")
                        if credits_raw == 0
                        else (Decimal(credits_consumed) - credits_raw) / credits_raw
                    ),
                    "pricing_scope": snapshot.get("pricing_scope", "_:_"),
                    "source_language_code": snapshot.get("source_language_code") or None,
                    "target_language_code": snapshot.get("target_language_code") or None,
                    "unit_breakdown": breakdown,
                    "window_seq": int(snapshot["window_seq"]),
                    "window_start_ms": int(float(snapshot["window_start_ms"])),
                    "last_event": snapshot.get("last_event_details"),
                },
            )
        except Exception:
            await self.redis.redis.delete(charge_dedupe_key)
            raise

    def _should_flush(self, snapshot: dict[str, str], now_ms: int, *, force: bool = False) -> bool:
        if force:
            return True
        started_at = int(float(snapshot.get("window_start_ms", now_ms)))
        age_ms = now_ms - started_at
        credits = self._micro_decimal_from_snapshot(snapshot, "micro_credits")
        return (
            age_ms >= self.settings.accumulator_flush_interval_seconds * 1000
            or credits >= Decimal(self.settings.max_credits_per_flush)
        )

    def _log_missing_rate_skip(
        self,
        *,
        event_source: str,
        charge_type: str,
        unit: str,
        provider: str,
        model: str,
        translation_room_id: str,
        source_language_code: str | None = None,
        target_language_code: str | None = None,
    ) -> None:
        self.logger.warning(
            "billing_event_skipped_missing_rate",
            metric_name="billing_event_skipped_missing_rate",
            event_source=event_source,
            charge_type=charge_type,
            unit=unit,
            provider=provider,
            model=model,
            translation_room_id=translation_room_id,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
        )

    @staticmethod
    def _accumulator_key(
        subscription_id: uuid.UUID,
        room_id: str,
        charge_type: str,
        source_language_code: str | None,
        target_language_code: str | None,
    ) -> str:
        pricing_scope = BillingSettlementWorker._pricing_scope(
            source_language_code,
            target_language_code,
        )
        return (
            f"{ACCUMULATOR_KEY_PREFIX}:{subscription_id}:{room_id}:{charge_type}:"
            f"{pricing_scope}"
        )

    @staticmethod
    def _pricing_scope(source_language_code: str | None, target_language_code: str | None) -> str:
        source = source_language_code or "_"
        target = target_language_code or "_"
        return f"{source}:{target}"

    @staticmethod
    def _to_micro(value: Decimal) -> int:
        return int((value * MICRO_SCALE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))

    @staticmethod
    def _from_micro(value: str | int | Decimal) -> Decimal:
        return Decimal(str(value)) / MICRO_SCALE

    def _micro_decimal_from_snapshot(self, snapshot: dict[str, str], key: str) -> Decimal:
        return self._from_micro(snapshot.get(key, "0"))

    def _unit_breakdown(self, snapshot: dict[str, str]) -> list[dict]:
        units = sorted(
            key.removeprefix("quantity_micro_")
            for key in snapshot
            if key.startswith("quantity_micro_")
        )
        breakdown: list[dict] = []
        for unit in units:
            quantity = self._from_micro(snapshot[f"quantity_micro_{unit}"])
            if quantity <= 0:
                continue
            unit_price_snapshot = self._from_micro(
                snapshot.get(f"rate_{unit}_price_micro", "0")
            )
            pricing_rate_card_id = snapshot.get(f"rate_{unit}_id")
            item = {
                "unit": unit,
                "quantity": str(quantity),
                "pricing_rate_card_id": pricing_rate_card_id,
                "unit_price_snapshot": str(unit_price_snapshot),
                "provider": snapshot.get(f"provider_{unit}"),
                "model": snapshot.get(f"model_{unit}"),
            }
            breakdown.append(item)
        return breakdown

    @staticmethod
    def _primary_breakdown_unit(breakdown: list[dict]) -> dict:
        if not breakdown:
            return {
                "unit": "unknown",
                "quantity": Decimal("0"),
                "pricing_rate_card_id": None,
                "unit_price_snapshot": None,
                "provider": None,
                "model": None,
            }
        primary = max(breakdown, key=lambda item: Decimal(item["quantity"]))
        return {
            **primary,
            "quantity": Decimal(primary["quantity"]),
            "pricing_rate_card_id": (
                uuid.UUID(primary["pricing_rate_card_id"])
                if primary.get("pricing_rate_card_id")
                else None
            ),
            "unit_price_snapshot": Decimal(primary["unit_price_snapshot"]),
        }

    @staticmethod
    def _decode_hash(data: dict) -> dict[str, str]:
        return {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in (data or {}).items()
        }

    @staticmethod
    def _decimal_from_snapshot(snapshot: dict[str, str], key: str) -> Decimal:
        try:
            return Decimal(str(snapshot.get(key, "0")))
        except (InvalidOperation, ValueError):
            return Decimal(0)

    @staticmethod
    def _json_details(details: dict) -> str:
        import json

        return json.dumps(details, separators=(",", ":"), sort_keys=True)

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._shutdown_event.set)
            except NotImplementedError:
                pass
