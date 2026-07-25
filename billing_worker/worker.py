"""Billing settlement worker.

Consumes billable AI pipeline streams (audio:chunks, translate:results,
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
import signal
import socket
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation

from billing_worker.db import BillingRepository
from shared.config import BillingSettings, RedisSettings, WorkerSettings
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage

logger = get_logger("worker.billing")

STT_CHARGE_TYPE = "STT"
TRANSLATION_CHARGE_TYPE = "TRANSLATION"
ACCUMULATOR_KEY_PREFIX = "billing:acc"
ACCUMULATOR_SEQUENCE_PREFIX = "billing:accseq"
CHARGE_DEDUPE_PREFIX = "billing:charge"
EVENT_DEDUPE_PREFIX = "billing:event"


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
        self.logger = logger
        self._consumer_name = f"billing-{socket.gethostname()}"
        self._shutdown_event = asyncio.Event()
        # translation_room_id -> (subscription_id, workspace_id, cached_at_monotonic)
        self._subscription_cache: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._register_signal_handlers()
        await self.redis.connect()
        await self.db.connect()
        self.logger.info("billing_worker_started")
        try:
            await asyncio.gather(
                self._consume_loop("audio:chunks", "billing-stt-workers", self._handle_stt),
                self._consume_loop(
                    "translate:results", "billing-translation-workers", self._handle_translation
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

    async def _consume_loop(self, stream: str, group: str, handler) -> None:
        while not self._shutdown_event.is_set():
            try:
                async for message_id, data in self.redis.consume(
                    stream=stream,
                    group=group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                ):
                    if self._shutdown_event.is_set():
                        break
                    try:
                        await handler(data)
                    except Exception:
                        self.logger.exception(
                            "settlement_error", stream=stream, message_id=message_id
                        )
                        # Already acked by consume() — in production this should go to a
                        # dead-letter stream instead of being silently dropped.
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error", stream=stream)
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Subscription resolution (cached per translation_room_id)
    # ------------------------------------------------------------------

    async def _resolve_subscription(self, translation_room_id: str) -> tuple | None:
        cached = self._subscription_cache.get(translation_room_id)
        now = time.monotonic()
        if cached and now - cached[2] < self.settings.subscription_cache_ttl_seconds:
            return cached[0], cached[1]

        resolved = await self.db.resolve_subscription(translation_room_id)
        if resolved is None:
            return None

        subscription_id, workspace_id = resolved
        self._subscription_cache[translation_room_id] = (subscription_id, workspace_id, now)
        return subscription_id, workspace_id

    # ------------------------------------------------------------------
    # Per-stream handlers
    # ------------------------------------------------------------------

    async def _handle_stt(self, data: dict) -> None:
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

        await self._accumulate_and_maybe_flush(BillingEvent(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=STT_CHARGE_TYPE,
            charge_type=STT_CHARGE_TYPE,
            quantity=audio_seconds,
            unit="second",
            credits_event=audio_seconds,
            event_idempotency_key=f"{STT_CHARGE_TYPE}:{msg.meeting_id}:{msg.speaker_id}:{msg.chunk_index}",
            force_flush=msg.is_final_chunk,
            details={"event_source": "audio_chunks", "sample_rate": msg.sample_rate},
        ))

    async def _handle_translation(self, data: dict) -> None:
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
            self.logger.warning(
                "segment_id_extraction_failed", raw_segment_id=msg.segment_id
            )

        if msg.source_lang.split("-")[0] == msg.target_lang.split("-")[0]:
            return

        quantity_chars = Decimal(len(msg.original_text))
        if quantity_chars <= 0:
            return

        await self._accumulate_and_maybe_flush(BillingEvent(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=TRANSLATION_CHARGE_TYPE,
            charge_type=TRANSLATION_CHARGE_TYPE,
            quantity=quantity_chars,
            unit="character",
            credits_event=quantity_chars / Decimal(100),
            event_idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{msg.segment_id}:{msg.target_lang}",
            force_flush=msg.is_final_chunk,
            details={
                "event_source": "translate_results",
                "source_lang": msg.source_lang,
                "target_lang": msg.target_lang,
                "transcript_segment_id": underlying_segment_id,
            },
        ))

    async def _handle_tts(self, data: dict) -> None:
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
            self.logger.warning(
                "segment_id_extraction_failed", raw_segment_id=msg.segment_id
            )

        quantity_s = Decimal(msg.duration_ms) / Decimal(1000)
        if quantity_s <= 0:
            return

        await self._accumulate_and_maybe_flush(BillingEvent(
            subscription_id=subscription_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=charge_type,
            charge_type=charge_type,
            quantity=quantity_s,
            unit="second",
            credits_event=quantity_s,
            event_idempotency_key=f"{charge_type}:{msg.segment_id}:{msg.target_lang}",
            force_flush=msg.is_final_chunk,
            details={
                "event_source": "tts_results",
                "clone_provider": msg.clone_provider,
                "voice_mode": msg.voice_mode,
                "char_count": msg.char_count,
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
        )
        now_ms = int(time.time() * 1000)

        if not await self.redis.redis.hexists(key, "window_seq"):
            window_seq = await self.redis.redis.incr(
                f"{ACCUMULATOR_SEQUENCE_PREFIX}:{event.subscription_id}:{event.translation_room_id}:{event.charge_type}"
            )
            await self.redis.redis.hset(
                key,
                mapping={
                    "subscription_id": str(event.subscription_id),
                    "workspace_id": str(event.workspace_id),
                    "translation_room_id": event.translation_room_id,
                    "usage_type": event.usage_type,
                    "charge_type": event.charge_type,
                    "unit": event.unit,
                    "window_seq": str(window_seq),
                    "window_start_ms": str(now_ms),
                },
            )

        pipe = self.redis.redis.pipeline(transaction=True)
        pipe.hincrbyfloat(key, "credits", float(event.credits_event))
        pipe.hincrbyfloat(key, f"quantity_{event.unit}", float(event.quantity))
        pipe.hincrby(key, "event_count", 1)
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

        credits_raw = self._decimal_from_snapshot(snapshot, "credits")
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
            f"{snapshot['charge_type']}:{snapshot['translation_room_id']}:{snapshot['window_seq']}"
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
            unit = snapshot["unit"]
            await self.db.record_usage_and_charge(
                subscription_id=uuid.UUID(snapshot["subscription_id"]),
                user_id=None,
                workspace_id=uuid.UUID(snapshot["workspace_id"]),
                translation_room_id=snapshot["translation_room_id"],
                usage_type=snapshot["usage_type"],
                charge_type=snapshot["charge_type"],
                reference_id=None,
                reference_type="billing_accumulator",
                quantity=float(self._decimal_from_snapshot(snapshot, f"quantity_{unit}")),
                unit=unit,
                credits_consumed=credits_consumed,
                transcript_segment_id=None,
                idempotency_key=idempotency_key,
                details={
                    "event_count": int(float(snapshot.get("event_count", "0"))),
                    "raw_credits": str(credits_raw),
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
        credits = self._decimal_from_snapshot(snapshot, "credits")
        return (
            age_ms >= self.settings.accumulator_flush_interval_seconds * 1000
            or credits >= Decimal(self.settings.max_credits_per_flush)
        )

    @staticmethod
    def _accumulator_key(subscription_id: uuid.UUID, room_id: str, charge_type: str) -> str:
        return f"{ACCUMULATOR_KEY_PREFIX}:{subscription_id}:{room_id}:{charge_type}"

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
            loop.add_signal_handler(sig, self._shutdown_event.set)
