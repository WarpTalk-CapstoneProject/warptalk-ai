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
import signal
import socket
import time

from shared.config import BillingSettings, RedisSettings, WorkerSettings
from shared.logger import get_logger
from shared.redis_client import RedisStreamClient
from shared.schemas import STTResultMessage, TranslationResultMessage, TTSResultMessage

from billing_worker.db import BillingRepository

logger = get_logger("worker.billing")

STT_CHARGE_TYPE = "STT"
TRANSLATION_CHARGE_TYPE = "TRANSLATION"


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
        self.db = BillingRepository(self.settings.database)
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
                self._consume_loop("stt:results", "billing-stt-workers", self._handle_stt),
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
        # Flat credits-per-second placeholder — real pricing is a usage_rate_card lookup
        # by source_language (see subscription.usage_rate_card), not implemented yet.
        await self.db.record_usage_and_charge(
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
            credits_consumed=max(1, round(quantity_s)),
            # NOTE: msg.segment_id is randomly generated per STTResultMessage
            # (Field(default_factory=uuid4) in shared/schemas.py) — it is NOT stable
            # across a Redis Streams redelivery of the *upstream* audio chunk, since
            # stt_worker.process() would mint a fresh one on retry. This idempotency
            # key only protects against redelivery/retry at THIS worker's own consumer
            # group, not against stt_worker re-processing the same audio chunk twice.
            idempotency_key=f"{STT_CHARGE_TYPE}:{msg.segment_id}:NA",
        )

    async def _handle_translation(self, data: dict) -> None:
        msg = TranslationResultMessage.from_redis(data)
        if not msg.translated_text.strip():
            return

        resolved = await self._resolve_subscription(msg.meeting_id)
        if resolved is None:
            self.logger.warning("no_subscription_for_room", translation_room_id=msg.meeting_id)
            return
        subscription_id, workspace_id = resolved

        quantity_chars = float(len(msg.translated_text))
        await self.db.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=msg.speaker_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=TRANSLATION_CHARGE_TYPE,
            charge_type=TRANSLATION_CHARGE_TYPE,
            reference_id=msg.segment_id,
            reference_type="translation_content",
            quantity=quantity_chars,
            unit="character",
            credits_consumed=max(1, round(quantity_chars / 100)),
            # msg.segment_id here IS deterministic (translation_worker builds it as
            # f"{stt_result.segment_id}-c{idx}"), so this key is redelivery-safe.
            idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{msg.segment_id}:{msg.target_lang}",
        )

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
        quantity_s = max(msg.duration_ms / 1000.0, 0.1)
        await self.db.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=msg.speaker_id,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=charge_type,
            charge_type=charge_type,
            reference_id=msg.segment_id,
            reference_type="audio_dubbing",
            quantity=quantity_s,
            unit="second",
            credits_consumed=max(1, round(quantity_s)),
            idempotency_key=f"{charge_type}:{msg.segment_id}:{msg.target_lang}",
            details={"clone_provider": msg.clone_provider, "voice_mode": msg.voice_mode},
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)
