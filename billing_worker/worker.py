"""Billing settlement worker.

Consumes the BILLABLE AI pipeline result streams (translate:results, tts:results, and
translate:backfill_results for translations produced after a meeting ended)
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

WT-605 — SPEECH SAID WHILE THE TRANSCRIPT IS PAUSED IS STILL BILLED. THIS IS DELIBERATE.
    Pausing the transcript stops the meeting being written DOWN. It does not stop the
    meeting: translation_worker still translates every segment and tts_worker still renders
    and publishes the dub, so listeners on the other language keep hearing the speaker for
    the whole pause. The service was delivered in full, on purpose — the host asked for no
    record, not for no interpreting.

    So this worker has no pause gate and must not grow one. Adding one would hand out free
    translation and free dubbing to anyone who pressed Pause Transcript first, and would do
    it invisibly, since nothing downstream compares billed minutes against transcript lines.
    If a later ticket reports "we charged for a paused meeting" as a bug, it is not one: the
    charge follows translate:results and tts:results, which exist only because real work was
    done.

    The gates that DO belong to WT-605 sit where the written record is produced —
    ai_assistant_worker (summary) and suggestion_worker (badges). See
    shared/transcript_pause.py for the flag and the cross-repo contract behind it.

BACKFILL TRANSLATION IS BILLED LIKE LIVE TRANSLATION.
    translation_worker/backfill_worker.py translates a saved transcript's missing lines (and redoes
    a corrected line's translations) with the same model, and publishes to
    translate:backfill_results so that tts_worker does not dub a meeting that ended hours ago. That
    separate stream is also why none of it was ever charged: this worker only read
    translate:results. Same charge type, same rate card, same unit (seconds of source speech), so
    a line costs the same whether it was translated during the meeting or after it. Only where the
    workspace and the user come from differs; see _handle_backfill_translation.

    A RETRANSLATION AFTER A TRANSCRIPT CORRECTION IS NOT BILLED. THIS IS DELIBERATE.
    The same stream carries the redo of a line whose transcript somebody corrected. That work
    exists only because the platform heard the line wrong, and the seconds it covers were already
    paid for once, when the line was first translated live or by backfill. Charging again bills
    the same audio twice for the platform's own mistake: the opposite of transcription being free
    (WT-344), and of transcript_corrections.reversal_credit_transaction_id, which was designed to
    REFUND a translation a human had to correct. Spend on it stays bounded by TranscriptService's
    per-transcript backfill budget, which corrections draw from.

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

from billing_worker.db import BillingRepository, SettlementOutcome
from shared.config import BillingSettings, RedisSettings, WorkerSettings
from shared.control_markers import is_system_speaker
from shared.health_probe import heartbeat_key
from shared.logger import get_logger
from shared.redis_client import BILLING_UNBILLED_KEY, BILLING_UNBILLED_REASONS, RedisStreamClient
from shared.schemas import (
    TranslationResultMessage,
    TTSResultMessage,
)

logger = get_logger("worker.billing")

TRANSLATION_CHARGE_TYPE = "TRANSLATION"
BACKFILL_RESULT_STREAM = "translate:backfill_results"
BILLED_STREAMS = ("translate:results", BACKFILL_RESULT_STREAM, "tts:results")
SettlementHandler = Callable[[Mapping[Any, Any]], Awaitable[None]]

# WT-699 / TC3705 — WHEN A CHARGE IS REFUSED, TRANSLATION STOPS.
#
# settle_usage_charge refuses a charge (applied=False, not a replay) when the subscription is
# suspended or the charge would cross the overage cap — which, with overage off, is the moment
# the workspace runs out of credits. This worker used to log that and move on, and the meeting
# went on translating and dubbing for free with nothing on screen saying anything was wrong.
#
# Now a refusal marks the ROOM (and the workspace) suspended in Redis. translation_worker reads the
# room key and stops translating; TranslationRoomService reads both and refuses Start
# Translation, saying why; the Gateway relays the command below so everybody in the room is told.
#
# The keys are the ones BillingService already writes for its own suspensions
# (RedisConstants.Keys.*AiService*), in the same shape, so every reader has one contract.
# They carry a short TTL that the watch loop keeps refreshing for as long as the subscription is
# still suspended, and deletes the moment it is not — so a top-up followed by a resume brings the
# room back on its own, and a crashed worker cannot leave a room stopped forever.
ROOM_SUSPENDED_KEY = "translationRoom:{room_id}:ai_service_suspended"
ROOM_STATE_KEY = "translationRoom:{room_id}:ai_service_state"
WORKSPACE_SUSPENDED_KEY = "workspace:{workspace_id}:ai_service_suspended"
WORKSPACE_STATE_KEY = "workspace:{workspace_id}:ai_service_state"
GATEWAY_COMMANDS_CHANNEL = "warptalk:translation-room:commands"
SUSPENDED_SERVICE_STATE = "suspended"
SUSPENSION_TTL_SECONDS = 120
SUSPENSION_RECHECK_SECONDS = 30

# THE LEAK THIS CLOSES (2026-09-24). A room whose workspace had NO active subscription — it expired
# the day before — translated and dubbed a 14-minute meeting for free. `resolve_subscription` asks
# for `is_active`, found nothing, and the handlers logged `no_subscription_for_room` and RETURNED:
# no charge, so no refusal, so the WT-699 stop path above never ran. "Nobody to bill" is now the
# same event as "the bill was refused": the room is stopped, the room is told why, and it is
# counted. This is the reason that travels with it — BillingService's
# SubscriptionConstants.SuspendedReasons.SubscriptionExpired, and what settle_usage_charge returns
# for an inactive row, so every reader has one word for it.
SUBSCRIPTION_EXPIRED_REASON = "subscription_expired"

# Billable work that was NOT billed, by reason — a hash the stateless metrics_exporter turns into
# warptalk_billing_unbilled_events_total{reason, charge_type}. A log line was the only trace of the
# leak, and nobody aggregates log lines; this is the number an alert can watch.
UNBILLED_METRIC_KEY = BILLING_UNBILLED_KEY
UNBILLED_METRIC_TTL_SECONDS = 7 * 24 * 60 * 60
UNBILLED_NO_SUBSCRIPTION, UNBILLED_CHARGE_REFUSED = BILLING_UNBILLED_REASONS


class NoActiveSubscription:
    """A room's workspace, resolved, that has no active subscription to bill.

    Returned instead of None so the caller still knows WHICH workspace cannot pay — the stop path
    marks the workspace as well as the room, and the watch loop asks the workspace, not a
    subscription that does not exist, whether it can pay again.
    """

    __slots__ = ("workspace_id",)

    def __init__(self, workspace_id: uuid.UUID) -> None:
        self.workspace_id = workspace_id

    def __repr__(self) -> str:
        return f"NoActiveSubscription(workspace_id={self.workspace_id})"


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


def _translation_quantity_seconds(msg: TranslationResultMessage) -> float:
    """Seconds of source speech a translation is priced on: one rule for live and backfill.

    A message with no usable span (an older backfill producer, a segment stored without timing)
    bills the same flat 1.0s the live path always has.
    """
    if msg.end_ms > msg.start_ms:
        return max((msg.end_ms - msg.start_ms) / 1000.0, 0.1)
    return 1.0


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
        # workspace_id -> (subscription_id, workspace_id, cached_at_monotonic), for backfills
        self._workspace_subscription_cache: dict[str, tuple[uuid.UUID, uuid.UUID, float]] = {}
        # WT-699 / TC3705: rooms this replica stopped because a charge was refused, or because
        # the workspace had no subscription to charge (subscription_id None).
        # translation_room_id -> (subscription_id | None, workspace_id)
        self._suspended_rooms: dict[str, tuple[uuid.UUID | None, uuid.UUID]] = {}

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
                self._suspension_watch_loop(),
                self._consume_loop(
                    "translate:results", "billing-translation-workers", self._handle_translation
                ),
                self._consume_loop(
                    "translate:backfill_results",
                    "billing-translation-backfill-workers",
                    self._handle_backfill_translation,
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
                    "streams": list(BILLED_STREAMS),
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
    # WT-699 / TC3705: a refused charge stops the room's translation
    # ------------------------------------------------------------------

    async def _note_settlement(
        self,
        translation_room_id: str,
        subscription_id: uuid.UUID,
        workspace_id: uuid.UUID,
        outcome: object,
        charge_type: str = TRANSLATION_CHARGE_TYPE,
    ) -> None:
        """Stop the room if the workspace could not pay for what it just used.

        A replay carries its original transaction and is not a refusal. Anything that is not a
        SettlementOutcome (an older repository, a test double) is left alone rather than guessed at.
        """
        if not isinstance(outcome, SettlementOutcome):
            return
        if outcome.applied or outcome.replayed:
            return
        await self._count_unbilled(UNBILLED_CHARGE_REFUSED, charge_type)
        if outcome.suspended_reason == SUBSCRIPTION_EXPIRED_REASON:
            # The cached subscription id is a plan that has ended. Forget it, so the next event
            # resolves afresh and finds the renewal the moment there is one.
            self._subscription_cache.pop(translation_room_id, None)
        try:
            await self._suspend_room(
                translation_room_id,
                subscription_id,
                workspace_id,
                outcome.service_state,
                outcome.suspended_reason,
            )
        except Exception:
            # The charge was already refused; failing the message here would only get it
            # redelivered and refused again. The next refusal retries the marking.
            self.logger.exception(
                "credit_suspension_mark_failed", translation_room_id=translation_room_id
            )

    async def _suspend_room(
        self,
        translation_room_id: str,
        subscription_id: uuid.UUID | None,
        workspace_id: uuid.UUID,
        service_state: str | None,
        suspended_reason: str | None,
    ) -> None:
        self._suspended_rooms[translation_room_id] = (subscription_id, workspace_id)
        state = json.dumps(
            {
                "translationRoomId": translation_room_id,
                "workspaceId": str(workspace_id),
                "serviceState": service_state or SUSPENDED_SERVICE_STATE,
                "suspendedReason": suspended_reason,
                "updatedAt": int(time.time() * 1000),
            }
        )
        room_key = ROOM_SUSPENDED_KEY.format(room_id=translation_room_id)
        # SET NX decides who announces it: one refusal per room reaches the UI, however many
        # replicas and however many refused segments follow it.
        newly_suspended = await self.redis.set_if_absent(room_key, "true", SUSPENSION_TTL_SECONDS)
        if not newly_suspended:
            await self.redis.expire(room_key, SUSPENSION_TTL_SECONDS)
        await self.redis.set_with_ttl(
            ROOM_STATE_KEY.format(room_id=translation_room_id), state, SUSPENSION_TTL_SECONDS
        )
        await self.redis.set_with_ttl(
            WORKSPACE_SUSPENDED_KEY.format(workspace_id=workspace_id),
            "true",
            SUSPENSION_TTL_SECONDS,
        )
        await self.redis.set_with_ttl(
            WORKSPACE_STATE_KEY.format(workspace_id=workspace_id), state, SUSPENSION_TTL_SECONDS
        )

        if newly_suspended:
            self.logger.warning(
                "translation_suspended_charge_refused",
                translation_room_id=translation_room_id,
                workspace_id=str(workspace_id),
                service_state=service_state,
                suspended_reason=suspended_reason,
            )
            await self._publish_gateway_command(
                {
                    "Command": "TranslationCreditsExhausted",
                    "RoomId": translation_room_id,
                    "Reason": suspended_reason or "",
                }
            )

    async def _stop_unpaid_room(
        self,
        translation_room_id: str,
        unpaid: NoActiveSubscription,
        charge_type: str,
    ) -> None:
        """Billable output for a room whose workspace has no subscription: treat it as refused.

        Warning, counted, and the room is stopped through the same keys and the same Gateway
        command as a refused charge — with its own reason, so the room is told "the subscription
        expired", not "you ran out of credits". Never raises: the message is acknowledged either
        way, because redelivering it cannot conjure a subscription.
        """
        self.logger.warning(
            "no_subscription_for_room",
            translation_room_id=translation_room_id,
            workspace_id=str(unpaid.workspace_id),
            charge_type=charge_type,
            action="translation_suspended",
        )
        await self._count_unbilled(UNBILLED_NO_SUBSCRIPTION, charge_type)
        try:
            await self._suspend_room(
                translation_room_id,
                None,
                unpaid.workspace_id,
                SUSPENDED_SERVICE_STATE,
                SUBSCRIPTION_EXPIRED_REASON,
            )
        except Exception:
            self.logger.exception(
                "credit_suspension_mark_failed", translation_room_id=translation_room_id
            )

    async def _count_unbilled(self, reason: str, charge_type: str) -> None:
        """Best effort, like every metric here: it must never fail the settlement it describes."""
        try:
            pipeline = self.redis.redis.pipeline(transaction=False)
            pipeline.hincrby(UNBILLED_METRIC_KEY, f"{reason}:{charge_type}", 1)
            pipeline.expire(UNBILLED_METRIC_KEY, UNBILLED_METRIC_TTL_SECONDS)
            await pipeline.execute()
        except Exception:
            self.logger.debug("unbilled_metric_failed", reason=reason, exc_info=True)

    async def _suspension_watch_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(SUSPENSION_RECHECK_SECONDS)
            try:
                await self._recheck_suspended_rooms()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("credit_suspension_recheck_failed")

    async def _workspace_payment_state(
        self, workspace_id: uuid.UUID
    ) -> tuple[str | None, str | None] | None:
        """(service_state, suspended_reason) of the workspace's ACTIVE subscription, or None.

        Asked of the workspace, not of the subscription the room was stopped on: a renewal after
        expiry creates a NEW subscription row, so the old one stays inactive forever and asking it
        would keep the room stopped after the customer has paid.
        """
        resolved = await self.db.resolve_subscription(str(workspace_id))
        if resolved is None:
            return None
        return await self.db.get_service_state(resolved[0])

    async def _recheck_suspended_rooms(self) -> None:
        """Keep a stopped room stopped while it cannot pay; let it go the moment it can."""
        for room_id, (_subscription_id, workspace_id) in list(self._suspended_rooms.items()):
            state = await self._workspace_payment_state(workspace_id)
            if state is None or state[0] == SUSPENDED_SERVICE_STATE:
                for key in (
                    ROOM_SUSPENDED_KEY.format(room_id=room_id),
                    ROOM_STATE_KEY.format(room_id=room_id),
                    WORKSPACE_SUSPENDED_KEY.format(workspace_id=workspace_id),
                    WORKSPACE_STATE_KEY.format(workspace_id=workspace_id),
                ):
                    await self.redis.expire(key, SUSPENSION_TTL_SECONDS)
                continue

            self._suspended_rooms.pop(room_id, None)
            # A cached id may be the subscription that ended; the room bills its renewal now.
            self._subscription_cache.pop(room_id, None)
            self._workspace_subscription_cache.pop(str(workspace_id), None)
            # Only the replica that actually removed the room key announces the resume.
            removed = await self.redis.redis.delete(
                ROOM_SUSPENDED_KEY.format(room_id=room_id),
                ROOM_STATE_KEY.format(room_id=room_id),
            )
            await self.redis.redis.delete(
                WORKSPACE_SUSPENDED_KEY.format(workspace_id=workspace_id),
                WORKSPACE_STATE_KEY.format(workspace_id=workspace_id),
            )
            if removed:
                self.logger.info(
                    "translation_resumed_service_restored",
                    translation_room_id=room_id,
                    service_state=state[0] if state else None,
                )
                await self._publish_gateway_command(
                    {"Command": "TranslationCreditsRestored", "RoomId": room_id}
                )

    async def _publish_gateway_command(self, payload: dict[str, str]) -> None:
        # PascalCase on purpose: the Gateway deserializes TranslationRoomCommandMessage with the
        # default (case-sensitive) options, the same envelope every other publisher here uses.
        try:
            await self.redis.redis.publish(GATEWAY_COMMANDS_CHANNEL, json.dumps(payload))
        except Exception:
            # Pub/sub is a courtesy to the UI; the Redis flags above are what stop translation.
            self.logger.exception("gateway_command_publish_failed", command=payload.get("Command"))

    # ------------------------------------------------------------------
    # Subscription resolution (cached per translation_room_id)
    # ------------------------------------------------------------------

    async def _resolve_subscription(
        self,
        translation_room_id: str,
    ) -> tuple[uuid.UUID, uuid.UUID] | NoActiveSubscription:
        cached = self._subscription_cache.get(translation_room_id)
        now = time.monotonic()
        if cached and now - cached[2] < self.settings.subscription_cache_ttl_seconds:
            return cached[0], cached[1]

        # `meeting:room:v2:` — the `v2` is NOT decoration, and dropping it stops all billing.
        #
        # MeetingService owns this projection and says so at MeetingRoomService.JoinMeetingAsync:
        # "Billing and AI workers consume this as the local room -> workspace projection." WT-428
        # bumped the key to v2 there (entries cached before requires_approval existed deserialize
        # with proto3's default FALSE, which fails OPEN on an approval gate, so the bump orphans
        # them rather than trusting them) — and this reader was not bumped with it.
        #
        # Nothing writes the unversioned key any more, and the old entries aged out on their 24h
        # TTL, so every settlement then found no projection, raised, retried and dead-lettered.
        # Silent, because a dead letter is not an error the meeting can see: the meeting works
        # perfectly and simply bills nothing.
        #
        # The two literals are a contract across two repos with no shared constant to hold them.
        # If MeetingService versions this key again, this line has to move on the same day.
        raw_room = await self.redis.get(f"meeting:room:v2:{translation_room_id}")
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
            # Not cached: a renewal must be seen on the very next event.
            return NoActiveSubscription(uuid.UUID(str(workspace_id)))

        subscription_id, workspace_id = resolved
        self._subscription_cache[translation_room_id] = (subscription_id, workspace_id, now)
        return subscription_id, workspace_id

    async def _resolve_workspace_subscription(
        self,
        workspace_id: str,
    ) -> tuple[uuid.UUID, uuid.UUID] | None:
        """The active subscription of a workspace named directly, for backfills.

        Not the room projection: a backfill runs whenever somebody reads the transcript, which is
        routinely days after `meeting:room:v2:` expired. TranscriptService names the workspace from
        the transcript row itself.
        """
        cached = self._workspace_subscription_cache.get(workspace_id)
        now = time.monotonic()
        if cached and now - cached[2] < self.settings.subscription_cache_ttl_seconds:
            return cached[0], cached[1]

        resolved = await self.db.resolve_subscription(workspace_id)
        if resolved is None:
            return None

        subscription_id, workspace_uuid = resolved
        self._workspace_subscription_cache[workspace_id] = (subscription_id, workspace_uuid, now)
        return subscription_id, workspace_uuid

    # ------------------------------------------------------------------
    # Per-stream handlers
    # ------------------------------------------------------------------

    def _is_unbillable(self, speaker_id: str | None, meeting_id: str) -> bool:
        """Whether this event is the platform talking to itself rather than a chargeable use.

        THE FAILURE THIS CLOSES
            `record_usage_and_charge` does `_as_uuid(user_id)`, and the __MEETING_END__ sentinel
            carries `speaker_id="system"`. `uuid.UUID("system")` raises, identically on all five
            deliveries, so the message was dead-lettered — twice per meeting, once on
            translate:results and once on tts:results. That is the entire content of the
            `WarpTalkDeadLetterPresent` alert: five meetings, ten entries, one bug.

            translation_worker now drops the sentinel before it can reach either stream, so in
            practice nothing should arrive here. This stays anyway, because the two guards fail
            differently: that one stops the waste, and this one stops a malformed speaker id —
            from any future synthetic event, from a replayed old message, from anything — from
            becoming a permanent alert that needs a human to drain a stream.

        A REFUSAL, NOT AN ERROR
            Returning rather than raising is the point. A settlement that cannot name a user is
            not a failed charge to retry; it is not a charge. Retrying it five times and then
            paging somebody was the system treating a category error as an outage.
        """
        if is_system_speaker(speaker_id):
            self.logger.debug(
                "settlement_skipped_system_speaker",
                translation_room_id=meeting_id,
            )
            return True

        try:
            uuid.UUID(str(speaker_id))
        except (ValueError, AttributeError, TypeError):
            # WARNING, not debug: "system" above is expected and silent, but any OTHER
            # unparseable speaker id means something upstream is emitting a shape nobody
            # designed, and that is worth a line.
            self.logger.warning(
                "settlement_skipped_unparseable_speaker",
                translation_room_id=meeting_id,
                speaker_id=speaker_id,
            )
            return True

        return False

    async def _is_external_speaker(self, meeting_id: str, speaker_id: str | None) -> bool:
        """Whether this speaker was a guest in the room's workspace when they were admitted.

        WT-446: the owner wants external spend separable from a workspace's own, so every usage
        record carries the fact. TranslationRoomService resolves it once per admission (it is the
        only service that can — workspace membership is a gRPC hop away from here) and writes it
        to `translationRoom:<room>:external_participants`, a sibling of the `:languages` hash the
        rest of the pipeline already reads.

        FAILS TOWARDS "INTERNAL", DELIBERATELY. Redis runs allkeys-lru and evicts live meeting
        state, so this hash can vanish mid-meeting. A missing field then means "not known to be
        external", which under-attributes external spend but never invents it, and — crucially —
        never fails a settlement. Money still moves correctly either way: this flag partitions
        usage for reporting, it does not price it.
        """
        if not speaker_id:
            return False
        try:
            raw = await self.redis.redis.hget(
                f"translationRoom:{meeting_id}:external_participants",
                speaker_id,
            )
        except Exception:
            # Attribution is a courtesy; a Redis hiccup must not cost us the usage record.
            return False
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return raw == "1"

    async def _handle_translation(self, data: Mapping[Any, Any]) -> None:
        msg = TranslationResultMessage.from_redis(data)
        if not msg.translated_text.strip():
            return
        # An early sentence — one the STT model finished mid-chunk and published before the
        # turn closed — is NOT free work, but it is already paid for by the completed segment
        # of the same chunk, which carries that chunk's WHOLE duration however much text went
        # out early (stt_worker/model.py: `"end": duration_s`). Charging both would bill the
        # same seconds twice.
        #
        # Charging it on its own terms would be worse than double-billing, in the other
        # direction: an early segment has start_ms == end_ms, so it falls into the
        # zero-duration fallback below and bills a flat 1.0s. A fifteen-second turn split into
        # four early sentences would be priced as four seconds.
        #
        # So the money still follows the audio, exactly as it did before early publishing
        # existed. This is the only place that reads the flag.
        if msg.is_early:
            self.logger.debug(
                "skipped_early_segment",
                translation_room_id=msg.meeting_id,
                segment_id=msg.segment_id,
            )
            return
        if self._is_unbillable(msg.speaker_id, msg.meeting_id):
            return

        resolved = await self._resolve_subscription(msg.meeting_id)
        if isinstance(resolved, NoActiveSubscription):
            await self._stop_unpaid_room(msg.meeting_id, resolved, TRANSLATION_CHARGE_TYPE)
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

        quantity_s = _translation_quantity_seconds(msg)
        outcome = await self.db.record_usage_and_charge(
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
            # WT-446. Rides in `details` rather than as a new settle_usage_charge argument: the
            # column on usage_records is GENERATED from exactly this key, so the 200-line
            # settlement function — which decides whether a workspace can pay at all — is not
            # touched to add a reporting dimension.
            details={
                "is_external": await self._is_external_speaker(msg.meeting_id, msg.speaker_id),
            },
        )
        await self._note_settlement(msg.meeting_id, subscription_id, workspace_id, outcome)

    async def _handle_backfill_translation(self, data: Mapping[Any, Any]) -> None:
        """Charge a translation produced after the meeting, priced exactly like a live one.

        Same charge type, rate card, unit and quantity rule as _handle_translation. Three inputs
        cannot come from where the live path gets them:

        * WORKSPACE: from the message (the owner of the transcript row), not the room projection,
          which has expired for any transcript read back more than a day later.
        * USER: the requester, not the speaker. Nobody spoke this; somebody asked for it. The user
          is nullable on usage_records, so a message without one is still charged.
        * IDEMPOTENCY: keyed on the bare segment id and language. A duplicated backfill (two
          readers racing, a run marker that expired mid-run) is charged once.

        A retranslation after a correction (previous_translation_content_id set) is not charged
        at all; see the module docstring.

        A message without a workspace is from a producer older than this contract. It is skipped,
        not retried: no number of redeliveries will add the field.
        """
        msg = TranslationResultMessage.from_redis(data)
        if not msg.translated_text.strip():
            return

        if msg.previous_translation_content_id:
            # Logged, not silent: the model call still happened, and this is the only record of
            # what the platform absorbed for its own transcription errors.
            self.logger.info(
                "retranslation_not_charged",
                translation_room_id=msg.meeting_id,
                workspace_id=msg.workspace_id,
                transcript_id=msg.transcript_id,
                segment_id=msg.segment_id,
                target_lang=msg.target_lang,
                seconds=_translation_quantity_seconds(msg),
            )
            return

        segment_id = _extract_underlying_segment_id(msg.segment_id)
        if segment_id is None:
            self.logger.warning(
                "backfill_settlement_skipped_bad_segment_id", raw_segment_id=msg.segment_id
            )
            return

        if not msg.workspace_id:
            self.logger.warning(
                "backfill_settlement_skipped_no_workspace",
                translation_room_id=msg.meeting_id,
                segment_id=segment_id,
            )
            return
        try:
            uuid.UUID(msg.workspace_id)
            uuid.UUID(msg.meeting_id)
        except ValueError:
            self.logger.warning(
                "backfill_settlement_skipped_unparseable_ids",
                workspace_id=msg.workspace_id,
                translation_room_id=msg.meeting_id,
            )
            return

        resolved = await self._resolve_workspace_subscription(msg.workspace_id)
        if resolved is None:
            # No room to stop — a backfill runs after the meeting — but never silent: it is
            # counted, so a lapsed workspace reading back its transcripts shows up as unbilled work.
            self.logger.warning(
                "no_subscription_for_workspace",
                workspace_id=msg.workspace_id,
                translation_room_id=msg.meeting_id,
                charge_type=TRANSLATION_CHARGE_TYPE,
            )
            await self._count_unbilled(UNBILLED_NO_SUBSCRIPTION, TRANSLATION_CHARGE_TYPE)
            return
        subscription_id, workspace_id = resolved

        requested_by = msg.requested_by_user_id
        if requested_by:
            try:
                uuid.UUID(requested_by)
            except ValueError:
                self.logger.warning(
                    "backfill_settlement_unparseable_requester", requested_by=requested_by
                )
                requested_by = None

        target_lang = msg.target_lang
        idempotency_key = f"{TRANSLATION_CHARGE_TYPE}:backfill:{segment_id}:{target_lang}"

        outcome = await self.db.record_usage_and_charge(
            subscription_id=subscription_id,
            user_id=requested_by,
            workspace_id=workspace_id,
            translation_room_id=msg.meeting_id,
            usage_type=TRANSLATION_CHARGE_TYPE,
            charge_type=TRANSLATION_CHARGE_TYPE,
            reference_id=segment_id,
            reference_type="translation_content",
            quantity=_translation_quantity_seconds(msg),
            unit="second",
            source_language_code=msg.source_lang,
            target_language_code=target_lang,
            transcript_segment_id=segment_id,
            idempotency_key=idempotency_key,
            details={
                # A reporting dimension that fails towards internal (see _is_external_speaker);
                # the requester already passed the transcript read check of the workspace.
                "is_external": False,
                "origin": "backfill",
                "transcript_id": msg.transcript_id,
            },
        )
        if isinstance(outcome, SettlementOutcome) and not (outcome.applied or outcome.replayed):
            await self._count_unbilled(UNBILLED_CHARGE_REFUSED, TRANSLATION_CHARGE_TYPE)

    async def _handle_tts(self, data: Mapping[Any, Any]) -> None:
        msg = TTSResultMessage.from_redis(data)
        if msg.cache_hit:
            return  # reused a previously synthesized clip — no new provider cost incurred
        if not msg.audio_data:
            return
        if self._is_unbillable(msg.speaker_id, msg.meeting_id):
            return

        charge_type = (
            "AUDIO_DUBBING_VOICE_CLONE" if msg.voice_type == "cloned" else "AUDIO_DUBBING_STANDARD"
        )

        resolved = await self._resolve_subscription(msg.meeting_id)
        if isinstance(resolved, NoActiveSubscription):
            await self._stop_unpaid_room(msg.meeting_id, resolved, charge_type)
            return
        subscription_id, workspace_id = resolved

        # Same composite-segment-id situation as _handle_translation above — extract the
        # real TranscriptSegment.Id before using it as a UUID column value.
        underlying_segment_id = _extract_underlying_segment_id(msg.segment_id)
        if underlying_segment_id is None:
            self.logger.warning("segment_id_extraction_failed", raw_segment_id=msg.segment_id)

        quantity_s = max(msg.duration_ms / 1000.0, 0.1)
        outcome = await self.db.record_usage_and_charge(
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
            details={
                "clone_provider": msg.clone_provider,
                "voice_mode": msg.voice_mode,
                # See _handle_translation — the usage_records column is generated from this key.
                "is_external": await self._is_external_speaker(msg.meeting_id, msg.speaker_id),
            },
        )
        await self._note_settlement(
            msg.meeting_id, subscription_id, workspace_id, outcome, charge_type
        )

    def _register_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)
