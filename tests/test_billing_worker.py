"""Tests for Billing Settlement Worker — segment-id extraction helper."""

from __future__ import annotations

import asyncio
import json
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from billing_worker import db as billing_db
from billing_worker.worker import (
    BillingSettlementWorker,
    NoActiveSubscription,
    _extract_underlying_segment_id,
)
from shared.health_probe import check_worker
from shared.schemas import TranslationResultMessage


class TestExtractUnderlyingSegmentId:
    """Byte-for-byte port of TranscriptRedisConsumerService.ExtractUnderlyingSegmentId (C#)."""

    def test_plain_guid_passthrough(self) -> None:
        guid = str(uuid.uuid4())
        assert _extract_underlying_segment_id(guid) == guid

    def test_composite_segment_id_single_digit_chunk(self) -> None:
        guid = str(uuid.uuid4())
        composite = f"{guid}-c0"
        assert _extract_underlying_segment_id(composite) == guid

    def test_composite_segment_id_multi_digit_chunk(self) -> None:
        guid = str(uuid.uuid4())
        composite = f"{guid}-c12"
        assert _extract_underlying_segment_id(composite) == guid

    def test_malformed_input_returns_none(self) -> None:
        assert _extract_underlying_segment_id("not-a-guid-at-all") is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_underlying_segment_id("") is None

    def test_none_like_falsy_returns_none(self) -> None:
        assert _extract_underlying_segment_id(None) is None  # type: ignore[arg-type]


async def test_subscription_resolution_uses_redis_room_projection_not_foreign_database() -> None:
    room_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    subscription_id = uuid.uuid4()
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(
        return_value=(f'{{"WorkspaceId":"{workspace_id}","Status":"IN_PROGRESS"}}').encode()
    )
    worker.db = MagicMock()
    worker.db.resolve_subscription = AsyncMock(
        return_value=(subscription_id, uuid.UUID(workspace_id))
    )

    resolved = await worker._resolve_subscription(room_id)

    assert resolved == (subscription_id, uuid.UUID(workspace_id))
    worker.db.resolve_subscription.assert_awaited_once_with(workspace_id)


async def test_subscription_resolution_reads_the_versioned_projection_key() -> None:
    """The key is a cross-repo contract, and nothing was pinning it.

    MeetingService owns `meeting:room:v2:<id>` and bumped it to v2 in WT-428. This reader kept
    asking for the unversioned key; nothing writes that any more, so every settlement found no
    projection, raised, retried and dead-lettered — while the meeting itself worked perfectly and
    simply billed nothing.

    The two tests either side of this one both stub `redis.get` with a blanket AsyncMock, so they
    pass whatever key is asked for. That is why the break was invisible. This one asserts the
    string.
    """
    room_id = str(uuid.uuid4())
    workspace_id = str(uuid.uuid4())
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(
        return_value=(f'{{"WorkspaceId":"{workspace_id}","Status":"IN_PROGRESS"}}').encode()
    )
    worker.db = MagicMock()
    worker.db.resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), uuid.UUID(workspace_id)))

    await worker._resolve_subscription(room_id)

    worker.redis.get.assert_awaited_once_with(f"meeting:room:v2:{room_id}")


def _externality_worker(hget_result: object) -> BillingSettlementWorker:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.redis = MagicMock()
    worker.redis.redis = MagicMock()
    if isinstance(hget_result, Exception):
        worker.redis.redis.hget = AsyncMock(side_effect=hget_result)
    else:
        worker.redis.redis.hget = AsyncMock(return_value=hget_result)
    return worker


async def test_a_guest_speaker_is_attributed_as_external() -> None:
    worker = _externality_worker(b"1")

    assert await worker._is_external_speaker("room-1", "speaker-1") is True
    worker.redis.redis.hget.assert_awaited_once_with(
        "translationRoom:room-1:external_participants", "speaker-1"
    )


async def test_a_member_speaker_is_not_external() -> None:
    assert await _externality_worker(b"0")._is_external_speaker("room-1", "speaker-1") is False


async def test_an_evicted_or_absent_field_attributes_as_internal() -> None:
    # Redis runs allkeys-lru and drops live meeting state. Under-attributing external spend is
    # tolerable; inventing it, or failing the settlement, is not.
    assert await _externality_worker(None)._is_external_speaker("room-1", "speaker-1") is False


async def test_a_redis_failure_never_costs_the_usage_record() -> None:
    worker = _externality_worker(ConnectionError("redis is down"))

    assert await worker._is_external_speaker("room-1", "speaker-1") is False


async def test_a_missing_speaker_id_is_not_external() -> None:
    # The __MEETING_END__ sentinel and friends reach here with no usable speaker.
    worker = _externality_worker(b"1")

    assert await worker._is_external_speaker("room-1", None) is False
    worker.redis.redis.hget.assert_not_awaited()


async def test_subscription_resolution_fails_when_room_projection_is_missing() -> None:
    room_id = str(uuid.uuid4())
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._subscription_cache = {}
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(return_value=None)
    worker.db = MagicMock()
    worker.logger = MagicMock()

    with pytest.raises(RuntimeError, match="Room projection is unavailable"):
        await worker._resolve_subscription(room_id)


def test_credit_charge_rounds_rate_card_cost_up_to_whole_credit() -> None:
    calculate_credit_charge = getattr(billing_db, "calculate_credit_charge", None)
    assert callable(calculate_credit_charge)
    assert calculate_credit_charge(1.0, Decimal("0.25")) == 1
    assert calculate_credit_charge(61.0, Decimal("0.25")) == 16


async def test_settlement_error_propagates_so_message_remains_pending() -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    handler = AsyncMock(side_effect=RuntimeError("database unavailable"))
    process_message = getattr(worker, "_process_settlement_message", None)
    assert callable(process_message)

    try:
        await process_message(
            "stt:results",
            b"1-0",
            {b"text": b"hello"},
            handler,
        )
    except RuntimeError as error:
        assert str(error) == "database unavailable"
    else:
        raise AssertionError("settlement failure was swallowed")


async def test_billing_heartbeat_satisfies_shared_health_probe(monkeypatch) -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker._consumer_name = "billing-host-1"
    worker.redis = MagicMock()
    worker.redis.set_with_ttl = AsyncMock()

    await worker._publish_heartbeat()

    _, payload, _ = worker.redis.set_with_ttl.await_args.args
    redis = AsyncMock()
    redis.mget = AsyncMock(return_value=[payload.encode()])
    client = MagicMock()
    client.redis = redis
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    monkeypatch.setenv("WORKER_HEALTH_NAME", "billing")

    with (
        patch("shared.health_probe.RedisStreamClient", return_value=client),
        patch("shared.health_probe.socket.gethostname", return_value="host-1"),
    ):
        assert await check_worker() is True

    assert isinstance(json.loads(payload)["last_progress_unix_ms"], int)


async def test_billing_heartbeat_loop_recovers_after_transient_redis_failure() -> None:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    worker.heartbeat_interval_seconds = 0
    worker._shutdown_event = asyncio.Event()
    attempts = 0

    async def publish_heartbeat() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Redis restarted")
        worker._shutdown_event.set()

    worker._publish_heartbeat = AsyncMock(side_effect=publish_heartbeat)

    await asyncio.wait_for(worker._heartbeat_loop(), timeout=1)

    assert worker._publish_heartbeat.await_count == 2
    worker.logger.exception.assert_called_once_with("billing_heartbeat_failed")


class TestBillableSurface:
    """WT-344 — what a meeting pays for, pinned.

    Transcription and the inline assistant are FREE on the owner's call; translation and
    dubbing are what a workspace spends credits on. This is a product decision that lives
    in exactly one place — which streams the settlement worker subscribes to — so it is
    worth asserting directly rather than inferring from behaviour.

    These would both have passed while the STT and suggestion handlers still existed, so
    they assert the ABSENCE of the handler as well as the absence of the stream: a future
    change that re-adds a handler without re-subscribing (or vice versa) fails here.
    """

    @staticmethod
    def _subscribed_streams() -> set[str]:
        import inspect

        source = inspect.getsource(BillingSettlementWorker.start)
        return {
            stream
            for stream in (
                "stt:results",
                "translate:results",
                "translate:backfill_results",
                "tts:results",
                "ai_assistant:results",
            )
            if f'"{stream}"' in source
        }

    def test_only_translation_and_dubbing_are_billed(self) -> None:
        # translate:backfill_results is translation too, produced after the meeting ended.
        assert self._subscribed_streams() == {
            "translate:results",
            "translate:backfill_results",
            "tts:results",
        }

    def test_free_pipelines_have_no_settlement_handler_left_behind(self) -> None:
        # A handler with no subscription is dead code that reads as a live feature — the
        # exact shape of defect this codebase has hit repeatedly.
        assert not hasattr(BillingSettlementWorker, "_handle_stt")
        assert not hasattr(BillingSettlementWorker, "_handle_suggestion")

    def test_the_billable_handlers_are_still_wired(self) -> None:
        assert hasattr(BillingSettlementWorker, "_handle_translation")
        assert hasattr(BillingSettlementWorker, "_handle_tts")


class TestEarlySegmentsAreNotCharged:
    """One turn must cost the same whether or not STT published it a sentence at a time.

    stt_worker publishes each sentence the model finishes MID-CHUNK, and the completed
    segment for that same chunk still carries the chunk's WHOLE duration
    (stt_worker/model.py: `"end": duration_s`). So charging both bills the same audio twice.

    Charging the early ones on their own terms would be wrong in the other direction: they
    carry start_ms == end_ms, which sends `_handle_translation` down its zero-duration
    fallback of a flat 1.0s — pricing a fifteen-second turn as four seconds.

    Both mistakes are silent and both are money, which is why this is pinned.
    """

    @staticmethod
    def _worker() -> BillingSettlementWorker:
        worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
        worker.logger = MagicMock()
        worker.db = MagicMock()
        worker.db.record_usage_and_charge = AsyncMock()
        worker._resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), uuid.uuid4()))
        worker._is_unbillable = MagicMock(return_value=False)
        return worker

    @staticmethod
    def _message(*, is_early: bool) -> dict[str, str]:
        return TranslationResultMessage(
            segment_id=f"{uuid.uuid4()}-vi-c0",
            meeting_id=str(uuid.uuid4()),
            speaker_id=str(uuid.uuid4()),
            original_text="Hello there.",
            translated_text="Xin chào.",
            source_lang="en",
            target_lang="vi",
            start_ms=1000,
            end_ms=1000 if is_early else 16000,
            is_early=is_early,
        ).to_redis()

    def test_an_early_sentence_is_not_charged(self) -> None:
        worker = self._worker()
        asyncio.run(worker._handle_translation(self._message(is_early=True)))
        worker.db.record_usage_and_charge.assert_not_awaited()

    def test_the_completed_segment_still_is(self) -> None:
        worker = self._worker()
        asyncio.run(worker._handle_translation(self._message(is_early=False)))
        worker.db.record_usage_and_charge.assert_awaited_once()
        # 15s of audio, billed as seconds — not the 1.0s zero-duration fallback.
        assert worker.db.record_usage_and_charge.await_args.kwargs["quantity"] == 15.0

    def test_a_message_from_before_the_flag_existed_is_still_charged(self) -> None:
        """Rolling deploy: an older translation worker publishes no `is_early` field at all."""
        worker = self._worker()
        payload = self._message(is_early=False)
        payload.pop("is_early")
        asyncio.run(worker._handle_translation(payload))
        worker.db.record_usage_and_charge.assert_awaited_once()


class _FakeConnection:
    """Records what was executed and replays canned rows, so settlement can be tested
    without a database. Only fetchrow is used by record_usage_and_charge."""

    def __init__(self, rate_row, settle_row) -> None:
        self._rate_row = rate_row
        self._settle_row = settle_row
        self.queries: list[str] = []
        self.settle_args: tuple = ()

    async def fetchrow(self, query, *args):
        self.queries.append(query)
        if "usage_rate_card" in query:
            return self._rate_row
        if "settle_usage_charge" in query:
            self.settle_args = args
            return self._settle_row
        raise AssertionError(f"unexpected query: {query}")

    async def execute(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("settlement must not write its own statements")

    async def fetchval(self, *args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("settlement must not write its own statements")


class _FakePool:
    def __init__(self, conn) -> None:
        self._conn = conn

    def acquire(self):
        conn = self._conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


def _repository(settle_row):
    repo = billing_db.BillingRepository.__new__(billing_db.BillingRepository)
    conn = _FakeConnection(
        rate_row={"id": uuid.uuid4(), "unit_price": Decimal("0.25"), "currency": "CRD"},
        settle_row=settle_row,
    )
    repo._pool = _FakePool(conn)
    return repo, conn


async def _settle(repo, **overrides):
    kwargs = dict(
        subscription_id=uuid.uuid4(),
        user_id=str(uuid.uuid4()),
        workspace_id=uuid.uuid4(),
        translation_room_id=str(uuid.uuid4()),
        usage_type="TRANSLATION",
        charge_type="TRANSLATION",
        reference_id=str(uuid.uuid4()),
        reference_type="translation_content",
        quantity=4.0,
        unit="second",
        idempotency_key="TRANSLATION:seg:vi",
    )
    kwargs.update(overrides)
    return await repo.record_usage_and_charge(**kwargs)


class TestSettlementGoesThroughTheDatabaseFunction:
    """The worker must not reimplement settlement.

    It used to write the usage record, the balance UPDATE and the credit transaction itself.
    That version had no overage, never wrote service_state, and raised when a workspace ran
    out of credits — turning an expected business state into a crash loop on redelivery.
    """

    async def test_it_calls_settle_usage_charge_and_writes_nothing_itself(self) -> None:
        repo, conn = _repository(
            {
                "applied": True,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": 900,
                "service_state": "healthy",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is True
        assert outcome.service_state == "healthy"
        assert outcome.balance_after == 900
        # _FakeConnection.execute/fetchval raise, so reaching here proves no hand-written
        # INSERT or UPDATE survived.
        assert any("settle_usage_charge" in query for query in conn.queries)

    async def test_running_out_of_credits_is_a_state_not_an_exception(self) -> None:
        # The whole point of the consolidation. The old code raised
        # "Insufficient credits for subscription ...", which crashed the handler and left the
        # Redis message pending forever.
        repo, _ = _repository(
            {
                "applied": True,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": -12,
                "service_state": "in_overage",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is True
        assert outcome.service_state == "in_overage"
        assert outcome.balance_after == -12

    async def test_a_suspended_subscription_is_refused_without_raising(self) -> None:
        repo, _ = _repository(
            {
                "applied": False,
                "transaction_id": None,
                "usage_record_id": None,
                "balance_after": 0,
                "service_state": "suspended",
                "suspended_reason": "overage_cap",
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is False
        assert outcome.replayed is False
        assert outcome.suspended_reason == "overage_cap"

    async def test_a_replay_is_distinguishable_from_a_refusal(self) -> None:
        # Both have applied=False. Only the replay carries the original transaction, and the
        # two must not be logged or reacted to the same way.
        repo, _ = _repository(
            {
                "applied": False,
                "transaction_id": uuid.uuid4(),
                "usage_record_id": uuid.uuid4(),
                "balance_after": 900,
                "service_state": "healthy",
                "suspended_reason": None,
            }
        )

        outcome = await _settle(repo)

        assert outcome.applied is False
        assert outcome.replayed is True

    async def test_a_missing_rate_card_still_raises(self) -> None:
        # A misconfiguration, not a business state — it must not be settled silently at zero.
        repo, conn = _repository({"applied": True})
        conn._rate_row = None

        with pytest.raises(RuntimeError, match="No active usage rate card"):
            await _settle(repo)


# ── WT-699 / TC3705: a refused charge stops the room's translation ─────────────────────────


def _suspension_worker() -> BillingSettlementWorker:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    worker._suspended_rooms = {}
    worker.redis = MagicMock()
    worker.redis.set_if_absent = AsyncMock(return_value=True)
    worker.redis.set_with_ttl = AsyncMock()
    worker.redis.expire = AsyncMock()
    worker.redis.redis = MagicMock()
    worker.redis.redis.publish = AsyncMock()
    worker.redis.redis.delete = AsyncMock(return_value=2)
    worker.redis.redis.pipeline = MagicMock(return_value=MagicMock(execute=AsyncMock()))
    worker.db = MagicMock()
    worker._subscription_cache = {}
    worker._workspace_subscription_cache = {}
    return worker


def _outcome(*, applied: bool, replayed: bool = False) -> billing_db.SettlementOutcome:
    return billing_db.SettlementOutcome(
        applied=applied,
        replayed=replayed,
        balance_after=0,
        service_state="suspended" if not applied else "healthy",
        suspended_reason="overage_cap" if not applied else None,
        credits_consumed=3,
    )


class TestRefusedChargeStopsTranslation:
    """Translation used to keep running at zero credits: the refusal was logged and ignored."""

    async def test_a_refused_charge_flags_the_room_and_tells_the_room(self) -> None:
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        subscription_id, workspace_id = uuid.uuid4(), uuid.uuid4()

        await worker._note_settlement(
            room_id, subscription_id, workspace_id, _outcome(applied=False)
        )

        worker.redis.set_if_absent.assert_awaited_once_with(
            f"translationRoom:{room_id}:ai_service_suspended", "true", 120
        )
        written = {call.args[0]: call.args[1] for call in worker.redis.set_with_ttl.await_args_list}
        assert written[f"workspace:{workspace_id}:ai_service_suspended"] == "true"
        state = json.loads(written[f"translationRoom:{room_id}:ai_service_state"])
        assert state["suspendedReason"] == "overage_cap"
        channel, message = worker.redis.redis.publish.await_args.args
        assert channel == "warptalk:translation-room:commands"
        assert json.loads(message) == {
            "Command": "TranslationCreditsExhausted",
            "RoomId": room_id,
            "Reason": "overage_cap",
        }
        assert worker._suspended_rooms[room_id] == (subscription_id, workspace_id)

    async def test_the_room_is_announced_once_however_many_charges_are_refused(self) -> None:
        worker = _suspension_worker()
        worker.redis.set_if_absent = AsyncMock(return_value=False)

        await worker._note_settlement(
            str(uuid.uuid4()), uuid.uuid4(), uuid.uuid4(), _outcome(applied=False)
        )

        worker.redis.redis.publish.assert_not_awaited()
        worker.redis.expire.assert_awaited()

    @pytest.mark.parametrize(
        "outcome",
        [_outcome(applied=True), _outcome(applied=False, replayed=True), MagicMock()],
    )
    async def test_a_paid_or_replayed_charge_stops_nothing(self, outcome: object) -> None:
        worker = _suspension_worker()

        await worker._note_settlement(str(uuid.uuid4()), uuid.uuid4(), uuid.uuid4(), outcome)

        worker.redis.set_if_absent.assert_not_awaited()
        worker.redis.redis.publish.assert_not_awaited()

    async def test_the_room_stays_stopped_while_the_subscription_is_suspended(self) -> None:
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        worker._suspended_rooms[room_id] = (uuid.uuid4(), uuid.uuid4())
        worker.db.resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), uuid.uuid4()))
        worker.db.get_service_state = AsyncMock(return_value=("suspended", "overage_cap"))

        await worker._recheck_suspended_rooms()

        assert room_id in worker._suspended_rooms
        worker.redis.redis.delete.assert_not_awaited()
        refreshed = {call.args[0] for call in worker.redis.expire.await_args_list}
        assert f"translationRoom:{room_id}:ai_service_suspended" in refreshed

    async def test_the_room_translates_again_once_the_workspace_can_pay(self) -> None:
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        worker._suspended_rooms[room_id] = (uuid.uuid4(), uuid.uuid4())
        worker.db.resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), uuid.uuid4()))
        worker.db.get_service_state = AsyncMock(return_value=("healthy", None))

        await worker._recheck_suspended_rooms()

        assert room_id not in worker._suspended_rooms
        channel, message = worker.redis.redis.publish.await_args.args
        assert json.loads(message) == {"Command": "TranslationCreditsRestored", "RoomId": room_id}

    async def test_the_translation_handler_reports_its_settlement(self) -> None:
        """The wiring, end to end through _handle_translation: a refusal there stops the room."""
        worker = _suspension_worker()
        worker._subscription_cache = {}
        worker.settings = MagicMock()
        worker.settings.subscription_cache_ttl_seconds = 300
        room_id = str(uuid.uuid4())
        subscription_id, workspace_id = uuid.uuid4(), uuid.uuid4()
        worker._resolve_subscription = AsyncMock(return_value=(subscription_id, workspace_id))
        worker._is_external_speaker = AsyncMock(return_value=False)
        worker.db.record_usage_and_charge = AsyncMock(return_value=_outcome(applied=False))
        worker.redis.redis.hget = AsyncMock(return_value=None)
        msg = TranslationResultMessage(
            segment_id=f"{uuid.uuid4()}-vi-c0",
            meeting_id=room_id,
            speaker_id=str(uuid.uuid4()),
            original_text="hello",
            translated_text="xin chao",
            source_lang="en",
            target_lang="vi",
            is_final_chunk=True,
        )

        await worker._handle_translation(msg.to_redis())

        assert worker._suspended_rooms[room_id] == (subscription_id, workspace_id)


# ── The expired-subscription leak: nobody to bill is a refusal, not a shrug ────────────────


def _unpaid_worker(workspace_id: uuid.UUID) -> BillingSettlementWorker:
    worker = _suspension_worker()
    worker.settings = MagicMock()
    worker.settings.subscription_cache_ttl_seconds = 300
    worker.redis.get = AsyncMock(
        return_value=(f'{{"WorkspaceId":"{workspace_id}","Status":"IN_PROGRESS"}}').encode()
    )
    worker.db.resolve_subscription = AsyncMock(return_value=None)
    worker.db.record_usage_and_charge = AsyncMock()
    return worker


def _translation(room_id: str) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id=f"{uuid.uuid4()}-vi-c0",
        meeting_id=room_id,
        speaker_id=str(uuid.uuid4()),
        original_text="hello",
        translated_text="xin chao",
        source_lang="en",
        target_lang="vi",
        is_final_chunk=True,
    )


class TestNoSubscriptionStopsTranslation:
    """2026-09-24: an expired workspace translated a 14-minute meeting free.

    `resolve_subscription` found no active row, the handler logged `no_subscription_for_room` and
    returned — no charge, so no refusal, so nothing ever stopped the room.
    """

    async def test_resolution_names_the_workspace_that_cannot_pay(self) -> None:
        workspace_id = uuid.uuid4()
        worker = _unpaid_worker(workspace_id)

        resolved = await worker._resolve_subscription(str(uuid.uuid4()))

        assert isinstance(resolved, NoActiveSubscription)
        assert resolved.workspace_id == workspace_id
        assert worker._subscription_cache == {}, "a renewal must be seen on the very next event"

    async def test_a_translation_with_no_subscription_stops_the_room_and_says_why(self) -> None:
        workspace_id = uuid.uuid4()
        worker = _unpaid_worker(workspace_id)
        room_id = str(uuid.uuid4())

        await worker._handle_translation(_translation(room_id).to_redis())

        worker.db.record_usage_and_charge.assert_not_awaited()
        worker.redis.set_if_absent.assert_awaited_once_with(
            f"translationRoom:{room_id}:ai_service_suspended", "true", 120
        )
        written = {call.args[0]: call.args[1] for call in worker.redis.set_with_ttl.await_args_list}
        assert written[f"workspace:{workspace_id}:ai_service_suspended"] == "true"
        state = json.loads(written[f"translationRoom:{room_id}:ai_service_state"])
        assert state["suspendedReason"] == "subscription_expired"
        channel, message = worker.redis.redis.publish.await_args.args
        assert channel == "warptalk:translation-room:commands"
        assert json.loads(message) == {
            "Command": "TranslationCreditsExhausted",
            "RoomId": room_id,
            "Reason": "subscription_expired",
        }
        assert worker._suspended_rooms[room_id] == (None, workspace_id)

    async def test_it_is_logged_as_a_warning_and_counted_never_silent(self) -> None:
        workspace_id = uuid.uuid4()
        worker = _unpaid_worker(workspace_id)
        room_id = str(uuid.uuid4())
        pipeline = MagicMock(execute=AsyncMock())
        worker.redis.redis.pipeline = MagicMock(return_value=pipeline)

        await worker._handle_translation(_translation(room_id).to_redis())

        warned = [call.args[0] for call in worker.logger.warning.call_args_list]
        assert "no_subscription_for_room" in warned
        pipeline.hincrby.assert_called_once_with(
            "warptalk:billing:unbilled", "no_active_subscription:TRANSLATION", 1
        )

    async def test_a_dub_with_no_subscription_stops_the_room_too(self) -> None:
        workspace_id = uuid.uuid4()
        worker = _unpaid_worker(workspace_id)
        room_id = str(uuid.uuid4())
        msg = MagicMock(
            cache_hit=False,
            audio_data=b"pcm",
            speaker_id=str(uuid.uuid4()),
            meeting_id=room_id,
            voice_type="standard",
        )

        with patch("billing_worker.worker.TTSResultMessage.from_redis", return_value=msg):
            await worker._handle_tts({})

        worker.db.record_usage_and_charge.assert_not_awaited()
        assert worker._suspended_rooms[room_id] == (None, workspace_id)

    async def test_a_room_stopped_mid_meeting_when_the_subscription_expires(self) -> None:
        """In flight: the room's subscription id is cached, and the plan ends under it.

        settle_usage_charge refuses an inactive row with 'subscription_expired'. The room stops,
        and the cached id is forgotten so a renewal is billed on the next event instead of the
        ended plan being refused again.
        """
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        subscription_id, workspace_id = uuid.uuid4(), uuid.uuid4()
        worker._subscription_cache[room_id] = (subscription_id, workspace_id, 0.0)
        refused = billing_db.SettlementOutcome(
            applied=False,
            replayed=False,
            balance_after=500,
            service_state="suspended",
            suspended_reason="subscription_expired",
            credits_consumed=3,
        )

        await worker._note_settlement(room_id, subscription_id, workspace_id, refused)

        assert room_id not in worker._subscription_cache
        assert worker._suspended_rooms[room_id] == (subscription_id, workspace_id)
        _, message = worker.redis.redis.publish.await_args.args
        assert json.loads(message)["Reason"] == "subscription_expired"

    async def test_the_room_stays_stopped_until_the_workspace_has_a_subscription_again(
        self,
    ) -> None:
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        workspace_id = uuid.uuid4()
        worker._suspended_rooms[room_id] = (None, workspace_id)
        worker.db.resolve_subscription = AsyncMock(return_value=None)
        worker.db.get_service_state = AsyncMock()

        await worker._recheck_suspended_rooms()

        assert room_id in worker._suspended_rooms
        worker.redis.redis.delete.assert_not_awaited()
        worker.db.resolve_subscription.assert_awaited_once_with(str(workspace_id))

    async def test_renewal_re_enables_the_room(self) -> None:
        """A renewal after expiry is a NEW subscription row; the room is released on it."""
        worker = _suspension_worker()
        room_id = str(uuid.uuid4())
        ended_subscription, workspace_id = uuid.uuid4(), uuid.uuid4()
        renewed = uuid.uuid4()
        worker._suspended_rooms[room_id] = (ended_subscription, workspace_id)
        worker._subscription_cache[room_id] = (ended_subscription, workspace_id, 0.0)
        worker.db.resolve_subscription = AsyncMock(return_value=(renewed, workspace_id))
        worker.db.get_service_state = AsyncMock(return_value=("healthy", None))

        await worker._recheck_suspended_rooms()

        worker.db.get_service_state.assert_awaited_once_with(renewed)
        assert room_id not in worker._suspended_rooms
        assert room_id not in worker._subscription_cache
        _, message = worker.redis.redis.publish.await_args.args
        assert json.loads(message) == {"Command": "TranslationCreditsRestored", "RoomId": room_id}
