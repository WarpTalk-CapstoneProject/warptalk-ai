"""Post-meeting translation is charged, and charged like live translation.

translation_worker/backfill_worker.py publishes to translate:backfill_results so that tts_worker
does not dub a meeting that ended hours ago. The billing worker only read translate:results, so
every backfilled line and every post-correction retranslation was a model call nobody paid for.

Pinned here: the stream is subscribed, the price is the live price (same charge type, same unit,
same quantity rule), the workspace comes from the message rather than the room projection that
has long expired, duplicates of the same work are charged once, and the redo of a line whose
transcript was corrected is not charged at all.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from unittest.mock import AsyncMock, MagicMock

from billing_worker.worker import BACKFILL_RESULT_STREAM, BillingSettlementWorker
from shared.schemas import TranslationResultMessage

SEGMENT = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ROOM = "dddddddd-dddd-dddd-dddd-dddddddddddd"
WORKSPACE = "33333333-3333-3333-3333-333333333333"
REQUESTER = "44444444-4444-4444-4444-444444444444"
TRANSCRIPT = "11111111-1111-1111-1111-111111111111"


def _worker(subscription: tuple[uuid.UUID, uuid.UUID] | None = None) -> BillingSettlementWorker:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    worker.settings = MagicMock(subscription_cache_ttl_seconds=300)
    worker.db = MagicMock()
    worker.db.record_usage_and_charge = AsyncMock()
    worker.db.resolve_subscription = AsyncMock(
        return_value=subscription or (uuid.uuid4(), uuid.UUID(WORKSPACE))
    )
    worker._workspace_subscription_cache = {}
    # The live path's room projection must never be consulted for a backfill.
    worker._resolve_subscription = AsyncMock(side_effect=AssertionError("room projection read"))
    return worker


def _backfill(**overrides) -> dict[str, str]:
    fields = dict(
        segment_id=SEGMENT,
        meeting_id=ROOM,
        speaker_id="",
        original_text="Xin chào mọi người.",
        translated_text="Hello everyone.",
        source_lang="vi",
        target_lang="en",
        translator_model="gpt-4.1",
        source_segment_id=SEGMENT,
        is_final_chunk=True,
        start_ms=12_000,
        end_ms=15_500,
        workspace_id=WORKSPACE,
        requested_by_user_id=REQUESTER,
        transcript_id=TRANSCRIPT,
    )
    fields.update(overrides)
    return TranslationResultMessage(**fields).to_redis()


def _charge(worker: BillingSettlementWorker) -> dict:
    worker.db.record_usage_and_charge.assert_awaited_once()
    return worker.db.record_usage_and_charge.await_args.kwargs


class TestTheBackfillStreamIsBilled:
    def test_the_worker_subscribes_to_it(self) -> None:
        source = inspect.getsource(BillingSettlementWorker.start)
        assert f'"{BACKFILL_RESULT_STREAM}"' in source
        assert "self._handle_backfill_translation" in source

    def test_it_is_not_the_stream_tts_reads(self) -> None:
        # Moving backfills onto translate:results to "get them billed" would dub them.
        assert BACKFILL_RESULT_STREAM != "translate:results"


class TestPricedLikeALiveTranslation:
    def test_same_charge_type_unit_and_seconds_of_speech(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        charge = _charge(worker)
        assert charge["charge_type"] == "TRANSLATION"
        assert charge["usage_type"] == "TRANSLATION"
        assert charge["unit"] == "second"
        assert charge["quantity"] == 3.5
        assert charge["source_language_code"] == "vi"
        assert charge["target_language_code"] == "en"
        assert charge["reference_type"] == "translation_content"
        assert charge["transcript_segment_id"] == SEGMENT

    def test_the_same_message_on_the_live_path_would_cost_the_same_quantity(self) -> None:
        live = _worker()
        live._resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), uuid.UUID(WORKSPACE)))
        live._is_unbillable = MagicMock(return_value=False)
        live._is_external_speaker = AsyncMock(return_value=False)
        asyncio.run(
            live._handle_translation(_backfill(segment_id=f"{SEGMENT}-en-c0", speaker_id=REQUESTER))
        )

        backfill = _worker()
        asyncio.run(backfill._handle_backfill_translation(_backfill()))

        assert _charge(live)["quantity"] == _charge(backfill)["quantity"]

    def test_an_older_producer_without_timing_falls_back_to_the_live_flat_second(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill(start_ms=0, end_ms=0)))
        assert _charge(worker)["quantity"] == 1.0


class TestWhoPays:
    def test_the_workspace_comes_from_the_message_not_the_expired_room_projection(self) -> None:
        subscription_id = uuid.uuid4()
        worker = _worker((subscription_id, uuid.UUID(WORKSPACE)))
        asyncio.run(worker._handle_backfill_translation(_backfill()))

        worker.db.resolve_subscription.assert_awaited_once_with(WORKSPACE)
        charge = _charge(worker)
        assert charge["subscription_id"] == subscription_id
        assert charge["workspace_id"] == uuid.UUID(WORKSPACE)
        assert charge["translation_room_id"] == ROOM

    def test_the_requester_is_the_user_because_nobody_spoke_it(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        assert _charge(worker)["user_id"] == REQUESTER

    def test_no_requester_is_still_charged_to_the_workspace(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill(requested_by_user_id=None)))
        assert _charge(worker)["user_id"] is None

    def test_a_malformed_requester_does_not_lose_the_charge(self) -> None:
        worker = _worker()
        asyncio.run(
            worker._handle_backfill_translation(_backfill(requested_by_user_id="not-a-uuid"))
        )
        assert _charge(worker)["user_id"] is None

    def test_a_message_without_a_workspace_is_skipped_not_retried(self) -> None:
        # From a producer older than this contract: redelivery will never add the field, so
        # raising would only dead-letter it five times over.
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill(workspace_id=None)))
        worker.db.record_usage_and_charge.assert_not_awaited()

    def test_a_workspace_without_a_subscription_is_not_charged(self) -> None:
        worker = _worker()
        worker.db.resolve_subscription = AsyncMock(return_value=None)
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        worker.db.record_usage_and_charge.assert_not_awaited()

    def test_empty_output_costs_nothing(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill(translated_text="  ")))
        worker.db.record_usage_and_charge.assert_not_awaited()

    def test_the_subscription_lookup_is_cached_per_workspace(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        asyncio.run(worker._handle_backfill_translation(_backfill(target_lang="ja")))
        worker.db.resolve_subscription.assert_awaited_once()


class TestIdempotency:
    @staticmethod
    def _key(**overrides) -> str:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill(**overrides)))
        return _charge(worker)["idempotency_key"]

    def test_a_duplicated_gap_fill_is_charged_once(self) -> None:
        # Two readers racing, or a run marker that expired mid-run, queue the same line twice.
        assert self._key() == self._key()

    def test_each_language_is_its_own_charge(self) -> None:
        assert self._key(target_lang="en") != self._key(target_lang="ja")

    def test_it_never_collides_with_the_live_charge_of_the_same_segment(self) -> None:
        live_key = f"TRANSLATION:{SEGMENT}-en-c0:en"
        assert self._key() != live_key

    def test_origin_is_recorded_for_reporting(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        details = _charge(worker)["details"]
        assert details["origin"] == "backfill"
        assert details["transcript_id"] == TRANSCRIPT
        assert details["is_external"] is False


class TestACorrectionIsNotBilled:
    """The redo exists because the platform heard the line wrong, over seconds already paid for.

    Charging it billed the same audio twice for the platform's own error, against WT-344
    (transcription is free) and the refund transcript_corrections.reversal_credit_transaction_id
    was designed for.
    """

    @staticmethod
    def _retranslation(**overrides) -> dict[str, str]:
        return _backfill(
            is_retranslated=True, previous_translation_content_id=str(uuid.uuid4()), **overrides
        )

    def test_a_retranslation_is_not_charged(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(self._retranslation()))
        worker.db.record_usage_and_charge.assert_not_awaited()

    def test_it_does_not_even_look_up_the_subscription(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(self._retranslation()))
        worker.db.resolve_subscription.assert_not_awaited()

    def test_the_absorbed_cost_is_still_logged(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(self._retranslation()))
        worker.logger.info.assert_called_once()
        event, fields = worker.logger.info.call_args.args[0], worker.logger.info.call_args.kwargs
        assert event == "retranslation_not_charged"
        assert fields["seconds"] == 3.5
        assert fields["workspace_id"] == WORKSPACE

    def test_a_gap_fill_of_the_same_line_is_still_charged(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_backfill_translation(_backfill()))
        worker.db.record_usage_and_charge.assert_awaited_once()


def test_the_live_translation_message_carries_no_backfill_fields() -> None:
    """translate:results must look exactly as it did: the new fields are absent unless set."""
    payload = TranslationResultMessage(
        segment_id=f"{SEGMENT}-en-c0",
        meeting_id=ROOM,
        speaker_id=REQUESTER,
        original_text="a",
        translated_text="b",
        source_lang="vi",
        target_lang="en",
    ).to_redis()
    assert not {"workspace_id", "requested_by_user_id", "transcript_id"} & payload.keys()


def test_backfill_fields_survive_a_round_trip() -> None:
    parsed = TranslationResultMessage.from_redis(_backfill())
    assert parsed.workspace_id == WORKSPACE
    assert parsed.requested_by_user_id == REQUESTER
    assert parsed.transcript_id == TRANSCRIPT
    assert (parsed.start_ms, parsed.end_ms) == (12_000, 15_500)
