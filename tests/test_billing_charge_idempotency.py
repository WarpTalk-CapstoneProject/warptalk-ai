"""One charge per (segment, target language, service), whatever delivers the event twice.

Pinned here, because a charge is the one pipeline effect a user can count after the fact:

* A translated STT segment is charged once per target language, not once per sentence chunk.
  translation_worker publishes one message per sentence and every one of them carries the whole
  segment's duration, so a chunk-keyed charge billed a three-sentence segment three times.
* A dub is charged once per (chunk, language) whichever voice rendered it: the charge type follows
  the voice, the idempotency key must not.
* A second charge for a key the database already holds is a replay, never an error and never a
  refusal — raising dead-letters it, and a refusal stops the room's translation.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg

from billing_worker.db import BillingRepository, SettlementOutcome
from billing_worker.worker import BillingSettlementWorker
from shared.schemas import TranslationResultMessage, TTSResultMessage

SEGMENT = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
ROOM = "dddddddd-dddd-dddd-dddd-dddddddddddd"
SPEAKER = "44444444-4444-4444-4444-444444444444"
WORKSPACE = uuid.UUID("33333333-3333-3333-3333-333333333333")


def _worker() -> BillingSettlementWorker:
    worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
    worker.logger = MagicMock()
    worker.db = MagicMock()
    worker.db.record_usage_and_charge = AsyncMock(
        return_value=SettlementOutcome(
            applied=True,
            replayed=False,
            balance_after=100,
            service_state="active",
            suspended_reason=None,
            credits_consumed=1,
        )
    )
    worker._resolve_subscription = AsyncMock(return_value=(uuid.uuid4(), WORKSPACE))
    worker._is_unbillable = MagicMock(return_value=False)
    worker._is_external_speaker = AsyncMock(return_value=False)
    worker._note_settlement = AsyncMock()
    return worker


def _chunk(idx: int, target_lang: str = "en") -> dict[str, str]:
    return TranslationResultMessage(
        segment_id=f"{SEGMENT}-{target_lang}-c{idx}",
        meeting_id=ROOM,
        speaker_id=SPEAKER,
        original_text=f"Câu {idx}.",
        translated_text=f"Sentence {idx}.",
        source_lang="vi",
        target_lang=target_lang,
        translator_model="gpt-4.1-mini",
        source_segment_id=SEGMENT,
        # Every chunk of one STT segment carries the segment's whole span.
        start_ms=10_000,
        end_ms=19_000,
        is_final_chunk=idx == 2,
    ).to_redis()


def _keys(worker: BillingSettlementWorker) -> list[str]:
    return [
        call.kwargs["idempotency_key"] for call in worker.db.record_usage_and_charge.await_args_list
    ]


class TestTranslationIsChargedPerSegmentNotPerChunk:
    def test_every_chunk_of_one_segment_names_the_same_charge(self) -> None:
        worker = _worker()
        for idx in range(3):
            asyncio.run(worker._handle_translation(_chunk(idx)))

        keys = _keys(worker)
        assert len(keys) == 3
        assert set(keys) == {f"TRANSLATION:{SEGMENT}:en"}, (
            "three sentences of one 9s segment must settle as ONE 9s charge; a key per chunk "
            "billed the same seconds of speech three times"
        )

    def test_each_target_language_is_its_own_charge(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_translation(_chunk(0, "en")))
        asyncio.run(worker._handle_translation(_chunk(0, "ja")))

        assert _keys(worker) == [f"TRANSLATION:{SEGMENT}:en", f"TRANSLATION:{SEGMENT}:ja"]

    def test_a_redelivered_chunk_names_the_same_charge_again(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_translation(_chunk(1)))
        asyncio.run(worker._handle_translation(_chunk(1)))

        first, second = _keys(worker)
        assert first == second

    def test_the_quantity_is_the_segments_speech_not_a_share_of_it(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_translation(_chunk(0)))

        assert worker.db.record_usage_and_charge.await_args.kwargs["quantity"] == 9.0


def _dub(voice_type: str) -> dict[str, str]:
    return TTSResultMessage(
        segment_id=f"{SEGMENT}-en-c0",
        meeting_id=ROOM,
        speaker_id=SPEAKER,
        audio_data=b"\x00\x01" * 800,
        duration_ms=2_400,
        voice_type=voice_type,
        target_lang="en",
    ).to_redis()


class TestDubbingIsChargedOncePerChunkWhateverTheVoice:
    def test_a_retry_in_a_different_voice_names_the_same_charge(self) -> None:
        worker = _worker()
        asyncio.run(worker._handle_tts(_dub("default")))
        asyncio.run(worker._handle_tts(_dub("cloned")))

        calls = worker.db.record_usage_and_charge.await_args_list
        # Priced by the voice that rendered it...
        assert [c.kwargs["charge_type"] for c in calls] == [
            "AUDIO_DUBBING_STANDARD",
            "AUDIO_DUBBING_VOICE_CLONE",
        ]
        # ...but one sentence, one language: one charge.
        assert set(_keys(worker)) == {f"AUDIO_DUBBING:{SEGMENT}-en-c0:en"}


class _Conn:
    def __init__(self, settle_error: BaseException | None) -> None:
        self._settle_error = settle_error

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        if "usage_rate_card" in sql:
            return {"id": uuid.uuid4(), "unit_price": Decimal("0.5"), "currency": "CRD"}
        if self._settle_error is not None:
            raise self._settle_error
        raise AssertionError("unexpected query")


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *_: object) -> None:
        return None


def _repository(settle_error: BaseException | None) -> BillingRepository:
    repo = BillingRepository.__new__(BillingRepository)
    pool = MagicMock()
    pool.acquire = MagicMock(return_value=_Acquire(_Conn(settle_error)))
    repo._pool = pool
    return repo


def _settle(repo: BillingRepository) -> SettlementOutcome:
    return asyncio.run(
        repo.record_usage_and_charge(
            subscription_id=uuid.uuid4(),
            user_id=SPEAKER,
            workspace_id=WORKSPACE,
            translation_room_id=ROOM,
            usage_type="AUDIO_DUBBING_VOICE_CLONE",
            charge_type="AUDIO_DUBBING_VOICE_CLONE",
            reference_id=SEGMENT,
            reference_type="audio_dubbing",
            quantity=2.4,
            unit="second",
            idempotency_key=f"AUDIO_DUBBING:{SEGMENT}-en-c0:en",
        )
    )


class TestAnAlreadyChargedKeyIsAReplay:
    def test_the_unique_index_on_the_key_reads_as_a_replay_not_an_error(self) -> None:
        outcome = _settle(_repository(asyncpg.UniqueViolationError("duplicate key")))

        assert outcome.applied is False
        assert outcome.replayed is True, (
            "a replay read as a refusal suspends the room (WT-699); read as an error it is "
            "redelivered five times into the dead-letter stream"
        )

    def test_any_other_database_error_still_propagates(self) -> None:
        repo = _repository(asyncpg.PostgresConnectionError("gone"))
        try:
            _settle(repo)
        except asyncpg.PostgresConnectionError:
            return
        raise AssertionError("a real failure must leave the event pending for a retry")
