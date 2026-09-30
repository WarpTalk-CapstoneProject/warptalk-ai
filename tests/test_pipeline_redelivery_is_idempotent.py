"""A message delivered twice must not reach the transcript, the dub or the bill twice.

Redis Streams deliver at least once. A second delivery happens when an attempt dies after its
output went out but before XACK, when an attempt raises after publishing part of its output
(the message stays pending and is retried from the top), and — with two replicas in one group,
which every surge rollout creates for a minute — when one replica's idle poll XAUTOCLAIMs an
entry the other is still working on.

Each stage below makes its own output idempotent in the unit the next stage keys on:

* STT: at most one transcription per `audio:chunks` entry. A segment id is derived from the
  recognised TEXT, and recognition is not deterministic, so a second transcription is a second
  set of segments nothing downstream can recognise — shown twice, dubbed twice, billed twice.
* Translation: one publish per chunk id (`{stt segment}-{lang}-c{idx}`), which is deterministic.
* Reclaim: an entry is only taken from another consumer once it cannot still be in progress.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.base_worker import BaseWorker
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.model import TranscribedSegment
from tests.test_stt_worker import TestProsodyMeasurement as _SttHarness
from tests.test_stt_worker import _pcm_tone
from tests.test_translation_worker import TestTranslationWorker as _TranslationHarness


def _in_memory_keys(client: Any) -> dict[str, Any]:
    """Give the mocked Redis real GET/SET/SETEX/DEL semantics, so a marker actually persists."""
    store: dict[str, Any] = {}

    async def _set(key: str, value: Any, nx: bool = False, ex: int | None = None, **_: Any) -> Any:
        if nx and key in store:
            return None
        store[key] = value
        return True

    async def _setex(key: str, _ttl: int, value: Any) -> bool:
        store[key] = value
        return True

    async def _get(key: str) -> Any:
        return store.get(key)

    async def _delete(*keys: str) -> int:
        return sum(1 for key in keys if store.pop(key, None) is not None)

    client._redis.set = AsyncMock(side_effect=_set)
    client._redis.setex = AsyncMock(side_effect=_setex)
    client._redis.get = AsyncMock(side_effect=_get)
    client._redis.delete = AsyncMock(side_effect=_delete)
    return store


def _published(client: Any, stream: str) -> list[Any]:
    return [call for call in client._redis.xadd.call_args_list if call.args[0] == stream]


class TestSttTranscribesAChunkAtMostOnce:
    def _worker(self, mock_redis_client: Any, worker_settings: Any) -> Any:
        worker = _SttHarness()._make_worker(mock_redis_client, worker_settings)
        _in_memory_keys(mock_redis_client)
        return worker

    @staticmethod
    def _chunk() -> dict[str, Any]:
        return AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=3,
            audio_data=_pcm_tone(140.0),
            language="en",
        ).to_redis()

    async def test_a_redelivered_chunk_is_not_transcribed_or_published_again(
        self, mock_redis_client: Any, worker_settings: Any
    ) -> None:
        worker = self._worker(mock_redis_client, worker_settings)

        await worker.process(b"1790610385615-0", self._chunk())
        # The retry transcribes differently — which is exactly why it must not run: a new text
        # is a new segment id, and nothing downstream could tell it was the same speech.
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello.", language="en", confidence=-0.2, start_ms=0, end_ms=990
                )
            ]
        )
        await worker.process(b"1790610385615-0", self._chunk())

        assert len(_published(mock_redis_client, "stt:results")) == 1
        worker.model.transcribe.assert_not_awaited()

    async def test_a_different_chunk_is_transcribed_as_usual(
        self, mock_redis_client: Any, worker_settings: Any
    ) -> None:
        worker = self._worker(mock_redis_client, worker_settings)

        await worker.process(b"1790610385615-0", self._chunk())
        await worker.process(b"1790610402675-0", self._chunk())

        assert len(_published(mock_redis_client, "stt:results")) == 2

    async def test_an_attempt_that_published_nothing_is_retried_in_full(
        self, mock_redis_client: Any, worker_settings: Any
    ) -> None:
        worker = self._worker(mock_redis_client, worker_settings)
        worker.model.transcribe = AsyncMock(side_effect=RuntimeError("socket closed"))

        await worker.process(b"1790610385615-0", self._chunk())
        assert _published(mock_redis_client, "stt:results") == []

        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello", language="en", confidence=-0.25, start_ms=0, end_ms=1000
                )
            ]
        )
        await worker.process(b"1790610385615-0", self._chunk())

        assert len(_published(mock_redis_client, "stt:results")) == 1


class TestTranslationPublishesEachChunkOnce:
    async def test_a_retry_after_a_partial_publish_does_not_repeat_the_first_sentence(
        self, mock_redis_client: Any, worker_settings: Any
    ) -> None:
        worker = _TranslationHarness()._make_worker(mock_redis_client, worker_settings)
        _in_memory_keys(mock_redis_client)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        # Sentence 0 translates alone; sentences 1..N in one batch, which fails the first time.
        worker.translator.translate_batch = AsyncMock(side_effect=RuntimeError("provider 503"))
        message = STTResultMessage(
            segment_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            meeting_id="m1",
            speaker_id="s1",
            text="Hello there. How are you today?",
            language="en",
            confidence=0.95,
            is_final_chunk=True,
        ).to_redis()

        try:
            await worker.process(b"5-0", message)
        except RuntimeError:
            pass  # left pending, retried from the top
        worker.translator.translate_batch = AsyncMock(return_value=["Hôm nay bạn thế nào?"])
        await worker.process(b"5-0", message)

        segment_ids = [
            call.args[1][b"segment_id"] if b"segment_id" in call.args[1] else None
            for call in _published(mock_redis_client, "translate:results")
        ]
        assert len(segment_ids) == 2, (
            "sentence 0 went out on the first attempt; the retry must publish only sentence 1, "
            f"or the listener hears sentence 0 dubbed twice (got {segment_ids})"
        )

    async def test_a_failed_publish_gives_the_claim_back(
        self, mock_redis_client: Any, worker_settings: Any
    ) -> None:
        worker = _TranslationHarness()._make_worker(mock_redis_client, worker_settings)
        store = _in_memory_keys(mock_redis_client)
        worker.publish = AsyncMock(side_effect=ConnectionError("redis down"))
        result = MagicMock(segment_id="seg-vi-c0")
        result.to_redis = MagicMock(return_value={})

        try:
            await worker._publish_once("m1", result)
        except ConnectionError:
            pass

        assert store == {}, "a chunk that never went out must be publishable on the retry"


class _SlowWorker(BaseWorker):
    worker_name = "slow"
    input_stream = "input"
    consumer_group = "slow-workers"

    async def load_model(self) -> None:
        return None

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        return None


class TestReclaimNeverTakesWorkStillInProgress:
    def test_the_threshold_is_above_the_processing_timeout(self) -> None:
        worker = _SlowWorker.__new__(_SlowWorker)
        assert worker._reclaim_min_idle_ms() > worker.processing_timeout_seconds * 1000

    def test_it_follows_a_worker_that_raises_its_timeout(self) -> None:
        worker = _SlowWorker.__new__(_SlowWorker)
        worker.processing_timeout_seconds = 600
        assert worker._reclaim_min_idle_ms() >= 630_000

    async def test_the_reclaim_asks_redis_for_that_threshold(self) -> None:
        worker = _SlowWorker.__new__(_SlowWorker)
        worker.logger = MagicMock()
        worker._consumer_name = "slow-host-2"
        worker.redis = MagicMock()
        worker.redis.reclaim_stale = AsyncMock(return_value=[])

        await worker._recover_stale_messages()

        assert worker.redis.reclaim_stale.await_args.kwargs["min_idle_ms"] > 120_000, (
            "at 60s against a 120s processing timeout, a 61s summary on one replica was taken "
            "and run again by the other"
        )
