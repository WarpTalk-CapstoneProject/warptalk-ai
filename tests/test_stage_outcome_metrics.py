"""Stage success rates: whether the pipeline WORKED, not only how fast it was.

WHY THIS EXISTS
    `warptalk_stage_latency_ms` could say how long STT, translation and TTS took, never whether
    they succeeded. A stage failing every call fast has an excellent p95. The STT and TTS workers
    also swallow vendor failures on purpose — so one refused chunk does not end a meeting — which
    meant a total Cartesia outage (402 quota, seen in production) returned normally from every
    `process()` call and looked, to any counter at the consume loop, like 100% success.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from metrics_exporter.metrics import collect_metrics
from shared.base_worker import BaseWorker
from shared.config import RedisSettings
from shared.redis_client import OUTCOME_KEY_PREFIX, RedisStreamClient

pytestmark = pytest.mark.asyncio


# ── Recording ────────────────────────────────────────────────────────────────────────────────


def _client() -> RedisStreamClient:
    client = RedisStreamClient(RedisSettings())
    redis = MagicMock()
    pipeline = MagicMock()
    pipeline.execute = AsyncMock(return_value=[1, True])
    redis.pipeline = MagicMock(return_value=pipeline)
    client._redis = redis
    client._pipeline = pipeline  # type: ignore[attr-defined]
    return client


async def test_an_outcome_is_one_increment_on_the_stage_hash_with_a_ttl() -> None:
    client = _client()

    await client.record_outcome("tts", "vendor_error")

    client._pipeline.hincrby.assert_called_once_with(f"{OUTCOME_KEY_PREFIX}tts", "vendor_error", 1)
    client._pipeline.expire.assert_called_once()


async def test_an_unknown_outcome_is_counted_as_an_error_not_as_a_new_label() -> None:
    client = _client()

    await client.record_outcome("stt", "exploded")

    client._pipeline.hincrby.assert_called_once_with(f"{OUTCOME_KEY_PREFIX}stt", "error", 1)


async def test_a_redis_failure_never_reaches_the_pipeline() -> None:
    client = _client()
    client._pipeline.execute = AsyncMock(side_effect=ConnectionError("redis gone"))

    await client.record_outcome("stt", "ok")  # must not raise


# ── BaseWorker ───────────────────────────────────────────────────────────────────────────────


class _Worker(BaseWorker):
    worker_name = "stt"
    input_stream = "audio:chunks"
    consumer_group = "stt-workers"
    processing_timeout_seconds = 0.2

    def __init__(self, behaviour: str) -> None:
        super().__init__()
        self.behaviour = behaviour
        self.redis = MagicMock()
        self.redis.record_outcome = AsyncMock()

    async def load_model(self) -> None:
        return None

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        if self.behaviour == "raise":
            raise RuntimeError("model refused")
        if self.behaviour == "hang":
            await asyncio.sleep(5)
        if self.behaviour == "swallow":
            # What stt_worker does when the vendor refuses a chunk: note it, then carry on.
            await asyncio.sleep(0)
            self.note_attempt_outcome("vendor_error")


def _recorded(worker: _Worker) -> list[Any]:
    return [call.args for call in worker.redis.record_outcome.await_args_list]


async def test_a_message_that_is_processed_counts_as_ok() -> None:
    worker = _Worker("ok")

    await worker._process_and_log_errors(b"1-0", {})

    assert _recorded(worker) == [("stt", "ok")]


async def test_a_raised_error_counts_and_still_propagates_so_the_message_stays_pending() -> None:
    worker = _Worker("raise")

    with pytest.raises(RuntimeError):
        await worker._process_and_log_errors(b"1-0", {})

    assert _recorded(worker) == [("stt", "error")]


async def test_a_timeout_is_its_own_outcome() -> None:
    worker = _Worker("hang")

    with pytest.raises(TimeoutError):
        await worker._process_and_log_errors(b"1-0", {})

    assert _recorded(worker) == [("stt", "timeout")]


async def test_a_swallowed_vendor_failure_is_not_counted_as_a_success() -> None:
    worker = _Worker("swallow")

    await worker._process_and_log_errors(b"1-0", {})

    assert _recorded(worker) == [("stt", "vendor_error")]


async def test_a_note_outside_an_attempt_is_a_no_op() -> None:
    worker = _Worker("ok")

    worker.note_attempt_outcome("vendor_error")  # must not raise, must not leak into the next

    await worker._process_and_log_errors(b"1-0", {})
    assert _recorded(worker) == [("stt", "ok")]


async def test_the_stt_and_tts_vendor_failure_paths_are_marked() -> None:
    """The two swallowed failures that hid the Cartesia and STT vendor outages."""
    import inspect

    from stt_worker.worker import STTWorker
    from tts_worker.worker import TTSWorker

    stt = inspect.getsource(STTWorker)
    tts = inspect.getsource(TTSWorker)
    assert 'self.note_attempt_outcome("vendor_error")' in stt
    assert stt.index('self.note_attempt_outcome("vendor_error")') < stt.index(
        'event_type="stt_unavailable"'
    )
    assert 'self.note_attempt_outcome("vendor_error")' in tts
    assert tts.index('self.note_attempt_outcome("vendor_error")') < tts.index(
        '"cartesia_synthesis_failed"'
    )


# ── Exporter ─────────────────────────────────────────────────────────────────────────────────


class _FakeRedis:
    def __init__(self, outcomes: dict[str, dict[str, str]]) -> None:
        self._outcomes = outcomes

    async def xinfo_groups(self, stream: str) -> list[dict[str, Any]]:
        return []

    async def xlen(self, stream: object) -> int:
        return 0

    async def hgetall(self, key: object) -> dict[str, str]:
        return self._outcomes.get(str(key), {})

    async def scan_iter(  # type: ignore[no-untyped-def]
        self,
        match: str,
        count: int = 100,
        _type: str | None = None,
    ):
        if match.startswith(OUTCOME_KEY_PREFIX):
            for key in self._outcomes:
                yield key


async def test_every_known_outcome_is_exported_for_a_stage_zeros_included() -> None:
    body = await collect_metrics(
        _FakeRedis({f"{OUTCOME_KEY_PREFIX}tts": {"ok": "90", "vendor_error": "10"}})
    )

    assert "# TYPE warptalk_stage_messages_total counter" in body
    assert 'warptalk_stage_messages_total{stage="tts",outcome="ok"} 90' in body
    assert 'warptalk_stage_messages_total{stage="tts",outcome="vendor_error"} 10' in body
    assert 'warptalk_stage_messages_total{stage="tts",outcome="timeout"} 0' in body
    assert 'warptalk_stage_messages_total{stage="tts",outcome="dead_letter"} 0' in body


async def test_no_outcomes_yet_is_still_valid_output() -> None:
    body = await collect_metrics(_FakeRedis({}))

    assert "warptalk_stage_messages_total" in body
    assert body.endswith("\n")
