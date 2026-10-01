"""A sentence Cartesia failed must be retried if the failure was momentary, and parked if not.

Before: `cartesia_synthesis_failed` was logged and process() returned normally, so the message
was acknowledged and the sentence was gone — no retry, and no record anywhere to replay it from.

The policy these tests pin:
  * a network drop, a 5xx or a timeout is retried, a bounded number of times, within a window;
  * a 4xx (402 quota, 401/403 key, 429 concurrency, 400) is NEVER retried — it answers the same
    way next time, and retrying a refusal only multiplies the load being refused;
  * whatever is not retried, or runs out of retries, goes to `translate:results:dead-letter`
    with the reason;
  * a retry happens inside the key's lock, so it is late but never out of order, and a
    RECLAIMED message that a newer line has already overtaken is parked, not spoken.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.redis_client import RedisStreamClient
from shared.schemas import TranslationResultMessage
from tests.test_tts_worker import _make_msg, _make_worker


class _StatusError(Exception):
    """Shaped like the Cartesia SDK's APIStatusError: the status is an attribute."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Error code: {status}")
        self.status_code = status


def _settings(**overrides: Any) -> TTSSettings:
    values: dict[str, Any] = {
        "prosody_continuity": False,
        "synthesis_retry_backoff_seconds": 0.0,
    }
    values.update(overrides)
    return TTSSettings(**values)


def _xadds(client: RedisStreamClient, suffix: str) -> list[dict[str, Any]]:
    return [
        call.args[1]
        for call in client._redis.xadd.call_args_list  # type: ignore[attr-defined]
        if call.args and str(call.args[0]).endswith(suffix)
    ]


def _dead_letters(client: RedisStreamClient) -> list[dict[str, Any]]:
    return _xadds(client, "translate:results:dead-letter")


def _events(client: RedisStreamClient, event_type: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in _xadds(client, "translationRoom:system_events")
        if entry.get("event_type") == event_type
    ]


def _results(client: RedisStreamClient) -> list[dict[str, Any]]:
    return _xadds(client, "tts:results")


# ---------------------------------------------------------------------------
# Retried, then spoken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transient",
    [
        _StatusError(503),
        httpx.ConnectError("connection reset by peer"),
        httpx.ReadTimeout("read timed out"),
    ],
    ids=["5xx", "network", "timeout"],
)
async def test_a_transient_failure_is_retried_and_the_sentence_is_spoken(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings, transient: Exception
) -> None:
    worker = _make_worker(mock_redis_client, worker_settings, tts_settings=_settings())
    worker.cartesia.synthesize = AsyncMock(
        side_effect=[transient, (b"\x00" * 44 + b"\x01\x02" * 50, 1000, "voice")]
    )

    await worker.process(b"1-0", _make_msg().to_redis())

    assert worker.cartesia.synthesize.await_count == 2
    assert len(_results(mock_redis_client)) == 1, "spoken and billed on the retry"
    assert _dead_letters(mock_redis_client) == []
    assert _events(mock_redis_client, "tts_unavailable") == []


# ---------------------------------------------------------------------------
# Refused: never retried, parked with the reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "outcome"),
    [(402, "quota"), (429, "rate_limited"), (401, "auth"), (400, "client_error")],
)
async def test_a_4xx_is_not_retried_and_goes_to_the_dead_letter(
    mock_redis_client: RedisStreamClient,
    worker_settings: WorkerSettings,
    status: int,
    outcome: str,
) -> None:
    worker = _make_worker(mock_redis_client, worker_settings, tts_settings=_settings())
    worker.cartesia.synthesize = AsyncMock(side_effect=_StatusError(status))

    await worker.process(b"7-0", _make_msg().to_redis())

    assert worker.cartesia.synthesize.await_count == 1
    [parked] = _dead_letters(mock_redis_client)
    assert parked["reason"] == f"not_retryable:{outcome}"
    assert parked["original_message_id"] == "7-0"
    assert parked["delivery_attempts"] == 1
    assert _results(mock_redis_client) == []
    assert len(_events(mock_redis_client, "tts_unavailable")) == 1


async def test_retries_run_out_and_the_sentence_is_parked_replayable(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _make_worker(
        mock_redis_client, worker_settings, tts_settings=_settings(synthesis_max_retries=2)
    )
    worker.cartesia.synthesize = AsyncMock(side_effect=_StatusError(503))
    message = _make_msg(is_final=True)

    # Returns normally: the message is acknowledged, because what is left of it is now recorded
    # on the dead-letter stream rather than pending forever.
    await worker.process(b"9-0", message.to_redis())

    assert worker.cartesia.synthesize.await_count == 3  # the try and both retries
    [parked] = _dead_letters(mock_redis_client)
    assert parked["reason"] == "retries_exhausted:server_error"
    assert parked["delivery_attempts"] == 3
    assert parked["consumer_group"] == "tts-workers"
    assert parked["segment_id"] == message.segment_id
    # The original payload, as-is, so the line can be replayed onto the stream it came from.
    replay = TranslationResultMessage.from_redis(json.loads(parked["payload"]))
    assert replay.translated_text == message.translated_text
    # The segment's bookkeeping still completes: billing and the transcript wait on it.
    assert len(_events(mock_redis_client, "final_chunk_processed")) == 1


async def test_a_failure_that_already_took_longer_than_the_window_is_not_retried(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _make_worker(
        mock_redis_client,
        worker_settings,
        tts_settings=_settings(synthesis_retry_window_seconds=0.0),
    )
    worker.cartesia.synthesize = AsyncMock(side_effect=httpx.ReadTimeout("slow"))

    await worker.process(b"1-0", _make_msg().to_redis())

    assert worker.cartesia.synthesize.await_count == 1
    [parked] = _dead_letters(mock_redis_client)
    assert parked["reason"] == "retry_window_exceeded:timeout"


async def test_a_half_spoken_sentence_is_not_retried_from_its_first_word(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    """The context died after part of the line reached the listener, and the fallback failed
    too. A retry would speak the opening a second time — parked instead."""
    from tests.test_tts_prosody_continuity import _FakePublisher, _FakeTurn

    worker = _make_worker(
        mock_redis_client,
        worker_settings,
        tts_settings=_settings(prosody_continuity=True, stream_to_livekit=True),
    )
    worker._turns = {}
    worker._turn_connections = {}
    publisher = _FakePublisher()
    publisher.retire_voice_variants = MagicMock()  # type: ignore[attr-defined]
    worker.livekit_publisher = publisher  # type: ignore[assignment]

    async def open_prosody_context(**kwargs: Any) -> Any:
        turn = _FakeTurn()
        turn.fail_after_streaming = True
        connection = MagicMock()
        connection.close = AsyncMock()
        return turn, connection

    worker.cartesia.open_prosody_context = open_prosody_context
    worker.cartesia.synthesize = AsyncMock(side_effect=_StatusError(503))

    await worker.process(b"1-0", _make_msg().to_redis())

    assert worker.cartesia.synthesize.await_count == 1
    [parked] = _dead_letters(mock_redis_client)
    assert parked["reason"] == "partially_spoken:server_error"


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


async def test_a_retry_holds_the_next_line_back_rather_than_letting_it_overtake(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _make_worker(
        mock_redis_client,
        worker_settings,
        tts_settings=_settings(synthesis_retry_backoff_seconds=0.05),
    )
    worker._key_locks = {}
    spoken: list[str] = []
    failed_once: set[str] = set()

    async def synthesize(*, text: str, **kwargs: Any) -> tuple[bytes, int, str]:
        if text == "first" and text not in failed_once:
            failed_once.add(text)
            raise httpx.ConnectError("reset")
        spoken.append(text)
        return b"\x00" * 44 + b"\x01\x02" * 50, 1000, "voice"

    worker.cartesia.synthesize = synthesize

    first = _make_msg(text="first")
    second = _make_msg(text="second")
    await asyncio.gather(
        worker._run_in_key_order(b"1-0", first.to_redis()),
        worker._run_in_key_order(b"2-0", second.to_redis()),
    )

    assert spoken == ["first", "second"]


def _at(position_ms: int, chunk: int = 0) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id=f"seg-{position_ms}-{chunk}",
        meeting_id="m1",
        speaker_id="s1",
        original_text="src",
        translated_text="line",
        source_lang="en",
        target_lang="vi",
        start_ms=position_ms,
        chunk_index=chunk,
    )


def _reclaiming_worker(client: RedisStreamClient, reclaimed: TranslationResultMessage) -> Any:
    from tts_worker.worker import TTSWorker

    worker = TTSWorker.__new__(TTSWorker)
    worker.logger = MagicMock()
    worker.redis = client
    worker._consumer_name = "tts-test"
    worker._key_locks = {}
    client._redis.xautoclaim.return_value = [  # type: ignore[attr-defined]
        b"0-0",
        [(b"5-0", {k.encode(): v.encode() for k, v in reclaimed.to_redis().items()})],
        [],
    ]
    client._redis.xpending_range.return_value = [  # type: ignore[attr-defined]
        {"message_id": b"5-0", "times_delivered": 2}
    ]
    worker._process_and_log_errors = AsyncMock()  # type: ignore[method-assign]
    return worker


async def test_a_reclaimed_line_that_newer_lines_overtook_is_parked_not_spoken(
    mock_redis_client: RedisStreamClient,
) -> None:
    worker = _reclaiming_worker(mock_redis_client, _at(1000, chunk=0))
    worker._note_dub_started(_at(1000, chunk=1))  # the next sentence of the same turn has played

    await worker._recover_stale_messages()

    worker._process_and_log_errors.assert_not_awaited()
    [parked] = _dead_letters(mock_redis_client)
    assert parked["reason"] == "superseded"
    assert parked["superseded_by_position"] == "1000:1"
    acked = [call.args[2] for call in mock_redis_client._redis.xack.call_args_list]  # type: ignore[attr-defined]
    assert acked == [b"5-0"]


async def test_a_reclaimed_line_nothing_has_overtaken_is_still_spoken(
    mock_redis_client: RedisStreamClient,
) -> None:
    worker = _reclaiming_worker(mock_redis_client, _at(2000))
    worker._note_dub_started(_at(1000, chunk=3))  # only EARLIER lines have played

    await worker._recover_stale_messages()

    worker._process_and_log_errors.assert_awaited_once()
    assert _dead_letters(mock_redis_client) == []
