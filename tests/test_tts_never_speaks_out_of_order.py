"""A dub is never spoken after a later line of the same speaker has been spoken.

The key lock in tts_worker plays a (meeting, speaker, language) in the order its lines ARRIVE.
Arrival is usually speech order, but not always: translation_worker holds a line for its
predecessor only up to a bound and then publishes it anyway, and a retried message comes back
late. Before this, such a line was spoken late and sped up (WT-528's catch-up) — the listener
heard B, then A. The owner's rule: fast but out of order is meaningless, a gap is acceptable, an
inversion is not.

What these pin, through the live entry point `_run_in_key_order`:
  * an overtaken line is not synthesized, is logged and counted, and still closes its turn;
  * lines of one turn and later turns are all spoken when they arrive in order;
  * a meeting-clock restart (lost transcript anchor) does not silence the speaker.
"""

from __future__ import annotations

from typing import Any

from shared.config import WorkerSettings
from shared.redis_client import RedisStreamClient
from shared.schemas import TranslationResultMessage
from tests.test_tts_worker import _make_worker

_EPOCH = 1_790_000_000_000


def _line(
    name: str,
    start_ms: int,
    *,
    published_ms: int | None = None,
    chunk: int = 0,
    final: bool = False,
    text: str | None = None,
) -> dict[str, str]:
    return TranslationResultMessage(
        segment_id=f"seg-{name}",
        meeting_id="m1",
        speaker_id="s1",
        original_text=name,
        translated_text=f"câu {name}" if text is None else text,
        source_lang="en",
        target_lang="vi",
        start_ms=start_ms,
        chunk_index=chunk,
        is_final_chunk=final,
        timestamp_ms=_EPOCH + start_ms if published_ms is None else published_ms,
    ).to_redis()


def _worker(client: RedisStreamClient, settings: WorkerSettings) -> Any:
    worker = _make_worker(client, settings)
    worker._key_locks = {}
    client._redis.get.return_value = None  # type: ignore[attr-defined]
    client._redis.hget.return_value = None  # type: ignore[attr-defined]
    return worker


def _spoken(worker: Any) -> list[str]:
    return [c.kwargs["text"] for c in worker.cartesia.synthesize.await_args_list]


def _events(client: RedisStreamClient, event_type: str) -> list[dict[str, Any]]:
    return [
        call.args[1]
        for call in client._redis.xadd.call_args_list  # type: ignore[attr-defined]
        if call.args
        and str(call.args[0]).endswith("translationRoom:system_events")
        and call.args[1].get("event_type") == event_type
    ]


async def test_a_line_that_arrives_after_a_later_one_was_spoken_is_skipped(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _worker(mock_redis_client, worker_settings)

    await worker._run_in_key_order(b"2-0", _line("B", 4000))
    await worker._run_in_key_order(b"1-0", _line("A", 1000, final=True))

    assert _spoken(worker) == ["câu B"]
    warnings = [c.args[0] for c in worker.logger.warning.call_args_list if c.args]
    assert "tts_dub_skipped_out_of_order" in warnings
    skipped = next(
        c.kwargs
        for c in worker.logger.warning.call_args_list
        if c.args and c.args[0] == "tts_dub_skipped_out_of_order"
    )
    assert skipped["segment_id"] == "seg-A"
    assert skipped["overtaken_by_position"] == "4000:0"
    assert skipped["behind_ms"] == 3000
    # Counted where the stage outcomes are, so the rate is on the dashboard, not only in logs.
    mock_redis_client._redis.pipeline.return_value.hincrby.assert_any_call(  # type: ignore[attr-defined]
        "warptalk:outcome:tts", "out_of_order", 1
    )
    # The skipped line still closes its turn for billing_worker and the transcript consumer.
    assert [e["payload"] for e in _events(mock_redis_client, "final_chunk_processed")] == [
        '{"segmentId": "seg-A"}'
    ]


async def test_lines_in_speech_order_are_all_spoken(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _worker(mock_redis_client, worker_settings)

    # Two sentences of one STT chunk (same start, same publish instant), then the next turn.
    await worker._run_in_key_order(b"1-0", _line("A1", 1000))
    await worker._run_in_key_order(b"1-1", _line("A2", 1000))
    await worker._run_in_key_order(b"2-0", _line("B", 4000))

    assert _spoken(worker) == ["câu A1", "câu A2", "câu B"]


async def test_the_meeting_s_first_line_is_judged_too(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    """start_ms 0 is the first chunk of a meeting, not "no position": it used to be exempt."""
    worker = _worker(mock_redis_client, worker_settings)

    await worker._run_in_key_order(b"2-0", _line("B", 2500))
    await worker._run_in_key_order(b"1-0", _line("A", 0))

    assert _spoken(worker) == ["câu B"]


async def test_a_restarted_meeting_clock_does_not_silence_the_speaker(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    """The transcript anchor lives in Redis; if it is lost mid-meeting, start_ms restarts near 0.
    Those lines are NEWER (their chunks were published later), so they are spoken — and so is
    everything after them, measured on the restarted clock."""
    worker = _worker(mock_redis_client, worker_settings)
    later = _EPOCH + 3_600_000

    await worker._run_in_key_order(b"1-0", _line("before", 1_800_000))
    await worker._run_in_key_order(b"2-0", _line("after1", 2_000, published_ms=later))
    await worker._run_in_key_order(b"3-0", _line("after2", 6_000, published_ms=later + 4_000))
    # And order still holds on the new clock.
    await worker._run_in_key_order(b"4-0", _line("late", 4_000, published_ms=later + 2_000))

    assert _spoken(worker) == ["câu before", "câu after1", "câu after2"]


async def test_another_speaker_or_language_is_never_judged_against_this_one(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    worker = _worker(mock_redis_client, worker_settings)
    other_speaker = TranslationResultMessage.from_redis(_line("X", 500))
    other_speaker.speaker_id = "s2"
    other_language = TranslationResultMessage.from_redis(_line("Y", 500))
    other_language.target_lang = "ja"

    await worker._run_in_key_order(b"1-0", _line("B", 4000))
    await worker._run_in_key_order(b"2-0", other_speaker.to_redis())
    await worker._run_in_key_order(b"3-0", other_language.to_redis())

    assert _spoken(worker) == ["câu B", "câu X", "câu Y"]


async def test_an_empty_final_marker_is_left_to_process(
    mock_redis_client: RedisStreamClient, worker_settings: WorkerSettings
) -> None:
    """Nothing to speak, so nothing to skip: process() owns its bookkeeping, unchanged."""
    worker = _worker(mock_redis_client, worker_settings)

    await worker._run_in_key_order(b"2-0", _line("B", 4000))
    await worker._run_in_key_order(b"1-0", _line("A", 1000, final=True, text=""))

    warnings = [c.args[0] for c in worker.logger.warning.call_args_list if c.args]
    assert "tts_dub_skipped_out_of_order" not in warnings
    assert len(_events(mock_redis_client, "final_chunk_processed")) == 1
