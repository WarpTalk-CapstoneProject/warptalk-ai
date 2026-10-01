"""Sentence A, then sentence B: the listener hears A first, then B. Always.

stt:results is in speech order. The translation worker translates several messages at once, and
it used to publish each one the moment its own translation landed. So a long A spoken before a
short B lost the race, B reached tts:results first, and tts_worker (strictly FIFO per speaker and
language) spoke B before A. The owner heard this as "if A has not been dubbed when B is, A is lost".

These tests drive the worker's real consume loop and the real consume_concurrent dispatch. Only
the model call and Redis are faked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import TranslationSettings, WorkerSettings
from shared.schemas import STTResultMessage, TranslationResultMessage
from translation_worker import worker as worker_module
from translation_worker.worker import TranslationWorker

SLOW_A = "Sentence A is long and was spoken first."
FAST_B = "B."


def _stt(text: str, start_ms: int, speaker: str = "s1") -> dict[bytes, bytes]:
    msg = STTResultMessage(
        segment_id=f"seg-{start_ms}",
        meeting_id="m1",
        speaker_id=speaker,
        text=text,
        language="en",
        confidence=0.95,
        start_ms=start_ms,
        end_ms=start_ms + 1000,
    )
    return {k.encode(): str(v).encode() for k, v in msg.to_redis().items()}


def _worker(mock_redis_client, worker_settings: WorkerSettings, delays: dict[str, float]):
    worker = TranslationWorker.__new__(TranslationWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.translation_settings = TranslationSettings()
    worker._paused_rooms = set()
    worker._route_states = {}
    worker._is_translation_active = lambda _room: True  # type: ignore[method-assign]
    worker._mt_glossaries = {}
    worker._recent_source_contexts = {}
    worker.worker_name = "translation"
    worker._consumer_name = "translation-test"
    worker._shutdown_event = asyncio.Event()
    worker._recover_stale_messages = AsyncMock()  # type: ignore[method-assign]

    async def translate(text: str, **_kwargs: object) -> tuple[str, None]:
        await asyncio.sleep(delays.get(text, 0.0))
        return f"vi:{text}", None

    translator = MagicMock()
    translator.model = "test-model"
    translator.translate_with_valence = translate
    translator.translate_batch = AsyncMock(return_value=[])
    worker.translator = translator

    published: list[TranslationResultMessage] = []

    async def publish(stream: str, _meeting_id: str, data: dict[str, str]) -> None:
        if stream == "translate:results":
            published.append(TranslationResultMessage.from_redis(data))

    worker.publish = publish  # type: ignore[method-assign]
    mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
    return worker, published


async def _run_batch(worker: TranslationWorker, messages: list[dict[bytes, bytes]]) -> None:
    """One xreadgroup batch through the real consume loop, then stop."""
    batches = [[(b"stt:results", [(f"{i}-0".encode(), m) for i, m in enumerate(messages)])]]

    async def xreadgroup(**_kwargs: object) -> list[object]:
        if batches:
            return batches.pop()
        worker._shutdown_event.set()
        return []

    worker.redis._redis.xreadgroup = xreadgroup
    await asyncio.wait_for(worker._consume_loop(), timeout=20)


async def test_a_slow_earlier_sentence_is_published_before_a_fast_later_one(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    worker, published = _worker(mock_redis_client, worker_settings, {SLOW_A: 0.3, FAST_B: 0.0})

    await _run_batch(worker, [_stt(SLOW_A, 1000), _stt(FAST_B, 4000)])

    assert [r.original_text for r in published] == [SLOW_A, FAST_B]


async def test_different_speakers_do_not_wait_for_each_other(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    worker, published = _worker(mock_redis_client, worker_settings, {SLOW_A: 0.3, FAST_B: 0.0})

    await _run_batch(worker, [_stt(SLOW_A, 1000, "s1"), _stt(FAST_B, 4000, "s2")])

    assert [r.original_text for r in published] == [FAST_B, SLOW_A]


async def test_a_stalled_earlier_sentence_holds_the_next_one_only_up_to_the_bound(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    """The bound is announced in the log. B goes out, and A still goes out when it lands.
    Neither sentence is dropped."""
    worker, published = _worker(mock_redis_client, worker_settings, {SLOW_A: 0.5, FAST_B: 0.0})

    with patch.object(worker_module, "_PUBLISH_ORDER_WAIT_SECONDS", 0.05):
        await _run_batch(worker, [_stt(SLOW_A, 1000), _stt(FAST_B, 4000)])

    assert [r.original_text for r in published] == [FAST_B, SLOW_A]
    events = [c.args[0] for c in worker.logger.warning.call_args_list if c.args]
    assert "translation_publish_order_wait_timeout" in events
    # A still reaches translate:results (the transcript needs it), and says it is late, because
    # tts_worker will not dub it — see test_tts_never_speaks_out_of_order.py.
    assert events.count("translation_published_after_successor") == 1


async def test_an_earlier_sentence_that_fails_does_not_block_the_next(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    worker, published = _worker(mock_redis_client, worker_settings, {})

    async def translate(text: str, **_kwargs: object) -> tuple[str, None]:
        if text == SLOW_A:
            await asyncio.sleep(0.1)
            raise RuntimeError("model down")
        return f"vi:{text}", None

    worker.translator.translate_with_valence = translate  # type: ignore[union-attr]

    await _run_batch(worker, [_stt(SLOW_A, 1000), _stt(FAST_B, 4000)])

    assert [r.original_text for r in published] == [FAST_B]


@pytest.mark.parametrize(("arrival", "expected"), [(["A", "B"], ["A", "B"]), (["B", "A"], ["B"])])
async def test_tts_never_speaks_a_sentence_after_a_later_one(
    mock_redis_client, worker_settings: WorkerSettings, arrival: list[str], expected: list[str]
) -> None:
    """The TTS half of the invariant. In order, both are spoken. If B was spoken before A
    arrived, A is NOT spoken: a gap, never an inversion. (It used to be spoken late and sped
    up, which is exactly the "B then A" the owner reported.)"""
    from tests.test_tts_worker import _make_worker

    tts = _make_worker(mock_redis_client, worker_settings)
    tts._key_locks = {}
    mock_redis_client._redis.get.return_value = None
    start = {"A": 1000, "B": 4000}
    for name in arrival:
        msg = TranslationResultMessage(
            segment_id=f"seg-{name}",
            meeting_id="m1",
            speaker_id="s1",
            original_text=name,
            translated_text=f"câu {name}",
            source_lang="en",
            target_lang="vi",
            start_ms=start[name],
            # The chunk each sentence was published in, in speech order, whatever the arrival.
            timestamp_ms=1_790_000_000_000 + start[name],
        )
        await tts._run_in_key_order(f"{name}-0".encode(), msg.to_redis())

    spoken = [c.kwargs["text"] for c in tts.cartesia.synthesize.await_args_list]
    assert spoken == [f"câu {name}" for name in expected]
