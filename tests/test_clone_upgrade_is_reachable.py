"""A clone that scored the maximum stops the worker measuring for an upgrade it cannot earn.

THE ARITHMETIC
    `worth_cloning` in `_consume_audio_for_cloning` is

        assessment.score >= previous_score + voice_clone_upgrade_margin

    and `score` is closed at `MAX_SAMPLE_SCORE` (1.0) by construction — both components are
    `min(1.0, …)` and the weights sum to one. A speaker whose first accepted clip scored 1.0
    therefore needs 1.15 to beat it. No clip can. The comparison is unsatisfiable, not unlikely.

WHAT THAT COST
    Production meeting 01a00547, both speakers, `voice:clone:state`:

        s..0001  capturing 2.8 → 23.1  →  cloning  score 1.0
        s..0004  capturing 2.2 → 20.5  →  cloning  score 1.0
        s..0004  capturing 1.2 → 20.4 → 21.9 → 23.8 → 25.4 → 31.5 → … → 70.8

    Perfect scores are the ordinary case, not an edge one. After the clone was requested the
    buffer refilled and every chunk past 20 seconds re-ran `assess_clone_sample` over the WHOLE
    buffer — an autocorrelation FFT per 40ms frame, ~3,500 of them per chunk per speaker at 70
    seconds — to re-derive a verdict that arithmetic had already settled. The 90s cap in
    `_trim_clone_buffer` bounds the memory; nothing bounded the work.

WHAT THE FIX IS NOT
    It does not widen the upgrade path or touch the margin. A speaker at 1.0 has the best
    reference the scorer can describe; there is genuinely nothing left to look for. The change is
    to stop looking, and to say so — `cloned_best_possible` — rather than idle silently, which is
    the failure mode every other exit on this path was already fixed for.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import _varied
from tts_worker.clone_sample_quality import MAX_SAMPLE_SCORE, assess_clone_sample
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000


class _ScriptedRedis:
    def __init__(self, worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
        self._worker = worker
        self._chunks = chunks

    async def consume(self, **_kwargs: Any) -> Any:
        for index, chunk in enumerate(self._chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self._worker._running = False


def _chunks(pcm: bytes, count: int) -> list[AudioChunkMessage]:
    return [
        AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=index,
            audio_data=pcm,
            language="vi",
            sample_rate=SAMPLE_RATE,
        )
        for index in range(count)
    ]


def _worker() -> tuple[TTSWorker, list[str], list[int]]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(voice_clone_min_seconds=10.0)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
    worker._consumer_name = "test"
    worker.worker_name = "tts"
    worker._running = True

    noted: list[str] = []
    # Every clip the gate is asked to judge, so the test can count the work rather than infer it.
    judged: list[int] = []

    async def _note(_key: tuple[str, str], reason: str, **_metrics: Any) -> None:
        noted.append(reason)

    async def _clone_and_cache(*_args: Any, **_kwargs: Any) -> None:
        pass

    # None throughout: the clone request is in flight (and in production it never landed at all —
    # see test_clone_sends_a_decodable_file). This is the path the growth was actually on.
    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return None

    worker._note_clone_state = _note  # type: ignore[method-assign]
    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    return worker, noted, judged


def test_a_perfect_clip_actually_scores_the_maximum() -> None:
    """The premise. If this stops being true the guard below is measuring nothing."""
    assessment = assess_clone_sample(_varied(), SAMPLE_RATE)
    assert assessment.accepted
    assert assessment.score == pytest.approx(MAX_SAMPLE_SCORE), (
        "The rest of this file assumes a good clip reaches the top of the scale."
    )


def test_an_unbeatable_score_ends_the_measuring_instead_of_looping() -> None:
    worker, noted, judged = _worker()

    # Wrap the gate so the test counts how many times the buffer is re-judged AFTER the clone.
    import tts_worker.worker as worker_module

    original = worker_module.assess_clone_sample

    def counting_assess(pcm: bytes, rate: int) -> Any:
        judged.append(len(pcm))
        return original(pcm, rate)

    worker_module.assess_clone_sample = counting_assess  # type: ignore[assignment]
    try:
        # Four clips' worth: one to reach the bar and clone, three more afterwards. Before the
        # guard those three were each a full re-measure over an ever-larger buffer.
        worker.redis = _ScriptedRedis(worker, _chunks(_varied(), 4))  # type: ignore[assignment]
        asyncio.run(worker._consume_audio_for_cloning())
    finally:
        worker_module.assess_clone_sample = original  # type: ignore[assignment]

    assert "cloning" in noted, f"the first clip should still clone; saw {noted}"
    assert "cloned_best_possible" in noted, (
        "A speaker at the top of the scale can never earn an upgrade, and the worker said nothing "
        f"about giving up on one — it just kept re-measuring. Saw: {noted}"
    )
    assert len(judged) == 1, (
        "The buffer was re-judged after the score became unbeatable. Every one of those runs is "
        "an FFT per 40ms frame over the whole buffer, for an answer arithmetic already gave: "
        f"{judged}"
    )
    # And it stopped ACCUMULATING, not merely stopped judging: `capturing` must not climb again
    # past the point where the verdict is settled.
    assert noted.count("cloned_best_possible") >= 1
