"""A refused in-meeting clone must not keep counting as the speaker's clone (WT-874).

The capture loop records the clip's score the moment it STARTS a clone, because the clone runs as
a background task. Nothing took it back when Cartesia refused. Production room 01a0e5dd, 28 Sep:
`clone_failed: 402 plan_upgrade_required`, then 16 seconds later `cloned_best_possible` — the UI
showed the speaker's voice as done while every listener heard a stock voice.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.test_clone_pitch_coverage import _varied
from tests.test_clone_upgrade import _chunks, _run, _worker
from tts_worker.worker import TTSWorker, _settle_live_clone


def _failing_worker(code: str) -> tuple[TTSWorker, list[bytes], list[str]]:
    worker, _ = _worker([], voice_clone_min_seconds=10.0)
    attempts: list[bytes] = []
    states: list[str] = []

    async def _clone_and_cache(
        _meeting: str,
        _speaker: str,
        audio: bytes,
        _language: str = "en",
        _sample_rate: int = 16000,
        _score: float | None = None,
        **_kwargs: Any,
    ) -> str:
        attempts.append(audio)
        return code

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return None

    async def _note_clone_state(_key: tuple[str, str], reason: str, **_kwargs: Any) -> None:
        states.append(reason)

    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign,assignment]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    worker._note_clone_state = _note_clone_state  # type: ignore[method-assign]
    return worker, attempts, states


@pytest.mark.asyncio
async def test_a_plan_refusal_stops_capturing_and_never_reads_as_cloned() -> None:
    worker, attempts, states = _failing_worker("PROVIDER_PLAN_REQUIRED")

    await _run(worker, [c for _ in range(4) for c in _chunks(_varied())])

    assert len(attempts) == 1, "no later clip can fix the account's plan; one refusal is enough"
    assert "cloned_best_possible" not in states
    assert states.count("cloning") == 1


@pytest.mark.asyncio
async def test_a_transient_failure_is_retried_on_a_later_clip() -> None:
    worker, attempts, states = _failing_worker("PROVIDER_BUSY")

    await _run(worker, [c for _ in range(4) for c in _chunks(_varied())])

    assert len(attempts) >= 2, "a rate limit is worth another attempt"
    assert "cloned_best_possible" not in states


async def _finished(outcome: Any) -> asyncio.Task[Any]:
    async def _result() -> Any:
        return outcome

    task = asyncio.ensure_future(_result())
    await task
    return task


@pytest.mark.asyncio
async def test_a_failed_upgrade_restores_the_previous_clone_and_refunds_it() -> None:
    key = ("m1", "s1")
    cloned_score = {key: 0.9}
    upgrades_used = {key: 1}
    refused: set[tuple[str, str]] = set()

    _settle_live_clone(
        await _finished("UNKNOWN"),
        key=key,
        attempted_score=0.9,
        previous_score=0.4,
        spent_upgrade=True,
        cloned_score=cloned_score,
        upgrades_used=upgrades_used,
        clone_refused=refused,
    )

    assert cloned_score == {key: 0.4}
    assert upgrades_used == {key: 0}
    assert refused == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["", None])
async def test_success_and_an_unknown_outcome_change_nothing(outcome: str | None) -> None:
    key = ("m1", "s1")
    cloned_score = {key: 0.9}

    _settle_live_clone(
        await _finished(outcome),
        key=key,
        attempted_score=0.9,
        previous_score=None,
        spent_upgrade=False,
        cloned_score=cloned_score,
        upgrades_used={},
        clone_refused=set(),
    )

    assert cloned_score == {key: 0.9}
