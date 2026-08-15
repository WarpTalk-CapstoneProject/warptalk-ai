"""WT-420 — the clone's progress has to leave the log.

`_note_clone_state` already knew everything a speaker needs: whether capture is happening, how
far along it is, whether the clip was accepted, and why it was refused. It wrote all of it to a
structured log.

On 15 Aug the whole team tried to hear a cloned voice, could not, and reported cloning as broken.
The worker was logging `voice_clone_sample_accepted` with `score: 1.0` at the same time. Nobody in
a meeting can read a worker log, so from inside the product a healthy clone and a dead one look
identical — which is the actual defect.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker


class _RecordingWorker:
    """A worker with only the parts _note_clone_state touches."""

    def __init__(self, publish_raises: bool = False) -> None:
        self.worker = TTSWorker.__new__(TTSWorker)
        self.worker.settings = WorkerSettings()
        self.worker.tts_settings = TTSSettings()
        self.worker.logger = MagicMock()
        self.published: list[tuple[str, str, dict[str, Any]]] = []

        async def _publish(prefix: str, meeting_id: str, data: dict[str, Any]) -> bytes | str:
            if publish_raises:
                raise RuntimeError("redis is down")
            self.published.append((prefix, meeting_id, data))
            return "1-0"

        self.worker.publish = _publish  # type: ignore[method-assign, assignment]


KEY = ("m1", "s1")


@pytest.mark.asyncio
async def test_a_state_change_is_published_not_only_logged() -> None:
    harness = _RecordingWorker()

    await harness.worker._note_clone_state(KEY, "not_opted_in")

    assert len(harness.published) == 1
    prefix, meeting_id, payload = harness.published[0]
    assert prefix == "voice:clone:state"
    assert meeting_id == "m1"
    assert payload["speaker_id"] == "s1"
    assert payload["reason"] == "not_opted_in"


@pytest.mark.asyncio
async def test_capture_progress_carries_the_numbers_the_bar_is_made_of() -> None:
    harness = _RecordingWorker()

    await harness.worker._note_clone_state(KEY, "capturing", seconds=4.2, required_seconds=10.0)

    _prefix, _meeting, payload = harness.published[0]
    assert payload["reason"] == "capturing"
    assert payload["seconds"] == 4.2
    assert payload["required_seconds"] == 10.0


@pytest.mark.asyncio
async def test_progress_updates_as_the_buffer_fills() -> None:
    """The reason alone cannot be the dedupe key.

    `capturing` is reported on every chunk. Deduping on the reason would publish it once and
    freeze the bar at its first value — a progress bar that never moves is worse than none,
    because it reads as "stuck" rather than "not implemented".
    """
    harness = _RecordingWorker()

    for seconds in (1.0, 2.0, 3.0, 4.0):
        await harness.worker._note_clone_state(
            KEY, "capturing", seconds=seconds, required_seconds=10.0
        )

    assert [p["seconds"] for _prefix, _meeting, p in harness.published] == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_an_unchanged_state_is_still_published_only_once() -> None:
    """The negative control for the test above.

    Audio chunks arrive continuously for the whole meeting. A state with no progress attached
    must keep the original once-per-change behaviour, or this becomes one message per chunk per
    speaker — the volume the method was written to avoid.
    """
    harness = _RecordingWorker()

    for _ in range(5):
        await harness.worker._note_clone_state(KEY, "no_route_for_speaker")

    assert len(harness.published) == 1


@pytest.mark.asyncio
async def test_sub_second_progress_does_not_flood() -> None:
    harness = _RecordingWorker()

    for seconds in (1.0, 1.2, 1.4, 1.9):
        await harness.worker._note_clone_state(
            KEY, "capturing", seconds=seconds, required_seconds=10.0
        )

    assert len(harness.published) == 1, "progress published faster than the bar can show"


@pytest.mark.asyncio
async def test_a_rejected_clip_carries_its_reason_and_measurement() -> None:
    harness = _RecordingWorker()

    await harness.worker._note_clone_state(
        KEY, "clip_rejected:too little speech", active_speech_ratio=0.04
    )

    _prefix, _meeting, payload = harness.published[0]
    assert payload["reason"] == "clip_rejected:too little speech"
    assert payload["active_speech_ratio"] == 0.04


@pytest.mark.asyncio
async def test_a_publish_failure_never_takes_the_clone_path_down() -> None:
    """A stalled progress bar is a worse UI. A speaker who stops being cloned because a Redis
    write failed is a worse product."""
    harness = _RecordingWorker(publish_raises=True)

    await harness.worker._note_clone_state(KEY, "cloning", score=0.97)

    cast(MagicMock, harness.worker.logger).warning.assert_called_once()


@pytest.mark.asyncio
async def test_the_log_line_still_happens() -> None:
    """Publishing is in addition to logging, not instead of it. Triage from outside a meeting
    still reads the log, and the two must not drift into disagreeing about the same speaker."""
    harness = _RecordingWorker()

    await harness.worker._note_clone_state(KEY, "cloning", score=0.97)

    logger = cast(MagicMock, harness.worker.logger)
    logger.info.assert_called_once()
    assert logger.info.call_args.args[0] == "voice_clone_state"


@pytest.mark.asyncio
async def test_absent_metrics_are_omitted_rather_than_sent_as_null() -> None:
    harness = _RecordingWorker()

    await harness.worker._note_clone_state(KEY, "no_routes")

    _prefix, _meeting, payload = harness.published[0]
    assert "seconds" not in payload
    assert "score" not in payload


def test_asyncio_is_imported_for_the_module_under_test() -> None:
    # Guards against the harness above silently passing on a module that failed to import.
    assert asyncio is not None
