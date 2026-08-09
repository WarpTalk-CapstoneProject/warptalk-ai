"""One reader per speaker, not one per track.

The audio-reader guard was keyed on `(room_name, track.sid)`. LiveKit issues a NEW sid every
time a participant republishes their microphone — a reconnect, a device change, a momentary
network drop — so the guard saw an unfamiliar key and started a SECOND reader for a speaker
who already had one, while the first stayed alive.

In production this reached three concurrent readers on a single microphone. Each carried its
own chunk counter, so the same sentence was published three times with three different
`chunk_index` values and near-identical audio fingerprints; the meeting transcribed every
utterance three times, and because each copy travelled the whole pipeline, the dubbed voice
spoke it three times as well.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "019f6a39-a32c-7745-886e-1fe622c1f747"
SPEAKER = "019f0d00-0de0-7000-9000-000000000002"
OTHER_SPEAKER = "019f0d00-0de0-7000-9000-000000000003"


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    return worker


def _track(sid: str) -> MagicMock:
    track = MagicMock()
    track.sid = sid
    return track


@pytest.fixture(autouse=True)
def _never_really_read(monkeypatch: pytest.MonkeyPatch):
    """Replace the audio pipeline with something that simply waits to be cancelled."""

    async def _idle(self, room_name: str, speaker_id: str, track) -> None:  # noqa: ANN001
        await asyncio.Event().wait()

    monkeypatch.setattr(LiveKitIngressWorker, "process_audio_track", _idle)


@pytest.mark.asyncio
async def test_republished_microphone_does_not_add_a_second_reader() -> None:
    worker = _worker()

    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_first")) is True
    # Same human, new sid — this is the reconnect that used to double the transcript.
    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_second")) is True

    live = [task for task in worker.audio_tasks.values() if not task.done()]
    assert len(live) == 1, f"expected one reader for one speaker, got {len(live)}"
    assert worker.audio_task_tracks[(ROOM, SPEAKER)] == "TR_second"

    for task in worker.audio_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_the_stale_reader_is_actually_cancelled() -> None:
    worker = _worker()

    worker._start_audio_task(ROOM, SPEAKER, _track("TR_first"))
    first = worker.audio_tasks[(ROOM, SPEAKER)]

    worker._start_audio_task(ROOM, SPEAKER, _track("TR_second"))
    await asyncio.sleep(0)

    assert first.cancelled() or first.done(), "the reader on the stale track kept running"

    for task in worker.audio_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_the_same_track_twice_is_refused() -> None:
    worker = _worker()

    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_only")) is True
    # A duplicate track_published for a microphone already being read changes nothing.
    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_only")) is False
    assert len(worker.audio_tasks) == 1

    for task in worker.audio_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_two_speakers_each_keep_their_own_reader() -> None:
    worker = _worker()

    worker._start_audio_task(ROOM, SPEAKER, _track("TR_a"))
    worker._start_audio_task(ROOM, OTHER_SPEAKER, _track("TR_b"))

    live = [task for task in worker.audio_tasks.values() if not task.done()]
    assert len(live) == 2, "keying on the speaker must not merge two people into one reader"

    for task in worker.audio_tasks.values():
        task.cancel()


@pytest.mark.asyncio
async def test_replacing_a_reader_leaves_the_live_one_registered() -> None:
    """The cancelled task's done-callback must not evict its own replacement."""
    worker = _worker()

    worker._start_audio_task(ROOM, SPEAKER, _track("TR_first"))
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_second"))
    replacement = worker.audio_tasks[(ROOM, SPEAKER)]

    # Let the cancelled predecessor run its done-callback.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert worker.audio_tasks.get((ROOM, SPEAKER)) is replacement
    assert worker.audio_task_tracks.get((ROOM, SPEAKER)) == "TR_second"

    replacement.cancel()
