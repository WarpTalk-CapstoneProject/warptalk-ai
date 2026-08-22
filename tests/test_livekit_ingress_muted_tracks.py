"""WT-542 — a muted microphone must not be read.

A muted LiveKit publication is not a silent one: the subscription stays alive and the reader
goes on receiving frames of room tone and encoder noise. Near-silence is the input Whisper
invents text from, so the transcript credited people who were muted for the whole meeting with
things they never said.

Production room 01a01e3f (2026-08-20): a participant muted throughout was credited with
thirteen segments of ENGLISH in a Vietnamese meeting — "I see that facility is new.", "Whose
child's time did you take?", and the profanity string Whisper is known to emit on empty audio.

There is no downstream filter that fixes this honestly, because there is no text a muted
microphone could legitimately produce. These tests pin the only correct behaviour: do not read
the track, and pick it up again the moment its owner unmutes.
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
OTHER_SPEAKER = "019f0d00-0de0-7000-9000-000000000004"


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
    # rtc.TrackKind.KIND_AUDIO is an int enum; the worker compares publications against it.
    track.kind = 1
    return track


def _publication(sid: str, *, muted: bool) -> MagicMock:
    pub = MagicMock()
    pub.sid = sid
    pub.muted = muted
    pub.kind = 1
    pub.track = _track(sid)
    return pub


def _room_with(*participants: tuple[str, MagicMock]) -> MagicMock:
    room = MagicMock()
    remote = {}
    for identity, pub in participants:
        participant = MagicMock()
        participant.identity = identity
        participant.track_publications = {pub.sid: pub}
        remote[identity] = participant
    room.remote_participants = remote
    return room


@pytest.fixture(autouse=True)
def _never_really_read(monkeypatch: pytest.MonkeyPatch):
    """Replace the audio pipeline with something that simply waits to be cancelled."""

    async def _idle(self, room_name: str, speaker_id: str, track) -> None:  # noqa: ANN001
        await asyncio.Event().wait()

    monkeypatch.setattr(LiveKitIngressWorker, "process_audio_track", _idle)


@pytest.mark.asyncio
async def test_the_sweep_does_not_attach_a_reader_to_a_muted_publication() -> None:
    worker = _worker()
    room = _room_with((SPEAKER, _publication("TR_muted", muted=True)))

    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert (ROOM, SPEAKER) not in worker.audio_tasks


@pytest.mark.asyncio
async def test_the_sweep_still_attaches_to_everyone_who_is_unmuted() -> None:
    worker = _worker()
    room = _room_with(
        (SPEAKER, _publication("TR_muted", muted=True)),
        (OTHER_SPEAKER, _publication("TR_live", muted=False)),
    )

    assert worker._start_pending_audio_tasks(ROOM, room) == 1
    assert (ROOM, OTHER_SPEAKER) in worker.audio_tasks

    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_muting_mid_meeting_stops_that_speakers_reader() -> None:
    worker = _worker()
    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_live")) is True

    assert worker._cancel_audio_task(ROOM, SPEAKER) is True

    assert (ROOM, SPEAKER) not in worker.audio_tasks
    # And nothing is left behind that would make the reaper think a reader is still live.
    assert (ROOM, SPEAKER) not in worker.audio_task_tracks


@pytest.mark.asyncio
async def test_cancelling_one_speaker_leaves_the_rest_of_the_room_reading() -> None:
    worker = _worker()
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_a"))
    worker._start_audio_task(ROOM, OTHER_SPEAKER, _track("TR_b"))

    worker._cancel_audio_task(ROOM, SPEAKER)

    assert (ROOM, OTHER_SPEAKER) in worker.audio_tasks
    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_cancelling_a_speaker_who_is_not_being_read_reports_no_change() -> None:
    worker = _worker()

    assert worker._cancel_audio_task(ROOM, SPEAKER) is False


@pytest.mark.asyncio
async def test_a_speaker_who_unmutes_is_read_again() -> None:
    """The half that must not be missed: mute-then-unmute cannot silence somebody for good."""
    worker = _worker()
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_live"))
    worker._cancel_audio_task(ROOM, SPEAKER)

    # Same sid — unmuting does not republish, so the reader must start on the track it left.
    assert worker._start_audio_task(ROOM, SPEAKER, _track("TR_live")) is True
    assert (ROOM, SPEAKER) in worker.audio_tasks

    worker._cancel_room_audio_tasks(ROOM)
