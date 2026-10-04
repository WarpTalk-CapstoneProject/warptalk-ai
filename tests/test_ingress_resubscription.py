"""A connection resume must not leave a speaker's reader on a dead handle.

Production room 01a0fbe6 (2026-10-02). The ingress worker's LiveKit connection failed at
16:26:48, hit a ping timeout at 16:27:39 and recovered at 16:27:41. On recovery the SDK
re-subscribed all three microphones, flagging every one of them `muted`, although the owners'
microphones were on. Two of the three speakers were never heard again: their transcripts stopped
at 16:26:26, and the host ended the meeting two minutes later, believing a worker had crashed.

Two guards, each correct on its own, stacked into that outage:

  * readers were de-duplicated by track SID, and a re-subscription keeps the sid. The reader
    still bound to the replaced handle looked alive and received nothing, and the fresh
    subscription was refused as a duplicate of it;
  * WT-542 refuses to read a muted publication, and believed a muted flag that no
    `track_muted` event had delivered, so neither the subscribe handler nor the sweep tried.

What these pin: a new Track object with a known sid replaces the reader, and a muted flag on a
microphone we are still reading (so no mute event ever stopped it) does not block that
replacement. WT-542 still holds everywhere else.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from unittest.mock import MagicMock

import pytest
from livekit import rtc

import livekit_ingress_worker.worker as worker_module
from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a0fbe6-5512-7cab-8b11-8a2535d6980b"
SPEAKER = "019f0d00-0de0-7000-9000-000000000001"


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
    track.kind = rtc.TrackKind.KIND_AUDIO
    return track


def _room_with(pub: MagicMock) -> MagicMock:
    participant = MagicMock()
    participant.identity = SPEAKER
    participant.track_publications = {pub.sid: pub}
    room = MagicMock()
    room.remote_participants = {SPEAKER: participant}
    return room


def _publication(track: MagicMock, *, muted: bool) -> MagicMock:
    pub = MagicMock()
    pub.sid = track.sid
    pub.kind = rtc.TrackKind.KIND_AUDIO
    pub.muted = muted
    pub.source = rtc.TrackSource.SOURCE_MICROPHONE
    pub.track = track
    return pub


@pytest.fixture(autouse=True)
def _never_really_read(monkeypatch: pytest.MonkeyPatch):
    async def _idle(self, room_name: str, speaker_id: str, track) -> None:  # noqa: ANN001
        await asyncio.Event().wait()

    monkeypatch.setattr(LiveKitIngressWorker, "process_audio_track", _idle)


@pytest.mark.asyncio
async def test_a_resubscribed_track_replaces_the_reader_on_the_old_handle() -> None:
    worker = _worker()
    before_resume = _track("TR_AMUn2MiQNikH4b")
    worker._start_audio_task(ROOM, SPEAKER, before_resume)
    zombie = worker.audio_tasks[(ROOM, SPEAKER)]

    after_resume = _track("TR_AMUn2MiQNikH4b")  # same sid, new handle
    assert worker._start_audio_task(ROOM, SPEAKER, after_resume) is True

    await asyncio.sleep(0)
    assert zombie.cancelled()
    assert worker.audio_tasks[(ROOM, SPEAKER)] is not zombie
    assert worker._audio_task_track_objects[(ROOM, SPEAKER)] is after_resume
    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_a_muted_flag_on_a_microphone_still_being_read_is_a_resubscription() -> None:
    worker = _worker()
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_live"))

    assert worker._is_resubscription_of_live_reader(ROOM, SPEAKER, "TR_live") is True
    # A different sid is a different publication: its muted flag is a real one.
    assert worker._is_resubscription_of_live_reader(ROOM, SPEAKER, "TR_other") is False
    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_the_sweep_moves_a_live_reader_onto_the_resubscribed_handle() -> None:
    # The 01a0fbe6 shape exactly: a live reader on the old handle, and the publication now
    # carrying a new handle flagged muted. Before the fix the sweep skipped it as muted and
    # reported reattached_readers=0 for the rest of the meeting.
    worker = _worker()
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_live"))

    resubscribed = _track("TR_live")
    room = _room_with(_publication(resubscribed, muted=True))

    assert worker._start_pending_audio_tasks(ROOM, room) == 1
    assert worker._audio_task_track_objects[(ROOM, SPEAKER)] is resubscribed
    # And the next sweep finds nothing left to do — no churn every fifteen seconds.
    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_wt542_still_holds_for_a_microphone_muted_by_an_event() -> None:
    # The mute event cancels the reader, so there is no live reader to vouch for the track and
    # the muted flag is believed — the sweep must not re-attach it.
    worker = _worker()
    track = _track("TR_live")
    worker._start_audio_task(ROOM, SPEAKER, track)
    worker._cancel_audio_task(ROOM, SPEAKER)  # what on_track_muted does

    room = _room_with(_publication(_track("TR_live"), muted=True))

    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert (ROOM, SPEAKER) not in worker.audio_tasks


@pytest.mark.asyncio
async def test_wt542_still_holds_for_someone_who_joins_muted() -> None:
    worker = _worker()
    room = _room_with(_publication(_track("TR_joined_muted"), muted=True))

    assert worker._start_pending_audio_tasks(ROOM, room) == 0


@pytest.mark.asyncio
async def test_cancelling_forgets_the_handle_too() -> None:
    worker = _worker()
    worker._start_audio_task(ROOM, SPEAKER, _track("TR_live"))
    worker._cancel_audio_task(ROOM, SPEAKER)

    assert (ROOM, SPEAKER) not in worker._audio_task_track_objects

    worker._start_audio_task(ROOM, SPEAKER, _track("TR_live"))
    worker._cancel_room_audio_tasks(ROOM)

    assert (ROOM, SPEAKER) not in worker._audio_task_track_objects


def test_the_subscribe_handler_asks_before_refusing_a_muted_track() -> None:
    # The handler is a closure inside join_room and cannot run without a live LiveKit room, so
    # this is asserted on the source: the muted branch must consult the live-reader check, or
    # the subscribe event — the first and fastest door — stays shut after every resume.
    tree = ast.parse(inspect.getsource(worker_module))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "on_track_subscribed"
    )
    called = {
        node.func.attr
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_is_resubscription_of_live_reader" in called
