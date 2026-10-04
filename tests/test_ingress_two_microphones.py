"""One reader per person, even while the server briefly holds two of their microphones.

PRODUCTION, 2026-10-03, ROOM 01a1010c
    The host's client did a LiveKit full reconnect at 09:23:36 and logged "failed to remove
    track" on the way: the old microphone publication could not be taken down. The web's fix is
    to re-publish the microphone — which, with the old one still there, leaves the server holding
    TWO speech publications for one identity until the old one finally goes.

WHY THAT MATTERS HERE
    Readers are keyed per (room, participant). Two things assumed one microphone per person:

      * the sweep handed every eligible publication to `_start_audio_task`, which replaces a
        reader on any other sid — so each sweep restarted the reader on the first and then on the
        second, throwing away the utterance in progress twice every fifteen seconds;
      * `track_muted` cancelled the person's reader whichever of their publications was muted, so
        the old microphone being muted on its way out stopped the reader on the new, live one.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from livekit import rtc

import livekit_ingress_worker.worker as worker_module
from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a1010c-62db-7d7a-9a94-3bdbebfe3a3b"
HOST = "019f0d00-0de0-7000-9000-0000000000a1"

pytestmark = pytest.mark.asyncio


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="s" * 32)
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    return worker


def _publication(sid: str, *, muted: bool = False) -> MagicMock:
    track = MagicMock()
    track.sid = sid
    track.kind = rtc.TrackKind.KIND_AUDIO
    pub = MagicMock()
    pub.sid = sid
    pub.kind = rtc.TrackKind.KIND_AUDIO
    pub.muted = muted
    pub.source = rtc.TrackSource.SOURCE_MICROPHONE
    pub.track = track
    return pub


def _participant(*pubs: MagicMock) -> MagicMock:
    participant = MagicMock()
    participant.identity = HOST
    participant.name = ""
    participant.track_publications = {pub.sid: pub for pub in pubs}
    return participant


def _room(participant: MagicMock) -> MagicMock:
    room = MagicMock()
    room.remote_participants = {participant.identity: participant}
    return room


@pytest.fixture(autouse=True)
def _never_really_read(monkeypatch: pytest.MonkeyPatch):
    async def _idle(self, room_name: str, speaker_id: str, track) -> None:  # noqa: ANN001
        await asyncio.Event().wait()

    monkeypatch.setattr(LiveKitIngressWorker, "process_audio_track", _idle)


class _FakeRoom:
    """Records the handlers join_room registers, so a test can fire LiveKit's events at them."""

    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}
        self.remote_participants: dict[str, Any] = {}

    def on(self, event: str):  # noqa: ANN201
        def register(handler):  # noqa: ANN001, ANN202
            self.handlers[event] = handler
            return handler

        return register

    async def connect(self, *_args: Any) -> None:
        return None


async def _joined(monkeypatch: pytest.MonkeyPatch, worker: LiveKitIngressWorker) -> _FakeRoom:
    monkeypatch.setattr(worker_module.rtc, "Room", _FakeRoom)
    room = await worker.join_room(ROOM)
    assert isinstance(room, _FakeRoom)
    return room


async def test_the_sweep_leaves_a_reader_on_one_of_two_live_microphones_alone() -> None:
    worker = _worker()
    old, new = _publication("TR_old"), _publication("TR_new")
    worker._start_audio_task(ROOM, HOST, new.track)
    reader = worker.audio_tasks[(ROOM, HOST)]
    room = _room(_participant(old, new))

    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert worker._start_pending_audio_tasks(ROOM, room) == 0

    assert worker.audio_tasks[(ROOM, HOST)] is reader
    assert worker.audio_task_tracks[(ROOM, HOST)] == "TR_new"
    worker._cancel_room_audio_tasks(ROOM)


async def test_the_sweep_attaches_exactly_one_reader_to_someone_with_none() -> None:
    worker = _worker()
    room = _room(_participant(_publication("TR_old"), _publication("TR_new")))

    assert worker._start_pending_audio_tasks(ROOM, room) == 1
    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert worker.audio_task_tracks[(ROOM, HOST)] == "TR_new"
    worker._cancel_room_audio_tasks(ROOM)


async def test_a_reader_on_a_microphone_that_is_gone_moves_to_the_one_that_is_there() -> None:
    worker = _worker()
    gone = _publication("TR_gone")
    worker._start_audio_task(ROOM, HOST, gone.track)

    assert worker._start_pending_audio_tasks(ROOM, _room(_participant(_publication("TR_new")))) == 1
    assert worker.audio_task_tracks[(ROOM, HOST)] == "TR_new"
    worker._cancel_room_audio_tasks(ROOM)


async def test_a_muted_microphone_is_still_not_chosen_over_nothing() -> None:
    # WT-542 is unchanged by choosing per person: muted alone is still never read.
    worker = _worker()
    room = _room(_participant(_publication("TR_muted", muted=True)))

    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert (ROOM, HOST) not in worker.audio_tasks


async def test_muting_the_old_microphone_does_not_stop_the_reader_on_the_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    room = await _joined(monkeypatch, worker)
    old, new = _publication("TR_old"), _publication("TR_new")
    participant = _participant(old, new)
    worker._start_audio_task(ROOM, HOST, new.track)
    reader = worker.audio_tasks[(ROOM, HOST)]

    old.muted = True
    room.handlers["track_muted"](participant, old)
    await asyncio.sleep(0)

    assert worker.audio_tasks[(ROOM, HOST)] is reader
    assert not reader.cancelled()
    worker._cancel_room_audio_tasks(ROOM)


async def test_muting_the_microphone_being_read_still_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    room = await _joined(monkeypatch, worker)
    live = _publication("TR_live")
    worker._start_audio_task(ROOM, HOST, live.track)

    live.muted = True
    room.handlers["track_muted"](_participant(live), live)

    assert (ROOM, HOST) not in worker.audio_tasks


async def test_a_reconnect_of_the_bot_itself_is_logged_and_rechecks_the_microphones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _worker()
    worker.logger = MagicMock()
    worker._schedule_subscription_checks = MagicMock()  # type: ignore[method-assign]
    room = await _joined(monkeypatch, worker)

    room.handlers["reconnecting"]()
    room.handlers["reconnected"]()

    events = [call.args[0] for call in worker.logger.warning.call_args_list]
    assert events[-2:] == ["livekit_room_reconnecting", "livekit_room_reconnected"]
    worker._schedule_subscription_checks.assert_called_once_with(ROOM)


async def test_a_participant_leaving_is_logged_but_their_reader_is_not_touched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A client's full reconnect is a disconnect and a connect of the same identity in either
    # order; cancelling here could stop the reader already moved onto the new session.
    worker = _worker()
    worker.logger = MagicMock()
    room = await _joined(monkeypatch, worker)
    live = _publication("TR_new")
    worker._start_audio_task(ROOM, HOST, live.track)
    reader = worker.audio_tasks[(ROOM, HOST)]

    room.handlers["participant_disconnected"](_participant(live))

    assert worker.audio_tasks[(ROOM, HOST)] is reader
    worker.logger.info.assert_any_call(
        "livekit_participant_disconnected", room=ROOM, participant=HOST, reading_track="TR_new"
    )
    worker._cancel_room_audio_tasks(ROOM)
