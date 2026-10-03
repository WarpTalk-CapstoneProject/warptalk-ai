"""A microphone the bot can see but was never handed is asked for again.

Production room 01a1009e, 3 Oct 2026. The census counted three humans from 14:19:10, yet Ngọc Kỳ's
microphone was subscribed at 14:24:47 — six minutes of speech nobody transcribed or dubbed — and
Tuấn's 44 seconds after he joined. Auto-subscribe is a request the server may drop, and nothing
re-sent it: every reattach path in this worker starts from `pub.track`, which a publication that
was never subscribed does not have.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from livekit import rtc

import livekit_ingress_worker.worker as worker_module
from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a1009e-f359-7c58-8a8c-c12ca1ccbe3f"
SPEAKER = "019f0d00-0de0-7000-9000-000000000003"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _worker(clock: _Clock) -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    worker._now = clock  # type: ignore[method-assign]
    return worker


def _publication(
    *,
    subscribed_track: bool = False,
    source: rtc.TrackSource.ValueType = rtc.TrackSource.SOURCE_MICROPHONE,
    muted: bool = False,
) -> MagicMock:
    pub = MagicMock()
    pub.sid = "TR_AMXfDYLuiYRqxV"
    pub.kind = rtc.TrackKind.KIND_AUDIO
    pub.source = source
    pub.muted = muted
    pub.track = MagicMock() if subscribed_track else None
    return pub


def _room(pub: MagicMock, identity: str = SPEAKER) -> MagicMock:
    participant = MagicMock()
    participant.identity = identity
    participant.track_publications = {pub.sid: pub}
    room = MagicMock()
    room.remote_participants = {identity: participant}
    room.isconnected.return_value = True
    return room


def test_an_ordinary_subscription_gets_its_grace() -> None:
    clock = _Clock()
    worker = _worker(clock)
    pub = _publication()

    assert worker._repair_missing_subscriptions(ROOM, _room(pub)) == 0
    pub.set_subscribed.assert_not_called()


def test_a_subscription_stuck_past_the_grace_is_requested_again() -> None:
    clock = _Clock()
    worker = _worker(clock)
    pub = _publication()
    room = _room(pub)

    worker._repair_missing_subscriptions(ROOM, room)
    clock.now += worker_module._SUBSCRIPTION_GRACE_S + 0.1

    assert worker._repair_missing_subscriptions(ROOM, room) == 1
    # Off then on: a bare True repeats the state the server already ignored once.
    assert [c.args[0] for c in pub.set_subscribed.call_args_list] == [False, True]


def test_a_muted_microphone_is_subscribed_too_so_unmuting_is_instant() -> None:
    clock = _Clock()
    worker = _worker(clock)
    pub = _publication(muted=True)
    room = _room(pub)

    worker._repair_missing_subscriptions(ROOM, room)
    clock.now += worker_module._SUBSCRIPTION_GRACE_S + 0.1

    assert worker._repair_missing_subscriptions(ROOM, room) == 1


@pytest.mark.parametrize(
    "pub, identity",
    [
        (_publication(subscribed_track=True), SPEAKER),
        (_publication(source=rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO), SPEAKER),
        (_publication(), f"ai-interpreter-en-{SPEAKER}"),
    ],
    ids=["already-subscribed", "screen-share-audio", "our-own-bot"],
)
def test_nothing_else_is_touched(pub: MagicMock, identity: str) -> None:
    clock = _Clock()
    worker = _worker(clock)
    room = _room(pub, identity)

    worker._repair_missing_subscriptions(ROOM, room)
    clock.now += 60.0

    assert worker._repair_missing_subscriptions(ROOM, room) == 0
    pub.set_subscribed.assert_not_called()


def test_a_server_that_stays_silent_is_asked_again_not_hammered() -> None:
    clock = _Clock()
    worker = _worker(clock)
    pub = _publication()
    room = _room(pub)

    worker._repair_missing_subscriptions(ROOM, room)
    clock.now += worker_module._SUBSCRIPTION_GRACE_S + 0.1
    worker._repair_missing_subscriptions(ROOM, room)
    clock.now += 0.5
    assert worker._repair_missing_subscriptions(ROOM, room) == 0
    clock.now += worker_module._SUBSCRIPTION_GRACE_S
    assert worker._repair_missing_subscriptions(ROOM, room) == 1


@pytest.mark.asyncio
async def test_a_join_schedules_quick_checks_instead_of_waiting_for_the_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_SUBSCRIPTION_REPAIR_DELAYS_S", (0.0, 0.01))
    clock = _Clock()
    worker = _worker(clock)
    pub = _publication()
    room = _room(pub)
    worker.rooms[ROOM] = room
    calls: list[str] = []
    worker._repair_missing_subscriptions = (  # type: ignore[method-assign]
        lambda name, _room: calls.append(name) or 0
    )

    worker._schedule_subscription_checks(ROOM)
    for _ in range(20):
        await asyncio.sleep(0.01)

    assert calls == [ROOM, ROOM]


def test_the_sweep_repairs_subscriptions_too() -> None:
    """The backstop for a join event this replica never saw."""
    source = worker_module.LiveKitIngressWorker._sweep_idle_rooms
    import inspect

    assert "_repair_missing_subscriptions" in inspect.getsource(source)
