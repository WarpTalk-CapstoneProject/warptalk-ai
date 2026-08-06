"""WT-314 — the ingress bot must leave a room nobody is in, without being told to.

MeetingRoomService summons "AIBot_{room}" on every JoinMeetingAsync, but the only exit
used to be _cleanup_room(), driven by a backend message the backend suppressed for any
room that never had audio routes. A meeting where nobody pressed Start Translation left
the bot connected forever, billing LiveKit connection minutes — and because the bot is
itself a participant, LiveKit's own empty_timeout never fires either.

These tests pin the worker's own safety net: it releases a room with no human remote
participants after a bounded grace, keeps a room that still has one, and does not touch a
room that has only just joined.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit_ingress_worker.worker import (
    _IDLE_ROOM_GRACE_S,
    _IDLE_SWEEP_INTERVAL_S,
    LiveKitIngressWorker,
)
from shared.base_worker import TERMINAL_ROOM_STATUSES
from shared.config import LiveKitSettings, WorkerSettings

ROOM = "019fd60a-e5f3-7342-804a-4366e3214786"
OTHER_ROOM = "019fd60a-e5f3-7342-804a-000000000002"


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(return_value=None)
    return worker


def _participant(identity: str) -> MagicMock:
    participant = MagicMock()
    participant.identity = identity
    return participant


def _room(*identities: str, connected: bool = True) -> MagicMock:
    """A stand-in for rtc.Room — the SDK is never dialled in these tests."""
    room = MagicMock()
    room.connect = AsyncMock()
    room.disconnect = AsyncMock()
    room.isconnected.return_value = connected
    room.remote_participants = {i: _participant(i) for i in identities}
    return room


class _Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _sweep_and_drain(worker: LiveKitIngressWorker) -> None:
    """Run one sweep, then let the disconnect tasks it spawned finish."""
    await worker._sweep_idle_rooms()
    for _ in range(5):
        pending = [t for t in list(worker._event_tasks) if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_room_with_no_remote_participants_is_released_after_the_grace_period():
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    room = _room()
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now

    clock.advance(_IDLE_ROOM_GRACE_S + 1)
    await _sweep_and_drain(worker)

    room.disconnect.assert_awaited_once()
    assert ROOM not in worker.rooms
    assert ROOM not in worker._room_last_occupied
    assert worker._idle_releases_total == 1


@pytest.mark.asyncio
async def test_room_with_a_human_participant_is_never_released():
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    room = _room("user-123")
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now

    # Far past the grace period, and silent the whole time — idleness is measured in
    # participants, not speech, so a quiet meeting must survive indefinitely.
    for _ in range(10):
        clock.advance(_IDLE_ROOM_GRACE_S)
        await _sweep_and_drain(worker)

    room.disconnect.assert_not_awaited()
    assert worker.rooms[ROOM] is room
    # Each sweep re-stamps the room as occupied.
    assert worker._room_last_occupied[ROOM] == clock.now


@pytest.mark.asyncio
async def test_freshly_joined_room_is_not_released_before_its_grace_period():
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    # The bot legitimately sits alone between joining and the human's SFU handshake.
    room = _room()
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now

    clock.advance(_IDLE_ROOM_GRACE_S - 1)
    await _sweep_and_drain(worker)

    room.disconnect.assert_not_awaited()
    assert worker.rooms[ROOM] is room

    # ... and the human arrives just in time.
    room.remote_participants = {"user-123": _participant("user-123")}
    clock.advance(_IDLE_SWEEP_INTERVAL_S)
    await _sweep_and_drain(worker)

    room.disconnect.assert_not_awaited()
    assert worker.rooms[ROOM] is room


@pytest.mark.asyncio
async def test_a_room_holding_only_our_own_bots_counts_as_idle():
    """The TTS publisher's interpreter bots are remote participants in this same room.

    Counting them as occupancy would mean a room containing nothing but our own machinery
    looks busy — exactly the state WT-314 leaks.
    """
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    room = _room("ai-interpreter-vi-abc", "AIBot_other")
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now

    clock.advance(_IDLE_ROOM_GRACE_S + 1)
    await _sweep_and_drain(worker)

    room.disconnect.assert_awaited_once()
    assert ROOM not in worker.rooms


@pytest.mark.asyncio
async def test_sweep_releases_only_the_idle_room_and_leaves_the_busy_one_alone():
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    idle = _room()
    busy = _room("user-123")
    worker.rooms[ROOM] = idle
    worker.rooms[OTHER_ROOM] = busy
    worker._room_last_occupied[ROOM] = clock.now
    worker._room_last_occupied[OTHER_ROOM] = clock.now

    # A running transcription pipeline in the busy room must survive the idle room's exit.
    keep = asyncio.create_task(asyncio.sleep(30))
    worker.audio_tasks[(OTHER_ROOM, "TR_1")] = keep
    drop = asyncio.create_task(asyncio.sleep(30))
    worker.audio_tasks[(ROOM, "TR_2")] = drop

    clock.advance(_IDLE_ROOM_GRACE_S + 1)
    await _sweep_and_drain(worker)

    idle.disconnect.assert_awaited_once()
    busy.disconnect.assert_not_awaited()
    assert list(worker.rooms) == [OTHER_ROOM]
    assert drop.cancelled() or drop.cancelling()
    assert not keep.done()

    keep.cancel()
    await asyncio.gather(keep, drop, return_exceptions=True)


@pytest.mark.asyncio
async def test_release_is_abandoned_if_a_human_joins_before_the_task_takes_the_lock():
    """A track_published can land between the sweep's decision and the disconnect.

    Disconnecting a room somebody just joined is a real outage, so the release re-checks
    occupancy under the room lock rather than trusting the sweep's snapshot.
    """
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    room = _room()
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now
    clock.advance(_IDLE_ROOM_GRACE_S + 1)

    async with worker._room_lock(ROOM):
        await worker._sweep_idle_rooms()
        # The sweep decided to release; while it waits on the lock, a human arrives.
        room.remote_participants = {"user-123": _participant("user-123")}

    await _sweep_and_drain(worker)

    room.disconnect.assert_not_awaited()
    assert worker.rooms[ROOM] is room


@pytest.mark.asyncio
async def test_a_disconnected_room_handle_is_retired_by_the_same_grace():
    worker = _worker()
    clock = _Clock()
    worker._now = clock  # type: ignore[method-assign]

    # LiveKit dropped us, but a human is still listed on the stale handle. Not connected is
    # not occupied.
    room = _room("user-123", connected=False)
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = clock.now

    clock.advance(_IDLE_ROOM_GRACE_S + 1)
    await _sweep_and_drain(worker)

    assert ROOM not in worker.rooms


@pytest.mark.asyncio
async def test_sweep_loop_survives_an_exception_and_keeps_sweeping():
    worker = _worker()
    calls: list[int] = []

    async def _flaky() -> None:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("probe blew up")
        if len(calls) >= 3:
            worker._shutdown_event.set()

    worker._sweep_idle_rooms = _flaky  # type: ignore[method-assign]
    with patch("livekit_ingress_worker.worker._IDLE_SWEEP_INTERVAL_S", 0.001):
        await asyncio.wait_for(worker._idle_sweep_loop(), timeout=2.0)

    assert len(calls) >= 3


@pytest.mark.asyncio
async def test_ensure_idle_sweeper_restarts_a_sweeper_that_died():
    """A sweeper that stopped is indistinguishable from no sweeper, and silently restores
    the leak it exists to prevent."""
    worker = _worker()

    worker._ensure_idle_sweeper()
    first = worker._idle_sweeper
    assert first is not None

    # Idempotent while alive.
    worker._ensure_idle_sweeper()
    assert worker._idle_sweeper is first

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)

    worker._ensure_idle_sweeper()
    assert worker._idle_sweeper is not None
    assert worker._idle_sweeper is not first

    worker._idle_sweeper.cancel()
    await asyncio.gather(worker._idle_sweeper, return_exceptions=True)


@pytest.mark.asyncio
async def test_cleanup_cancels_the_sweeper_and_disconnects_every_room():
    worker = _worker()
    room = _room("user-123")
    worker.rooms[ROOM] = room
    worker._room_last_occupied[ROOM] = 1.0
    worker._ensure_idle_sweeper()
    sweeper = worker._idle_sweeper

    await worker._cleanup()

    assert sweeper is not None and sweeper.cancelled()
    assert worker._idle_sweeper is None
    room.disconnect.assert_awaited_once()
    assert worker.rooms == {}
    assert worker._room_last_occupied == {}


@pytest.mark.asyncio
async def test_cleanup_room_forgets_the_idle_timer():
    worker = _worker()
    worker._room_last_occupied[ROOM] = 1.0
    worker.rooms[ROOM] = _room()

    worker._cleanup_room(ROOM)

    assert ROOM not in worker._room_last_occupied
    for task in list(worker._event_tasks):
        await asyncio.gather(task, return_exceptions=True)


def test_expired_is_terminal_and_timeout_is_not_a_real_status():
    """EXPIRED is a real RoomStatus (Domain/Enums/RoomStatus.cs) that the worker ignored,
    so an expired room's bot was never released. TIMEOUT was never a status the backend
    publishes, so that entry matched nothing."""
    assert "EXPIRED" in TERMINAL_ROOM_STATUSES
    assert "TIMEOUT" not in TERMINAL_ROOM_STATUSES
    assert {"FAILED", "ENDED", "CANCELLED"} <= TERMINAL_ROOM_STATUSES


@pytest.mark.asyncio
async def test_expired_room_status_releases_the_bot():
    worker = _worker()
    room = _room()
    worker.rooms[ROOM] = room

    await worker._handle_route_update_message(
        {
            "type": "pmessage",
            "channel": f"translationRoom:{ROOM}:events",
            "data": '{"type": "AUDIO_ROUTES_UPDATED", "data": {"room_status": "EXPIRED"}}',
        }
    )

    for _ in range(5):
        pending = [t for t in list(worker._event_tasks) if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)

    assert ROOM not in worker.rooms
    room.disconnect.assert_awaited_once()
