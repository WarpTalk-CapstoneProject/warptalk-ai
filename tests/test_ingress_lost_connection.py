"""WT-395 — a meeting that stops being heard, while everything reports healthy.

THE PRODUCTION EVIDENCE (2026-08-14, room 01a00058)

    Trần Mạnh Tuấn   27 segments   12:56:58 -> 13:01:09
    Ngọc Kỳ          27 segments   12:57:22 -> 13:01:13
    System            1 segment    13:11:52   (the meeting-end marker)

    Two speakers stopped within FOUR SECONDS of each other and never resumed. The room and
    the translation session stayed alive until 13:11:52; the participants were still there;
    Redis was at 129 MB of 768 MB with no new evictions, so this is not the eviction that
    caused WT-387. Four seconds apart is not two independent audio readers dying — it is the
    one thing that serves both of them going away.

THE BRANCH

    `_sweep_idle_rooms` read a room it could no longer reach as a room with nobody in it:

        count = self._human_participant_count(room) if room.isconnected() else 0

    Those are opposite situations. "Nobody is here" means we are not needed; "we cannot see"
    means we have failed and must reconnect. Collapsing them was safe for the question
    WT-314 asked (a leaked bot costs LiveKit minutes) and wrong for this one, because
    retirement is only harmless if something re-joins — and nothing does. `_release_idle_room`
    also discards the room from `_deferred_rooms`, and the only other way back is a
    `meeting.track_published` event, which in a room where everyone has already published
    their mic never arrives again.

    So one dropped connection ended audio ingestion for the rest of the meeting, silently.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit_ingress_worker.worker import _IDLE_ROOM_GRACE_S, LiveKitIngressWorker

pytestmark = pytest.mark.asyncio

ROOM = "room-1"


def _room(*, connected: bool, humans: int = 0) -> Any:
    room = MagicMock()
    room.isconnected = MagicMock(return_value=connected)
    room.remote_participants = {f"user-{i}": MagicMock(identity=f"user-{i}") for i in range(humans)}
    return room


def _worker(room: Any, *, now: float = 1000.0) -> LiveKitIngressWorker:
    worker: LiveKitIngressWorker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.rooms = {ROOM: room}
    worker.audio_tasks = {}
    worker.audio_task_tracks = {}
    worker._room_last_occupied = {}
    worker._deferred_rooms = set()
    worker._owned_rooms = set()
    worker._event_tasks = set()
    worker._idle_releases_total = 0
    worker._room_locks = {}
    worker.logger = MagicMock()
    worker._now = MagicMock(return_value=now)  # type: ignore[method-assign]
    worker._release_room_ownership = AsyncMock()  # type: ignore[method-assign]
    return worker


async def test_a_lost_connection_is_requeued_rather_than_retired() -> None:
    """The fix. `_claim_deferred_rooms` runs immediately after the sweep in the same tick,
    so the cost of a dropped connection becomes one sweep interval instead of the rest of
    the meeting."""
    worker = _worker(_room(connected=False))

    await worker._sweep_idle_rooms()

    assert ROOM in worker._deferred_rooms
    assert ROOM not in worker.rooms


async def test_a_lost_connection_does_not_wait_out_the_idle_grace() -> None:
    # It used to sit in `_room_last_occupied` for 120 seconds before anything happened, and
    # then be retired permanently. Neither half is right: nothing can be learned by waiting
    # on a connection we do not have.
    worker = _worker(_room(connected=False))

    await worker._sweep_idle_rooms()

    assert ROOM not in worker._room_last_occupied
    worker._release_room_ownership.assert_awaited_once_with(ROOM)


async def test_ownership_is_dropped_so_the_reclaim_path_can_take_it() -> None:
    # `_claim_deferred_rooms` gates on winning the claim. Holding a lease for a room we are
    # no longer connected to would block our own recovery for a full lease TTL.
    worker = _worker(_room(connected=False))

    await worker._sweep_idle_rooms()

    worker._release_room_ownership.assert_awaited_once()


async def test_an_occupied_room_is_left_alone() -> None:
    worker = _worker(_room(connected=True, humans=2))

    await worker._sweep_idle_rooms()

    assert ROOM in worker.rooms
    assert ROOM not in worker._deferred_rooms


async def test_a_genuinely_empty_room_still_waits_out_the_grace_window() -> None:
    """WT-314 must survive this change.

    A connected room with no humans is the leak that idle-release exists to stop, and it
    must NOT be requeued — re-dialling an empty room recreates exactly the bot that sits
    there billing connection minutes for the life of the process.
    """
    worker = _worker(_room(connected=True, humans=0), now=1000.0)
    # Seeded in the past, just inside the window. An OCCUPIED room would have this stamped
    # forward to now; an empty one must not, or the grace can never elapse and the WT-314
    # leak comes back. Asserting only "== now" cannot tell those two apart — it holds either
    # way — so the clock is what this pins.
    entered_grace_at = 1000.0 - _IDLE_ROOM_GRACE_S + 1
    worker._room_last_occupied[ROOM] = entered_grace_at

    await worker._sweep_idle_rooms()

    assert worker._room_last_occupied[ROOM] == entered_grace_at
    assert ROOM in worker.rooms
    assert ROOM not in worker._deferred_rooms
    worker._release_room_ownership.assert_not_awaited()


async def test_an_empty_room_past_the_grace_window_is_released_not_requeued() -> None:
    worker = _worker(_room(connected=True, humans=0), now=1000.0)
    worker._room_last_occupied[ROOM] = 1000.0 - _IDLE_ROOM_GRACE_S - 1
    released: list[str] = []
    worker._release_idle_room = AsyncMock(side_effect=released.append)  # type: ignore[method-assign]

    await worker._sweep_idle_rooms()

    assert ROOM not in worker._deferred_rooms


async def test_our_own_bots_do_not_keep_a_room_looking_occupied() -> None:
    # The TTS publisher's ai-interpreter-* participants are remote participants in this same
    # room. Counting them would make an empty room look busy forever, which is WT-314.
    room = _room(connected=True, humans=0)
    room.remote_participants = {"ai": MagicMock(identity="ai-interpreter-en-user1")}
    worker = _worker(room, now=1000.0)

    await worker._sweep_idle_rooms()

    assert worker._room_last_occupied[ROOM] == 1000.0
    assert ROOM not in worker._deferred_rooms
