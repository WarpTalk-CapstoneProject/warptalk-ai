"""A room the ingress bot could not (re)join must stay in the re-dial queue.

PRODUCTION, 2026-10-03, ROOM 01a1010c (Google Meet bridge)
    At 09:23:36 the host's client did a LiveKit full reconnect. From about then on NOTHING in the
    room was transcribed — not the host, not the far side — in the popup or the main window, and
    it stayed that way after the host reloaded and re-published their microphone.

THE GAP THESE PIN
    Every path that dials a room does it right after WINNING the room's claim, and winning the
    claim takes the room out of `_deferred_rooms`. `_connect_room` then returned without a
    connection in two ways — a backoff from an earlier failure still running, or the dial itself
    failing — and in both the room was left in neither `rooms` nor `_deferred_rooms`. The sweep
    renews only rooms it holds and re-dials only rooms it has deferred, so this process never
    tried that room again; only a fresh `meeting.track_published` could bring it back, and a
    re-publish that lands inside the backoff window is spent on the same dead end.

    A lost connection is requeued by the sweep (WT-395) — and the very first re-dial after a
    LiveKit hiccup is the one most likely to fail.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit_ingress_worker.worker import _ROOM_OWNER_KEY_PREFIX, LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a1010c-62db-7d7a-9a94-3bdbebfe3a3b"
OWNER_KEY = f"{_ROOM_OWNER_KEY_PREFIX}{ROOM}"

pytestmark = pytest.mark.asyncio


def _event(track_id: str = "TR_1") -> dict[str, Any]:
    return {
        "event_type": "meeting.track_published",
        "schema_version": 1,
        "producer": "meeting-service",
        "payload": {"room_name": ROOM, "participant_identity": "host", "track_id": track_id},
    }


@pytest.fixture
def livekit():
    """A LiveKit whose dials fail while `state["fail"]` holds an exception."""
    with (
        patch("livekit_ingress_worker.worker.rtc") as mock_rtc,
        patch("livekit_ingress_worker.worker.api") as mock_api,
    ):
        state: dict[str, Any] = {"fail": None, "dials": 0}

        def _new_room() -> MagicMock:
            room = MagicMock()

            async def _connect(*_args: Any) -> None:
                state["dials"] += 1
                if state["fail"] is not None:
                    raise state["fail"]

            room.connect = AsyncMock(side_effect=_connect)
            room.disconnect = AsyncMock()
            room.isconnected.return_value = True
            room.remote_participants = {}
            return room

        mock_rtc.Room.side_effect = _new_room
        token = MagicMock()
        token.with_identity.return_value = token
        token.with_name.return_value = token
        token.with_grants.return_value = token
        token.to_jwt.return_value = "fake-jwt"
        mock_api.AccessToken.return_value = token
        yield state


def _replica(shared: FakeSharedRedis, name: str = "replica-a") -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = shared
    worker._consumer_name = f"livekit_ingress-{name}"
    # The sweep loop itself is not under test; only that one is asked for.
    worker._ensure_idle_sweeper = MagicMock()  # type: ignore[method-assign]
    return worker


def _clear_backoff(worker: LiveKitIngressWorker) -> None:
    worker._connect_not_before[ROOM] = asyncio.get_running_loop().time() - 1


async def test_a_failed_dial_leaves_the_room_queued_for_the_sweep(livekit) -> None:
    worker = _replica(FakeSharedRedis())
    livekit["fail"] = Exception("connection reset by peer")

    await worker.handle_track_published(_event())

    assert ROOM not in worker.rooms
    assert ROOM in worker._deferred_rooms, "nothing in this process would ever dial it again"


async def test_the_sweep_rejoins_once_the_backoff_has_run_out(livekit) -> None:
    worker = _replica(FakeSharedRedis())
    livekit["fail"] = Exception("connection reset by peer")
    await worker.handle_track_published(_event())

    livekit["fail"] = None
    _clear_backoff(worker)
    await worker._claim_deferred_rooms()

    assert ROOM in worker.rooms
    assert ROOM not in worker._deferred_rooms
    assert livekit["dials"] == 2


async def test_a_claim_inside_the_backoff_does_not_dial_and_does_not_drop_the_room(
    livekit,
) -> None:
    worker = _replica(FakeSharedRedis())
    livekit["fail"] = Exception("connection reset by peer")
    await worker.handle_track_published(_event())

    # The next sweep, still inside the backoff: no dial, and the room is still queued.
    await worker._claim_deferred_rooms()
    # A re-publish (the host reloading) inside the same window: also no dial, also kept.
    await worker.handle_track_published(_event(track_id="TR_2"))

    assert livekit["dials"] == 1
    assert ROOM in worker._deferred_rooms


async def test_the_lost_connection_path_survives_a_failed_first_redial(livekit) -> None:
    """WT-395 requeues a dropped connection; this is the redial after it failing once."""
    worker = _replica(FakeSharedRedis())
    await worker.handle_track_published(_event())
    worker.rooms[ROOM].isconnected.return_value = False  # the LiveKit hiccup

    livekit["fail"] = Exception("signal connection failed")
    await worker._sweep_idle_rooms()
    await worker._claim_deferred_rooms()
    assert ROOM not in worker.rooms
    assert ROOM in worker._deferred_rooms

    livekit["fail"] = None
    _clear_backoff(worker)
    await worker._claim_deferred_rooms()

    assert ROOM in worker.rooms


async def test_the_claim_is_kept_so_a_rate_limited_room_is_not_handed_to_the_other_replica(
    livekit,
) -> None:
    # Releasing the claim on failure would let the standby dial at once, with no knowledge of
    # this replica's backoff — the WT-269 storm with one replica taken out of it.
    shared = FakeSharedRedis()
    replica_a = _replica(shared, "replica-a")
    replica_b = _replica(shared, "replica-b")
    livekit["fail"] = Exception("connect failed: 429 Too Many Requests")

    await replica_a.handle_track_published(_event())
    await replica_b.handle_track_published(_event())
    await replica_a._claim_deferred_rooms()
    await replica_b._claim_deferred_rooms()

    assert livekit["dials"] == 1
    assert shared.values[OWNER_KEY] == "livekit_ingress-replica-a"
    assert ROOM in replica_a._deferred_rooms


async def test_a_finished_meeting_is_not_requeued(livekit) -> None:
    worker = _replica(FakeSharedRedis())
    worker._route_states[ROOM] = "ENDED"
    livekit["fail"] = Exception("room not found")

    await worker._connect_room(ROOM)

    assert ROOM not in worker._deferred_rooms


async def test_the_requeue_is_logged(livekit) -> None:
    worker = _replica(FakeSharedRedis())
    worker.logger = MagicMock()
    livekit["fail"] = Exception("connection reset by peer")

    await worker.handle_track_published(_event())

    events = [call.args[0] for call in worker.logger.warning.call_args_list]
    assert "livekit_room_connect_requeued" in events
