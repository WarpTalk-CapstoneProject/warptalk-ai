"""A restarted ingress worker must pick up meetings that are already running.

THE INCIDENT (2026-08-15, 12:17:07 local)
    The cgroup OOM killer took livekit-ingress-worker mid-meeting, two people talking. It
    restarted eight seconds later, opened its listeners — and then did nothing at all for
    FOUR MINUTES AND NINETEEN SECONDS while the meeting continued without it. No transcript,
    no translation, no dub. It only rejoined at 12:21:44, when somebody happened to publish
    a new track.

WHY IT COULD NOT RECOVER ON ITS OWN
    `meeting.track_published` arrives over Redis Pub/Sub, and pub/sub has no replay. Every
    participant in that room had already published before the crash, so the event that would
    have summoned the worker back was gone. In a settled meeting it never comes again.

    `_deferred_rooms` already exists for the neighbouring case (another replica dying) and
    the sweeper already retries the claim for everything in it — but it is an in-memory set,
    so the one replica that most needs to recover its rooms is the one that just lost the
    record of them.

The durable record needed no new publisher: the backend already writes
`translationRoom:{id}:audio_routes` with `room_status` on every route change.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from livekit_ingress_worker.worker import LiveKitIngressWorker

LIVE = "01a003d5-f7fd-7764-83fc-f678867535c1"
OTHER_LIVE = "01a003d0-0e1c-7a97-bad7-f471d356ffe5"
FINISHED = "01a0033f-d993-734f-a2ff-0dfc82194626"


class _FakeRedis:
    """Only the two calls _rediscover_active_rooms makes."""

    def __init__(self, snapshots: dict[str, dict[str, Any]], *, scan_fails: bool = False) -> None:
        self._snapshots = snapshots
        self._scan_fails = scan_fails
        self.scanned: list[str] = []

    async def scan_keys(self, match: str, count: int = 200) -> list[str]:
        if self._scan_fails:
            raise ConnectionError("redis is down")
        self.scanned.append(match)
        return [f"translationRoom:{room}:audio_routes" for room in self._snapshots]

    async def get(self, key: str) -> str | None:
        room = key.split(":")[1]
        snapshot = self._snapshots.get(room)
        return json.dumps(snapshot) if snapshot is not None else None


def _worker(snapshots: dict[str, dict[str, Any]], **kwargs: Any) -> LiveKitIngressWorker:
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker.redis = _FakeRedis(snapshots, **kwargs)  # type: ignore[assignment]
    worker._deferred_rooms = set()
    worker._route_states = {}
    worker._idle_sweeper = None

    claimed: list[str] = []

    async def _claim_deferred_rooms() -> None:
        claimed.extend(sorted(worker._deferred_rooms))

    worker._claim_deferred_rooms = _claim_deferred_rooms  # type: ignore[method-assign]
    worker._ensure_idle_sweeper = lambda: None  # type: ignore[method-assign]
    worker.claimed = claimed  # type: ignore[attr-defined]
    return worker


def _snapshot(status: str) -> dict[str, Any]:
    return {"routes": [], "room_status": status, "translation_active": False}


@pytest.mark.asyncio
async def test_a_meeting_already_in_progress_is_reclaimed_after_a_restart() -> None:
    """The incident, in one assertion. Before this, the set was empty and stayed empty."""
    worker = _worker({LIVE: _snapshot("IN_PROGRESS")})

    recovered = await worker._rediscover_active_rooms()

    assert recovered == 1
    assert LIVE in worker._deferred_rooms, (
        "A meeting that was in progress across the restart was not picked back up. This is "
        "the four minutes of dead air: nothing else will summon this worker, because every "
        "participant had already published before it died."
    )


@pytest.mark.asyncio
async def test_the_recovered_rooms_are_claimed_immediately_not_at_the_next_sweep() -> None:
    """Seeding the set is not the point — connecting is. Waiting for the sweep would hand
    back a slice of the very gap this closes."""
    worker = _worker({LIVE: _snapshot("AUDIO_ROUTING_ACTIVE")})

    await worker._rediscover_active_rooms()

    assert worker.claimed == [LIVE], f"the seeded room was never claimed; got {worker.claimed}"


@pytest.mark.asyncio
async def test_a_finished_meeting_is_not_dialled_back_into() -> None:
    """The snapshot key outlives the meeting by its TTL (12h). Without a live-status filter,
    every restart would reconnect the bot to every meeting that ended today — billable
    connection minutes for rooms with nobody in them, which is WT-314 by another door."""
    worker = _worker(
        {
            LIVE: _snapshot("IN_PROGRESS"),
            FINISHED: _snapshot("ENDED"),
        }
    )

    recovered = await worker._rediscover_active_rooms()

    assert recovered == 1
    assert LIVE in worker._deferred_rooms
    assert FINISHED not in worker._deferred_rooms, "the bot would have rejoined a dead meeting"


@pytest.mark.asyncio
async def test_an_unrecognised_status_is_left_alone() -> None:
    """An allow-list, not a deny-list. A status this worker has never heard of must not be
    dialled into on the strength of not being one of the two we happen to exclude."""
    worker = _worker({LIVE: _snapshot("SOME_FUTURE_STATE")})

    assert await worker._rediscover_active_rooms() == 0
    assert not worker._deferred_rooms


@pytest.mark.asyncio
async def test_several_live_meetings_are_all_recovered() -> None:
    worker = _worker(
        {
            LIVE: _snapshot("IN_PROGRESS"),
            OTHER_LIVE: _snapshot("IN_PROGRESS"),
            FINISHED: _snapshot("ENDED"),
        }
    )

    assert await worker._rediscover_active_rooms() == 2
    assert worker._deferred_rooms == {LIVE, OTHER_LIVE}


@pytest.mark.asyncio
async def test_redis_being_unreachable_does_not_stop_the_worker_starting() -> None:
    """Recovery is best effort. A worker that refuses to start because it could not
    enumerate is strictly worse than one that starts and misses the recovery — it turns a
    missed reconnect into an outage."""
    worker = _worker({LIVE: _snapshot("IN_PROGRESS")}, scan_fails=True)

    assert await worker._rediscover_active_rooms() == 0
    assert not worker._deferred_rooms


@pytest.mark.asyncio
async def test_the_recovered_status_is_remembered_not_just_the_room() -> None:
    """`_route_states` is what the rest of the worker reads for lifecycle decisions, and it
    is just as empty after a restart as `_deferred_rooms` was."""
    worker = _worker({LIVE: _snapshot("IN_PROGRESS")})

    await worker._rediscover_active_rooms()

    assert worker._route_states[LIVE] == "IN_PROGRESS"
