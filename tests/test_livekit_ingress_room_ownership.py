"""S1 — exactly one ingress bot per LiveKit room, across replicas.

meeting.track_published travels over Redis Pub/Sub, which fans out: EVERY replica receives
EVERY message. The bot identity is "AIBot_{room_name}" with no replica discriminator and
the chart runs replicas: 2, so both replicas dialled the same room under the same identity
and LiveKit resolved the collision by evicting one. The per-room asyncio.Lock and the
"reuse the live connection" check are per-process and cannot see the other replica at all.

These tests drive two workers against one FakeSharedRedis — the shape of the real
deployment — and pin: only one replica dials, the other stands down, ownership survives
renewal, and the room is picked back up if its owner goes away.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit_ingress_worker.worker import (
    _ROOM_OWNER_KEY_PREFIX,
    LiveKitIngressWorker,
)
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "019fd60a-e5f3-7342-804a-4366e3214786"
OWNER_KEY = f"{_ROOM_OWNER_KEY_PREFIX}{ROOM}"


def _event(room_name: str = ROOM, track_id: str = "TR_1") -> dict[str, Any]:
    return {
        "event_type": "meeting.track_published",
        "schema_version": 1,
        "producer": "meeting-service",
        "payload": {
            "room_name": room_name,
            "participant_identity": "user-123",
            "track_id": track_id,
        },
    }


@pytest.fixture
def mock_livekit_sdk():
    with (
        patch("livekit_ingress_worker.worker.rtc") as mock_rtc,
        patch("livekit_ingress_worker.worker.api") as mock_api,
    ):
        rooms: list[MagicMock] = []
        identities: list[str] = []

        def _new_room() -> MagicMock:
            room = MagicMock()
            room.connect = AsyncMock()
            room.disconnect = AsyncMock()
            room.isconnected.return_value = True
            room.remote_participants = {}
            rooms.append(room)
            return room

        mock_rtc.Room.side_effect = _new_room

        token_builder = MagicMock()

        def _with_identity(identity: str) -> MagicMock:
            identities.append(identity)
            return token_builder

        token_builder.with_identity.side_effect = _with_identity
        token_builder.with_name.return_value = token_builder
        token_builder.with_grants.return_value = token_builder
        token_builder.to_jwt.return_value = "fake-jwt"
        mock_api.AccessToken.return_value = token_builder

        yield {"rooms": rooms, "identities": identities}


def _replica(shared: FakeSharedRedis, name: str) -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = shared
    worker._consumer_name = f"livekit_ingress-{name}"
    return worker


class TestOneBotPerRoomAcrossReplicas:
    async def test_fanned_out_event_dials_livekit_exactly_once(self, mock_livekit_sdk) -> None:
        """THE BUG. Pub/Sub delivers the same event to both replicas.

        Before the fix both replicas ran _connect_room and both built an AccessToken with
        identity "AIBot_{room}" — two rtc.Room objects, two dials, one eviction.
        """
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")
        replica_b = _replica(shared, "replica-b")

        # One PUBLISH, delivered to every subscriber.
        await replica_a.handle_track_published(_event())
        await replica_b.handle_track_published(_event())

        assert len(mock_livekit_sdk["rooms"]) == 1, "two replicas must not both dial LiveKit"
        assert mock_livekit_sdk["identities"] == [f"AIBot_{ROOM}"]
        assert ROOM in replica_a.rooms
        assert ROOM not in replica_b.rooms

    async def test_loser_remembers_the_room_so_it_can_take_over(self, mock_livekit_sdk) -> None:
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")
        replica_b = _replica(shared, "replica-b")

        await replica_a.handle_track_published(_event())
        await replica_b.handle_track_published(_event())

        assert shared.values[OWNER_KEY] == "livekit_ingress-replica-a"
        assert replica_b._deferred_rooms == {ROOM}
        assert replica_b._owned_rooms == set()

    async def test_owner_keeps_the_room_through_repeated_events(self, mock_livekit_sdk) -> None:
        """Renewal must not be mistaken for a fresh claim by the room's own owner."""
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")

        await replica_a.handle_track_published(_event(track_id="TR_1"))
        await replica_a.handle_track_published(_event(track_id="TR_2"))
        await replica_a.handle_track_published(_event(track_id="TR_3"))

        assert len(mock_livekit_sdk["rooms"]) == 1
        assert replica_a._connects_total == 1
        assert shared.values[OWNER_KEY] == "livekit_ingress-replica-a"

    async def test_standby_takes_over_when_the_owner_stops_renewing(self, mock_livekit_sdk) -> None:
        """Electing an owner must not cost the failover eviction accidentally provided.

        The replica that lost the race used to reconnect and carry on. Here the owner's
        lease elapses (it died), and the standby must pick the room up on its next sweep.
        """
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")
        replica_b = _replica(shared, "replica-b")

        await replica_a.handle_track_published(_event())
        await replica_b.handle_track_published(_event())
        assert ROOM not in replica_b.rooms

        shared.expire_lease(OWNER_KEY)  # replica-a is gone; its TTL elapsed
        await replica_b._claim_deferred_rooms()

        assert ROOM in replica_b.rooms
        assert shared.values[OWNER_KEY] == "livekit_ingress-replica-b"
        assert len(mock_livekit_sdk["rooms"]) == 2  # a's, then b's — never concurrent

    async def test_owner_that_lost_its_lease_stands_down(self, mock_livekit_sdk) -> None:
        """Two owners is the state the lease exists to prevent, in either direction.

        If our claim was taken over while we still held the connection, the correct move is
        to drop the room — not to keep the connection and fight for the identity.
        """
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")

        await replica_a.handle_track_published(_event())
        room = replica_a.rooms[ROOM]

        shared.values[OWNER_KEY] = "livekit_ingress-replica-b"  # somebody else took over
        await replica_a._renew_room_ownership()

        assert ROOM not in replica_a.rooms
        room.disconnect.assert_awaited_once()
        assert replica_a._deferred_rooms == {ROOM}

    async def test_release_does_not_delete_another_replicas_claim(self, mock_livekit_sdk) -> None:
        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")
        replica_b = _replica(shared, "replica-b")

        await replica_a.handle_track_published(_event())
        await replica_b.handle_track_published(_event())

        await replica_b._release_room_ownership(ROOM)

        assert shared.values[OWNER_KEY] == "livekit_ingress-replica-a"

    async def test_ownership_is_declined_when_redis_is_unreachable(self, mock_livekit_sdk) -> None:
        """Fail closed: a bot that cannot reach Redis publishes no audio chunks anyway."""
        from redis.exceptions import ConnectionError as RedisConnectionError

        shared = FakeSharedRedis()
        replica_a = _replica(shared, "replica-a")
        replica_a.redis = MagicMock()
        replica_a.redis.get = AsyncMock(return_value=None)
        replica_a.redis.set_if_absent = AsyncMock(side_effect=RedisConnectionError("down"))

        await replica_a.handle_track_published(_event())

        assert mock_livekit_sdk["rooms"] == []
        assert ROOM not in replica_a.rooms
