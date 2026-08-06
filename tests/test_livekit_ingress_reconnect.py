"""WT-269 — the ingress worker must not answer a published track by rebuilding the room.

The worker used to disconnect() and join_room() the entire room on every
meeting.track_published, from unserialised tasks and with no backoff. LiveKit Cloud
rate-limited the project (HTTP 429 in every region) and no meeting could carry media.
These tests pin the corrected behaviour: reuse the live connection, serialise per room,
back off on failure, and treat 429 as "stop", not "retry".
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit_ingress_worker.worker import (
    _RATE_LIMITED_DELAY_S,
    _RECONNECT_JITTER_RATIO,
    _RECONNECT_STORM_THRESHOLD,
    LiveKitIngressWorker,
    _is_rate_limited_error,
)
from shared.config import LiveKitSettings, WorkerSettings

ROOM = "019f6a39-a32c-7745-886e-1fe622c1f747"
OTHER_ROOM = "019f6a39-a32c-7745-886e-000000000002"


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
    """Patch the LiveKit SDK the same way tests/test_livekit_publisher.py does."""
    with (
        patch("livekit_ingress_worker.worker.rtc") as mock_rtc,
        patch("livekit_ingress_worker.worker.api") as mock_api,
    ):
        rooms: list[MagicMock] = []

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
        token_builder.with_identity.return_value = token_builder
        token_builder.with_name.return_value = token_builder
        token_builder.with_grants.return_value = token_builder
        token_builder.to_jwt.return_value = "fake-jwt"
        mock_api.AccessToken.return_value = token_builder

        yield {"rtc": mock_rtc, "api": mock_api, "rooms": rooms}


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    # _hydrate_room_status is the only Redis touch on this path.
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(return_value=None)
    return worker


class TestTrackPublishedDoesNotRebuildTheRoom:
    async def test_first_event_connects_once(self, mock_livekit_sdk) -> None:
        worker = _worker()

        await worker.handle_track_published(_event())

        assert len(mock_livekit_sdk["rooms"]) == 1
        mock_livekit_sdk["rooms"][0].connect.assert_awaited_once_with(
            "ws://livekit:7880", "fake-jwt"
        )
        assert worker.rooms[ROOM] is mock_livekit_sdk["rooms"][0]

    async def test_second_track_reuses_the_live_connection(self, mock_livekit_sdk) -> None:
        """The bug, pinned: a new published track must NOT disconnect the room."""
        worker = _worker()

        await worker.handle_track_published(_event(track_id="TR_1"))
        await worker.handle_track_published(_event(track_id="TR_2"))
        await worker.handle_track_published(_event(track_id="TR_3"))

        assert len(mock_livekit_sdk["rooms"]) == 1, "must not build a second Room"
        room = mock_livekit_sdk["rooms"][0]
        room.connect.assert_awaited_once()
        room.disconnect.assert_not_awaited()
        assert worker._connects_total == 1

    async def test_concurrent_events_do_not_interleave_teardown_and_rejoin(
        self, mock_livekit_sdk
    ) -> None:
        """Unserialised handlers used to dial twice with the same AIBot_{room} identity."""
        worker = _worker()

        connect_started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_connect(*_args: Any, **_kwargs: Any) -> None:
            connect_started.set()
            await release.wait()

        original_new_room = mock_livekit_sdk["rtc"].Room.side_effect

        def _new_slow_room() -> MagicMock:
            room = original_new_room()
            room.connect = AsyncMock(side_effect=_slow_connect)
            return room

        mock_livekit_sdk["rtc"].Room.side_effect = _new_slow_room

        first = asyncio.create_task(worker.handle_track_published(_event(track_id="TR_1")))
        await connect_started.wait()
        second = asyncio.create_task(worker.handle_track_published(_event(track_id="TR_2")))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)

        assert worker._connects_total == 1
        assert len(mock_livekit_sdk["rooms"]) == 1

    async def test_genuine_disconnection_does_rejoin(self, mock_livekit_sdk) -> None:
        worker = _worker()

        await worker.handle_track_published(_event())
        stale = mock_livekit_sdk["rooms"][0]
        stale.isconnected.return_value = False

        await worker.handle_track_published(_event(track_id="TR_2"))

        stale.disconnect.assert_awaited_once()
        assert len(mock_livekit_sdk["rooms"]) == 2
        assert worker.rooms[ROOM] is mock_livekit_sdk["rooms"][1]
        assert worker._connects_total == 2


class TestAudioTaskIsolation:
    async def test_rejoining_one_room_leaves_another_rooms_pipelines_alone(
        self, mock_livekit_sdk
    ) -> None:
        """The old teardown cleared self.audio_tasks wholesale, killing every room."""
        worker = _worker()

        async def _never() -> None:
            await asyncio.Event().wait()

        other_task = asyncio.create_task(_never())
        worker.audio_tasks[(OTHER_ROOM, "TR_OTHER")] = other_task
        doomed = asyncio.create_task(_never())
        worker.audio_tasks[(ROOM, "TR_MINE")] = doomed

        await worker.handle_track_published(_event())
        mock_livekit_sdk["rooms"][0].isconnected.return_value = False
        await worker.handle_track_published(_event(track_id="TR_2"))
        await asyncio.sleep(0)

        assert doomed.cancelled() or doomed.cancelling()
        assert not other_task.done(), "another room's audio pipeline must survive"
        assert (OTHER_ROOM, "TR_OTHER") in worker.audio_tasks

        other_task.cancel()


class TestBackoff:
    async def test_rate_limit_detection(self) -> None:
        assert _is_rate_limited_error(Exception("server rejected: 429 Too Many Requests"))
        assert _is_rate_limited_error(Exception("rate limit exceeded"))
        assert not _is_rate_limited_error(Exception("dns failure"))

    async def test_429_stops_dialling_instead_of_retrying(self, mock_livekit_sdk) -> None:
        worker = _worker()
        mock_livekit_sdk["rtc"].Room.side_effect = None
        failing = MagicMock()
        failing.connect = AsyncMock(side_effect=Exception("connect failed: 429 Too Many Requests"))
        failing.disconnect = AsyncMock()
        failing.isconnected.return_value = False
        failing.remote_participants = {}
        mock_livekit_sdk["rtc"].Room.return_value = failing

        await worker.handle_track_published(_event(track_id="TR_1"))
        await worker.handle_track_published(_event(track_id="TR_2"))
        await worker.handle_track_published(_event(track_id="TR_3"))

        assert failing.connect.await_count == 1, "a 429 must not be retried immediately"
        assert ROOM not in worker.rooms
        held_off_for = worker._connect_not_before[ROOM] - asyncio.get_running_loop().time()
        assert held_off_for >= _RATE_LIMITED_DELAY_S * (1 - _RECONNECT_JITTER_RATIO) - 1

    async def test_ordinary_failure_backs_off_exponentially(self, mock_livekit_sdk) -> None:
        worker = _worker()
        mock_livekit_sdk["rtc"].Room.side_effect = None
        failing = MagicMock()
        failing.connect = AsyncMock(side_effect=Exception("dns failure"))
        failing.disconnect = AsyncMock()
        failing.isconnected.return_value = False
        failing.remote_participants = {}
        mock_livekit_sdk["rtc"].Room.return_value = failing

        delays: list[float] = []
        for _ in range(4):
            worker._connect_not_before.pop(ROOM, None)
            await worker._connect_room(ROOM)
            delays.append(worker._connect_not_before[ROOM] - asyncio.get_running_loop().time())

        assert worker._connect_failures[ROOM] == 4
        # Jitter is ±25%, so consecutive doublings still have to be strictly increasing.
        assert delays[0] < delays[1] < delays[2] < delays[3]

    async def test_successful_connect_clears_the_backoff(self, mock_livekit_sdk) -> None:
        worker = _worker()
        worker._connect_failures[ROOM] = 3
        worker._connect_not_before[ROOM] = 0.0

        await worker._connect_room(ROOM)

        assert ROOM not in worker._connect_failures
        assert ROOM not in worker._connect_not_before


class TestReconnectVisibility:
    async def test_storm_is_reported_at_error_level(self, mock_livekit_sdk) -> None:
        worker = _worker()
        worker.logger = MagicMock()

        for _ in range(_RECONNECT_STORM_THRESHOLD):
            worker._record_connect_attempt(ROOM)

        logged = [call.args[0] for call in worker.logger.error.call_args_list]
        assert "livekit_reconnect_storm_suspected" in logged
        attempts = [call.args[0] for call in worker.logger.info.call_args_list]
        assert attempts.count("livekit_room_connect_attempt") == _RECONNECT_STORM_THRESHOLD

    async def test_reused_connection_is_logged_as_such(self, mock_livekit_sdk) -> None:
        worker = _worker()
        await worker.handle_track_published(_event(track_id="TR_1"))
        worker.logger = MagicMock()

        await worker.handle_track_published(_event(track_id="TR_2"))

        logged = [call.args[0] for call in worker.logger.info.call_args_list]
        assert "track_published_reusing_connection" in logged
