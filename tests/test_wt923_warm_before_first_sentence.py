"""WT-923 — the first sentence of a meeting must not pay for the warm-up.

Everything that gets ready for a speaker — the ingress bot's LiveKit connection and the STT
Realtime socket — was summoned by meeting.track_published, i.e. by the first microphone. A
person who joins muted publishes no microphone until they unmute, and they unmute to speak, so
the warm-up ran at the same moment as the first sentence and the host's transcript stayed empty
for it. meeting.participant_joined now summons both when the person connects.

The warm pool behind the STT socket also aged out as a block after a quiet 50 minutes, so the
next speaker paid the handshake anyway. It is rotated ahead of the cap now.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from livekit_ingress_worker.worker import (
    LiveKitIngressWorker,
    _parse_participant_joined_event,
)
from shared.config import LiveKitSettings, WorkerSettings
from stt_worker.model import WARM_SOCKET_ROTATE_AGE_S, OpenAISTT
from stt_worker.worker import STTWorker
from tests.conftest import FakeSharedRedis

ROOM = "019fd60a-e5f3-7342-804a-4366e3214786"


def _joined(identity: str = "user-123", room_name: str = ROOM) -> dict[str, Any]:
    return {
        "event_type": "meeting.participant_joined",
        "schema_version": 1,
        "producer": "meeting-service",
        "payload": {
            "room_name": room_name,
            "participant_identity": identity,
            "joined_at": "2026-10-02T07:00:00Z",
        },
    }


def _published(identity: str = "user-123") -> dict[str, Any]:
    return {
        "event_type": "meeting.track_published",
        "schema_version": 1,
        "producer": "meeting-service",
        "payload": {"room_name": ROOM, "participant_identity": identity, "track_id": "TR_1"},
    }


class _FakePubSub:
    """Hands out queued messages, then stops the worker's loop."""

    def __init__(self, shutdown: asyncio.Event, messages: list[dict[str, Any]]) -> None:
        self._shutdown = shutdown
        self._messages = list(messages)
        self.channels: tuple[str, ...] = ()

    async def subscribe(self, *channels: str) -> None:
        self.channels = channels

    async def get_message(self, **_: Any) -> dict[str, Any] | None:
        if self._messages:
            return self._messages.pop(0)
        self._shutdown.set()
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def mock_livekit_sdk():
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
        yield rooms


def _ingress() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    worker._consumer_name = "livekit_ingress-a"
    return worker


class TestIngressJoinsWithTheFirstPerson:
    async def test_a_join_dials_the_room_before_any_microphone(self, mock_livekit_sdk) -> None:
        worker = _ingress()

        await worker.handle_participant_joined(_joined())

        assert len(mock_livekit_sdk) == 1
        assert ROOM in worker.rooms

    async def test_the_later_unmute_reuses_that_connection(self, mock_livekit_sdk) -> None:
        """The whole point: the first microphone must not start a second dial."""
        worker = _ingress()

        await worker.handle_participant_joined(_joined())
        await worker.handle_track_published(_published())

        assert len(mock_livekit_sdk) == 1

    @pytest.mark.parametrize("identity", [f"AIBot_{ROOM}", "ai-interpreter-vi"])
    async def test_our_own_bots_joining_summon_nothing(self, mock_livekit_sdk, identity) -> None:
        worker = _ingress()

        await worker.handle_participant_joined(_joined(identity))

        assert mock_livekit_sdk == []
        assert worker.rooms == {}

    def test_parser_rejects_a_join_without_an_identity(self) -> None:
        event = _joined()
        event["payload"]["participant_identity"] = None
        assert _parse_participant_joined_event(event) is None
        assert _parse_participant_joined_event(_published()) is None
        assert _parse_participant_joined_event(_joined()) == (ROOM, "user-123")

    async def test_the_listener_routes_each_channel_to_its_handler(self) -> None:
        """Wiring, not just a handler: a handler nobody subscribes to is a fix that never runs."""
        worker = _ingress()
        pubsub = _FakePubSub(
            worker._shutdown_event,
            [
                {"channel": b"meeting.participant_joined", "data": json.dumps(_joined())},
                {"channel": b"meeting.track_published", "data": json.dumps(_published())},
            ],
        )
        worker.redis = MagicMock()
        worker.redis.redis.pubsub.return_value = pubsub
        worker._rediscover_active_rooms = AsyncMock(return_value=0)
        worker.handle_participant_joined = AsyncMock()
        worker.handle_track_published = AsyncMock()

        await worker._consume_loop()
        await asyncio.gather(*list(worker._event_tasks))

        assert set(pubsub.channels) == {"meeting.track_published", "meeting.participant_joined"}
        worker.handle_participant_joined.assert_awaited_once_with(_joined())
        worker.handle_track_published.assert_awaited_once_with(_published())


def _stt_worker(mock_redis_client, worker_settings: WorkerSettings) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker._stt_prompts = {}
    worker._stt_keywords = {}
    worker._room_languages = {}
    worker.model = MagicMock()
    worker.model.prepare_session = AsyncMock()
    return worker


class TestSttPrewarmsOnJoin:
    async def test_a_join_prepares_the_speakers_socket(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hget.return_value = b"vi"
        mock_redis_client._redis.hgetall.return_value = {b"user-123": b"vi"}
        mock_redis_client._redis.get.return_value = None

        await worker._prewarm_from_track_event(json.dumps(_joined()))

        worker.model.prepare_session.assert_awaited_once()
        assert worker.model.prepare_session.await_args.args == (ROOM, "user-123")
        assert worker.model.prepare_session.await_args.kwargs["language"] == "vi"

    async def test_the_listener_subscribes_to_joins(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker._shutdown_event = asyncio.Event()
        pubsub = _FakePubSub(worker._shutdown_event, [])
        worker.redis = MagicMock()
        worker.redis.redis.pubsub.return_value = pubsub

        await worker._listen_for_track_prewarm()

        assert "meeting.participant_joined" in pubsub.channels
        assert "meeting.track_published" in pubsub.channels


class TestWarmPoolRotation:
    def _model(self, ages: list[float], now: float) -> OpenAISTT:
        model = OpenAISTT.__new__(OpenAISTT)
        model._warm_sessions = deque({"opened_at": now - age, "age": age} for age in ages)
        model._close_session = AsyncMock()
        model._schedule_warm_refill = MagicMock()
        return model

    async def test_sockets_near_the_cap_are_replaced_before_anyone_claims_them(self) -> None:
        now = 10_000.0
        young, old = 60.0, WARM_SOCKET_ROTATE_AGE_S + 1
        model = self._model([old, young, old, young], now)

        retired = await model.rotate_warm_pool(now=now)
        await asyncio.sleep(0)

        assert retired == 2
        assert [s["age"] for s in model._warm_sessions] == [young, young]
        assert model._close_session.await_count == 2
        model._schedule_warm_refill.assert_called_once()

    async def test_an_unstamped_socket_counts_as_old(self) -> None:
        model = self._model([], 0.0)
        model._warm_sessions.append({"opened_at": None})

        assert await model.rotate_warm_pool(now=0.0) == 1
        assert not model._warm_sessions

    async def test_a_short_pool_is_refilled_even_with_nothing_to_retire(self) -> None:
        """A refill that stopped on a provider error used to wait for the next claim."""
        model = self._model([], 0.0)

        assert await model.rotate_warm_pool(now=0.0) == 0
        model._schedule_warm_refill.assert_called_once()
