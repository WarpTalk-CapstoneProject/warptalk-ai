"""Transcription and translation are separate features, so they need separate signals.

`translation_active` is the backend's own answer, computed from the room's active
TranslationRoomSession. Before it existed the workers read the room STATUS, and a room is
IN_PROGRESS from the moment it is opened — so "the meeting is live" and "translation is
running" were one value, and transcript-only was not expressible.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from shared.base_worker import BaseWorker


class _Worker(BaseWorker):
    worker_name = "signal"
    input_stream = "input"
    consumer_group = "signal-workers"

    async def load_model(self) -> None:
        return None

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        return None


def _make_worker() -> _Worker:
    worker = _Worker.__new__(_Worker)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._translation_active = {}
    worker._paused_rooms = set()
    worker._room_routes = {}
    return worker


def _event(room_id: str, **data: object) -> dict[str, object]:
    return {
        "type": "pmessage",
        "channel": f"translationRoom:{room_id}:events",
        "data": json.dumps({"type": "AUDIO_ROUTES_UPDATED", "data": data}),
    }


async def test_an_open_room_is_not_translating_until_somebody_starts_it() -> None:
    worker = _make_worker()

    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=False)
    )

    assert worker._is_translation_active("room-1") is False
    # ...and the room is emphatically not paused, which is what keeps the transcript running.
    assert "room-1" not in worker._paused_rooms


async def test_starting_translation_opens_the_gate() -> None:
    worker = _make_worker()

    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=True)
    )

    assert worker._is_translation_active("room-1") is True


async def test_stopping_translation_closes_the_gate_without_pausing_the_room() -> None:
    worker = _make_worker()
    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=True)
    )

    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=False)
    )

    assert worker._is_translation_active("room-1") is False
    assert "room-1" not in worker._paused_rooms


async def test_a_backend_that_does_not_send_the_flag_keeps_the_old_meaning() -> None:
    """A rolling deploy must not silence translation fleet-wide."""
    worker = _make_worker()

    await worker._handle_route_update_message(_event("room-1", room_status="IN_PROGRESS"))

    assert worker._is_translation_active("room-1") is True


async def test_a_room_nobody_has_reported_is_not_translating() -> None:
    assert _make_worker()._is_translation_active("never-heard-of-it") is False


async def test_pausing_still_stops_the_room_listening() -> None:
    worker = _make_worker()
    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=True)
    )

    await worker._handle_route_update_message(
        _event("room-1", room_status="PAUSED", translation_active=False)
    )

    assert "room-1" in worker._paused_rooms
    assert worker._is_translation_active("room-1") is False


async def test_a_finished_room_is_forgotten_entirely() -> None:
    worker = _make_worker()
    await worker._handle_route_update_message(
        _event("room-1", room_status="IN_PROGRESS", translation_active=True)
    )

    worker._cleanup_room("room-1")

    assert "room-1" not in worker._translation_active
    assert worker._is_translation_active("room-1") is False
