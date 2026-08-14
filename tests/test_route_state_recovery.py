"""Recovering a room's route state when no broadcast was ever heard for it.

WT-373. Every worker learned `translation_active` from one Redis pub/sub event and nothing else.
Pub/sub has no replay, and a deploy restarts every worker, so a worker that came back while a
meeting was already translating never learned that it was — and `_is_translation_active` answers
a room it has never heard of with False. In translation_worker that False is the gate that
discards a paid-for STT result, behind a `logger.debug` that production's INFO level never prints.

The failure mode is therefore: dub stops for every meeting that was live across a deploy, silently,
until the room ends. These tests pin the recovery path and — as much as anything — pin that an
unknown room is a question, not an answer.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from shared.base_worker import BaseWorker


class _Worker(BaseWorker):
    worker_name = "probe"
    input_stream = "input"
    consumer_group = "probe-workers"

    async def load_model(self) -> None:
        return None

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        return None


def _worker(snapshot: object = None, *, raises: bool = False) -> _Worker:
    """A worker built the way the suite builds them: `__new__`, no `__init__`."""
    worker = _Worker.__new__(_Worker)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._translation_active = {}
    worker._room_routes = {}
    worker._paused_rooms = set()
    worker.redis = MagicMock()
    if raises:
        worker.redis.get = AsyncMock(side_effect=RuntimeError("redis is down"))
    else:
        payload = None if snapshot is None else json.dumps(snapshot).encode()
        worker.redis.get = AsyncMock(return_value=payload)
    return worker


ROOM = "019fff06-2b98-7e1d-a923-1f53d10b455a"

# The shape the backend really writes, verbatim from production on 2026-08-14.
LIVE_SNAPSHOT = {
    "routes": [{"Id": "r1", "SourceLanguage": "vi", "TargetLanguage": "en"}],
    "version": 639222870647560901,
    "generated_at": "2026-08-14T06:51:04.7560903Z",
    "room_status": "IN_PROGRESS",
    "translation_active": True,
    "room_languages": ["en", "vi"],
}


async def test_a_worker_that_missed_the_broadcast_recovers_the_answer_from_redis() -> None:
    """The WT-373 case. Restarted mid-meeting, no event ever seen, room is translating."""
    worker = _worker(LIVE_SNAPSHOT)

    assert worker._is_translation_active(ROOM) is False, "precondition: nothing in memory"
    assert await worker._translation_active_for(ROOM) is True
    worker.redis.get.assert_awaited_once_with(f"translationRoom:{ROOM}:audio_routes")


async def test_recovery_restores_the_routes_and_status_too_not_only_the_flag() -> None:
    # `_room_routes` feeds voice-clone lookups and `_route_states` feeds the pause gate. Restoring
    # one field of three would trade a silent pipeline for a subtly wrong one.
    worker = _worker(LIVE_SNAPSHOT)

    await worker._translation_active_for(ROOM)

    assert worker._room_routes[ROOM] == LIVE_SNAPSHOT["routes"]
    assert worker._route_states[ROOM] == "IN_PROGRESS"


async def test_a_paused_room_is_still_paused_after_recovery() -> None:
    worker = _worker({**LIVE_SNAPSHOT, "room_status": "PAUSED"})

    await worker._translation_active_for(ROOM)

    assert ROOM in worker._paused_rooms


async def test_a_room_that_really_is_not_translating_stays_off() -> None:
    """Recovery must not become "translate everything". The production snapshot for the WT-373
    room said `translation_active: False` with the room IN_PROGRESS — a live meeting nobody had
    pressed Start Translation on — and that is a correct reason to skip."""
    worker = _worker({**LIVE_SNAPSHOT, "translation_active": False})

    assert await worker._translation_active_for(ROOM) is False


async def test_a_broadcast_already_in_memory_is_not_re_fetched() -> None:
    # The hot path. This runs per STT result; a Redis GET per utterance would be a real cost for
    # an answer the worker already has.
    worker = _worker(LIVE_SNAPSHOT)
    worker._translation_active[ROOM] = True

    assert await worker._translation_active_for(ROOM) is True
    worker.redis.get.assert_not_awaited()


async def test_a_fresher_broadcast_wins_over_the_recovered_snapshot() -> None:
    # Stop Translation arrives as an event. If the snapshot could overwrite it the room would
    # resume translating on the next utterance.
    worker = _worker(LIVE_SNAPSHOT)
    await worker._translation_active_for(ROOM)
    assert worker._translation_active[ROOM] is True

    await worker._handle_route_update_message(
        {
            "type": "pmessage",
            "channel": f"translationRoom:{ROOM}:events".encode(),
            "data": json.dumps(
                {
                    "type": "AUDIO_ROUTES_UPDATED",
                    "status": "IN_PROGRESS",
                    "data": {"routes": [], "translation_active": False},
                }
            ),
        }
    )

    assert await worker._translation_active_for(ROOM) is False


async def test_no_snapshot_falls_back_to_the_old_answer_rather_than_failing() -> None:
    # A room with no key at all — already ended, or never routed. The pre-existing status-based
    # answer is deliberately unchanged here.
    worker = _worker(None)
    worker._route_states[ROOM] = "IN_PROGRESS"

    assert await worker._translation_active_for(ROOM) is True


async def test_redis_being_down_does_not_take_the_worker_with_it() -> None:
    worker = _worker(raises=True)

    assert await worker._translation_active_for(ROOM) is False
    worker.logger.warning.assert_called()


async def test_a_corrupt_snapshot_is_ignored_rather_than_raised() -> None:
    worker = _worker(None)
    worker.redis.get = AsyncMock(return_value=b"{not json")

    assert await worker._translation_active_for(ROOM) is False
    worker.logger.warning.assert_called()


async def test_a_snapshot_that_is_not_an_object_is_ignored() -> None:
    worker = _worker([1, 2, 3])

    assert await worker._translation_active_for(ROOM) is False
