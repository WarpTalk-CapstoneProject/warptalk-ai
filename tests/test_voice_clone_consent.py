"""Why nobody's voice was ever cloned in production, and why nothing said so.

THE OBSERVATION (2026-08-14)
    A tester: the dubbed voice "giọng AI lắm" — it does not sound like the speaker. Production
    Redis agreed, and was blunter than the report:

        97 of 97 recent tts:results segments   voice_type=default, voice_mode=default,
                                               clone_strength=0.0, clone_provider=""
        voice:* keys in Redis                  0

    There was no clone to be a poor likeness of. Every dub was a stock Cartesia catalog voice
    picked by hashing the speaker id. Tuning clone_strength or the model would have changed
    nothing, because the cloning branch was never entered.

TWO DEFECTS, ONE VISIBLE
    1. `is_voice_clone_consented` reads `_room_routes`, which is populated ONLY by the
       AUDIO_ROUTES_UPDATED pub/sub broadcast. Pub/sub has no replay, and every deploy restarts
       every worker — so a worker that comes up mid-meeting never learns that room's routes and
       answers "no consent" for every speaker until the room ends. This is the same asymmetry
       `_translation_active_for` was added to close for the translation gate (WT-373); the
       consent gate reads the same cache and never got the same treatment.

    2. Every exit before the clone call returned in silence. "Nobody opted in" and "this worker
       was never told anything about this room" produced identical output: none. That is why the
       first defect survived to be found by a tester rather than by a log.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from shared.base_worker import BaseWorker

ROOM = "room-1"
SPEAKER = "019f0d00-0de0-7000-9000-000000000001"


class _Worker(BaseWorker):
    """The smallest concrete BaseWorker. The two abstract methods are never called here — the
    consent gate touches no I/O beyond the route cache and the snapshot read."""

    worker_name = "test"
    input_stream = "test:in"
    consumer_group = "test-workers"

    async def load_model(self) -> None: ...

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None: ...


def _worker(routes: dict[str, list[dict[str, Any]]], snapshot: dict | None = None) -> BaseWorker:
    """Built with __new__ because BaseWorker.__init__ wants a Redis connection and a model; the
    pattern is used throughout this suite."""
    worker: BaseWorker = _Worker.__new__(_Worker)
    worker._room_routes = routes  # type: ignore[attr-defined]

    async def _load(room_id: str) -> bool:
        if snapshot is None:
            return False
        routes[room_id] = snapshot.get("routes", [])
        return True

    worker._load_route_snapshot = _load  # type: ignore[assignment]
    return worker


def _route(enabled: bool, user_id: str = SPEAKER) -> dict[str, Any]:
    return {"SourceUserId": user_id, "VoiceCloneEnabled": enabled}


pytestmark = pytest.mark.asyncio


async def test_an_opted_in_speaker_is_consented() -> None:
    consented, reason = await _worker({ROOM: [_route(True)]}).voice_clone_consent_state(
        ROOM, SPEAKER
    )

    assert (consented, reason) == (True, "consented")


async def test_a_known_room_where_nobody_opted_in_says_so() -> None:
    consented, reason = await _worker({ROOM: [_route(False)]}).voice_clone_consent_state(
        ROOM, SPEAKER
    )

    assert (consented, reason) == (False, "not_opted_in")


async def test_an_unknown_room_is_reported_as_unknown_not_as_an_opt_out() -> None:
    """The distinction is the whole point.

    Both answers refuse to clone, and must — cloning a voice without a confirmed opt-in is
    capturing biometric data. But one is a user's choice and the other is a worker that missed a
    broadcast, and they need different fixes. Reporting them identically is what hid this.
    """
    consented, reason = await _worker({}).voice_clone_consent_state(ROOM, SPEAKER)

    assert consented is False
    assert reason == "routes_unknown"


async def test_an_unknown_room_recovers_consent_from_the_redis_snapshot() -> None:
    # The failure this fixes: a deploy restarts the worker mid-meeting, the AUDIO_ROUTES_UPDATED
    # broadcast has already been and gone, and nothing else ever tells it. The backend writes the
    # same payload to a durable key, which is what this reads.
    worker = _worker({}, snapshot={"routes": [_route(True)]})

    consented, reason = await worker.voice_clone_consent_state(ROOM, SPEAKER)

    assert (consented, reason) == (True, "consented")


async def test_the_snapshot_is_not_consulted_when_routes_are_already_known() -> None:
    # A speaker who has opted out must stay opted out. Re-reading Redis on every chunk would
    # also be a request per audio chunk per speaker, which is the hot path.
    worker = _worker({ROOM: [_route(False)]})
    worker._load_route_snapshot = AsyncMock(return_value=True)  # type: ignore[assignment]

    consented, reason = await worker.voice_clone_consent_state(ROOM, SPEAKER)

    assert (consented, reason) == (False, "not_opted_in")
    worker._load_route_snapshot.assert_not_awaited()


async def test_a_snapshot_that_grants_nobody_consent_is_not_read_as_consent() -> None:
    worker = _worker({}, snapshot={"routes": [_route(False)]})

    consented, reason = await worker.voice_clone_consent_state(ROOM, SPEAKER)

    assert consented is False
    assert reason == "not_opted_in"


async def test_another_speakers_consent_does_not_carry() -> None:
    worker = _worker({ROOM: [_route(True, user_id="somebody-else")]})

    consented, reason = await worker.voice_clone_consent_state(ROOM, SPEAKER)

    assert (consented, reason) == (False, "not_opted_in")


async def test_consent_matching_is_case_insensitive_on_the_user_id() -> None:
    # The route carries the id the backend denormalised; the AI pipeline carries the one STT
    # tagged the chunk with. A case difference between them would silently deny every speaker.
    worker = _worker({ROOM: [_route(True, user_id=SPEAKER.upper())]})

    consented, _ = await worker.voice_clone_consent_state(ROOM, SPEAKER)

    assert consented is True
