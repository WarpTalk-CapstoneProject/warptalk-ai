"""WT-529: the ingress worker writes down how to say each speaker's name.

The producer half. The resolver in `ai_assistant_worker.speaker_names` is worthless if nothing
ever fills the map — that is the shape of failure this repo keeps finding, so it is asserted
separately from the reading side.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from livekit_ingress_worker.worker import LiveKitIngressWorker

ROOM = "01a02db4-3030-743b-9ca7-53d5dedf4c1c"
IDENTITY = "speaker-019f0d00-0de0-7000-9000-000000000003"


def _worker():
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker._speaker_name_tasks = set()
    worker.redis = SimpleNamespace(hset=AsyncMock(), expire=AsyncMock())
    worker.logger = SimpleNamespace(
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
    )
    return worker


def _participant(identity: str = IDENTITY, name: str | None = "Ngọc Kỳ"):
    return SimpleNamespace(identity=identity, name=name)


async def _drain(worker) -> None:
    """Let the fire-and-forget write run."""
    if worker._speaker_name_tasks:
        await asyncio.gather(*list(worker._speaker_name_tasks))


async def test_a_named_participant_is_written_to_the_rooms_hash():
    worker = _worker()

    worker._remember_speaker_name(ROOM, _participant())
    await _drain(worker)

    worker.redis.hset.assert_awaited_once_with(f"meeting:{ROOM}:speaker_names", IDENTITY, "Ngọc Kỳ")


async def test_the_key_is_given_a_ttl_so_a_finished_room_expires_on_its_own():
    worker = _worker()

    worker._remember_speaker_name(ROOM, _participant())
    await _drain(worker)

    key, ttl = worker.redis.expire.await_args.args
    assert key == f"meeting:{ROOM}:speaker_names"
    # Must outlive a real meeting — the summary is written when it ends.
    assert ttl >= 60 * 60


async def test_a_participant_with_no_name_writes_nothing():
    # No `name` claim on the token. Writing an empty label would render as a transcript line
    # nobody spoke; the summariser's pseudonyms cover this properly.
    worker = _worker()

    worker._remember_speaker_name(ROOM, _participant(name=None))
    worker._remember_speaker_name(ROOM, _participant(name="   "))
    await _drain(worker)

    worker.redis.hset.assert_not_awaited()


async def test_a_name_that_is_just_the_identity_is_not_written():
    # Some clients send the identity as the display name. Writing it would move the uuid out
    # of the transcript and into the map, which fixes nothing.
    worker = _worker()

    worker._remember_speaker_name(ROOM, _participant(name=IDENTITY))
    await _drain(worker)

    worker.redis.hset.assert_not_awaited()


async def test_a_redis_failure_never_reaches_the_caller():
    # Reading audio must not depend on this write. A room with no names still summarises.
    worker = _worker()
    worker.redis.hset = AsyncMock(side_effect=RuntimeError("redis is down"))

    worker._remember_speaker_name(ROOM, _participant())
    await _drain(worker)  # must not raise


async def test_the_task_is_held_then_released():
    # Held so the loop cannot collect a task nobody awaits; released so the set does not grow
    # for the life of the process.
    worker = _worker()

    worker._remember_speaker_name(ROOM, _participant())
    assert len(worker._speaker_name_tasks) == 1

    await _drain(worker)
    await asyncio.sleep(0)
    assert worker._speaker_name_tasks == set()


class TestItIsActuallyWired:
    """That `_remember_speaker_name` is CALLED — the failure mode this repo keeps finding.

    Both handlers that learn about a speaker's audio are closures defined inside `_connect_room`
    and cannot be reached without a live LiveKit room, so this is asserted on the source. A
    resolver, a producer and nothing joining them is a fix that ships and does nothing.
    """

    @staticmethod
    def _source() -> str:
        from pathlib import Path

        import livekit_ingress_worker.worker as module

        return Path(module.__file__).read_text(encoding="utf-8")

    def test_the_track_subscribed_handler_records_the_name(self):
        source = self._source()
        handler = source.split('@room.on("track_subscribed")', 1)[1].split("@room.on(", 1)[0]

        assert "_remember_speaker_name(room_name, participant)" in handler, (
            "A speaker whose audio we subscribe to is exactly who can appear in the "
            "transcript — their name has to be written down there."
        )

    def test_the_rediscovery_sweep_records_it_too(self):
        # _start_pending_audio_tasks attaches to tracks published before we connected, or
        # re-attaches after a reconnect. Somebody who joined during that window is a speaker
        # the track_subscribed handler never saw.
        source = self._source()
        sweep = source.split("def _start_pending_audio_tasks", 1)[1].split("\n    def ", 1)[0]

        assert "_remember_speaker_name(room_name, participant)" in sweep
