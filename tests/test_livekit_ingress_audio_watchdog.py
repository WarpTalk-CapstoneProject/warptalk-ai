"""WT-404 — a connected room with people in it is not the same as a room being HEARD.

Production 2026-08-14, room 01a0015d. Somebody spoke for eight minutes. `audio:chunks`
received the last of it at 17:43:14, seventy-six seconds in, and one transcript segment was
ever saved. The room stayed connected the whole time and the idle sweep went on reporting it
occupied, because the only question that sweep asked was "is anyone in here".

`process_audio_track` ends silently when its AudioStream ends without raising: the task
completes, `_forget_audio_task` drops it, and the sole trace was one INFO line worded exactly
like an ordinary teardown. `_start_audio_task` is reached from two places — the
`track_subscribed` event and joining the room — so a speaker whose stream ended without a
fresh subscribe was never read again.

These pin the two halves of the fix: the sweep now re-attaches a missing reader, and it
refuses to do so where a reader was stopped on purpose.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a0015d-c945-758d-a622-8794cb537dfb"
SPEAKER = "019f0d00-0de0-7000-9000-000000000003"


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    worker._consumer_name = "livekit_ingress-replica-a"
    # The VAD pipeline itself is covered elsewhere; here only the decision to start one matters.
    worker.process_audio_track = AsyncMock()  # type: ignore[method-assign]
    return worker


def _audio_track(sid: str = "TR_audio_1") -> MagicMock:
    import livekit.rtc as rtc

    track = MagicMock()
    track.sid = sid
    track.kind = rtc.TrackKind.KIND_AUDIO
    return track


def _room_with_speaker(track: MagicMock | None, identity: str = SPEAKER) -> MagicMock:
    participant = MagicMock()
    participant.identity = identity
    publication = MagicMock()
    publication.track = track
    participant.track_publications = {"pub-1": publication}

    room = MagicMock()
    room.isconnected.return_value = True
    room.disconnect = AsyncMock()
    room.remote_participants = {identity: participant}
    return room


# ── the gap this closes ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_speaker_whose_reader_died_is_read_again() -> None:
    """The failure itself.

    No reader, a live room, a still-published track. Before WT-404 the sweep counted the room
    as occupied and moved on, and that speaker was silent for the rest of the meeting.
    """
    worker = _worker()
    track = _audio_track()
    worker.rooms = {ROOM: _room_with_speaker(track)}

    await worker._sweep_idle_rooms()

    assert (ROOM, SPEAKER) in worker.audio_tasks, "the sweep left a live room unheard"


@pytest.mark.asyncio
async def test_a_speaker_already_being_read_is_left_alone() -> None:
    # Re-attaching a healthy reader would publish the same speech twice, under two chunk
    # counters — worse than the silence being fixed.
    worker = _worker()
    track = _audio_track()
    room = _room_with_speaker(track)
    worker.rooms = {ROOM: room}

    await worker._sweep_idle_rooms()
    first = worker.audio_tasks[(ROOM, SPEAKER)]
    await worker._sweep_idle_rooms()

    assert worker.audio_tasks[(ROOM, SPEAKER)] is first, "a healthy reader was replaced"


@pytest.mark.asyncio
async def test_a_participant_with_no_published_track_is_not_invented() -> None:
    # Somebody present but not publishing audio — muted at the source, or video only. There is
    # nothing to read, and starting a reader on None would crash the sweep for the whole fleet.
    worker = _worker()
    worker.rooms = {ROOM: _room_with_speaker(track=None)}

    await worker._sweep_idle_rooms()

    assert worker.audio_tasks == {}


@pytest.mark.asyncio
async def test_a_room_nobody_is_in_is_still_released() -> None:
    """WT-314 must survive this.

    An empty room has to be given up — that is what stops a leaked bot billing LiveKit minutes
    forever. The watchdog runs in the OCCUPIED branch precisely so it cannot keep a dead room
    alive, and this is the assertion that says the release still happens.

    Asserting on the release rather than on `audio_tasks`: an empty room has no human to attach
    to and a bot is filtered out anyway, so an audio_tasks check here can never fail and would
    be a test that proves nothing.
    """
    worker = _worker()
    room = MagicMock()
    room.isconnected.return_value = True
    room.disconnect = AsyncMock()
    room.remote_participants = {}
    worker.rooms = {ROOM: room}
    worker._room_last_occupied[ROOM] = worker._now() - 10_000

    await worker._sweep_idle_rooms()
    for _ in range(5):
        pending = [t for t in list(worker._event_tasks) if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)

    assert ROOM not in worker.rooms, "an empty room was kept, which is WT-314 returning"


@pytest.mark.asyncio
async def test_a_bot_is_never_read_back_into_the_pipeline() -> None:
    # Subscribing to the interpreter's own synthesised speech is a feedback loop. The watchdog
    # runs over the same participants as the join path and must inherit the same filter.
    worker = _worker()
    worker.rooms = {ROOM: _room_with_speaker(_audio_track(), identity="ai-interpreter-en-x")}

    await worker._sweep_idle_rooms()

    assert worker.audio_tasks == {}


# ── and the log that would have shown it ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_reader_that_stops_itself_is_not_logged_as_a_normal_teardown() -> None:
    """The line that was missing.

    A cancelled reader and one whose stream simply ended used the same INFO wording, so a
    speaker going silent mid-meeting read as routine cleanup. Telling them apart is what makes
    the failure visible without a database query.
    """
    worker = _worker()
    del worker.process_audio_track  # exercise the real one
    worker.logger = MagicMock()
    worker._publish_speech_chunk = AsyncMock()  # type: ignore[method-assign]

    import livekit.rtc as rtc

    class _EndingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    rtc.AudioStream = MagicMock(return_value=_EndingStream())  # type: ignore[assignment]
    worker._vad_model = MagicMock()

    await worker.process_audio_track(ROOM, SPEAKER, _audio_track())

    warned = [c.args[0] for c in worker.logger.warning.call_args_list]
    assert "audio_stream_ended_on_its_own" in warned, (
        "a reader that stopped by itself was logged as an ordinary teardown"
    )
