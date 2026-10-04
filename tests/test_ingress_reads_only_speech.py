"""WT-631 — a shared tab is not a participant speaking.

The ingress worker attached a reader to every human audio track it could see and asked only
whether the kind was audio. So anything a participant PLAYED rather than said — a screen
share's audio, a shared browser tab — became that participant's speech: transcribed,
translated, dubbed and billed, attributed to somebody who had not spoken. Because readers are
keyed per (room, participant), it also displaced the reader on their real microphone.

What these pin:
  * the predicate — microphone and undeclared (the external bridge) are read; screen-share
    audio is not;
  * the reaper sweep, observed rather than inferred: it reads the microphone and leaves the
    share's audio alone, even when the share is listed first;
  * every path that can attach a reader asks first, and the mute path asks too. There are three
    attaching paths (subscribe, unmute, and the sweep that re-attaches anything with no live
    reader); a check missing from the sweep would be silently undone one sweep later, exactly as
    the WT-542 mute bug was.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from livekit import rtc

from livekit_ingress_worker.worker import LiveKitIngressWorker, _carries_speech
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "019f6a39-a32c-7745-886e-1fe622c1f747"
SPEAKER = "019f0d00-0de0-7000-9000-000000000002"
BRIDGE = "019f0d00-0de0-7000-9000-0000000000b1"


class _Publication:
    """Only what the predicate reads. rtc.RemoteTrackPublication cannot be built in a test."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.sid = "TR_test"


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    return worker


def _audio_publication(sid: str, source: Any) -> MagicMock:
    track = MagicMock()
    track.sid = sid
    track.kind = rtc.TrackKind.KIND_AUDIO
    pub = MagicMock()
    pub.sid = sid
    pub.kind = rtc.TrackKind.KIND_AUDIO
    pub.muted = False
    pub.source = source
    pub.track = track
    return pub


def _room_with(identity: str, *publications: MagicMock) -> MagicMock:
    participant = MagicMock()
    participant.identity = identity
    # Insertion order is the iteration order, so a test can put the share FIRST — the order in
    # which it used to win the per-participant reader slot.
    participant.track_publications = {pub.sid: pub for pub in publications}
    room = MagicMock()
    room.remote_participants = {identity: participant}
    return room


@pytest.fixture(autouse=True)
def _never_really_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the audio pipeline with something that simply waits to be cancelled."""

    async def _idle(
        self: LiveKitIngressWorker, room_name: str, speaker_id: str, track: Any
    ) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(LiveKitIngressWorker, "process_audio_track", _idle)


# ── the predicate ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "source",
    [
        rtc.TrackSource.SOURCE_MICROPHONE,
        # The external bridge. warptalk-web's bridge-inbound-connection.ts publishes the far
        # side of a Meet call with no source, which LiveKit records as unknown — a
        # microphone-only rule would deafen it, and the far side would speak an entire meeting
        # without one line of transcript.
        rtc.TrackSource.SOURCE_UNKNOWN,
    ],
)
def test_speech_is_read(source: Any) -> None:
    assert _carries_speech(_Publication(source)) is True


def test_audio_a_participant_is_playing_is_not_read() -> None:
    assert _carries_speech(_Publication(rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)) is False


# ── the sweep, observed ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sweep_reads_the_microphone_not_the_shared_tab() -> None:
    worker = _worker()
    room = _room_with(
        SPEAKER,
        _audio_publication("TR_tab", rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO),
        _audio_publication("TR_mic", rtc.TrackSource.SOURCE_MICROPHONE),
    )

    assert worker._start_pending_audio_tasks(ROOM, room) == 1
    assert worker.audio_task_tracks[(ROOM, SPEAKER)] == "TR_mic"

    # And a second sweep does not swap the share in either.
    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert worker.audio_task_tracks[(ROOM, SPEAKER)] == "TR_mic"

    worker._cancel_room_audio_tasks(ROOM)


@pytest.mark.asyncio
async def test_a_participant_who_only_shares_a_tab_is_not_read() -> None:
    worker = _worker()
    room = _room_with(
        SPEAKER, _audio_publication("TR_tab", rtc.TrackSource.SOURCE_SCREENSHARE_AUDIO)
    )

    assert worker._start_pending_audio_tasks(ROOM, room) == 0
    assert (ROOM, SPEAKER) not in worker.audio_tasks


@pytest.mark.asyncio
async def test_the_bridge_far_side_is_still_read() -> None:
    worker = _worker()
    room = _room_with(BRIDGE, _audio_publication("TR_far", rtc.TrackSource.SOURCE_UNKNOWN))

    assert worker._start_pending_audio_tasks(ROOM, room) == 1
    assert worker.audio_task_tracks[(ROOM, BRIDGE)] == "TR_far"

    worker._cancel_room_audio_tasks(ROOM)


# ── every door asks ──────────────────────────────────────────────────────────────────────────


def _functions_in_worker() -> dict[str, ast.FunctionDef]:
    source = Path(inspect.getsourcefile(_carries_speech) or "")
    tree = ast.parse(source.read_text())
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _calls(node: ast.AST) -> set[str]:
    return {
        child.func.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
    }


def _asks_carries_speech(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Name) and child.id == "_carries_speech" for child in ast.walk(node)
    )


def test_every_path_that_attaches_a_reader_asks_first() -> None:
    # An allow-list applied at two of three doors is not an allow-list. The handlers are
    # closures inside join_room and cannot be reached without a live LiveKit room, so this is
    # asserted on the source.
    functions = _functions_in_worker()
    attaching = {name for name, node in functions.items() if "_start_audio_task" in _calls(node)}

    assert attaching == {
        "_start_pending_audio_tasks",
        "on_track_subscribed",
        "on_track_unmuted",
    }, f"a new path attaches a reader and this test has not been told about it: {sorted(attaching)}"

    for name in attaching:
        assert _asks_carries_speech(functions[name]), (
            f"{name} attaches an audio reader without asking whether it is speech"
        )


def test_muting_a_shared_tab_does_not_stop_the_microphone() -> None:
    # Readers are keyed per participant, so the mute handler cancelling on a screen share's
    # audio would silence the participant's still-live microphone until the next sweep.
    functions = _functions_in_worker()
    cancelling = {name for name, node in functions.items() if "_cancel_audio_task" in _calls(node)}

    assert cancelling == {"on_track_muted"}, (
        f"a new path cancels a reader and this test does not know it: {sorted(cancelling)}"
    )
    assert _asks_carries_speech(functions["on_track_muted"])
