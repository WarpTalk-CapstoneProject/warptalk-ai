"""The voice a speaker CHOSE to be dubbed in — WT-396.

A tester uploaded a recording of their own voice, the UI listed the profile as active, and the
dub still came back in a stock catalogue voice. It had to: `_resolve_voice_variants` looked in
exactly one place for the speaker's voice — a clone built live from the meeting's own microphone
audio — and nothing anywhere read the profile they had picked.

The confusion underneath is that `voice_profiles` already meant something else. Rows written by
SetPreferredVoiceAsync are documented as "the library voice this user HEARS", a listener
preference; an uploaded recording is the opposite direction. The two shared a table, so a choice
about how somebody SOUNDS was stored beside choices about what they hear, and read by neither.

These pin the read side: the choice reaches the worker, it outranks a live clone, and every way
of not having one still lands on the behaviour everybody had before.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker

ROOM = "01a0015d-c945-758d-a622-8794cb537dfb"
SPEAKER = "019f0d00-0de0-7000-9000-000000000003"
CHOSEN = "voice-the-user-picked"
LIVE_CLONE = "voice-cloned-in-this-meeting"


def _worker(routes: list[dict] | None = None) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker._room_routes = {ROOM: routes if routes is not None else []}
    worker.redis = AsyncMock()
    worker.redis.hgetall = AsyncMock(return_value={})
    worker.redis.hget = AsyncMock(return_value=None)
    return worker


def _route(**overrides) -> dict:
    route = {
        "SourceUserId": SPEAKER,
        "TargetUserId": "someone-else",
        "VoiceCloneEnabled": False,
        "SourceDubVoiceId": None,
    }
    route.update(overrides)
    return route


# ── the choice reaches the worker ────────────────────────────────────────────────────────────


def test_the_speakers_choice_is_read_from_the_route_snapshot() -> None:
    worker = _worker([_route(SourceDubVoiceId=CHOSEN)])

    assert worker.chosen_dub_voice(ROOM, SPEAKER) == CHOSEN


def test_another_persons_choice_is_not_borrowed() -> None:
    # Routes carry every pairing in the room. Matching loosely would dub one person in another
    # person's voice, which is worse than the bug being fixed.
    worker = _worker([_route(SourceUserId="somebody-else", SourceDubVoiceId=CHOSEN)])

    assert worker.chosen_dub_voice(ROOM, SPEAKER) is None


def test_an_unknown_room_means_no_choice_not_an_error() -> None:
    assert _worker().chosen_dub_voice("a-room-nobody-told-us-about", SPEAKER) is None


def test_a_backend_that_does_not_send_the_field_yet_is_not_a_crash() -> None:
    # Half the fleet talks to an older backend during a rolling deploy. A missing field must read
    # as "no choice", never as a failure.
    worker = _worker([{"SourceUserId": SPEAKER, "VoiceCloneEnabled": True}])

    assert worker.chosen_dub_voice(ROOM, SPEAKER) is None


# ── and it outranks the live clone ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_chosen_voice_beats_a_voice_cloned_live_in_the_meeting() -> None:
    """The precedence that makes the feature real.

    Somebody who picked a voice and then spoke long enough to be cloned must keep the voice they
    picked. A live clone silently overriding it is the same failure as the pick never being read.
    """
    worker = _worker([_route(SourceDubVoiceId=CHOSEN)])
    worker._get_voice_id = AsyncMock(return_value=LIVE_CLONE)

    variants = await worker._resolve_voice_variants(ROOM, SPEAKER, "en")

    voice_id, voice_type, voice_key = variants[0]
    assert voice_id == CHOSEN
    assert voice_type == "profile"
    assert voice_key == ""


@pytest.mark.asyncio
async def test_without_a_choice_the_live_clone_is_still_used() -> None:
    # The path everybody was on before this existed has to survive unchanged.
    worker = _worker([_route()])
    worker._get_voice_id = AsyncMock(return_value=LIVE_CLONE)

    variants = await worker._resolve_voice_variants(ROOM, SPEAKER, "en")

    assert variants[0][:2] == (LIVE_CLONE, "cloned")


@pytest.mark.asyncio
async def test_with_neither_it_still_falls_back_to_a_hashed_catalogue_voice() -> None:
    worker = _worker([_route()])
    worker._get_voice_id = AsyncMock(return_value=None)
    worker._hashed_default_voice_id = AsyncMock(return_value="catalogue-voice")

    variants = await worker._resolve_voice_variants(ROOM, SPEAKER, "en")

    assert variants[0][:2] == ("catalogue-voice", "default")


@pytest.mark.asyncio
async def test_a_chosen_voice_does_not_ask_for_a_live_clone_at_all() -> None:
    # Not just precedence — the lookup is skipped. Asking anyway would be a Redis round trip per
    # utterance whose answer is discarded.
    worker = _worker([_route(SourceDubVoiceId=CHOSEN)])
    worker._get_voice_id = AsyncMock(return_value=LIVE_CLONE)

    await worker._resolve_voice_variants(ROOM, SPEAKER, "en")

    worker._get_voice_id.assert_not_awaited()
