"""Each dub bot says whose voice it speaks in, so the client can honour the SPEAKER's choice.

WHAT WAS REPORTED
    Kỳ and Tuấn in one meeting: Kỳ turned voice clone on, and Tuấn only heard Kỳ's cloned voice
    after Tuấn ALSO turned his own switch on. The server was not the cause — every one of Kỳ's
    dubs in that meeting (01a10296, 3 Oct) was synthesised as "cloned". The listener's switch
    decided playback, because the client had no way to know whose voice a dub track carried.

WHAT THESE PIN
    - A bot created after the kind is known announces it before speaking.
    - A change of kind (stock voice -> clone once it is ready) is re-announced; a repeat is not.
    - A failed attribute update never raises into the sentence.
    - The meeting's kinds are forgotten when the meeting is retired.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tts_worker.livekit_publisher import VOICE_KIND_ATTRIBUTE, LiveKitTTSPublisher

MEETING = "m-1"
SPEAKER = "s-1"


def _publisher() -> LiveKitTTSPublisher:
    return LiveKitTTSPublisher(SimpleNamespace(url="ws://x", api_key="k", api_secret="s"))


def _bot() -> dict:
    room = MagicMock()
    room.local_participant.set_attributes = AsyncMock()
    return {"room": room, "source": MagicMock(), "last_used": 0.0}


@pytest.mark.asyncio
async def test_kind_is_announced_on_an_existing_bot_and_only_when_it_changes() -> None:
    publisher = _publisher()
    bot = _bot()
    publisher._bots[(MEETING, SPEAKER, "en", "")] = bot

    await publisher.set_voice_kind(MEETING, SPEAKER, "en", "default")
    await publisher.set_voice_kind(MEETING, SPEAKER, "en", "default")
    await publisher.set_voice_kind(MEETING, SPEAKER, "en", "cloned")

    calls = bot["room"].local_participant.set_attributes.await_args_list
    assert [c.args[0] for c in calls] == [
        {VOICE_KIND_ATTRIBUTE: "default"},
        {VOICE_KIND_ATTRIBUTE: "cloned"},
    ]


@pytest.mark.asyncio
async def test_kind_decided_before_the_bot_exists_is_kept_for_it() -> None:
    publisher = _publisher()
    await publisher.set_voice_kind(MEETING, SPEAKER, "ja", "cloned")
    assert publisher._voice_kinds[(MEETING, SPEAKER, "ja", "")] == "cloned"


@pytest.mark.asyncio
async def test_a_failed_announcement_does_not_raise() -> None:
    publisher = _publisher()
    bot = _bot()
    bot["room"].local_participant.set_attributes.side_effect = RuntimeError("closed")
    publisher._bots[(MEETING, SPEAKER, "en", "")] = bot

    await publisher.set_voice_kind(MEETING, SPEAKER, "en", "cloned")


def test_retiring_the_meeting_forgets_its_kinds() -> None:
    publisher = _publisher()
    publisher._voice_kinds[(MEETING, SPEAKER, "en", "")] = "cloned"
    publisher._voice_kinds[("other", SPEAKER, "en", "")] = "default"

    publisher.retire_meeting(MEETING, "ended")

    assert list(publisher._voice_kinds) == [("other", SPEAKER, "en", "")]
