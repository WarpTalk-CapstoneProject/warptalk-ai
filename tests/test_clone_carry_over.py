"""A clone made in one meeting is kept, used at t=0 in the next, and improved on (WT-B).

Before this, `voice:{meeting}:{speaker}` was keyed BY MEETING and expired, so every meeting
re-cloned from nothing and paid its first twenty seconds in a stock catalogue voice — and the
clone could never get better across meetings, because the score it was judged against lived in
a process-local dict that died with the worker.

Four things here are load-bearing and each is pinned below:

  * The RENAME comes before the publish. The orphan sweep judges a voice by its name, so a
    voice handed to AuthService while still named `speaker-` is a stored row pointing at
    something that gets deleted within 24 hours.
  * A carried voice is used at t=0, but a clone made in THIS meeting outranks it.
  * The carried SCORE becomes the bar. Without it, the first acceptable clip of every meeting
    looks like an improvement on nothing and re-clones immediately.
  * A carried clone is read from its OWN route field, never from SourceDubVoiceId. Merging the
    two would make the worker treat it as a deliberate pick, stop capturing, and freeze every
    speaker at the first clone they ever earned.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker

ROOM = "11111111-1111-1111-1111-111111111111"
SPEAKER = "22222222-2222-2222-2222-222222222222"
CARRIED = "voice-from-last-meeting"


class _Redis:
    def __init__(self, cached_voice: str | None = None) -> None:
        self.cached_voice = cached_voice
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.hset_calls: list[tuple[str, str, str]] = []

    async def hget(self, key: str, field: str) -> str | None:
        return self.cached_voice

    async def hset(self, key: str, field: str, value: str) -> None:
        self.hset_calls.append((key, field, value))

    async def expire(self, key: str, ttl_seconds: int) -> None:
        return None

    async def publish(self, stream: str, data: dict[str, Any]) -> str:
        self.published.append((stream, data))
        return "1-1"

    async def publish_system_event(self, **kwargs: Any) -> None:
        return None


class _Cartesia:
    def __init__(self, *, rename_ok: bool = True) -> None:
        self.rename_ok = rename_ok
        self.renamed: list[tuple[str, str]] = []
        self.deleted: list[str] = []

    async def clone_voice(self, audio: bytes, label: str, language: str) -> str:
        return "brand-new-voice"

    async def rename_voice(self, voice_id: str, name: str) -> bool:
        self.renamed.append((voice_id, name))
        return self.rename_ok

    async def delete_voice(self, voice_id: str) -> bool:
        self.deleted.append(voice_id)
        return True


def _worker(
    routes: list[dict[str, Any]] | None = None,
    redis: _Redis | None = None,
    cartesia: _Cartesia | None = None,
    **overrides: Any,
) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._consumer_name = "tts-test"
    worker.worker_name = "tts"
    worker.cartesia = cartesia or _Cartesia()  # type: ignore[assignment]
    worker.redis = redis or _Redis()  # type: ignore[assignment]
    worker._room_routes = {ROOM: routes if routes is not None else []}
    worker._clone_state = {}
    return worker


def _route(**overrides: Any) -> dict[str, Any]:
    route = {
        "SourceUserId": SPEAKER,
        "VoiceCloneEnabled": True,
        "SourceDubVoiceId": None,
        "SourceAutoCloneVoiceId": None,
        "SourceAutoCloneScore": None,
    }
    route.update(overrides)
    return route


# --------------------------------------------------------------------------------------
# Reading the carried clone off the route
# --------------------------------------------------------------------------------------


def test_reads_the_carried_voice_and_its_score() -> None:
    worker = _worker([_route(SourceAutoCloneVoiceId=CARRIED, SourceAutoCloneScore="0.742")])

    assert worker.carried_clone(ROOM, SPEAKER) == (CARRIED, 0.742)


def test_a_deliberate_dub_pick_is_not_a_carried_clone() -> None:
    """The separation the whole feature rests on.

    Read as a carried clone, a deliberate pick would keep being "improved on" and eventually
    overwritten. Read as a pick, a carried clone would stop the worker capturing and freeze the
    speaker at their first ever clone. They travel on separate fields for exactly this reason.
    """
    worker = _worker([_route(SourceDubVoiceId="a-voice-they-chose")])

    assert worker.carried_clone(ROOM, SPEAKER) == (None, None)


def test_a_carried_voice_with_no_score_reports_an_unset_bar_not_zero() -> None:
    """NULL is "not measured". Zero would grade as the worst possible sample and invite
    replacement by anything at all."""
    worker = _worker([_route(SourceAutoCloneVoiceId=CARRIED, SourceAutoCloneScore=None)])

    assert worker.carried_clone(ROOM, SPEAKER) == (CARRIED, None)


def test_an_unreadable_score_keeps_the_voice_and_drops_the_bar() -> None:
    worker = _worker([_route(SourceAutoCloneVoiceId=CARRIED, SourceAutoCloneScore="not-a-number")])

    assert worker.carried_clone(ROOM, SPEAKER) == (CARRIED, None)


def test_says_nothing_for_a_room_it_has_not_been_told_about() -> None:
    """A rolling deploy has half the fleet talking to a backend that does not send this yet."""
    worker = _worker([_route(SourceAutoCloneVoiceId=CARRIED)])

    assert worker.carried_clone("some-other-room", SPEAKER) == (None, None)


def test_does_not_hand_one_speakers_carried_voice_to_another() -> None:
    worker = _worker([_route(SourceUserId="somebody-else", SourceAutoCloneVoiceId=CARRIED)])

    assert worker.carried_clone(ROOM, SPEAKER) == (None, None)


# --------------------------------------------------------------------------------------
# Using it at t=0
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_carried_voice_is_used_before_this_meeting_has_cloned_anyone() -> None:
    """The point of the feature: no stock-voice opening while the clone is rebuilt."""
    worker = _worker(
        [_route(SourceAutoCloneVoiceId=CARRIED, SourceAutoCloneScore="0.7")],
        redis=_Redis(cached_voice=None),
    )

    assert await worker._get_voice_id(ROOM, SPEAKER) == CARRIED


@pytest.mark.asyncio
async def test_a_clone_made_in_this_meeting_outranks_the_carried_one() -> None:
    """Newer evidence about how this person sounds today wins."""
    worker = _worker(
        [_route(SourceAutoCloneVoiceId=CARRIED, SourceAutoCloneScore="0.7")],
        redis=_Redis(cached_voice="cloned-just-now"),
    )

    assert await worker._get_voice_id(ROOM, SPEAKER) == "cloned-just-now"


@pytest.mark.asyncio
async def test_withdrawn_consent_drops_the_carried_voice_too() -> None:
    """The gate guards biometric processing and must not be escapable through the new path."""
    worker = _worker(
        [_route(VoiceCloneEnabled=False, SourceAutoCloneVoiceId=CARRIED)],
        redis=_Redis(cached_voice=None),
    )

    assert await worker._get_voice_id(ROOM, SPEAKER) is None


# --------------------------------------------------------------------------------------
# Handing a finished clone over
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_finished_clone_is_renamed_out_of_the_sweeps_sights_then_announced() -> None:
    cartesia = _Cartesia()
    redis = _Redis()
    worker = _worker(redis=redis, cartesia=cartesia)

    await worker._offer_carry_over(SPEAKER, "vi", "new-voice-id", 0.812)

    assert cartesia.renamed == [("new-voice-id", f"profile-{SPEAKER[:8]}-vi")]
    assert redis.published == [
        (
            "voice:auto_clone_ready",
            {
                "user_id": SPEAKER,
                "language": "vi",
                "voice_id": "new-voice-id",
                "score": "0.812",
            },
        )
    ]


@pytest.mark.asyncio
async def test_a_failed_rename_announces_nothing() -> None:
    """The invariant that stops this feature creating dead rows.

    A voice announced while still named `speaker-` is swept within 24 hours, leaving AuthService
    holding a row that names an id Cartesia has never heard of — and the person opening a later
    meeting in a stranger's voice, with the row still looking perfectly correct. Publishing
    nothing leaves an ordinary in-meeting clone, swept on schedule, re-cloned next meeting: the
    behaviour we already had.
    """
    cartesia = _Cartesia(rename_ok=False)
    redis = _Redis()
    worker = _worker(redis=redis, cartesia=cartesia)

    await worker._offer_carry_over(SPEAKER, "en", "new-voice-id", 0.9)

    assert cartesia.renamed == [("new-voice-id", f"profile-{SPEAKER[:8]}-en")]
    assert redis.published == []


@pytest.mark.asyncio
async def test_an_unmeasured_clone_travels_with_an_empty_score_not_a_zero() -> None:
    redis = _Redis()
    worker = _worker(redis=redis)

    await worker._offer_carry_over(SPEAKER, "en", "new-voice-id", None)

    assert redis.published[0][1]["score"] == ""


@pytest.mark.asyncio
async def test_a_carry_over_that_fails_never_breaks_the_clone_that_just_succeeded() -> None:
    class _Exploding(_Cartesia):
        async def rename_voice(self, voice_id: str, name: str) -> bool:
            raise RuntimeError("cartesia is down")

    worker = _worker(cartesia=_Exploding())

    # Must not raise: the clone is already cached and in use for this meeting.
    await worker._offer_carry_over(SPEAKER, "en", "new-voice-id", 0.5)


# --------------------------------------------------------------------------------------
# The carried score becomes the bar the next clip has to beat
# --------------------------------------------------------------------------------------
#
# Driven through the real capture loop, reusing the harness the upgrade tests already use, so
# these exercise the branch order rather than a re-description of it.

from tests.test_clone_pitch_coverage import _flat, _varied  # noqa: E402
from tests.test_clone_upgrade import _chunks, _run  # noqa: E402
from tests.test_clone_upgrade import _worker as _capture_worker  # noqa: E402


def _with_carried(worker: TTSWorker, voice_id: str | None, score: str | None) -> None:
    worker._room_routes = {
        "m1": [
            {
                "SourceUserId": "s1",
                "VoiceCloneEnabled": True,
                "SourceAutoCloneVoiceId": voice_id,
                "SourceAutoCloneScore": score,
            }
        ]
    }


@pytest.mark.asyncio
async def test_a_narrow_clip_does_not_displace_a_good_carried_clone() -> None:
    """The regression this seeding exists to prevent.

    Without the carried score, `previous_score is None` on the first clip of every meeting, so
    ANY clip that clears the floors reads as an improvement on nothing. A speaker who earned a
    wide-range clone last week would have it thrown away by the first "alo alo" of the next
    meeting — and burn their one upgrade doing it.
    """
    worker, cloned_from = _capture_worker([], voice_clone_min_seconds=10.0)
    _with_carried(worker, CARRIED, "0.95")

    await _run(worker, _chunks(_flat()))

    assert cloned_from == []


@pytest.mark.asyncio
async def test_a_materially_better_clip_still_replaces_a_carried_clone() -> None:
    """Seeding the bar must not weld the voice shut — improving it is the other half of B."""
    worker, cloned_from = _capture_worker([], voice_clone_min_seconds=10.0)
    _with_carried(worker, CARRIED, "0.10")

    await _run(worker, _chunks(_varied()))

    assert len(cloned_from) == 1


@pytest.mark.asyncio
async def test_a_carried_clone_with_no_score_is_replaced_by_the_first_good_clip() -> None:
    """An unset bar is not a high one. A row from before scores existed should not pin a
    speaker to a voice nobody ever measured."""
    worker, cloned_from = _capture_worker([], voice_clone_min_seconds=10.0)
    _with_carried(worker, CARRIED, None)

    await _run(worker, _chunks(_varied()))

    assert len(cloned_from) == 1


# --------------------------------------------------------------------------------------
# Carrying out a deletion AuthService asked for
# --------------------------------------------------------------------------------------


class _DeleteRequests:
    """Yields one batch of delete requests, then stops the worker's outer loop."""

    def __init__(self, worker: TTSWorker, voice_ids: list[str]) -> None:
        self._worker = worker
        self._voice_ids = voice_ids

    async def consume(self, **_kwargs: Any) -> Any:
        for index, voice_id in enumerate(self._voice_ids):
            yield f"{index}-0".encode(), {b"voice_id": voice_id.encode()}
        self._worker._running = False


@pytest.mark.asyncio
async def test_a_requested_deletion_reaches_cartesia() -> None:
    """The half of the consent promise that only this side can keep.

    AuthService holds the row and can forget a voice; only this side holds the key that can
    destroy it. While a clone died with its meeting, "stops being used" was the whole promise —
    keeping the clone is what makes actually deleting it the rest of it.
    """
    cartesia = _Cartesia()
    worker = _worker(cartesia=cartesia)
    worker._running = True
    worker.redis = _DeleteRequests(worker, ["voice-to-destroy"])  # type: ignore[assignment]

    await worker._consume_voice_delete_requests()

    assert cartesia.deleted == ["voice-to-destroy"]
