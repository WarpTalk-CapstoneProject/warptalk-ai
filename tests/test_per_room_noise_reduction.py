"""WT-427 — denoising is a property of the ROOM, not of the deployment.

STT_NOISE_REDUCTION is one environment variable for the whole platform, and its default is "off"
for a measured reason: a second denoising pass on top of the browser's own distorted clean
close-mic speech in replay tests.

It is also wrong for the other half of the estate. A laptop picking a room up from two metres away
needs exactly that pass, and without it the transcript degrades into whatever the microphone is
hearing — which is the report: "chỉ bắt voice ở gần mic thì transcript khá chính xác, nới ra thì
transcript tệ hẳn".

One variable made that an all-or-nothing choice, so whichever way it was set half the meetings were
configured for the other half's room.

AND THE ROOM WAS STILL TOO COARSE
    Denoising describes a microphone. The mixed room — a headset and a laptop two metres from a
    fan, in the same call — is wrong for one participant whichever way a single room-wide value is
    set, and the transcription session was already keyed per (meeting, speaker). So a speaker's own
    setting outranks the room's, and the room remains the default for everyone who has not chosen.

WHY NONE OF IT EVER RAN
    The room key had a reader and no writer, in any repo. And this lookup cached per room for as
    long as the worker remembered it, which cancelled the mid-meeting change
    OpenAISTT._get_or_create_session goes out of its way to support. Both are fixed here; the
    write half lives in warptalk-backend and warptalk-web.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import stt_worker.worker as worker_module
from shared.config import STTSettings, WorkerSettings
from stt_worker.model import OpenAISTT
from stt_worker.worker import STTWorker


class _ModeRedis:
    def __init__(self, value: str | None = None, fail: bool = False) -> None:
        self._value = value
        self._fail = fail
        self.reads = 0

    async def get(self, _key: str) -> str | None:
        if self._fail:
            raise RuntimeError("redis is down")
        self.reads += 1
        return self._value


class _KeyRedis:
    """Answers depend on the key, so the speaker -> room -> default chain is observable."""

    def __init__(self, values: dict[str, str | None]) -> None:
        self.values = values
        self.reads: list[str] = []

    async def get(self, key: str) -> str | None:
        self.reads.append(key)
        return self.values.get(key)


def _worker(redis: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker._room_noise_reduction = {}
    worker._speaker_noise_reduction = {}
    worker.redis = redis
    return worker


def _room_key(meeting_id: str) -> str:
    return f"translationRoom:{meeting_id}:noise_reduction"


def _speaker_key(meeting_id: str, speaker_id: str) -> str:
    return f"translationRoom:{meeting_id}:participant:{speaker_id}:noise_reduction"


def _model(default: str) -> OpenAISTT:
    model = OpenAISTT.__new__(OpenAISTT)
    model.model = "gpt-transcribe"
    model.noise_reduction = default
    return model


# ── the payload ──────────────────────────────────────────────


def test_a_room_override_beats_the_deployment_default() -> None:
    payload = _model("off")._session_payload("vi", None, None, None, noise_reduction="far_field")

    assert payload["audio"]["input"]["noise_reduction"] == {"type": "far_field"}


def test_no_override_keeps_the_deployment_default() -> None:
    # The negative control: every room that says nothing must behave exactly as it did before.
    assert (
        "noise_reduction"
        not in _model("off")._session_payload("vi", None, None, None)["audio"]["input"]
    )

    payload = _model("far_field")._session_payload("vi", None, None, None)
    assert payload["audio"]["input"]["noise_reduction"] == {"type": "far_field"}


def test_a_room_may_turn_it_off_against_a_far_field_default() -> None:
    # Both directions. "off" is a real choice, not an absent one, so it must survive as an
    # override — otherwise a close-mic room on a far-field deployment cannot opt out.
    payload = _model("far_field")._session_payload("vi", None, None, None, noise_reduction="off")

    assert "noise_reduction" not in payload["audio"]["input"]


# ── the room lookup ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_configured_room_reports_its_mode() -> None:
    assert await _worker(_ModeRedis("far_field"))._get_room_noise_reduction("m1") == "far_field"


@pytest.mark.asyncio
async def test_an_unconfigured_room_reports_nothing_rather_than_a_guess() -> None:
    assert await _worker(_ModeRedis(None))._get_room_noise_reduction("m1") is None


@pytest.mark.asyncio
async def test_an_unrecognised_mode_is_refused_not_forwarded() -> None:
    """An unknown string fails the whole session update.

    It would take the language hint and the keywords down with it — _degrade_session_config exists
    because exactly that has happened before.
    """
    worker = _worker(_ModeRedis("aggressive"))

    assert await worker._get_room_noise_reduction("m1") is None
    cast(MagicMock, worker.logger).warning.assert_called_once()


@pytest.mark.asyncio
async def test_the_mode_is_read_at_most_once_per_window() -> None:
    # It is on the hot path — every chunk of every speaker. The TTL exists to let a change
    # through, not to move this lookup onto Redis per sentence.
    redis = _ModeRedis("far_field")
    worker = _worker(redis)

    for _ in range(8):
        await worker._get_room_noise_reduction("m1")

    assert redis.reads == 1


@pytest.mark.asyncio
async def test_a_change_made_during_the_meeting_reaches_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression that made this whole feature unreachable.

    OpenAISTT._get_or_create_session compares `noise_reduction` and issues a session.update on the
    LIVE socket when it differs — deliberately, with a comment saying a value the update never
    notices changed "is a setting that silently does nothing". But this lookup cached per room for
    as long as the worker remembered it, so the new value could never arrive. Somebody moving the
    control mid-meeting saw nothing happen, for the rest of the meeting.
    """
    monkeypatch.setattr(worker_module, "_NOISE_REDUCTION_TTL_S", 0.0)
    redis = _KeyRedis({_room_key("m1"): None})
    worker = _worker(redis)

    assert await worker._get_room_noise_reduction("m1") is None

    redis.values[_room_key("m1")] = "far_field"

    assert await worker._get_room_noise_reduction("m1") == "far_field"


# ── the speaker override ─────────────────────────────────────


@pytest.mark.asyncio
async def test_a_speakers_own_microphone_setting_wins_over_the_room() -> None:
    """Denoising describes a microphone, not a meeting.

    The mixed room is the one that needs this: a headset and a laptop two metres from a fan in the
    same call. One room-wide value is wrong for one of them whichever way it is set, and the
    transcription session is already keyed per (meeting, speaker).
    """
    redis = _KeyRedis({_room_key("m1"): "off", _speaker_key("m1", "s1"): "far_field"})

    assert await _worker(redis)._get_noise_reduction("m1", "s1") == "far_field"


@pytest.mark.asyncio
async def test_an_explicit_off_on_a_speaker_beats_a_room_set_to_far_field() -> None:
    """ "Off" is an answer, not the absence of one.

    The close-mic distortion the deployment default was measured against is exactly what a headset
    user needs to opt out of, even in a room the host configured for far-field.
    """
    redis = _KeyRedis({_room_key("m1"): "far_field", _speaker_key("m1", "s1"): "off"})

    assert await _worker(redis)._get_noise_reduction("m1", "s1") == "off"


@pytest.mark.asyncio
async def test_a_speaker_who_has_chosen_nothing_inherits_the_room() -> None:
    redis = _KeyRedis({_room_key("m1"): "far_field"})

    assert await _worker(redis)._get_noise_reduction("m1", "s1") == "far_field"


@pytest.mark.asyncio
async def test_two_speakers_in_one_room_get_their_own_answers() -> None:
    redis = _KeyRedis(
        {
            _room_key("m1"): "off",
            _speaker_key("m1", "headset"): "off",
            _speaker_key("m1", "laptop"): "far_field",
        }
    )
    worker = _worker(redis)

    assert await worker._get_noise_reduction("m1", "headset") == "off"
    assert await worker._get_noise_reduction("m1", "laptop") == "far_field"


@pytest.mark.asyncio
async def test_the_speaker_id_is_case_normalised_before_it_becomes_a_redis_key() -> None:
    """Redis keys are case-sensitive; this id's casing is not dependable.

    speaker_id is the auth user id as LiveKit reported it, and base_worker compares SourceUserId
    with .lower() on both sides because the two do not reliably agree. An un-normalised id here
    would be a write half that never meets its reader — the exact failure this change ends.
    """
    speaker = "3F2504E0-4F89-11D3-9A0C-0305E82C3301"
    redis = _KeyRedis({_speaker_key("m1", speaker.lower()): "far_field"})

    assert await _worker(redis)._get_noise_reduction("m1", speaker) == "far_field"


@pytest.mark.asyncio
async def test_nobody_configured_anything_and_the_deployment_default_stands() -> None:
    redis = _KeyRedis({})

    assert await _worker(redis)._get_noise_reduction("m1", "s1") is None


@pytest.mark.asyncio
async def test_ending_a_room_forgets_both_of_its_denoising_caches() -> None:
    """Otherwise one entry per meeting, and one per (meeting, speaker), held for the process life.

    WT-427's cache was missing from _cleanup_room entirely; the per-speaker one would have
    repeated that.
    """
    worker = _worker(_KeyRedis({}))
    # What _cleanup_room touches beyond this feature. Built with __new__, so nothing exists yet.
    worker._route_states = {}
    worker._translation_active = {}
    worker._paused_rooms = set()
    worker._room_routes = {}
    worker._stt_prompts = {}
    worker._room_languages = {}
    worker._speaker_locks = {}
    worker._room_noise_reduction["m1"] = ("far_field", 0.0)
    worker._speaker_noise_reduction[("m1", "s1")] = ("off", 0.0)
    worker._speaker_noise_reduction[("m2", "s1")] = ("off", 0.0)

    STTWorker._cleanup_room(worker, "m1")

    assert "m1" not in worker._room_noise_reduction
    assert ("m1", "s1") not in worker._speaker_noise_reduction
    # A different meeting is untouched.
    assert ("m2", "s1") in worker._speaker_noise_reduction


@pytest.mark.asyncio
async def test_an_unreadable_redis_falls_back_to_the_deployment_default() -> None:
    worker = _worker(_ModeRedis(fail=True))

    assert await worker._get_room_noise_reduction("m1") is None
    cast(MagicMock, worker.logger).warning.assert_called_once()
