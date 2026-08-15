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
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

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


def _worker(redis: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker._room_noise_reduction = {}
    worker.redis = redis
    return worker


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
async def test_the_mode_is_read_once_per_room() -> None:
    # It is on the hot path — every chunk of every speaker.
    redis = _ModeRedis("far_field")
    worker = _worker(redis)

    for _ in range(8):
        await worker._get_room_noise_reduction("m1")

    assert redis.reads == 1


@pytest.mark.asyncio
async def test_an_unreadable_redis_falls_back_to_the_deployment_default() -> None:
    worker = _worker(_ModeRedis(fail=True))

    assert await worker._get_room_noise_reduction("m1") is None
    cast(MagicMock, worker.logger).warning.assert_called_once()
