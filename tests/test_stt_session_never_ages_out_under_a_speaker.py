"""A speaker's Realtime socket is never closed under them, and an empty streamed commit is retried.

Room 01a100d8, 3 Oct 2026. The host's prewarm claimed a pool socket that had already lived ~45
minutes; the max-age sweep closed it five minutes into the meeting (`age_s 3000`), the next turn
found no session (`append_failed`) and paid a 2.8s reconnect inline. Earlier the same host's first
sentence — 2.9s of clear speech — had come back from a streamed commit as nothing at all.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import stt_worker.model as stt_model
from shared.openai_options import REALTIME_SESSION_MAX_AGE_S
from stt_worker.model import (
    SESSION_RENEW_LEAD_S,
    WARM_SOCKET_MIN_REMAINING_S,
    OpenAISTT,
)

KEY = ("meeting-1", "speaker-1")
MINUTE = 60.0


def _pool_model(ages: list[float], now: float) -> OpenAISTT:
    model = OpenAISTT.__new__(OpenAISTT)
    model._warm_sessions = deque({"opened_at": now - age, "age": age} for age in ages)
    model._close_session = AsyncMock()  # type: ignore[method-assign]
    return model


class TestClaimHandsOutSocketsWithRoomForAMeeting:
    async def test_the_youngest_socket_is_claimed_first(self) -> None:
        now = time.monotonic()
        model = _pool_model([10 * MINUTE, 1 * MINUTE], now)

        claimed = await model._claim_warm_socket()

        assert claimed is not None and claimed["age"] == 1 * MINUTE

    async def test_a_socket_too_close_to_the_cap_is_not_handed_out(self) -> None:
        now = time.monotonic()
        age = REALTIME_SESSION_MAX_AGE_S - WARM_SOCKET_MIN_REMAINING_S + MINUTE
        model = _pool_model([age], now)

        assert await model._claim_warm_socket() is None
        await asyncio.sleep(0)
        assert model._close_session.await_count == 1


class _Conn:
    def __init__(self) -> None:
        self.session = MagicMock()
        self.session.update = AsyncMock()


def _renewing_model(age_s: float, **state: Any) -> tuple[OpenAISTT, dict[str, Any], _Conn]:
    model = OpenAISTT.__new__(OpenAISTT)
    model._client = MagicMock()
    model.model = "gpt-realtime-whisper"
    old = {
        "manager": MagicMock(),
        "conn": MagicMock(),
        "epoch": 1,
        "opened_at": time.monotonic() - age_s,
        "last_used": time.monotonic(),
        "language": "vi",
        "prompt": None,
        "languages": ("vi",),
        "allowed_languages": {"vi", "en"},
        "keywords": ("WarpTalk",),
        "noise_reduction": None,
        "dirty": False,
        "in_flight": False,
        **state,
    }
    model._sessions = {KEY: old}
    model._session_epoch = 1
    fresh = _Conn()
    model._claim_warm_socket = AsyncMock(  # type: ignore[method-assign]
        return_value={"manager": MagicMock(), "conn": fresh, "opened_at": time.monotonic()}
    )
    model._schedule_warm_refill = MagicMock()  # type: ignore[method-assign]
    model._session_payload = MagicMock(return_value={})  # type: ignore[method-assign]
    model._close_session = AsyncMock()  # type: ignore[method-assign]
    model._close_session_later = AsyncMock()  # type: ignore[method-assign]
    return model, old, fresh


async def _settle(model: OpenAISTT) -> None:
    for _ in range(200):
        if not getattr(model, "_renewing", None):
            return
        await asyncio.sleep(0.01)


class TestActiveSessionsAreRenewedBetweenTurns:
    async def test_an_aging_session_is_swapped_for_a_configured_fresh_one(self) -> None:
        model, old, fresh = _renewing_model(REALTIME_SESSION_MAX_AGE_S - SESSION_RENEW_LEAD_S + 1)

        assert model.renew_aging_sessions() == 1
        await _settle(model)

        renewed = model._sessions[KEY]
        assert renewed["conn"] is fresh
        assert renewed["epoch"] == 2, "a commit naming the old epoch must resend its audio"
        assert renewed["language"] == "vi" and renewed["keywords"] == ("WarpTalk",)
        fresh.session.update.assert_awaited_once()
        # The old socket stays open for a commit still reading from it, then closes.
        model._close_session_later.assert_awaited_once()

    async def test_a_young_session_is_left_alone(self) -> None:
        model, _old, _fresh = _renewing_model(5 * MINUTE)

        assert model.renew_aging_sessions() == 0

    async def test_a_turn_in_the_buffer_is_never_split_across_sockets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stt_model, "_RENEW_POLL_S", 0.01)
        model, old, fresh = _renewing_model(
            REALTIME_SESSION_MAX_AGE_S - SESSION_RENEW_LEAD_S + 1, dirty=True
        )

        model.renew_aging_sessions()
        await asyncio.sleep(0.05)
        assert model._sessions[KEY] is old, "swapped while frames of a turn were buffered"

        old["dirty"] = False  # the turn was committed
        await _settle(model)
        assert model._sessions[KEY]["conn"] is fresh

    async def test_a_commit_in_flight_holds_the_swap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(stt_model, "_RENEW_POLL_S", 0.01)
        model, old, _fresh = _renewing_model(
            REALTIME_SESSION_MAX_AGE_S - SESSION_RENEW_LEAD_S + 1, in_flight=True
        )

        model.renew_aging_sessions()
        await asyncio.sleep(0.05)
        assert model._sessions[KEY] is old
        old["in_flight"] = False
        await _settle(model)
        assert model._sessions[KEY] is not old

    async def test_a_session_replaced_meanwhile_is_not_overwritten(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(stt_model, "_RENEW_POLL_S", 0.01)
        model, _old, _fresh = _renewing_model(
            REALTIME_SESSION_MAX_AGE_S - SESSION_RENEW_LEAD_S + 1, dirty=True
        )

        model.renew_aging_sessions()
        await asyncio.sleep(0.03)
        replacement = {"conn": MagicMock(), "epoch": 5}
        model._sessions[KEY] = replacement  # a language change reopened it
        await _settle(model)

        assert model._sessions[KEY] is replacement
        model._close_session.assert_awaited()  # the prepared socket is not leaked


def _transcribing_model(results: list[tuple[str, float]]) -> tuple[OpenAISTT, list[Any]]:
    model = OpenAISTT.__new__(OpenAISTT)
    model.model = "gpt-realtime-whisper"
    model._sessions = {}
    calls: list[Any] = []

    async def _via_session(*_args: Any, **kwargs: Any) -> tuple[str, float]:
        calls.append(kwargs.get("streamed_epoch"))
        return results[len(calls) - 1]

    model._transcribe_via_session = _via_session  # type: ignore[method-assign]
    return model, calls


class TestEmptyStreamedCommitIsRetried:
    async def test_speech_that_came_back_empty_is_sent_again(self) -> None:
        model, calls = _transcribing_model([("", -1.0), ("Xin chào mọi người.", -0.1)])

        segments = await model.transcribe(
            b"\x10\x00" * 16000 * 3,
            sample_rate=16000,
            language="vi",
            meeting_id=KEY[0],
            speaker_id=KEY[1],
            streamed_epoch=3,
            speech_ms=2900,
        )

        assert calls == [3, None], "the retry must send the audio, not commit the buffer again"
        assert [s.text for s in segments] == ["Xin chào mọi người."]

    async def test_a_cough_is_believed(self) -> None:
        model, calls = _transcribing_model([("", -1.0)])

        segments = await model.transcribe(
            b"\x10\x00" * 16000,
            sample_rate=16000,
            language="vi",
            meeting_id=KEY[0],
            speaker_id=KEY[1],
            streamed_epoch=3,
            speech_ms=300,
        )

        assert segments == [] and calls == [3]

    async def test_a_non_streamed_empty_result_is_not_retried(self) -> None:
        model, calls = _transcribing_model([("", -1.0)])

        await model.transcribe(
            b"\x10\x00" * 16000 * 3,
            sample_rate=16000,
            language="vi",
            meeting_id=KEY[0],
            speaker_id=KEY[1],
            speech_ms=2900,
        )

        assert calls == [None]
