"""A live meeting is transcribed before anybody presses Start Translation.

Production, reported after a release: captions and the transcript panel stayed empty for the
whole first half of a meeting, and only filled once translation was started. The web client was
correct and so was the ingress worker — ai#36 had already removed this exact gate there, under
the heading "transcribe every live meeting, translate only when asked". It was not removed, only
moved: STT still refused any room whose ``room_status`` was not IN_PROGRESS.

WAITING is not a hypothetical state for a room with people talking in it. The JOIN path itself
publishes the routes payload, so a participant arriving before the host starts the meeting is
what writes ``room_status: "WAITING"`` into the very key this gate reads.
"""

import pytest

from stt_worker.worker import STTWorker


class _NoRedis:
    """The gate must not reach Redis once the state is already cached in memory."""

    async def get(self, key: str):  # pragma: no cover - called only if the cache misses
        raise AssertionError(f"unexpected Redis read for {key}")


def _worker(state: str | None) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker._route_states = {} if state is None else {"room-1": state}
    worker.redis = _NoRedis()
    return worker


class TestRoomStateGate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "state", ["WAITING", "SCHEDULED", "IN_PROGRESS", "AUDIO_ROUTING_ACTIVE"]
    )
    async def test_a_live_or_not_yet_started_room_is_transcribed(self, state: str):
        assert await _worker(state)._room_state_allows_stt("room-1") is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("state", ["ENDED", "CANCELLED", "FAILED", "EXPIRED"])
    async def test_a_finished_room_still_refuses_stale_audio(self, state: str):
        # The one thing this gate is still for: a queued chunk must not append to the
        # transcript of a meeting that is over.
        assert await _worker(state)._room_state_allows_stt("room-1") is False

    @pytest.mark.asyncio
    async def test_an_unknown_state_fails_open(self):
        # A room whose status nobody has published yet, an older payload, an unreadable
        # cache — all of them are live speech until something says otherwise.
        worker = STTWorker.__new__(STTWorker)
        worker._route_states = {}

        class _EmptyRedis:
            async def get(self, key: str):
                return None

        worker.redis = _EmptyRedis()
        assert await worker._room_state_allows_stt("room-1") is True
