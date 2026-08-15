"""WT-421 — a transcript's clock must be the meeting's, not a track's.

`chunk_offset_ms` was `chunk.chunk_index * chunk_duration_ms`: the chunk's position within its
TRACK. `chunk_index` restarts at 0 whenever an ingress track reconnects, so every reconnect sent
the transcript's clock back to the start of the meeting.

The web side already defended against the symptom — dedupeTranscriptSegments carries "Arrival
order stays valid when a reconnected ingress track resets startTimeMs" — while publishing under a
contract that says "elapsed ms from room start".

Production, 15 Aug: the same sentence was stamped 15:33 for one participant and 15:28 for another.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker.worker import STTWorker

ANCHOR_MS = 1_700_000_000_000


class _AnchorRedis:
    """A Redis with working SET NX semantics, and a counter so the caching can be observed."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, str] = {}
        self.set_calls = 0
        self._fail = fail

    async def set_if_absent(self, key: str, value: bytes | str, _ttl: int) -> bool:
        if self._fail:
            raise RuntimeError("redis is down")
        self.set_calls += 1
        if key in self.store:
            return False
        self.store[key] = value if isinstance(value, str) else value.decode()
        return True

    async def get(self, key: str) -> str | None:
        if self._fail:
            raise RuntimeError("redis is down")
        return self.store.get(key)


def _worker(redis: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker._transcript_anchors = {}
    worker.redis = redis
    return worker


def _chunk(index: int, timestamp_ms: int, meeting_id: str = "m1") -> AudioChunkMessage:
    return AudioChunkMessage(
        meeting_id=meeting_id,
        speaker_id="s1",
        chunk_index=index,
        audio_data=b"\x00\x00",
        language="vi",
        sample_rate=16000,
        timestamp_ms=timestamp_ms,
    )


@pytest.mark.asyncio
async def test_elapsed_is_measured_from_the_meeting_not_the_track() -> None:
    worker = _worker(_AnchorRedis())

    first = await worker._elapsed_ms(_chunk(0, ANCHOR_MS))
    later = await worker._elapsed_ms(_chunk(5, ANCHOR_MS + 30_000))

    assert first == 0
    assert later == 30_000


@pytest.mark.asyncio
async def test_a_reconnect_does_not_send_the_clock_back_to_zero() -> None:
    """The bug itself.

    An ingress reconnect restarts chunk_index at 0. Under the old arithmetic that produced offset
    0 — thirty minutes into the meeting.
    """
    worker = _worker(_AnchorRedis())

    await worker._elapsed_ms(_chunk(0, ANCHOR_MS))
    after_reconnect = await worker._elapsed_ms(_chunk(0, ANCHOR_MS + 1_800_000))

    assert after_reconnect == 1_800_000, "a reconnect reset the transcript clock"


@pytest.mark.asyncio
async def test_two_replicas_agree_on_the_same_meeting() -> None:
    """The other half of the report: two seats, two timestamps for one sentence.

    SET NX means the first worker to see the room claims the anchor and every other worker reads
    back the value that won — so a second replica joining later measures from the same instant.
    """
    shared_redis = _AnchorRedis()
    first_replica = _worker(shared_redis)
    second_replica = _worker(shared_redis)

    await first_replica._elapsed_ms(_chunk(0, ANCHOR_MS))
    from_second = await second_replica._elapsed_ms(_chunk(0, ANCHOR_MS + 45_000))

    assert from_second == 45_000


@pytest.mark.asyncio
async def test_the_anchor_is_resolved_once_per_room() -> None:
    # It is on the hot path — every chunk of every speaker. A Redis round trip per chunk would be
    # a real cost for a value that cannot change.
    redis = _AnchorRedis()
    worker = _worker(redis)

    for index in range(10):
        await worker._elapsed_ms(_chunk(index, ANCHOR_MS + index * 1000))

    assert redis.set_calls == 1


@pytest.mark.asyncio
async def test_each_meeting_gets_its_own_anchor() -> None:
    # The negative control for the caching above: one dict keyed by meeting, not one value.
    worker = _worker(_AnchorRedis())

    await worker._elapsed_ms(_chunk(0, ANCHOR_MS, meeting_id="m1"))
    other = await worker._elapsed_ms(_chunk(0, ANCHOR_MS + 999_999, meeting_id="m2"))

    assert other == 0, "a second meeting measured from the first meeting's start"


@pytest.mark.asyncio
async def test_clock_skew_cannot_produce_a_negative_offset() -> None:
    # Skew between the gateway that stamps the chunk and this worker is real, and a negative
    # offset formats into nonsense on the panel rather than failing loudly.
    worker = _worker(_AnchorRedis())

    await worker._elapsed_ms(_chunk(0, ANCHOR_MS))
    earlier = await worker._elapsed_ms(_chunk(1, ANCHOR_MS - 5_000))

    assert earlier == 0


@pytest.mark.asyncio
async def test_an_unavailable_redis_keeps_transcribing() -> None:
    """Wrong by a constant is recoverable. Refusing to transcribe is not."""
    worker = _worker(_AnchorRedis(fail=True))

    first = await worker._elapsed_ms(_chunk(0, ANCHOR_MS))
    later = await worker._elapsed_ms(_chunk(1, ANCHOR_MS + 20_000))

    assert first == 0
    assert later == 20_000
    cast(MagicMock, worker.logger).warning.assert_called()
