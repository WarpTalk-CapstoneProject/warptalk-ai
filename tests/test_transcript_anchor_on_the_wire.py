"""WT-655 — a transcript offset is useless without the instant it is measured from.

WT-421 gave the pipeline a stable per-room anchor (`STTWorker._elapsed_ms`) so `start_ms` and
`end_ms` stopped being a per-track chunk counter. It never published the anchor itself, so what
reached the database was a list of durations: 0ms, 4200ms, 9800ms, with nothing saying what
moment 0 was. "Click a transcript line, the recording seeks to that moment" is fully built on the
.NET and web side and dead in production for exactly that reason.

These pin the contract the persisting consumer is already written against:

  * the field is named `anchor_ms` on the wire, and it round-trips;
  * a message published before the field existed reads back as 0, and 0 means NOT STATED —
    the consumer stores the anchor only when it is > 0, first write wins;
  * all three of the worker's publish sites carry it, not just the completed segment. In flash
    mode most lines are published early, and a final-chunk marker may be the only message a
    silent meeting ever emits.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.model import TranscribedSegment
from stt_worker.worker import STTWorker

# A plausible wall clock rather than a small integer: the field is a unix epoch millisecond, and
# a test that passes for 1234 would also pass for a value the consumer reads as 1970.
ANCHOR_MS = 1_700_000_000_000


# ── the wire format ─────────────────────────────────────────────────────────────


class TestAnchorOnTheWire:
    def test_roundtrips(self) -> None:
        original = STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text="Hello",
            language="en",
            start_ms=4_200,
            end_ms=6_100,
            anchor_ms=ANCHOR_MS,
        )

        restored = STTResultMessage.from_redis(original.to_redis())

        assert restored.anchor_ms == ANCHOR_MS

    def test_the_field_is_named_anchor_ms(self) -> None:
        # The consumer in the .NET repo reads this Redis stream field by name and parses it as a
        # long. Renaming it does not break a test over there; it silently stops seeking working.
        payload = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en", anchor_ms=ANCHOR_MS
        ).to_redis()

        assert payload["anchor_ms"] == str(ANCHOR_MS)

    def test_an_older_message_reads_back_as_not_stated(self) -> None:
        """What every message already in a live stream looks like: no such field.

        0 has to survive as "no origin stated" rather than becoming an epoch timestamp, because
        the stored anchor is first-write-wins and can never be corrected afterwards.
        """
        payload = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en", anchor_ms=ANCHOR_MS
        ).to_redis()
        del payload["anchor_ms"]

        assert STTResultMessage.from_redis(payload).anchor_ms == 0

    def test_not_stated_is_sent_explicitly_rather_than_omitted(self) -> None:
        # Unlike `prosody`, 0 is a real answer here and not a placeholder for a skipped
        # measurement — so the field is always on the wire and the consumer's ">0" gate is the
        # only place the distinction is made.
        payload = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text="hi", language="en"
        ).to_redis()

        assert payload["anchor_ms"] == "0"


# ── what the worker publishes ───────────────────────────────────────────────────


def _worker(redis: Any, worker_settings: WorkerSettings) -> STTWorker:
    """The __new__ construction every STT worker test uses; no __init__, no event loop."""
    worker = STTWorker.__new__(STTWorker)
    worker.settings = worker_settings
    worker.stt_settings = STTSettings()
    worker.redis = redis
    worker.logger = MagicMock()
    worker._paused_rooms = set()
    worker._stt_prompts = {}
    worker._room_languages = {}
    worker.model = MagicMock()
    return worker


def _anchor_in_redis(redis: Any) -> None:
    """Answer the anchor key with ANCHOR_MS and every other key exactly as the fixture did.

    Scoped to the one key on purpose: `process()` reads route state, keywords and noise reduction
    through the same `get`, and a blanket return value would be changing what this test is about.
    """
    redis._redis.get = AsyncMock(
        side_effect=lambda key, *_a, **_kw: (
            str(ANCHOR_MS).encode() if "transcript_anchor_ms" in str(key) else None
        )
    )


def _published_results(redis: Any) -> list[dict[str, str]]:
    """Every stt:results payload, once each.

    BaseWorker.publish() XADDs the same payload to the room stream and the global one, so the
    raw call list holds each message twice; deduplicating by segment_id keeps the counts below
    about how many SEGMENTS were published rather than about the fan-out.
    """
    unique: dict[str, dict[str, str]] = {}
    for stream, data in (c.args for c in redis._redis.xadd.call_args_list):
        if "stt:results" in str(stream):
            unique.setdefault(data["segment_id"], data)
    return list(unique.values())


@pytest.mark.asyncio
async def test_a_completed_segment_carries_the_rooms_anchor(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    _anchor_in_redis(mock_redis_client)
    worker = _worker(mock_redis_client, worker_settings)
    worker.model.transcribe = AsyncMock(
        return_value=[
            TranscribedSegment(
                text="Hello", language="en", confidence=-0.25, start_ms=0, end_ms=1000
            )
        ]
    )

    await worker.process(
        b"msg-1",
        AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            language="auto",
        ).to_redis(),
    )

    published = _published_results(mock_redis_client)
    assert published, "nothing reached stt:results"
    for data in published:
        assert int(data["anchor_ms"]) == ANCHOR_MS


@pytest.mark.asyncio
async def test_an_early_sentence_carries_it_too(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    """In flash mode MOST spoken sentences arrive down the early path.

    An anchor on the completed segment alone would leave the majority of a meeting's lines
    unseekable, which reads in the UI as "seeking works sometimes" — the worst of the states.
    """
    _anchor_in_redis(mock_redis_client)
    worker = _worker(mock_redis_client, worker_settings)

    async def fake_transcribe(*_args: Any, **kwargs: Any) -> list[TranscribedSegment]:
        early = kwargs.get("on_early_segment")
        assert early is not None
        await early(
            TranscribedSegment(
                text="Hello there.", language="en", confidence=0.0, start_ms=0, end_ms=0
            )
        )
        return [
            TranscribedSegment(
                text="How are you?", language="en", confidence=0.0, start_ms=0, end_ms=1000
            )
        ]

    worker.model.transcribe = AsyncMock(side_effect=fake_transcribe)

    await worker.process(
        b"msg-1",
        AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            is_final_chunk=True,
        ).to_redis(),
    )

    by_text = {data["text"]: data for data in _published_results(mock_redis_client)}
    assert set(by_text) == {"Hello there.", "How are you?"}
    assert int(by_text["Hello there."]["anchor_ms"]) == ANCHOR_MS
    assert int(by_text["How are you?"]["anchor_ms"]) == ANCHOR_MS


@pytest.mark.asyncio
async def test_the_final_chunk_marker_carries_it(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    # A meeting whose audio produced no text still has an origin, and this empty marker can be
    # the only message it ever publishes.
    _anchor_in_redis(mock_redis_client)
    worker = _worker(mock_redis_client, worker_settings)
    worker.model.transcribe = AsyncMock(return_value=[])

    await worker.process(
        b"msg-1",
        AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            is_final_chunk=True,
        ).to_redis(),
    )

    published = _published_results(mock_redis_client)
    assert len(published) == 1
    assert published[0]["text"] == ""
    assert int(published[0]["anchor_ms"]) == ANCHOR_MS


@pytest.mark.asyncio
async def test_the_anchor_is_read_from_the_cache_not_resolved_again(
    mock_redis_client: Any,
    worker_settings: WorkerSettings,
    sample_audio_bytes: bytes,
) -> None:
    """One Redis round trip per CHUNK, not one per segment.

    `_elapsed_ms` has already resolved and cached the anchor by the time anything is published.
    Resolving again per segment would be a round trip on the hot path for a value that cannot
    change — and on the Redis-unavailable fallback it could hand back a different number than the
    offsets published beside it were computed from.
    """
    _anchor_in_redis(mock_redis_client)
    worker = _worker(mock_redis_client, worker_settings)
    worker.model.transcribe = AsyncMock(
        return_value=[
            TranscribedSegment(text=str(i), language="en", confidence=-0.2, start_ms=0, end_ms=100)
            for i in range(4)
        ]
    )

    await worker.process(
        b"msg-1",
        AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        ).to_redis(),
    )

    anchor_reads = [
        c
        for c in mock_redis_client._redis.get.call_args_list
        if "transcript_anchor_ms" in str(c.args[0])
    ]
    assert len(anchor_reads) == 1, "the anchor was re-resolved per segment"
    assert len(_published_results(mock_redis_client)) == 4


def test_a_worker_that_never_resolved_an_anchor_states_none() -> None:
    """The honest degradation. 0 travels as "not stated" and the consumer ignores it.

    Inventing a wall clock here would be worse than saying nothing: the stored origin is
    first-write-wins, so a wrong first write can never be corrected afterwards.
    """
    worker = STTWorker.__new__(STTWorker)

    # getattr-defensive by design — the attribute genuinely does not exist on a worker built
    # with __new__, which is how every test suite in this repo builds one.
    assert worker._cached_transcript_anchor("meeting-1") == 0
