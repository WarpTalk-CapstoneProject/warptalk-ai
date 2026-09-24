"""End to end through the worker: STT segments in, clean sentences on `transcript:clean` out.

The contract being pinned here is the one the backend reads:

    revision 0, source="prepass"  — published the moment the sentence closes
    revision 1, source="llm"      — same sentence_id, same segment_ids, only when it verifies

and the failure shape that matters most: when the model is slow, wrong or absent, revision 0 is
still there and is still the line.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from shared.config import WorkerSettings
from shared.control_markers import MEETING_END_MARKER, SYSTEM_SPEAKER_ID
from shared.schemas import STTResultMessage
from shared.transcript_pause import transcript_paused_key
from transcript_clean_worker.config import TranscriptCleanSettings
from transcript_clean_worker.llm_cleaner import CleanedSentence
from transcript_clean_worker.worker import TranscriptCleanWorker

MEETING_ID = "11111111-1111-1111-1111-111111111111"
SPEAKER = "22222222-2222-2222-2222-222222222222"


class FakeRedis:
    """Only what this worker touches: one GET (the pause flag) and the stream publishes."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.published: list[tuple[str, dict[str, str]]] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def publish(self, stream: str, data: dict[str, str]) -> bytes:
        self.published.append((stream, data))
        return b"1-0"


class FakeCleaner:
    """Stands in for LLMCleaner: returns canned answers, or hangs, or is unavailable."""

    def __init__(
        self,
        results: list[CleanedSentence | None] | None = None,
        *,
        available: bool = True,
        hang: bool = False,
    ) -> None:
        self.results = results or []
        self._available = available
        self.hang = hang
        self.calls: list[tuple[str, str, str | None, str]] = []
        self.rejections: dict[str, int] = {}

    @property
    def is_available(self) -> bool:
        return self._available

    async def load(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def clean(
        self,
        raw: str,
        language: str,
        *,
        prepass_text: str | None = None,
        previous_line: str = "",
    ) -> CleanedSentence | None:
        self.calls.append((raw, language, prepass_text, previous_line))
        if self.hang:
            await asyncio.sleep(30)
        return self.results.pop(0) if self.results else None


def build_worker(
    cleaner: Any = None, **overrides: Any
) -> tuple[TranscriptCleanWorker, FakeRedis, Any]:
    fake_cleaner = cleaner if cleaner is not None else FakeCleaner(available=False)
    worker = TranscriptCleanWorker(
        clean_settings=TranscriptCleanSettings(**overrides),
        cleaner=fake_cleaner,
        settings=WorkerSettings(),
    )
    redis = FakeRedis()
    worker.redis = redis  # type: ignore[assignment]
    return worker, redis, fake_cleaner


def stt(
    text: str,
    *,
    language: str = "en",
    speaker_id: str = SPEAKER,
    segment_id: str | None = None,
    start_ms: int = 0,
    end_ms: int = 1000,
    clean_text: str | None = None,
) -> dict[bytes, bytes]:
    payload = STTResultMessage(
        segment_id=segment_id or str(uuid.uuid4()),
        meeting_id=MEETING_ID,
        speaker_id=speaker_id,
        text=text,
        language=language,
        start_ms=start_ms,
        end_ms=end_ms,
        clean_text=clean_text,
    ).to_redis()
    return {key.encode(): value.encode() for key, value in payload.items()}


def clean_messages(redis: FakeRedis) -> list[dict[str, str]]:
    """Messages on the GLOBAL transcript:clean stream — the one the backend consumes."""
    return [data for stream, data in redis.published if stream == "transcript:clean"]


async def settle() -> None:
    """Let the detached LLM tasks run."""
    for _ in range(5):
        await asyncio.sleep(0)


class TestTheWireContract:
    async def test_revision_zero_then_revision_one(self):
        cleaner = FakeCleaner(
            [
                CleanedSentence(
                    text="We should ship it today.", self_repair=False, deleted_indices=(0,)
                )
            ]
        )
        worker, redis, _ = build_worker(cleaner)
        first, second = str(uuid.uuid4()), str(uuid.uuid4())

        await worker.process(b"1-0", stt("um we should ship it", segment_id=first, end_ms=1000))
        await worker.process(b"1-1", stt("today.", segment_id=second, start_ms=1200, end_ms=2000))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        await settle()

        messages = clean_messages(redis)
        assert [message["revision"] for message in messages] == ["0", "1"]
        prepass_message, llm_message = messages
        assert prepass_message["source"] == "prepass"
        assert prepass_message["clean_text"] == "We should ship it today."
        assert json.loads(prepass_message["segment_ids"]) == [first, second]
        assert prepass_message["speaker_id"] == SPEAKER
        assert prepass_message["language"] == "en"
        assert llm_message["source"] == "llm"
        assert llm_message["sentence_id"] == prepass_message["sentence_id"]
        assert json.loads(llm_message["segment_ids"]) == [first, second]

    async def test_it_publishes_to_the_per_meeting_stream_too(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("We should ship it today."))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        streams = [stream for stream, _ in redis.published]
        assert streams == [f"transcript:clean:{MEETING_ID}", "transcript:clean"]

    async def test_a_self_repair_is_flagged_on_revision_one(self):
        cleaner = FakeCleaner(
            [CleanedSentence(text="Họp thứ ba.", self_repair=True, deleted_indices=(1, 2, 3, 4))]
        )
        worker, redis, _ = build_worker(cleaner)
        await worker.process(b"1-0", stt("họp thứ hai, à không, thứ ba", language="vi"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        await settle()

        revision_one = clean_messages(redis)[1]
        assert revision_one["flags"] == "self_repair"
        assert revision_one["clean_text"] == "Họp thứ ba."

    async def test_a_non_guid_segment_id_is_dropped_rather_than_breaking_the_row(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("We should ship it today.", segment_id="not-a-guid"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        assert json.loads(clean_messages(redis)[0]["segment_ids"]) == []


class TestWhenTheModelDoesNotAnswer:
    async def test_a_hanging_llm_leaves_only_revision_zero(self):
        worker, redis, _ = build_worker(FakeCleaner(hang=True))
        await worker.process(b"1-0", stt("um we should ship it today"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        await settle()

        assert [message["revision"] for message in clean_messages(redis)] == ["0"]
        assert clean_messages(redis)[0]["clean_text"] == "We should ship it today."

    async def test_a_refused_answer_leaves_only_revision_zero(self):
        worker, redis, _ = build_worker(FakeCleaner([None]))
        await worker.process(b"1-0", stt("um we should ship it today"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        await settle()

        assert [message["revision"] for message in clean_messages(redis)] == ["0"]

    async def test_without_a_model_the_prepass_line_is_the_line(self):
        worker, redis, cleaner = build_worker()
        await worker.process(b"1-0", stt("um we should ship it today"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        await settle()

        assert cleaner.calls == []
        assert len(clean_messages(redis)) == 1


class TestGates:
    async def test_the_meeting_end_marker_flushes_and_is_not_speech(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("We will ship it and"))
        assert clean_messages(redis) == []

        await worker.process(
            b"1-1", stt(MEETING_END_MARKER, speaker_id=SYSTEM_SPEAKER_ID, language="system")
        )

        [message] = clean_messages(redis)
        # Cut where the meeting ended, with no words added. The full stop is the shared
        # terminal-punctuation policy (shared/disfluency/punctuation.py), which every
        # single-segment line already goes through — not this stage completing a sentence.
        assert message["clean_text"] == "We will ship it and."
        assert MEETING_ID not in worker._segmenters

    async def test_a_paused_transcript_publishes_nothing(self):
        worker, redis, _ = build_worker()
        redis.values[transcript_paused_key(MEETING_ID)] = "1"

        await worker.process(b"1-0", stt("We should ship it today."))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        assert clean_messages(redis) == []

    async def test_a_disabled_worker_consumes_and_publishes_nothing(self):
        worker, redis, _ = build_worker(enabled=False)
        await worker.process(b"1-0", stt("We should ship it today."))
        assert redis.published == []

    async def test_a_filler_only_turn_produces_no_line(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("Ummm"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")
        assert clean_messages(redis) == []

    async def test_a_terminal_room_status_flushes_the_open_line(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("We will ship it and"))
        await worker._on_route_status_changed(MEETING_ID, "ENDED")

        assert len(clean_messages(redis)) == 1


class TestThePrepassTier:
    async def test_it_runs_the_prepass_when_the_producer_did_not(self):
        worker, redis, _ = build_worker()
        await worker.process(b"1-0", stt("um so we we should ship it"))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        message = clean_messages(redis)[0]
        assert message["clean_text"] == "So we should ship it."
        # The prepass was unsure about the discourse marker "so" and said so.
        assert message["flags"] == "escalate"

    async def test_it_uses_the_producers_clean_text_when_there_is_one(self):
        worker, redis, _ = build_worker()
        await worker.process(
            b"1-0", stt("um we should ship it today", clean_text="We should ship it today.")
        )
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        assert clean_messages(redis)[0]["clean_text"] == "We should ship it today."

    async def test_the_idle_timer_closes_a_line_nobody_finished(self):
        worker, redis, _ = build_worker(idle_flush_ms=4000)
        await worker.process(b"1-0", stt("We will ship it and"))

        segmenter = worker._segmenters[MEETING_ID]
        arrived = segmenter._buffer.segments[-1].arrived_at_ms  # type: ignore[union-attr]
        for sentence in segmenter.flush_idle(arrived + 5000):
            await worker._publish(MEETING_ID, sentence)

        assert len(clean_messages(redis)) == 1

    async def test_a_backchannel_does_not_cut_the_speakers_sentence(self):
        worker, redis, _ = build_worker()
        other = "33333333-3333-3333-3333-333333333333"
        await worker.process(b"1-0", stt("I think that", end_ms=1000))
        await worker.process(b"1-1", stt("Yeah.", speaker_id=other, start_ms=1100, end_ms=1300))
        await worker.process(b"1-2", stt("we should ship it.", start_ms=1400, end_ms=2400))
        await worker._flush_meeting(MEETING_ID, reason="meeting_end")

        texts = [
            (message["speaker_id"], message["clean_text"]) for message in clean_messages(redis)
        ]
        assert texts == [(other, "Yeah."), (SPEAKER, "I think that we should ship it.")]
