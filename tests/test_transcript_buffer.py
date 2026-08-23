"""WT-536: a finished meeting whose summary said the transcript was empty.

Measured on production before this fix: 37 finished meetings carried the refusal while their
transcript sat safely in the database — 1789 segments, the worst meeting 751 lines. A further
112 refusals were correct, on meetings that genuinely had no speech; those must keep refusing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from ai_assistant_worker.transcript_buffer import (
    buffer_key,
    choose_segments,
    decode_segments,
    encode_segment,
)

SEG_A = ("speaker-1", "we should ship on Friday", 1000)
SEG_B = ("speaker-2", "agreed", 2000)


class TestEncoding:
    def test_a_segment_survives_the_round_trip(self):
        assert decode_segments([encode_segment(SEG_A)]) == [SEG_A]

    def test_non_ascii_survives_it_too(self):
        segment = ("speaker-1", "Chào mọi người", 5)
        assert decode_segments([encode_segment(segment)]) == [segment]

    def test_bytes_from_redis_decode(self):
        assert decode_segments([encode_segment(SEG_A).encode()]) == [SEG_A]

    def test_a_half_written_entry_costs_only_that_line(self):
        # Refusing the whole meeting over one bad entry would reintroduce the very failure
        # this module exists to remove.
        raw = [encode_segment(SEG_A), "{not json", encode_segment(SEG_B)]
        assert decode_segments(raw) == [SEG_A, SEG_B]

    def test_entries_of_the_wrong_shape_are_skipped(self):
        raw = [
            json.dumps(["only-two", 1]),
            json.dumps([1, "speaker is not a string", 2]),
            json.dumps(["s", 7, 2]),
            encode_segment(SEG_A),
        ]
        assert decode_segments(raw) == [SEG_A]

    def test_a_missing_buffer_is_no_segments(self):
        assert decode_segments(None) == []
        assert decode_segments([]) == []


class TestChoosingWhichCopyToSummarise:
    def test_a_restart_mid_meeting_uses_the_buffer(self):
        # Memory holds only what arrived after the process came back; the buffer holds it all.
        assert choose_segments([SEG_B], [SEG_A, SEG_B]) == [SEG_A, SEG_B]

    def test_a_redis_outage_uses_memory(self):
        # The mirror image, and the reason this is not simply "prefer the buffer".
        assert choose_segments([SEG_A, SEG_B], [SEG_A]) == [SEG_A, SEG_B]

    def test_the_ordinary_case_uses_memory(self):
        both = [SEG_A, SEG_B]
        assert choose_segments(both, list(both)) is both

    def test_a_meeting_with_no_speech_still_has_nothing(self):
        # The 112 correct refusals must keep refusing.
        assert choose_segments([], []) == []


class TestTheWorkerUsesIt:
    @staticmethod
    def _worker(in_memory, buffered):
        from ai_assistant_worker.worker import AIAssistantWorker

        worker = AIAssistantWorker.__new__(AIAssistantWorker)
        worker._transcripts = {"m1": list(in_memory)} if in_memory else {}

        async def _lrange(key: str, start: int = 0, stop: int = -1):
            return [encode_segment(s).encode() for s in buffered]

        worker.redis = SimpleNamespace(
            lrange=AsyncMock(side_effect=_lrange),
            rpush_capped=AsyncMock(),
            delete=AsyncMock(),
            get=AsyncMock(return_value=None),
            hgetall=AsyncMock(return_value={}),
            hset=AsyncMock(),
        )
        worker.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        worker.publish = AsyncMock()

        captured: dict[str, str] = {}

        async def _summarize(transcript_text: str, **kwargs):
            captured["transcript"] = transcript_text
            return "a summary"

        async def _structured(transcript_text: str, **kwargs):
            return {"insufficientData": False, "sections": []}

        async def _extract(transcript_text: str, **kwargs):
            return []

        worker._require_assistant = lambda: SimpleNamespace(
            summarize=_summarize,
            extract_action_items=_extract,
            generate_structured_summary=_structured,
        )
        return worker, captured

    async def test_a_meeting_lost_to_a_restart_is_still_summarised(self):
        # The reported bug: memory empty because the process died mid-meeting.
        worker, captured = self._worker(in_memory=[], buffered=[SEG_A, SEG_B])

        await worker._generate_summary("m1")

        assert "we should ship on Friday" in captured["transcript"]
        assert "agreed" in captured["transcript"]
        worker.publish.assert_awaited()

    async def test_a_partial_restart_summarises_the_whole_meeting(self):
        worker, captured = self._worker(in_memory=[SEG_B], buffered=[SEG_A, SEG_B])

        await worker._generate_summary("m1")

        assert "we should ship on Friday" in captured["transcript"]

    async def test_a_meeting_with_no_speech_publishes_nothing(self):
        # Still correct to refuse — 112 of the production refusals were this case.
        worker, captured = self._worker(in_memory=[], buffered=[])

        await worker._generate_summary("m1")

        assert "transcript" not in captured
        worker.publish.assert_not_awaited()

    async def test_a_redis_failure_does_not_lose_a_summary_memory_could_produce(self):
        worker, captured = self._worker(in_memory=[SEG_A], buffered=[])
        worker.redis.lrange = AsyncMock(side_effect=RuntimeError("redis is down"))

        await worker._generate_summary("m1")

        assert "we should ship on Friday" in captured["transcript"]

    async def test_the_buffer_is_released_when_the_meeting_is_done(self):
        worker, _ = self._worker(in_memory=[], buffered=[SEG_A])

        await worker._generate_summary("m1")

        worker.redis.delete.assert_awaited_with(buffer_key("m1"))

    async def test_the_buffer_is_released_for_a_silent_meeting_too(self):
        worker, _ = self._worker(in_memory=[], buffered=[])

        await worker._generate_summary("m1")

        worker.redis.delete.assert_awaited_with(buffer_key("m1"))

    async def test_recovering_from_the_buffer_does_not_crash_on_cleanup(self):
        # `del self._transcripts[id]` used to be unconditional, and a meeting recovered from the
        # buffer has no entry in memory at all — a KeyError there would throw away a summary
        # that had already been published and stored.
        worker, _ = self._worker(in_memory=[], buffered=[SEG_A])

        await worker._generate_summary("m1")  # must not raise

        assert "m1" not in worker._transcripts


class TestTheWriteSide:
    """The buffer only helps if something fills it — and `process` is the only writer."""

    @staticmethod
    def _worker():
        from ai_assistant_worker.worker import AIAssistantWorker

        worker = AIAssistantWorker.__new__(AIAssistantWorker)
        worker._transcripts = {}
        worker.redis = SimpleNamespace(rpush_capped=AsyncMock())
        worker.logger = SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
        return worker

    @staticmethod
    def _stt(text: str = "hello"):
        from shared.schemas import STTResultMessage

        return STTResultMessage(
            meeting_id="m1",
            speaker_id="speaker-1",
            text=text,
            language="en",
            confidence=0.9,
            start_ms=0,
            end_ms=500,
            timestamp_ms=1000,
            segment_id="seg-1",
        ).to_redis()

    async def test_every_segment_is_mirrored_to_the_buffer(self):
        worker = self._worker()

        await worker.process(b"1-1", self._stt("we should ship on Friday"))

        key, value = worker.redis.rpush_capped.await_args.args
        assert key == buffer_key("m1")
        assert decode_segments([value])[0][1] == "we should ship on Friday"

    async def test_the_buffer_is_bounded_and_expires(self):
        # A room left open overnight must not grow a list nobody bounded.
        worker = self._worker()

        await worker.process(b"1-1", self._stt())

        kwargs = worker.redis.rpush_capped.await_args.kwargs
        assert kwargs["max_len"] > 0
        assert kwargs["ttl_seconds"] >= 60 * 60

    async def test_a_redis_failure_does_not_drop_the_segment_from_memory(self):
        # This runs on the live path. The insurance failing must cost the insurance and
        # nothing else.
        worker = self._worker()
        worker.redis.rpush_capped = AsyncMock(side_effect=RuntimeError("redis is down"))

        await worker.process(b"1-1", self._stt("still counted"))

        assert worker._transcripts["m1"][0][1] == "still counted"
