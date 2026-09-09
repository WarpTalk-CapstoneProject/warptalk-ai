"""Pausing the transcript keeps speech out of the record. WT-605.

THE BUG THESE PIN
    A host paused transcript recording and kept talking. The saved transcript held three
    entries and none of them were those sentences — correctly, because the backend stops
    persisting while paused. The SUMMARY quoted them anyway, in ACTION ITEMS and OPEN
    QUESTIONS, with citations at 0:11 and 0:45 pointing at moments the transcript does not
    contain. `ai_assistant_worker` reads `stt:results` directly and had no idea a pause
    existed; `suggestion_worker` was pinning badges to transcript rows that were never written.

WHAT MUST KEEP WORKING
    Pausing the transcript is not pausing the meeting. The stream keeps flowing because
    translation and dubbing still depend on it, so every test here starts from segments that
    really do arrive while paused — and the most important one is that `__MEETING_END__` still
    gets through, because a host who forgets to press Resume must still get a summary.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from ai_assistant_worker.summary_templates import PAUSE_MARKER_SPEAKER
from ai_assistant_worker.transcript_buffer import buffer_key
from ai_assistant_worker.worker import AIAssistantWorker
from shared.config import SuggestionSettings, WorkerSettings
from shared.control_markers import MEETING_END_MARKER
from shared.schemas import STTResultMessage
from shared.transcript_pause import means_paused, transcript_paused_key
from suggestion_worker.suggester import GeneratedSuggestion, SuggestionDecision
from suggestion_worker.worker import SuggestionWorker
from tests.test_suggestion_worker import RecordingSuggester

MEETING = "019f6a39-a32c-7745-886e-1fe622c1f747"


class PauseAwareRedis:
    """Only what the two gated workers ask of Redis.

    `paused` mirrors the real key's shape: the backend writes a truthy string while recording
    is paused and DELETES the key otherwise, so None here means running — never "0".
    """

    def __init__(self, paused: bool = False, raise_on_get: bool = False) -> None:
        self.paused = paused
        self.raise_on_get = raise_on_get
        self.buffered: dict[str, list[str]] = {}
        self.deleted: list[str] = []
        self.published: list[tuple[str, dict[str, str]]] = []
        self.values: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    async def get(self, key: str) -> str | None:
        if self.raise_on_get:
            raise ConnectionError("redis is down")
        if key == transcript_paused_key(MEETING):
            return "1" if self.paused else None
        if key.endswith(":ai_policy"):
            return '{"allow_external_llm": true}'
        return self.values.get(key)

    async def rpush_capped(self, key: str, value: str, **_: Any) -> None:
        self.buffered.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> list[str]:
        return list(self.buffered.get(key, []))

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.buffered.pop(key, None)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {}

    async def hset(self, key: str, field: str, value: str) -> None:
        self.values[f"{key}#{field}"] = value

    async def set_if_absent(self, key: str, value: str, ttl_seconds: int) -> bool:
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def incr_with_ttl(self, key: str, ttl_seconds: int) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    async def publish(self, stream: str, data: dict[str, str]) -> bytes:
        self.published.append((stream, data))
        return b"1-0"


def _assistant(paused: bool = False, raise_on_get: bool = False) -> tuple[Any, PauseAwareRedis]:
    worker = AIAssistantWorker(settings=WorkerSettings())
    redis = PauseAwareRedis(paused=paused, raise_on_get=raise_on_get)
    worker.redis = redis  # type: ignore[assignment]
    worker._generate_summary = AsyncMock()  # type: ignore[method-assign]
    return worker, redis


def _stt(
    text: str = "con số quý ba là bốn tỷ",
    *,
    timestamp_ms: int = 1_000,
    speaker_id: str = "speaker-1",
) -> dict[bytes, bytes]:
    payload = STTResultMessage(
        segment_id="segment-1",
        meeting_id=MEETING,
        speaker_id=speaker_id,
        text=text,
        language="vi",
        confidence=-0.1,
        timestamp_ms=timestamp_ms,
    ).to_redis()
    return {key.encode(): value.encode() for key, value in payload.items()}


class TestTheCrossRepoContract:
    """The key name and value shape warptalk-backend writes. Changing either breaks both sides
    silently — this side would simply stop seeing pauses and go back to recording them."""

    def test_the_key_is_spelled_the_way_the_backend_writes_it(self) -> None:
        assert transcript_paused_key("abc") == "translationRoom:abc:transcript_paused"

    @pytest.mark.parametrize("raw", ["1", "true", "True", b"1", "PAUSED"])
    def test_a_truthy_flag_means_paused(self, raw: bytes | str) -> None:
        assert means_paused(raw) is True

    @pytest.mark.parametrize("raw", [None, "", "0", "false", b"0"])
    def test_an_absent_or_cleared_flag_means_recording(self, raw: bytes | str | None) -> None:
        """The backend deletes the key, but a flag left behind as "0" must not read as paused:
        that would stop recording a meeting nobody paused, which is the expensive direction."""
        assert means_paused(raw) is False


class TestTheSummaryPath:
    """`ai_assistant_worker` — the path that produced the quoted sentences."""

    @pytest.mark.asyncio
    async def test_a_segment_spoken_while_paused_reaches_neither_copy(self) -> None:
        """Both copies, because either one alone would put it back in the summary.

        `_generate_summary` reads whichever of memory and the Redis buffer is longer, so
        gating one and not the other would fix the bug only until the next restart.
        """
        worker, redis = _assistant(paused=True)

        await worker.process(b"1-0", _stt())

        assert worker._transcripts.get(MEETING, []) == []
        assert redis.buffered == {}

    @pytest.mark.asyncio
    async def test_a_segment_spoken_normally_is_still_accumulated(self) -> None:
        worker, redis = _assistant(paused=False)

        await worker.process(b"1-0", _stt())

        assert worker._transcripts[MEETING] == [("speaker-1", "con số quý ba là bốn tỷ", 1_000)]
        assert len(redis.buffered[buffer_key(MEETING)]) == 1

    @pytest.mark.asyncio
    async def test_the_end_of_meeting_marker_survives_a_room_left_paused(self) -> None:
        """THE ONE THAT MATTERS MOST.

        `__MEETING_END__` travels down `stt:results` like any segment. A pause gate placed in
        front of the control-marker test swallows it whenever the host forgets to press Resume
        before ending the meeting — and then no summary is ever generated, `_transcripts` is
        never released, and the failure is silent at every layer.
        """
        worker, _ = _assistant(paused=True)

        await worker.process(b"1-0", _stt(text=MEETING_END_MARKER, speaker_id="system"))

        worker._generate_summary.assert_awaited_once_with(MEETING)

    @pytest.mark.asyncio
    async def test_an_unreadable_pause_flag_keeps_the_segment(self) -> None:
        """Fail OPEN, matching `IsRoomTranscriptPausedAsync` on the backend.

        One stray sentence in a summary because Redis was down is a cheaper mistake than a
        convincing, complete-looking, empty record of a real meeting.
        """
        worker, _ = _assistant(raise_on_get=True)

        await worker.process(b"1-0", _stt())

        assert worker._transcripts[MEETING] == [("speaker-1", "con số quý ba là bốn tỷ", 1_000)]

    @pytest.mark.asyncio
    async def test_a_pause_is_not_read_as_a_lull_in_the_meeting(self) -> None:
        """WT-605 second half: the model is told the gap was deliberate.

        Left unmarked, a pause reaches the model as a jump in `t=` and nothing else — which
        invites it to join the sentence before to the sentence after, or to report that the
        room went quiet for a minute.
        """
        worker, redis = _assistant()

        await worker.process(b"1-0", _stt(text="mở đầu cuộc họp", timestamp_ms=1_000))
        redis.paused = True
        await worker.process(b"1-0", _stt(text="chuyện riêng tư", timestamp_ms=5_000))
        await worker.process(b"1-0", _stt(text="vẫn chuyện riêng tư", timestamp_ms=9_000))
        redis.paused = False
        await worker.process(b"1-0", _stt(text="chốt lại như vậy", timestamp_ms=20_000))

        rendered = worker._render_transcript(
            MEETING,
            worker._transcripts[MEETING],
            SimpleNamespace(name_for=lambda speaker: "Nhi"),
            1_000,
        )

        lines = rendered.splitlines()
        assert len(lines) == 3
        assert PAUSE_MARKER_SPEAKER in lines[1]
        # One marker for one pause, not one per dropped segment.
        assert worker._pause_gaps[MEETING] == [[5_000, 9_000]]
        # And what was said inside it is nowhere in what the model reads.
        assert "riêng tư" not in rendered

    @pytest.mark.asyncio
    async def test_a_finished_meeting_releases_its_pause_windows_too(self) -> None:
        worker, redis = _assistant(paused=True)
        await worker.process(b"1-0", _stt())
        assert worker._pause_gaps[MEETING]

        await worker._forget_meeting(MEETING)

        assert MEETING not in worker._pause_gaps
        assert MEETING not in worker._gap_open


class TestTheSuggestionPath:
    """`suggestion_worker` — badges pinned to transcript rows that are never written."""

    def _worker(self, paused: bool) -> tuple[SuggestionWorker, RecordingSuggester]:
        suggester = RecordingSuggester(
            decision=SuggestionDecision(
                should_suggest=True, category="action", confidence=0.9, token_count=40
            ),
            suggestion=GeneratedSuggestion(
                content="Chưa ai nhận phần tích hợp.", detail="", category="action", token_count=1
            ),
        )
        worker = SuggestionWorker(
            suggestion_settings=SuggestionSettings(enabled=True),
            suggester=suggester,
            settings=WorkerSettings(),
        )
        worker.redis = PauseAwareRedis(paused=paused)  # type: ignore[assignment]
        return worker, suggester

    @pytest.mark.asyncio
    async def test_no_badge_and_no_tokens_while_recording_is_paused(self) -> None:
        """Both stages, not just publishing: decide and generate are paid calls each."""
        worker, suggester = self._worker(paused=True)

        await worker.process(b"1-0", _stt(text="chúng ta cần chốt deadline cho tích hợp này"))

        assert suggester.decide_calls == []
        assert suggester.generate_calls == 0
        assert worker.redis.published == []  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_paused_speech_does_not_linger_in_the_context_window(self) -> None:
        """The slower version of the same leak.

        The window is what the decide stage is shown. Remembering a paused segment would put
        it back into a prompt one segment later, after the host took it off the record.
        """
        worker, _ = self._worker(paused=True)

        await worker.process(b"1-0", _stt(text="chuyện riêng tư không muốn ghi lại đâu nhé"))

        assert not worker._windows.get(MEETING)

    @pytest.mark.asyncio
    async def test_an_ordinary_segment_still_gets_its_badge(self) -> None:
        worker, suggester = self._worker(paused=False)

        await worker.process(b"1-0", _stt(text="chúng ta cần chốt deadline cho tích hợp này"))

        assert len(suggester.decide_calls) == 1
        assert suggester.generate_calls == 1
