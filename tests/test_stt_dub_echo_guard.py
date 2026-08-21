"""The dub-echo guard — the pipeline must not transcribe the room's own translation.

Production meeting "Hieu Clone" (21 Aug, room 01a0202b-c5a7-78df-a409-e1d697973a25): 77 English
segments in ten minutes credited to a participant who had not spoken. Each one was the English
dub of the other speaker's Vietnamese — played through the listener's speakers, re-captured by
their microphone, transcribed as new speech, then re-translated, re-synthesized, billed, and fed
to the language-override loop as evidence that the listener speaks English. The echo stopped at
00:30 because the person muted themselves, which is the manual version of this guard.

The client half (half-duplex-mic.tsx in warptalk-web) gates on the tab running current code and
on LiveKit's isSpeaking signal, and the meeting above is what got through it. This is the server
half, sitting where every path converges: a segment whose text matches a line the room's own TTS
was just told to speak, in that dub's language and within the recency window, is dropped before
it can become transcript, translation, or language evidence.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker.model import (
    TranscribedSegment,
    _filter_segments,
    _matches_recent_dub,
    _normalize_overheard_text,
)
from stt_worker.worker import STTWorker

# The literal pair from the production meeting: seq 46 (Tuấn, vi) and seq 50 — the dub of it,
# credited to the person whose speakers played it.
_DUB_LINE = "The machine probably only has time now, oh, I'm heading back."
_DUB_RECENT = [_normalize_overheard_text(_DUB_LINE)]


class TestMatchesRecentDub:
    def test_the_production_pair_matches_exactly(self) -> None:
        heard = _normalize_overheard_text(_DUB_LINE)
        assert _matches_recent_dub(heard, _DUB_RECENT)

    def test_case_and_trailing_punctuation_do_not_hide_the_match(self) -> None:
        heard = _normalize_overheard_text(
            "the machine probably only has time now, oh, I'm heading back"
        )
        assert _matches_recent_dub(heard, _DUB_RECENT)

    def test_a_fragment_of_the_dub_line_matches_by_containment(self) -> None:
        # STT often splits one dubbed sentence into pieces; each piece is still the echo.
        heard = _normalize_overheard_text("oh, I'm heading back.")
        assert _matches_recent_dub(heard, _DUB_RECENT)

    def test_small_mishearings_match_by_fuzz(self) -> None:
        heard = _normalize_overheard_text(
            "The machine probably only has time now, oh, I am heading back"
        )
        assert _matches_recent_dub(heard, _DUB_RECENT)

    def test_short_back_channel_is_never_treated_as_echo(self) -> None:
        # "Yeah." said near a dub that also said "Yeah." is ordinary back-channel. Dropping
        # real speech is the one failure mode this guard must not have.
        recent = [_normalize_overheard_text("Yeah.")]
        assert not _matches_recent_dub(_normalize_overheard_text("Yeah."), recent)

    def test_unrelated_speech_does_not_match(self) -> None:
        heard = _normalize_overheard_text("Let's review the deployment plan for tomorrow.")
        assert not _matches_recent_dub(heard, _DUB_RECENT)

    def test_no_recent_dubs_means_no_match(self) -> None:
        assert not _matches_recent_dub(_normalize_overheard_text(_DUB_LINE), [])


class TestFilterSegmentsDropsDubEcho:
    def _segment(self) -> list[dict[str, Any]]:
        return [
            {
                "text": _DUB_LINE,
                "start": 0.0,
                "end": 2.5,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        ]

    def test_the_same_segment_is_kept_without_the_guard_and_dropped_with_it(self) -> None:
        # The mutation pair: identical input, so the ONLY thing separating kept from dropped
        # is recent_dub_texts actually being consulted.
        kept = _filter_segments(self._segment(), "en", 0, {"vi", "en"})
        assert [seg.text for seg in kept] == [_DUB_LINE]

        dropped = _filter_segments(
            self._segment(), "en", 0, {"vi", "en"}, recent_dub_texts=_DUB_RECENT
        )
        assert dropped == []

    def test_real_speech_in_the_other_language_survives_the_guard(self) -> None:
        segments = [
            {
                "text": "Chúng ta xem lại kế hoạch triển khai ngày mai nhé.",
                "start": 0.0,
                "end": 2.0,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.0,
            }
        ]
        kept = _filter_segments(segments, "vi", 0, {"vi", "en"}, recent_dub_texts=_DUB_RECENT)
        assert len(kept) == 1

    def test_echo_mislabelled_with_the_declared_language_is_still_dropped(self) -> None:
        # The Hieu Clone shape exactly: English dub text arriving from a speaker declared vi.
        # Latin text carries no unambiguous language evidence, so the label resolves to the
        # DECLARATION — a guard conditioned on the label matching the dub's language misses
        # precisely this. It is also the evidence protection: _learn_language_evidence only
        # ever sees what _filter_segments returns, so an empty return IS the protection.
        dropped = _filter_segments(
            self._segment(), "vi", 0, {"vi", "en"}, recent_dub_texts=_DUB_RECENT
        )
        assert dropped == []


def _worker(redis: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker.redis = redis
    return worker


class _StreamRedis:
    """Raw-client stub: only xrevrange, counting calls so the cache is observable."""

    def __init__(self, entries: list[Any] | None = None, fail: bool = False) -> None:
        self._entries = entries or []
        self._fail = fail
        self.calls = 0

    @property
    def redis(self) -> _StreamRedis:
        return self

    async def xrevrange(self, _stream: str, count: int = 0) -> list[Any]:
        self.calls += 1
        if self._fail:
            raise RuntimeError("redis is down")
        return self._entries


def _entry(text: str, target_lang: str, age_ms: int) -> tuple[bytes, dict[bytes, bytes]]:
    stamp = int(time.time() * 1000) - age_ms
    return (
        b"1-0",
        {
            b"translated_text": text.encode(),
            b"target_lang": target_lang.encode(),
            b"timestamp_ms": str(stamp).encode(),
        },
    )


class TestGetRecentDubTexts:
    async def test_recent_lines_come_back_normalized_and_stale_ones_do_not(self) -> None:
        redis = _StreamRedis(
            entries=[
                _entry(_DUB_LINE, "en", age_ms=3_000),
                # Newest-first stream order: the first stale entry ends the scan.
                _entry("An old line from a minute ago.", "en", age_ms=60_000),
                _entry("Behind the stale one, never reached.", "vi", age_ms=2_000),
            ]
        )
        texts = await _worker(redis)._get_recent_dub_texts("meeting-1")

        assert texts == [_normalize_overheard_text(_DUB_LINE)]

    async def test_the_lookup_fails_open(self) -> None:
        texts = await _worker(_StreamRedis(fail=True))._get_recent_dub_texts("meeting-1")
        assert texts == []

    async def test_the_window_is_cached_briefly(self) -> None:
        redis = _StreamRedis(entries=[_entry(_DUB_LINE, "en", age_ms=3_000)])
        worker = _worker(redis)

        first = await worker._get_recent_dub_texts("meeting-1")
        second = await worker._get_recent_dub_texts("meeting-1")

        assert first == second
        # An echo cannot arrive before its dub has been synthesized and played (seconds), so a
        # 2s-stale read can never miss the line it needs — and chunks arrive about once a
        # second per speaker, so caching is what keeps this off the per-chunk hot path.
        assert redis.calls == 1


class TestProcessWiresTheGuard:
    async def test_process_hands_transcribe_the_recent_dub_lines(
        self,
        mock_redis_client: Any,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        # A guard that exists and is not consulted is the bug with a function above it —
        # the fetch must reach transcribe(), not merely exist beside it.
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}
        mock_redis_client._redis.xrevrange = AsyncMock(
            return_value=[_entry(_DUB_LINE, "en", age_ms=3_000)]
        )

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello", language="en", confidence=-0.25, start_ms=0, end_ms=1000
                )
            ]
        )

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            language="auto",
        )
        await worker.process(b"msg-1", chunk.to_redis())

        assert worker.model.transcribe.await_args.kwargs["recent_dub_texts"] == [
            _normalize_overheard_text(_DUB_LINE)
        ]
