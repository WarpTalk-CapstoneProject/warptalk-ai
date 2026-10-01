"""The STT-side text dedupe: a stand-in line that repeats a named WarpTalk speaker is dropped.

Behind the ingress overlap gate, the same sentence can still reach STT twice — once under the
WarpTalk user's own identity and once inside the bridge stand-in's Meet feed. The named copy is
the one worth keeping; the stand-in copy is dropped when it matches a recent named line.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.config import STTSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.far_side_dedupe import (
    DedupeConfig,
    NamedSegmentRef,
    find_far_side_duplicate,
    languages_compatible,
)
from stt_worker.model import TranscribedSegment, _normalize_overheard_text
from stt_worker.worker import STTWorker

T0 = 1_759_300_000_000
LINE = "we should ship the bridge build on friday"


def _ref(text: str = LINE, language: str = "en", ts: int = T0, speaker: str = "alice") -> Any:
    return NamedSegmentRef(
        speaker_id=speaker,
        text=_normalize_overheard_text(text),
        language=language,
        timestamp_ms=ts,
    )


class TestFindFarSideDuplicate:
    def test_exact_copy_matches(self) -> None:
        assert find_far_side_duplicate(LINE, "en", T0 + 400, [_ref()]) is not None

    def test_fragment_of_the_named_line_matches(self) -> None:
        match = find_far_side_duplicate("ship the bridge build", "en", T0 + 400, [_ref()])
        assert match is not None and match.speaker_id == "alice"

    def test_small_mishearing_matches_at_the_default_ratio(self) -> None:
        heard = "we should ship the bridge built on friday"
        assert find_far_side_duplicate(heard, "en", T0 + 400, [_ref()]) is not None

    def test_ratio_threshold_is_respected(self) -> None:
        heard = "we could skip the fridge build on monday"
        loose = DedupeConfig(min_ratio=0.6)
        strict = DedupeConfig(min_ratio=0.95)
        assert find_far_side_duplicate(heard, "en", T0, [_ref()], loose) is not None
        assert find_far_side_duplicate(heard, "en", T0, [_ref()], strict) is None

    def test_unrelated_line_does_not_match(self) -> None:
        heard = "can everyone see my screen right now"
        assert find_far_side_duplicate(heard, "en", T0, [_ref()]) is None

    def test_short_back_channel_is_never_dropped(self) -> None:
        # "okay" said by a real far-side person while Alice also said "okay".
        refs = [_ref("okay")]
        assert find_far_side_duplicate("okay", "en", T0, refs) is None

    def test_min_chars_is_configurable(self) -> None:
        refs = [_ref("okay")]
        assert (
            find_far_side_duplicate("okay", "en", T0, refs, DedupeConfig(min_chars=3)) is not None
        )

    def test_outside_the_time_window_does_not_match(self) -> None:
        cfg = DedupeConfig(window_ms=5_000)
        assert find_far_side_duplicate(LINE, "en", T0 + 6_000, [_ref()], cfg) is None
        assert find_far_side_duplicate(LINE, "en", T0 - 6_000, [_ref()], cfg) is None
        assert find_far_side_duplicate(LINE, "en", T0 + 4_000, [_ref()], cfg) is not None

    def test_language_mismatch_does_not_match_when_required(self) -> None:
        assert find_far_side_duplicate(LINE, "vi", T0, [_ref()]) is None
        relaxed = DedupeConfig(same_language=False)
        assert find_far_side_duplicate(LINE, "vi", T0, [_ref()], relaxed) is not None

    def test_language_tags_are_normalized(self) -> None:
        assert languages_compatible("en-US", "en")
        assert languages_compatible("auto", "vi")
        assert languages_compatible("", "en")
        assert not languages_compatible("vi", "en")

    def test_a_longer_stand_in_line_containing_the_named_one_is_kept(self) -> None:
        # The rest of this line is somebody on the far side; dropping it loses their words.
        heard = LINE + " and also we need to book the demo room for the client next week"
        assert find_far_side_duplicate(heard, "en", T0, [_ref()]) is None

    def test_best_ratio_wins_among_several_refs(self) -> None:
        refs = [
            _ref("we should ship the bridge build on monday", speaker="bob"),
            _ref(LINE, speaker="alice"),
        ]
        heard = "we should ship the bridge build on fridays"
        match = find_far_side_duplicate(heard, "en", T0, refs)
        assert match is not None and match.speaker_id == "alice"


# --- worker wiring --------------------------------------------------------------------------


def _stream_entry(fields: dict[str, str]) -> tuple[bytes, dict[bytes, bytes]]:
    return (b"1-0", {k.encode(): v.encode() for k, v in fields.items()})


def _named_result_entry(text: str, speaker: str = "alice", age_ms: int = 500) -> Any:
    return _stream_entry(
        {
            "speaker_id": speaker,
            "text": text,
            "language": "en",
            "timestamp_ms": str(int(time.time() * 1000) - age_ms),
        }
    )


def _worker(mock_redis_client: Any, streams: dict[str, list[Any]], **stt: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.stt_settings = STTSettings(**stt)
    worker._paused_rooms = set()
    worker._stt_prompts = {}
    worker._room_languages = {}

    async def xrevrange(key: str, count: int = 0) -> list[Any]:
        for prefix, entries in streams.items():
            if key.startswith(prefix):
                return entries
        return []

    mock_redis_client._redis.xrevrange = AsyncMock(side_effect=xrevrange)
    published: list[STTResultMessage] = []

    async def capture(result: STTResultMessage, _mid: bytes | None = None) -> STTResultMessage:
        published.append(result)
        return result

    worker._publish_stt_result = capture  # type: ignore[method-assign]
    worker.published = published  # type: ignore[attr-defined]
    return worker


def _chunk(speaker: str, audio: bytes, is_final: bool = True) -> AudioChunkMessage:
    return AudioChunkMessage(
        meeting_id="room-1",
        speaker_id=speaker,
        chunk_index=0,
        audio_data=audio,
        language="en",
        is_final_chunk=is_final,
    )


def _segments(*texts: str) -> list[TranscribedSegment]:
    return [
        TranscribedSegment(text=t, language="en", confidence=-0.2, start_ms=0, end_ms=1000)
        for t in texts
    ]


class TestWorkerDropsFarSideDuplicates:
    async def test_stand_in_copy_is_dropped_and_other_lines_kept(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_named_result_entry(LINE.capitalize())]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=_segments(LINE.capitalize() + ".", "Thanks, that works for our team.")
        )
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )

        texts = [r.text for r in worker.published]  # type: ignore[attr-defined]
        assert texts == ["Thanks, that works for our team."]
        assert worker._far_side_duplicates_dropped == 1
        events = [c.args[0] for c in worker.logger.info.call_args_list]
        assert "filtered_far_side_duplicate" in events

    async def test_named_speakers_are_never_deduped(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_named_result_entry(LINE, speaker="bob")]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(b"msg-1", _chunk("alice", sample_audio_bytes).to_redis())
        assert [r.text for r in worker.published] == [LINE]  # type: ignore[attr-defined]

    async def test_stand_in_refs_ignore_the_stand_in_itself(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        # The stand-in's own earlier line is not a "named speaker" line.
        worker = _worker(
            mock_redis_client,
            {"stt:results:room-1": [_named_result_entry(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID)]},
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert [r.text for r in worker.published] == [LINE]  # type: ignore[attr-defined]

    async def test_disabled_flag_keeps_the_copy(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {"stt:results:room-1": [_named_result_entry(LINE)]},
            far_side_dedupe_enabled=False,
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert [r.text for r in worker.published] == [LINE]  # type: ignore[attr-defined]

    async def test_a_final_chunk_whose_only_line_was_a_duplicate_still_closes_the_turn(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, {"stt:results:room-1": [_named_result_entry(LINE)]})
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        published = worker.published  # type: ignore[attr-defined]
        assert len(published) == 1
        assert published[0].text == "" and published[0].is_final_chunk

    async def test_ref_lookup_fails_open(self, mock_redis_client: Any) -> None:
        worker = _worker(mock_redis_client, {})
        mock_redis_client._redis.xrevrange = AsyncMock(side_effect=RuntimeError("down"))
        assert await worker._get_named_speaker_refs("room-1") == []

    async def test_stale_named_lines_are_not_refs(self, mock_redis_client: Any) -> None:
        worker = _worker(
            mock_redis_client,
            {
                "stt:results:room-1": [
                    _named_result_entry("a fresh line from alice", age_ms=1_000),
                    _named_result_entry("an ancient line from alice", age_ms=120_000),
                ]
            },
        )
        refs = await worker._get_named_speaker_refs("room-1")
        assert [r.text for r in refs] == ["a fresh line from alice"]
