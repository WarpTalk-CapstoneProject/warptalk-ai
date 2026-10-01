"""The STT-side text dedupe for bridge rooms: which copy of a doubled sentence survives.

Behind the ingress overlap gate, the same sentence can still reach STT twice — once under a
WarpTalk user's own identity and once inside the bridge stand-in's Meet feed. WHICH copy is the
real one depends on whose AUDIO started first, never on which line was published first:

* forward (user is also in the Meet): named first, stand-in ~300-900 ms later -> drop stand-in;
* leak (host on laptop speakers): stand-in first or simultaneous, named mic picks up a leaked
  copy -> keep the stand-in; drop the named copy only with STT_FAR_SIDE_LEAK_DEDUPE_ENABLED;
* timing unknown on either side -> keep both.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.far_side_dedupe import (
    DedupeConfig,
    LeakConfig,
    NamedSegmentRef,
    audio_start_epoch_ms,
    find_far_side_duplicate,
    find_far_side_leak,
    languages_compatible,
)
from stt_worker.model import TranscribedSegment, _normalize_overheard_text
from stt_worker.worker import STTWorker

T0 = 1_759_300_000_000
LINE = "we should ship the bridge build on friday"


def _ref(
    text: str = LINE,
    language: str = "en",
    ts: int = T0,
    speaker: str = "alice",
    audio_start: int | None = T0,
) -> Any:
    return NamedSegmentRef(
        speaker_id=speaker,
        text=_normalize_overheard_text(text),
        language=language,
        timestamp_ms=ts,
        audio_start_ms=audio_start,
    )


def _dup(
    text: str,
    lag_ms: int | None = 400,
    refs: list[Any] | None = None,
    language: str = "en",
    cfg: DedupeConfig | None = None,
) -> Any:
    """Forward check of a stand-in line whose audio starts `lag_ms` after the ref's (T0)."""
    return find_far_side_duplicate(
        text,
        language,
        T0 + (lag_ms or 0),
        refs if refs is not None else [_ref()],
        cfg,
        audio_start_ms=None if lag_ms is None else T0 + lag_ms,
    )


def _leak(
    text: str,
    named_delay_ms: int | None = 100,
    refs: list[Any] | None = None,
    language: str = "en",
    leak: LeakConfig | None = None,
) -> Any:
    """Reverse check of a NAMED line whose audio starts `named_delay_ms` after the stand-in's."""
    standin = refs if refs is not None else [_ref(speaker=EXTERNAL_BRIDGE_SPEAKER_ID)]
    return find_far_side_leak(
        text,
        language,
        T0 + (named_delay_ms or 0),
        standin,
        None,
        leak,
        audio_start_ms=None if named_delay_ms is None else T0 + named_delay_ms,
    )


class TestAudioStart:
    def test_anchor_plus_offset(self) -> None:
        assert audio_start_epoch_ms(T0, 1_500) == T0 + 1_500

    def test_no_anchor_is_unknown(self) -> None:
        assert audio_start_epoch_ms(0, 1_500) is None

    def test_clamped_first_chunk_is_unknown(self) -> None:
        # _elapsed_ms clamps the room's first chunk(s) to 0, where anchor+0 is the chunk's END.
        assert audio_start_epoch_ms(T0, 0) is None


class TestForwardDirection:
    @pytest.mark.parametrize("lag_ms", [150, 300, 600, 900, 2_000])
    def test_stand_in_after_named_is_dropped(self, lag_ms: int) -> None:
        assert _dup(LINE, lag_ms) is not None

    @pytest.mark.parametrize("lag_ms", [-1_000, -300, 0, 100, 149])
    def test_stand_in_first_or_simultaneous_is_kept(self, lag_ms: int) -> None:
        # The leak case: the stand-in carries the true Meet-side speech, never drop it.
        assert _dup(LINE, lag_ms) is None

    def test_beyond_max_lag_is_kept(self) -> None:
        assert _dup(LINE, 2_001) is None
        assert _dup(LINE, 2_500, cfg=DedupeConfig(max_lag_ms=3_000)) is not None

    def test_min_lag_is_configurable(self) -> None:
        assert _dup(LINE, 100, cfg=DedupeConfig(min_lag_ms=50)) is not None

    def test_missing_timing_keeps_both(self) -> None:
        assert _dup(LINE, None) is None
        assert _dup(LINE, 400, refs=[_ref(audio_start=None)]) is None

    def test_fragment_of_the_named_line_matches(self) -> None:
        match = _dup("ship the bridge build")
        assert match is not None and match.speaker_id == "alice"

    def test_small_mishearing_matches_at_the_default_ratio(self) -> None:
        assert _dup("we should ship the bridge built on friday") is not None

    def test_ratio_threshold_is_respected(self) -> None:
        heard = "we could skip the fridge build on monday"
        assert _dup(heard, cfg=DedupeConfig(min_ratio=0.6)) is not None
        assert _dup(heard, cfg=DedupeConfig(min_ratio=0.95)) is None

    def test_unrelated_line_does_not_match(self) -> None:
        assert _dup("can everyone see my screen right now") is None

    def test_short_back_channel_is_never_dropped(self) -> None:
        # "okay" said by a real far-side person while Alice also said "okay".
        assert _dup("okay", refs=[_ref("okay")]) is None

    def test_min_chars_is_configurable(self) -> None:
        assert _dup("okay", refs=[_ref("okay")], cfg=DedupeConfig(min_chars=3)) is not None

    def test_outside_the_timestamp_window_does_not_match(self) -> None:
        cfg = DedupeConfig(window_ms=5_000)
        far = find_far_side_duplicate(
            LINE, "en", T0 + 6_000, [_ref()], cfg, audio_start_ms=T0 + 400
        )
        assert far is None

    def test_different_language_is_kept(self) -> None:
        assert _dup(LINE, language="vi") is None
        assert _dup(LINE, language="vi", cfg=DedupeConfig(same_language=False)) is not None

    def test_language_tags_are_normalized(self) -> None:
        assert languages_compatible("en-US", "en")
        assert languages_compatible("auto", "vi")
        assert languages_compatible("", "en")
        assert not languages_compatible("vi", "en")

    def test_a_longer_stand_in_line_containing_the_named_one_is_kept(self) -> None:
        # The rest of this line is somebody on the far side; dropping it loses their words.
        assert _dup(LINE + " and also we need to book the demo room for the client") is None

    def test_best_ratio_wins_among_several_refs(self) -> None:
        refs = [
            _ref("we should ship the bridge build on monday", speaker="bob"),
            _ref(LINE, speaker="alice"),
        ]
        match = _dup("we should ship the bridge build on fridays", refs=refs)
        assert match is not None and match.speaker_id == "alice"


class TestLeakDirection:
    @pytest.mark.parametrize("named_delay_ms", [-149, -100, 0, 50, 300, 1_000])
    def test_named_copy_after_or_with_the_stand_in_is_a_leak(self, named_delay_ms: int) -> None:
        match = _leak(LINE, named_delay_ms)
        assert match is not None and match.speaker_id == EXTERNAL_BRIDGE_SPEAKER_ID

    @pytest.mark.parametrize("named_delay_ms", [-150, -400, -900])
    def test_named_first_is_the_forward_case_and_kept(self, named_delay_ms: int) -> None:
        assert _leak(LINE, named_delay_ms) is None

    def test_named_far_after_the_stand_in_is_kept(self) -> None:
        assert _leak(LINE, 1_001) is None

    def test_missing_timing_keeps_both(self) -> None:
        assert _leak(LINE, None) is None
        refs = [_ref(speaker=EXTERNAL_BRIDGE_SPEAKER_ID, audio_start=None)]
        assert _leak(LINE, 100, refs=refs) is None

    def test_no_stand_in_lines_means_no_leak(self) -> None:
        # Not a bridge room (or nothing from Meet yet).
        assert _leak(LINE, 100, refs=[]) is None

    def test_short_phrases_are_never_dropped(self) -> None:
        # Two people saying the same short phrase is ordinary; the leak floor is stricter.
        refs = [_ref("sounds good", speaker=EXTERNAL_BRIDGE_SPEAKER_ID)]
        assert _leak("sounds good", refs=refs) is None
        refs = [_ref("okay", speaker=EXTERNAL_BRIDGE_SPEAKER_ID)]
        assert _leak("okay", refs=refs) is None

    def test_different_language_is_kept(self) -> None:
        assert _leak(LINE, language="vi") is None

    def test_thresholds_are_stricter_than_forward(self) -> None:
        heard = "we should ship the bridge built on friday"  # ~0.98: passes both
        assert _leak(heard) is not None
        loose = "we should ship the bridge on monday"  # ratio ~0.84: passes 0.8, not 0.85
        assert _dup(loose) is not None
        assert _leak(loose) is None

    def test_named_line_containing_more_than_the_stand_in_is_kept(self) -> None:
        # The rest is the host's own speech.
        assert _leak(LINE + " and i will write the release notes tonight") is None

    def test_leaked_fragment_matches(self) -> None:
        # The leak is quieter; its VAD often catches only part of the sentence.
        assert _leak("ship the bridge build on friday") is not None

    @pytest.mark.parametrize("lag_ms", range(-1_200, 2_200, 50))
    def test_one_pair_never_satisfies_both_rules(self, lag_ms: int) -> None:
        # lag = stand-in audio start - named audio start.
        named = _ref(LINE, audio_start=T0)
        standin = _ref(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID, audio_start=T0 + lag_ms)
        fwd = find_far_side_duplicate(LINE, "en", T0, [named], audio_start_ms=T0 + lag_ms)
        rev = find_far_side_leak(LINE, "en", T0, [standin], audio_start_ms=T0)
        assert not (fwd is not None and rev is not None)


# --- worker wiring --------------------------------------------------------------------------

ANCHOR = 1_000_000  # epoch ms the room's offsets count from (test value)


def _stream_entry(fields: dict[str, str]) -> tuple[bytes, dict[bytes, bytes]]:
    return (b"1-0", {k.encode(): v.encode() for k, v in fields.items()})


def _result_entry(
    text: str,
    speaker: str = "alice",
    age_ms: int = 500,
    start_ms: int | None = 10_000,
    language: str = "en",
) -> Any:
    fields = {
        "speaker_id": speaker,
        "text": text,
        "language": language,
        "timestamp_ms": str(int(time.time() * 1000) - age_ms),
    }
    if start_ms is not None:
        fields["start_ms"] = str(start_ms)
        fields["anchor_ms"] = str(ANCHOR)
    return _stream_entry(fields)


def _worker(mock_redis_client: Any, streams: dict[str, list[Any]], **stt: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.stt_settings = STTSettings(**stt)
    worker._paused_rooms = set()
    worker._stt_prompts = {}
    worker._room_languages = {}
    # The room's agreed origin, as _elapsed_ms would have resolved it.
    worker._transcript_anchors = {"room-1": ANCHOR}

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


def _segments(*texts: str, start_ms: int = 10_400) -> list[TranscribedSegment]:
    """Segments whose audio starts at ANCHOR + start_ms (the default: 400 ms after refs)."""
    return [
        TranscribedSegment(
            text=t, language="en", confidence=-0.2, start_ms=start_ms, end_ms=start_ms + 1000
        )
        for t in texts
    ]


def _texts(worker: STTWorker) -> list[str]:
    return [r.text for r in worker.published]  # type: ignore[attr-defined]


def _events(worker: STTWorker) -> list[str]:
    return [c.args[0] for c in worker.logger.info.call_args_list]  # type: ignore[attr-defined]


class TestWorkerForwardDedupe:
    async def test_stand_in_echo_is_dropped_and_other_lines_kept(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_result_entry(LINE.capitalize())]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=_segments(LINE.capitalize() + ".", "Thanks, that works for our team.")
        )
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )

        assert _texts(worker) == ["Thanks, that works for our team."]
        assert worker._far_side_duplicates_dropped == 1
        assert "filtered_far_side_duplicate" in _events(worker)

    async def test_stand_in_first_is_kept_the_leak_case(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        # The host's mic line (named) starts 200 ms AFTER the stand-in's: the stand-in is the
        # true Meet-side copy and must survive.
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_result_entry(LINE, start_ms=10_200)]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_000))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert _texts(worker) == [LINE]

    async def test_missing_timing_keeps_the_stand_in(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_result_entry(LINE, start_ms=None)]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert _texts(worker) == [LINE]

    async def test_named_speakers_are_never_forward_deduped(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client, {"stt:results:room-1": [_result_entry(LINE, speaker="bob")]}
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(b"msg-1", _chunk("alice", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]

    async def test_stand_in_refs_ignore_the_stand_in_itself(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {"stt:results:room-1": [_result_entry(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID)]},
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert _texts(worker) == [LINE]

    async def test_disabled_flag_keeps_the_copy(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {"stt:results:room-1": [_result_entry(LINE)]},
            far_side_dedupe_enabled=False,
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE))
        await worker.process(
            b"msg-1", _chunk(EXTERNAL_BRIDGE_SPEAKER_ID, sample_audio_bytes).to_redis()
        )
        assert _texts(worker) == [LINE]

    async def test_a_final_chunk_whose_only_line_was_a_duplicate_still_closes_the_turn(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, {"stt:results:room-1": [_result_entry(LINE)]})
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

    async def test_stale_lines_are_not_refs(self, mock_redis_client: Any) -> None:
        worker = _worker(
            mock_redis_client,
            {
                "stt:results:room-1": [
                    _result_entry("a fresh line from alice", age_ms=1_000),
                    _result_entry("an ancient line from alice", age_ms=120_000),
                ]
            },
        )
        refs = await worker._get_named_speaker_refs("room-1")
        assert [r.text for r in refs] == ["a fresh line from alice"]
        assert refs[0].audio_start_ms == ANCHOR + 10_000


class TestWorkerLeakDedupe:
    def _standin_room(self, **entry: Any) -> dict[str, list[Any]]:
        return {
            "stt:results:room-1": [
                _result_entry(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID, start_ms=10_000, **entry)
            ]
        }

    async def test_flag_is_off_by_default(self) -> None:
        assert STTSettings().far_side_leak_dedupe_enabled is False

    async def test_leaked_named_copy_is_dropped_when_enabled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, self._standin_room(), far_side_leak_dedupe_enabled=True)
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=_segments(LINE, "and that is my own point", start_ms=10_200)
        )
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        assert _texts(worker) == ["and that is my own point"]
        assert worker._far_side_leaks_dropped == 1
        assert "filtered_far_side_leak" in _events(worker)

    async def test_simultaneous_named_copy_is_dropped_when_enabled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, self._standin_room(), far_side_leak_dedupe_enabled=True)
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_000))
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        published = worker.published  # type: ignore[attr-defined]
        # Only the empty turn-closing marker.
        assert len(published) == 1 and published[0].text == "" and published[0].is_final_chunk

    async def test_leaked_copy_is_kept_when_disabled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, self._standin_room())
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_200))
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]

    async def test_named_line_first_is_kept_even_when_enabled(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        # Forward case seen from the named side: the stand-in copy landed first in the stream
        # but its AUDIO started 500 ms after the named line's. Publish order must not decide.
        worker = _worker(mock_redis_client, self._standin_room(), far_side_leak_dedupe_enabled=True)
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=9_500))
        await worker.process(b"msg-1", _chunk("alice", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]

    async def test_non_bridge_room_is_untouched(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {"stt:results:room-1": [_result_entry(LINE, speaker="bob", start_ms=10_000)]},
            far_side_leak_dedupe_enabled=True,
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_100))
        await worker.process(b"msg-1", _chunk("alice", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]

    async def test_missing_timing_keeps_the_named_line(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {
                "stt:results:room-1": [
                    _result_entry(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID, start_ms=None)
                ]
            },
            far_side_leak_dedupe_enabled=True,
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_100))
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]

    async def test_short_and_other_language_named_lines_are_kept(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(
            mock_redis_client,
            {
                "stt:results:room-1": [
                    _result_entry("sounds good", speaker=EXTERNAL_BRIDGE_SPEAKER_ID),
                    _result_entry(LINE, speaker=EXTERNAL_BRIDGE_SPEAKER_ID, language="vi"),
                ]
            },
            far_side_leak_dedupe_enabled=True,
        )
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=_segments("Sounds good.", LINE, start_ms=10_100)
        )
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        assert _texts(worker) == ["Sounds good.", LINE]

    async def test_leak_check_fails_open(
        self, mock_redis_client: Any, sample_audio_bytes: bytes
    ) -> None:
        worker = _worker(mock_redis_client, {}, far_side_leak_dedupe_enabled=True)
        mock_redis_client._redis.xrevrange = AsyncMock(side_effect=RuntimeError("down"))
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(return_value=_segments(LINE, start_ms=10_100))
        await worker.process(b"msg-1", _chunk("host", sample_audio_bytes).to_redis())
        assert _texts(worker) == [LINE]
