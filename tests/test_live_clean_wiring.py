"""Clean transcript on the LIVE path (WT-716 tier 1).

The rule tier (`shared.disfluency.prepass`) has its own fixture suite. This file is about the
WIRING: that every final STT segment is cleaned at the single choke point in front of
`stt:results`, that the raw `text` survives it untouched, that nothing about it can cost a
meeting a segment, and that translation — and therefore the dub — works from the clean line.

What each test here is protecting:

  * the prepass runs where the segments are PUBLISHED, not somewhere a later caller could
    bypass, so the early per-sentence flush and the end-of-turn segment are both covered;
  * `text` is the record. Billing, retranscribe and corrections read it, and a cleaner that
    rewrites it would quietly delete the only copy of what was said;
  * a rule that throws must cost nothing but the clean fields — the sentence still goes out;
  * TRANSCRIPT_CLEAN_ENABLED=false must produce byte-for-byte the pre-WT-716 message, because
    that is the whole value of a kill switch;
  * "Ờ." answering a question is a word; the same "ờ" opening a speaker's own sentence is a
    hesitation — so the previous turn has to be tracked, per meeting, per speaker.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, TranslationSettings, WorkerSettings
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker import worker as stt_worker_module
from stt_worker.model import TranscribedSegment
from stt_worker.worker import STTWorker
from translation_worker.worker import TranslationWorker

pytestmark = pytest.mark.asyncio

# The English example from the ticket, and what the rule tier makes of it.
_RAW_EN = "um so we uh we need to finalize the budget"
_CLEAN_EN = "So we need to finalize the budget."
# The Japanese example: fillers go, the sentence stays.
_RAW_JA = "えーと、あのー、来週の会議ですが"


def _stt_worker(mock_redis_client: Any, settings: WorkerSettings) -> STTWorker:
    """A worker wired exactly like the ones in test_stt_worker.py, minus the model."""
    worker = STTWorker.__new__(STTWorker)
    worker.settings = settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.stt_settings = STTSettings()
    worker._paused_rooms = set()
    worker._stt_prompts = {}
    worker._room_languages = {}
    # Only _cleanup_room needs these; they are the BaseWorker fields __init__ would have made.
    worker._route_states = {}
    worker._translation_active = {}
    worker._room_routes = {}
    worker._speaker_locks = {}
    worker._prosody_baselines = {}
    worker.model = MagicMock()
    return worker


def _transcribes(*segments: TranscribedSegment) -> AsyncMock:
    return AsyncMock(return_value=list(segments))


def _segment(text: str, language: str = "en") -> TranscribedSegment:
    return TranscribedSegment(
        text=text, language=language, confidence=-0.25, start_ms=0, end_ms=1000
    )


def _chunk(sample_audio_bytes: bytes, speaker_id: str = "speaker-1") -> AudioChunkMessage:
    return AudioChunkMessage(
        meeting_id="meeting-1",
        speaker_id=speaker_id,
        chunk_index=0,
        audio_data=sample_audio_bytes,
        language="auto",
    )


def _published(mock_redis_client: Any) -> list[dict[str, str]]:
    # The per-room stream only. BaseWorker.publish writes every message twice — once to
    # `stt:results:{meeting}` and once to the global `stt:results` — and counting both would
    # make every assertion below report double.
    return [
        data
        for stream, data in (c.args for c in mock_redis_client._redis.xadd.call_args_list)
        if str(stream).startswith("stt:results:")
    ]


class TestPrepassRunsAtPublish:
    async def test_english_segment_carries_the_clean_line_and_the_raw_one(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment(_RAW_EN))

        await worker.process(b"msg-1", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert published["clean_text"] == _CLEAN_EN
        # THE RECORD IS UNTOUCHED. Everything downstream that has to reproduce what was
        # actually said — billing, retranscribe, a human correcting the transcript — reads
        # this field, and there is no second copy of it anywhere.
        assert published["text"] == _RAW_EN
        assert "fillers_removed" in published["clean_flags"].split(",")

    async def test_japanese_fillers_are_removed_without_touching_the_sentence(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment(_RAW_JA, language="ja"))

        await worker.process(b"msg-ja", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert "えーと" not in published["clean_text"]
        assert "あのー" not in published["clean_text"]
        assert "来週の会議ですが" in published["clean_text"]
        assert published["text"] == _RAW_JA

    async def test_the_early_sentence_path_is_cleaned_by_the_same_choke_point(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        """Flash mode publishes MOST of a meeting's lines mid-chunk.

        A cleaner wired only into the end-of-turn loop would leave the majority of a live
        room's subtitles uncleaned, which is why both paths go through _publish_stt_result.
        """
        worker = _stt_worker(mock_redis_client, worker_settings)

        async def fake_transcribe(*_args: Any, **kwargs: Any) -> list[TranscribedSegment]:
            early = kwargs["on_early_segment"]
            await early(
                TranscribedSegment(
                    text=_RAW_EN, language="en", confidence=0.0, start_ms=0, end_ms=0
                )
            )
            return []

        worker.model.transcribe = AsyncMock(side_effect=fake_transcribe)

        await worker.process(b"msg-early", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert published["text"] == _RAW_EN
        assert published["clean_text"] == _CLEAN_EN

    async def test_a_filler_only_turn_publishes_an_empty_clean_line_not_a_missing_one(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        # "" and absent are different answers: "" means the prepass looked and there was
        # nothing but filler, absent means nobody looked. Consumers act on that difference.
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment("Ummm"))

        await worker.process(b"msg-filler", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert published["clean_text"] == ""
        assert "filler_only" in published["clean_flags"].split(",")
        assert published["text"] == "Ummm"

    async def test_an_unsupported_language_is_left_alone(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        # Running English rules over Korean would be guessing. No clean version was computed,
        # and the message says so by omitting the fields.
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment("안녕하세요 여러분", language="ko"))

        await worker.process(b"msg-ko", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert "clean_text" not in published


class TestCleaningNeverCostsASegment:
    async def test_a_prepass_failure_still_publishes_the_segment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """The one invariant that matters more than any cleaning.

        A live meeting must not lose a line because a rule tripped over an input nobody
        anticipated. The failure is logged — rate-limited, because it will fire on every
        sentence — and the message goes out exactly as it would have before WT-716.
        """

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("a rule tripped over something")

        monkeypatch.setattr(stt_worker_module, "prepass", boom)
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment(_RAW_EN))

        await worker.process(b"msg-boom", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert published["text"] == _RAW_EN
        assert "clean_text" not in published
        assert "clean_flags" not in published
        assert worker.logger.warning.called

    async def test_repeated_failures_are_logged_once_not_once_per_sentence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("a rule tripped over something")

        monkeypatch.setattr(stt_worker_module, "prepass", boom)
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment(_RAW_EN))

        for index in range(5):
            await worker.process(f"msg-{index}".encode(), _chunk(sample_audio_bytes).to_redis())

        warnings = [c for c in worker.logger.warning.call_args_list if c.args]
        assert [c.args[0] for c in warnings].count("transcript_clean_failed") == 1
        # Nothing is hidden: the ones folded into the window are counted on the next warning.
        assert len(_published(mock_redis_client)) == 5

    async def test_the_kill_switch_restores_the_pre_wt716_message(
        self, mock_redis_client, redis_settings: Any, sample_audio_bytes: bytes
    ) -> None:
        settings = WorkerSettings(
            log_level="DEBUG",
            chunk_duration_ms=1000,
            redis=redis_settings,
            transcript_clean_enabled=False,
        )
        worker = _stt_worker(mock_redis_client, settings)
        worker.model.transcribe = _transcribes(_segment(_RAW_EN))

        await worker.process(b"msg-off", _chunk(sample_audio_bytes).to_redis())

        (published,) = _published(mock_redis_client)
        assert published["text"] == _RAW_EN
        assert "clean_text" not in published
        assert "clean_flags" not in published


class TestPreviousTurnIsTracked:
    """ "Ờ" is an answer or a hesitation depending on who spoke last, and what they said."""

    async def test_an_answer_to_someone_elses_question_keeps_its_opening_word(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)

        worker.model.transcribe = _transcribes(_segment("Anh gửi báo cáo chưa?", language="vi"))
        await worker.process(b"msg-q", _chunk(sample_audio_bytes, "speaker-A").to_redis())

        worker.model.transcribe = _transcribes(_segment("Ờ, em gửi rồi.", language="vi"))
        await worker.process(b"msg-a", _chunk(sample_audio_bytes, "speaker-B").to_redis())

        answer = _published(mock_redis_client)[-1]
        assert answer["clean_text"] == "Ờ, em gửi rồi."

    async def test_the_same_opening_word_inside_one_speakers_own_turn_is_a_hesitation(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        # The mutation pair for the test above: same second sentence, same speaker order
        # except that the question is the SPEAKER'S OWN — so nothing is being answered.
        worker = _stt_worker(mock_redis_client, worker_settings)

        worker.model.transcribe = _transcribes(_segment("Anh gửi báo cáo chưa?", language="vi"))
        await worker.process(b"msg-q", _chunk(sample_audio_bytes, "speaker-B").to_redis())

        worker.model.transcribe = _transcribes(_segment("Ờ, em gửi rồi.", language="vi"))
        await worker.process(b"msg-a", _chunk(sample_audio_bytes, "speaker-B").to_redis())

        answer = _published(mock_redis_client)[-1]
        assert answer["clean_text"] == "Em gửi rồi."

    async def test_a_statement_before_the_turn_leaves_it_a_hesitation(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        worker = _stt_worker(mock_redis_client, worker_settings)

        worker.model.transcribe = _transcribes(_segment("Báo cáo đã xong.", language="vi"))
        await worker.process(b"msg-s", _chunk(sample_audio_bytes, "speaker-A").to_redis())

        worker.model.transcribe = _transcribes(_segment("Ờ, em gửi rồi.", language="vi"))
        await worker.process(b"msg-a", _chunk(sample_audio_bytes, "speaker-B").to_redis())

        answer = _published(mock_redis_client)[-1]
        assert answer["clean_text"] == "Em gửi rồi."

    async def test_the_tracking_is_dropped_when_the_room_ends(
        self, mock_redis_client, worker_settings: WorkerSettings, sample_audio_bytes: bytes
    ) -> None:
        # One entry per meeting ever seen would otherwise live for the life of the process.
        worker = _stt_worker(mock_redis_client, worker_settings)
        worker.model.transcribe = _transcribes(_segment("Anh gửi báo cáo chưa?", language="vi"))
        await worker.process(b"msg-q", _chunk(sample_audio_bytes, "speaker-A").to_redis())
        assert "meeting-1" in worker._last_final_turn

        worker._cleanup_room("meeting-1")

        assert "meeting-1" not in worker._last_final_turn


def _translation_worker(mock_redis_client: Any, settings: WorkerSettings) -> TranslationWorker:
    """The harness from test_translation_worker.py — same shape, same gates declared open."""
    worker = TranslationWorker.__new__(TranslationWorker)
    worker.settings = settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.translation_settings = TranslationSettings()
    worker._paused_rooms = set()
    worker._route_states = {}
    worker._is_translation_active = lambda _room: True  # type: ignore[method-assign]
    worker._mt_glossaries = {}
    worker._recent_source_contexts = {}
    worker.worker_name = "translation"
    translator = MagicMock()
    translator.model = "gpt-4.1-mini"
    translator.translate_with_valence = AsyncMock(
        return_value=("Chúng ta cần chốt ngân sách.", None)
    )
    translator.translate_batch = AsyncMock(return_value=[])
    worker.translator = translator
    return worker


def _translated(mock_redis_client: Any) -> list[dict[str, str]]:
    return [
        data
        for stream, data in (c.args for c in mock_redis_client._redis.xadd.call_args_list)
        if str(stream).startswith("translate:results:")
    ]


class TestTranslationWorksFromTheCleanLine:
    async def test_the_model_is_given_the_clean_sentence(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Fillers in the MT input are paid for twice: once in tokens, once in the dub.

        A translator handed "um so we uh we need to finalize the budget" is free to render the
        hesitations as words of the target language, and tts_worker then SPEAKS them.
        """
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        message = STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text=_RAW_EN,
            language="en",
            confidence=0.95,
            clean_text=_CLEAN_EN,
            clean_flags=("fillers_removed",),
        )
        await worker.process(b"msg-clean", message.to_redis())

        call = worker.translator.translate_with_valence.call_args
        assert call.args[0] == _CLEAN_EN
        published = _translated(mock_redis_client)
        assert [data["original_text"] for data in published] == [_CLEAN_EN]
        # The context fed back to the model is the clean history, for the same reason.
        assert list(worker._recent_source_contexts["m1"]) == [_CLEAN_EN]

    async def test_a_message_without_a_clean_version_is_translated_exactly_as_before(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        # A replica that predates WT-716, or a deployment with the kill switch off.
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        message = STTResultMessage(
            meeting_id="m1", speaker_id="s1", text=_RAW_EN, language="en", confidence=0.95
        )
        await worker.process(b"msg-raw", message.to_redis())

        assert worker.translator.translate_with_valence.call_args.args[0] == _RAW_EN

    async def test_a_filler_only_segment_is_never_translated_or_published(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        message = STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text="Ummm",
            language="en",
            confidence=0.95,
            clean_text="",
            clean_flags=("filler_only", "fillers_removed"),
        )
        await worker.process(b"msg-filler", message.to_redis())

        assert worker.translator.translate_with_valence.await_count == 0
        assert _translated(mock_redis_client) == []

    async def test_a_filler_only_segment_that_closes_the_turn_still_marks_the_turn_closed(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Nothing is translated, but the turn boundary is not swallowed.

        `is_final_chunk` is what downstream consumers wait on to close a turn; dropping the
        marker would strand it rather than tidy it.
        """
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        message = STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text="Ummm",
            language="en",
            confidence=0.95,
            is_final_chunk=True,
            clean_text="",
            clean_flags=("filler_only",),
        )
        await worker.process(b"msg-filler-final", message.to_redis())

        assert worker.translator.translate_with_valence.await_count == 0
        published = _translated(mock_redis_client)
        assert [(d["original_text"], d["translated_text"]) for d in published] == [("", "")]
        assert published[0]["is_final_chunk"] == "1"
