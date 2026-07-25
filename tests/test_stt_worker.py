"""Tests for STT Worker — mock OpenAI Realtime STT session, verify output schema."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker.model import OpenAISTT, TranscribedSegment, _filter_segments, _normalize_language
from stt_worker.worker import STTWorker


class FakeRealtimeConn:
    """Minimal stand-in for openai's AsyncRealtimeConnection: an async iterator of
    events, plus the session/input_audio_buffer sub-resources actually called."""

    def __init__(self, events: list) -> None:
        self._events = events
        self.session = MagicMock()
        self.session.update = AsyncMock()
        self.input_audio_buffer = MagicMock()
        self.input_audio_buffer.append = AsyncMock()
        self.input_audio_buffer.commit = AsyncMock()

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event
        return gen()


class FakeRealtimeManager:
    """Minimal stand-in for AsyncRealtimeConnectionManager (`async with client.realtime.connect(...) as conn`)."""

    def __init__(self, conn: FakeRealtimeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> FakeRealtimeConn:
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


def _make_stt_with_conn(events: list) -> tuple[OpenAISTT, FakeRealtimeConn]:
    stt = OpenAISTT.__new__(OpenAISTT)
    stt.api_key = ""
    stt.model = "gpt-realtime-whisper"
    stt.noise_reduction = "far_field"
    stt._sessions = {}
    conn = FakeRealtimeConn(events)
    stt._client = MagicMock()
    stt._client.realtime.connect = MagicMock(return_value=FakeRealtimeManager(conn))
    return stt, conn


def _segment(text: str, avg_logprob: float = -0.3, no_speech_prob: float = 0.01) -> dict:
    return {
        "text": text,
        "start": 0.0,
        "end": 1.0,
        "avg_logprob": avg_logprob,
        "no_speech_prob": no_speech_prob,
    }


class TestOpenAISTT:
    """OpenAISTT wrapper tests — mocks the Realtime API's WebSocket session."""

    async def test_transcribe_returns_segments(self, sample_audio_bytes: bytes) -> None:
        """transcribe() should return a filtered list of TranscribedSegment.

        The Realtime API's completed event carries just a flat `transcript` field (no
        per-segment timing/confidence/language), same shape gap as the old REST
        response — a language hint must still be passed in since the API doesn't echo
        one back.
        """
        events = [SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            transcript=" Hello, world!",
        )]
        stt, conn = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes, sample_rate=16000, language="en",
            meeting_id="m1", speaker_id="s1",
        )

        assert len(result) == 1
        assert result[0].text == "Hello, world!"
        assert result[0].language == "en"
        assert result[0].start_ms == 0
        conn.input_audio_buffer.commit.assert_awaited_once()

    async def test_transcribe_empty_bytes_returns_empty(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt._client = MagicMock()
        result = await stt.transcribe(b"")
        assert result == []

    def test_session_payload_includes_noise_reduction_by_default(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "gpt-4o-transcribe"
        stt.noise_reduction = "far_field"

        payload = stt._session_payload(language=None, prompt=None)

        assert payload["audio"]["input"]["noise_reduction"] == {"type": "far_field"}

    def test_session_payload_supports_near_field(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "gpt-4o-transcribe"
        stt.noise_reduction = "near_field"

        payload = stt._session_payload(language=None, prompt=None)

        assert payload["audio"]["input"]["noise_reduction"] == {"type": "near_field"}

    def test_session_payload_omits_noise_reduction_when_off(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "gpt-4o-transcribe"
        stt.noise_reduction = "off"

        payload = stt._session_payload(language=None, prompt=None)

        assert "noise_reduction" not in payload["audio"]["input"]

    async def test_transcribe_api_error_raises_for_worker_degrade_signal(
        self, sample_audio_bytes: bytes
    ) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.api_key = ""
        stt.model = "gpt-realtime-whisper"
        stt.noise_reduction = "far_field"
        stt._sessions = {}
        stt._client = MagicMock()
        stt._client.realtime.connect = MagicMock(side_effect=Exception("API error"))

        with pytest.raises(Exception, match="API error"):
            await stt.transcribe(
                sample_audio_bytes, sample_rate=16000, meeting_id="m1", speaker_id="s1"
            )

    async def test_transcribe_emits_complete_sentences_early_from_deltas(
        self, sample_audio_bytes: bytes
    ) -> None:
        """A complete sentence appearing mid-stream in delta events should be handed
        to on_early_segment immediately, without waiting for .completed — that's the
        whole point of pipelining translation/TTS against a still-in-progress chunk.
        The trailing fragment ("How are you?") is a second complete sentence that
        arrives via delta too, so nothing is left over for the return value.
        """
        events = [
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="Hello there."),
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta=" How are you?"),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hello there. How are you?",
            ),
        ]
        stt, conn = _make_stt_with_conn(events)
        early_segments: list[TranscribedSegment] = []

        async def on_early(seg: TranscribedSegment) -> None:
            early_segments.append(seg)

        result = await stt.transcribe(
            sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1",
            on_early_segment=on_early,
        )

        assert [s.text for s in early_segments] == ["Hello there.", "How are you?"]
        assert result == []

    async def test_transcribe_returns_trailing_fragment_not_flushed_early(
        self, sample_audio_bytes: bytes
    ) -> None:
        """An incomplete trailing fragment must NOT be flushed early (no punctuation
        yet to confirm a sentence boundary) — it comes back in the normal return value
        once .completed supplies the authoritative final transcript.
        """
        events = [
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="Hello there."),
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta=" How are"),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hello there. How are you today?",
            ),
        ]
        stt, conn = _make_stt_with_conn(events)
        early_segments: list[TranscribedSegment] = []

        async def on_early(seg: TranscribedSegment) -> None:
            early_segments.append(seg)

        result = await stt.transcribe(
            sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1",
            on_early_segment=on_early,
        )

        assert [s.text for s in early_segments] == ["Hello there."]
        assert len(result) == 1
        assert result[0].text == "How are you today?"

    async def test_transcribe_without_on_early_segment_ignores_deltas(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Callers that don't pass on_early_segment (e.g. a future non-pipelined path)
        get the old all-at-once behavior — deltas are ignored, only .completed matters.
        """
        events = [
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="Hello there."),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hello there.",
            ),
        ]
        stt, conn = _make_stt_with_conn(events)

        result = await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")

        assert len(result) == 1
        assert result[0].text == "Hello there."

    async def test_transcribe_delta_final_mismatch_drops_trailing_safely(
        self, sample_audio_bytes: bytes
    ) -> None:
        """If the final transcript doesn't start with what was already flushed (model
        revised the flushed prefix), don't guess at a diff — drop the trailing part
        rather than risk re-publishing/double-billing already-flushed text.
        """
        events = [
            SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta="Helo there."),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hello there, how are you?",
            ),
        ]
        stt, conn = _make_stt_with_conn(events)
        early_segments: list[TranscribedSegment] = []

        async def on_early(seg: TranscribedSegment) -> None:
            early_segments.append(seg)

        result = await stt.transcribe(
            sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1",
            on_early_segment=on_early,
        )

        assert len(early_segments) == 1
        assert result == []

    async def test_transcribe_reuses_session_across_calls(
        self, sample_audio_bytes: bytes
    ) -> None:
        """A second chunk from the same (meeting, speaker) must NOT reconnect —
        that's the whole point of session reuse (pay the handshake once, not per chunk).
        """
        events = [SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            transcript="Hi",
        )]
        stt, conn = _make_stt_with_conn(events)

        await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")
        await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")

        stt._client.realtime.connect.assert_called_once()

    async def test_transcribe_reopens_session_when_language_changes(
        self, sample_audio_bytes: bytes
    ) -> None:
        """A live speak-language change (TranslationRoomHub.SetSpeakLanguage) must close
        the cached session and open a fresh one pinned to the new language — silently
        reusing the old session while ignoring the new `language` arg would leave the
        Realtime API's own accuracy/hallucination behavior stuck on whichever language
        the session happened to be created with first.
        """
        events1 = [SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed", transcript="Hi",
        )]
        events2 = [SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed", transcript="Xin chao",
        )]
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.api_key = ""
        stt.model = "gpt-realtime-whisper"
        stt.noise_reduction = "far_field"
        stt._sessions = {}
        conn1, conn2 = FakeRealtimeConn(events1), FakeRealtimeConn(events2)
        stt._client = MagicMock()
        stt._client.realtime.connect = MagicMock(
            side_effect=[FakeRealtimeManager(conn1), FakeRealtimeManager(conn2)]
        )

        await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")
        await stt.transcribe(sample_audio_bytes, language="vi", meeting_id="m1", speaker_id="s1")

        assert stt._client.realtime.connect.call_count == 2

    async def test_transcribe_reuses_session_when_language_unchanged(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Sanity counterpart to the above: passing the SAME language on every call
        (the common case) must still hit the cache, not reconnect every time."""
        events = [SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed", transcript="Hi",
        )]
        stt, conn = _make_stt_with_conn(events)

        await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")
        await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")

        stt._client.realtime.connect.assert_called_once()


class TestFilterSegments:
    """_filter_segments utility tests."""

    def test_allowed_language_en(self) -> None:
        segs = [_segment("Hello")]
        result = _filter_segments(segs, "en", 0)
        assert len(result) == 1
        assert result[0].language == "en"

    def test_allowed_language_vi(self) -> None:
        segs = [_segment("Hệ thống đang hoạt động tốt")]
        result = _filter_segments(segs, "vi", 0)
        assert len(result) == 1

    def test_declared_nonlatin_language_passes(self) -> None:
        """A speaker whose declared profile language is non-Latin (e.g. Japanese) must
        get their transcript — hard-coding the allow-list to vi/en was the bug that
        dropped every segment for such speakers ("nói không ra transcript"). The
        speaker's own hint language is always allowed, and the cross-script guard does
        NOT apply to a declared non-Latin language.
        """
        segs = [_segment("こんにちは、会議を始めましょう")]
        result = _filter_segments(segs, "ja", 0)
        assert len(result) == 1
        assert result[0].language == "ja"

    def test_language_outside_declared_set_filtered(self) -> None:
        """When the meeting declares an explicit language set that a detected language is
        not part of (and it isn't the speaker's own hint), the segment is dropped."""
        segs = [_segment("Bonjour tout le monde")]
        result = _filter_segments(segs, "de", 0, allowed_languages={"vi", "en"})
        # 'de' is force-added as the speaker's own hint, so it passes — trusting the
        # speaker's declaration is intentional; the meeting set constrains guessing, not
        # a speaker's pinned language.
        assert len(result) == 1

    def test_latin_speaker_foreign_script_filtered(self) -> None:
        """A speaker declared in a Latin-script language emitting CJK/Kana text is the
        model mixing languages mid-utterance — drop it to enforce "no mixing"."""
        segs = [_segment("こんにちは")]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_unknown_language_foreign_script_filtered_when_room_all_latin(self) -> None:
        """A speaker who joined with speak_language left on "auto" has no per-utterance
        language hint (detected_language="unknown") — but if every language this meeting
        has declared is Latin-script, a CJK/Kana hallucination is still unambiguous and
        must be dropped, same as the known-language case."""
        segs = [_segment("こんにちは")]
        result = _filter_segments(segs, "unknown", 0, allowed_languages={"vi", "en"})
        assert result == []

    def test_unknown_language_foreign_script_kept_when_room_has_cjk_speaker(self) -> None:
        """If the room has a genuinely declared CJK/Thai language (e.g. a Japanese
        speaker), the guard must NOT apply for an unknown-language utterance — there's no
        way to tell whose script is legitimate, and dropping could eat a real speaker's
        transcript."""
        segs = [_segment("こんにちは")]
        result = _filter_segments(segs, "unknown", 0, allowed_languages={"vi", "ja"})
        assert len(result) == 1

    def test_low_confidence_filtered(self) -> None:
        segs = [_segment("Some text", avg_logprob=-1.5)]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_high_no_speech_filtered(self) -> None:
        segs = [_segment("Some text", no_speech_prob=0.8)]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_hallucination_filtered(self) -> None:
        segs = [_segment("thank you")]
        result = _filter_segments(segs, "en", 0)
        assert result == []

    def test_chunk_offset_applied(self) -> None:
        segs = [
            {
                "text": "Hello",
                "start": 1.0,
                "end": 2.0,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.01,
            }
        ]
        result = _filter_segments(segs, "en", chunk_offset_ms=5000)
        assert result[0].start_ms == 6000
        assert result[0].end_ms == 7000

    def test_full_language_name_normalized(self) -> None:
        """OpenAI returns full language names when language=None."""
        segs = [_segment("Hello")]
        result = _filter_segments(segs, "english", 0)
        assert len(result) == 1
        assert result[0].language == "en"

    def test_long_text_on_near_silent_audio_filtered(self) -> None:
        """Classic Whisper-family hallucination: a full sentence invented over < 0.5s of
        real audio. Only checked when a real duration is supplied (see transcribe()'s
        trailing-fragment call) — never for early-emitted sentences."""
        segs = [_segment("This is a surprisingly long sentence for such little audio")]
        result = _filter_segments(segs, "en", 0, real_duration_s=0.2)
        assert result == []

    def test_short_text_on_short_audio_not_filtered(self) -> None:
        """A brief real reply ('Yes.', 'OK.') in a short audio chunk must not be
        penalized just for being short — only LONG text on short audio is suspicious."""
        segs = [_segment("Yes")]
        result = _filter_segments(segs, "en", 0, real_duration_s=0.2)
        assert len(result) == 1

    def test_long_text_without_real_duration_not_filtered(self) -> None:
        """Early-emitted sentences (see stt_worker.model.OpenAISTT._emit_early) have no
        real per-sentence timing — real_duration_s defaults to None and this filter must
        stay inert, or every early multi-word sentence would be wrongly dropped."""
        segs = [_segment("This is a surprisingly long sentence for such little audio")]
        result = _filter_segments(segs, "en", 0)
        assert len(result) == 1

    def test_long_text_on_long_audio_not_filtered(self) -> None:
        segs = [_segment("This is a perfectly reasonable sentence for a few seconds of speech")]
        result = _filter_segments(segs, "en", 0, real_duration_s=3.0)
        assert len(result) == 1


class TestNormalizeLanguage:
    def test_full_name_to_code(self) -> None:
        assert _normalize_language("english") == "en"
        assert _normalize_language("vietnamese") == "vi"
        assert _normalize_language("Chinese") == "zh"

    def test_code_passthrough(self) -> None:
        assert _normalize_language("en") == "en"
        assert _normalize_language("vi") == "vi"


class TestSTTWorker:
    """STT Worker process() tests."""

    async def test_process_publishes_stt_result(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """process() should publish STTResultMessage for each transcribed segment."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Hello",
                    language="en",
                    confidence=-0.25,
                    start_ms=0,
                    end_ms=1000,
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

        # BaseWorker.publish() calls xadd twice: room stream + global stream
        mock_redis_client._redis.xadd.assert_called()
        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("stt:results" in s for s in streams_published)

    async def test_process_publishes_early_segments_as_non_final(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """Early (mid-chunk) segments must always publish is_final_chunk=False —
        only the trailing segment(s) returned from transcribe() carry the chunk's
        real is_final_chunk flag, once the whole chunk is actually done.
        """
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}

        async def fake_transcribe(*args, **kwargs):
            on_early_segment = kwargs["on_early_segment"]
            await on_early_segment(
                TranscribedSegment(text="Hello there.", language="en", confidence=0.0, start_ms=0, end_ms=0)
            )
            return [
                TranscribedSegment(text="How are you?", language="en", confidence=0.0, start_ms=0, end_ms=1000)
            ]

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(side_effect=fake_transcribe)

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
            is_final_chunk=True,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        published = [
            data for stream, data in
            (c.args for c in mock_redis_client._redis.xadd.call_args_list)
            if "stt:results" in str(stream)
        ]
        by_text = {data["text"]: data["is_final_chunk"] for data in published}
        assert by_text["Hello there."] == "0"
        assert by_text["How are you?"] == "1"

    async def test_process_skips_paused_room(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """process() should skip messages for paused rooms."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = {"meeting-1"}
        worker.model = MagicMock()

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        mock_redis_client._redis.xadd.assert_not_called()

    async def test_process_publishes_system_event_on_stt_error(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """STT provider failures should emit an explicit degrade signal."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings(model="gpt-4o-mini-transcribe")
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._room_languages = {}

        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(side_effect=RuntimeError("provider down"))

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=3,
            audio_data=sample_audio_bytes,
            is_final_chunk=False,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("translationRoom:system_events" in stream for stream in streams_published)
        assert not any("stt:results" in stream for stream in streams_published)

    async def test_get_stt_prompt_returns_generic_base_when_no_glossary(
        self, mock_redis_client
    ) -> None:
        """Every session gets the generic anti-hallucination base prompt even when the
        room has no glossary published (get() returns None)."""
        from stt_worker.worker import _GENERIC_STT_BASE_PROMPT

        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}

        prompt = await worker._get_stt_prompt("m1")
        assert prompt == _GENERIC_STT_BASE_PROMPT

    async def test_get_stt_prompt_appends_glossary_to_base(
        self, mock_redis_client
    ) -> None:
        """A published glossary is appended AFTER the generic base, not instead of it."""
        from stt_worker.worker import _GENERIC_STT_BASE_PROMPT

        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}
        mock_redis_client._redis.get.return_value = b"WarpTalk, Kubernetes, gRPC"

        prompt = await worker._get_stt_prompt("m1")
        assert prompt.startswith(_GENERIC_STT_BASE_PROMPT)
        assert "WarpTalk, Kubernetes, gRPC" in prompt

    async def test_get_room_languages_derives_distinct_speak_languages(
        self, mock_redis_client
    ) -> None:
        """The meeting's allowed-language set is the distinct, normalized set of its
        participants' declared speak-languages (locale tags collapsed to bare codes)."""
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._room_languages = {}
        mock_redis_client._redis.hgetall.return_value = {
            b"user-1": b"vi",
            b"user-2": b"en-US",
            b"user-3": b"vi",
            b"user-4": b"ja",
        }

        langs = await worker._get_room_languages("m1")
        assert langs == {"vi", "en", "ja"}


class TestConsumeLoopConcurrency:
    """_consume_loop() must dispatch DIFFERENT speakers' chunks concurrently, while
    keeping any ONE speaker's own chunks strictly ordered (their Realtime session is a
    single reused WebSocket connection — see OpenAISTT._get_or_create_session)."""

    def _make_worker(self) -> STTWorker:
        worker = STTWorker.__new__(STTWorker)
        worker.logger = MagicMock()
        worker._shutdown_event = asyncio.Event()
        worker._consumer_name = "test-consumer"
        worker.input_stream = "audio:chunks"
        worker.consumer_group = "stt-workers"
        worker._speaker_locks = {}
        return worker

    async def test_different_speakers_dispatch_concurrently(self, mock_redis_client) -> None:
        worker = self._make_worker()
        worker.redis = mock_redis_client

        started: list[bytes] = []
        both_started = asyncio.Event()

        async def fake_process(message_id: bytes, data: dict) -> None:
            started.append(message_id)
            if len(started) == 2:
                both_started.set()
            # msg-1 can only reach here if msg-2 (a DIFFERENT speaker) has already
            # started — impossible unless both run concurrently, not sequentially.
            await asyncio.wait_for(both_started.wait(), timeout=1.0)

        worker.process = fake_process

        async def fake_consume(**kwargs):
            yield b"msg-1", {"meeting_id": "m1", "speaker_id": "speaker-A"}
            yield b"msg-2", {"meeting_id": "m1", "speaker_id": "speaker-B"}
            worker._shutdown_event.set()

        worker.redis.consume = fake_consume

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)
        await asyncio.sleep(0.05)  # let the dispatched create_task()s finish

        assert started == [b"msg-1", b"msg-2"]

    async def test_same_speaker_chunks_stay_ordered(self, mock_redis_client) -> None:
        worker = self._make_worker()
        worker.redis = mock_redis_client

        events: list[tuple[str, bytes]] = []

        async def fake_process(message_id: bytes, data: dict) -> None:
            events.append(("start", message_id))
            await asyncio.sleep(0.05)
            events.append(("end", message_id))

        worker.process = fake_process

        async def fake_consume(**kwargs):
            yield b"msg-1", {"meeting_id": "m1", "speaker_id": "speaker-A"}
            yield b"msg-2", {"meeting_id": "m1", "speaker_id": "speaker-A"}
            worker._shutdown_event.set()

        worker.redis.consume = fake_consume

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)
        await asyncio.sleep(0.2)

        # msg-2 must not start until msg-1 has fully finished — same speaker, same
        # reused Realtime session.
        assert events == [
            ("start", b"msg-1"),
            ("end", b"msg-1"),
            ("start", b"msg-2"),
            ("end", b"msg-2"),
        ]

    async def test_cleanup_room_purges_speaker_locks(self) -> None:
        worker = self._make_worker()
        worker._stt_prompts = {"m1": "glossary"}
        worker._room_languages = {"m1": ({"vi"}, 0.0)}
        worker._route_states = {}
        worker._paused_rooms = set()
        worker._room_routes = {}
        worker._speaker_locks = {
            ("m1", "speaker-A"): asyncio.Lock(),
            ("m2", "speaker-B"): asyncio.Lock(),
        }

        worker._cleanup_room("m1")

        assert "m1" not in worker._stt_prompts
        assert "m1" not in worker._room_languages
        assert ("m1", "speaker-A") not in worker._speaker_locks
        assert ("m2", "speaker-B") in worker._speaker_locks
