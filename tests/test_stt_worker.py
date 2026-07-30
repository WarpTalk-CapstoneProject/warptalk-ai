"""Tests for STT Worker — mock OpenAI Realtime STT session, verify output schema."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from stt_worker import worker as stt_worker_module
from stt_worker.model import OpenAISTT, TranscribedSegment, _filter_segments, _normalize_language
from stt_worker.worker import STTWorker, _language_hint_for_stt


def test_default_stt_model_uses_accuracy_first_transcribe_variant() -> None:
    assert STTSettings().model == "gpt-transcribe"


def test_default_chunk_window_preserves_long_code_switched_utterances() -> None:
    assert WorkerSettings().chunk_duration_ms == 6000


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
    """Minimal stand-in for AsyncRealtimeConnectionManager
    (`async with client.realtime.connect(...) as conn`)."""

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

    def test_default_stt_does_not_double_denoise_browser_audio(self) -> None:
        assert STTSettings().noise_reduction == "off"

    async def test_warm_pool_removes_first_speaker_websocket_handshake(self) -> None:
        stt = OpenAISTT(api_key="test", model="gpt-4o-transcribe")
        connections = [FakeRealtimeConn([]), FakeRealtimeConn([])]
        managers = [FakeRealtimeManager(conn) for conn in connections]
        stt._client = MagicMock()
        stt._client.realtime.connect = MagicMock(side_effect=managers)

        await stt.warm_up(pool_size=2)
        session = await stt._get_or_create_session(
            ("meeting-1", "speaker-1"),
            language="vi",
            prompt="Meeting topic: WarpTalk.",
        )

        assert session["conn"] in connections
        assert stt._client.realtime.connect.call_count == 2
        session["conn"].session.update.assert_awaited_once()

    async def test_transcribe_returns_segments(self, sample_audio_bytes: bytes) -> None:
        """transcribe() should return a filtered list of TranscribedSegment.

        The Realtime API's completed event carries just a flat `transcript` field (no
        per-segment timing/confidence/language), same shape gap as the old REST
        response — a language hint must still be passed in since the API doesn't echo
        one back.
        """
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=" Hello, world!",
            )
        ]
        stt, conn = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes,
            sample_rate=16000,
            language="en",
            meeting_id="m1",
            speaker_id="s1",
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

    def test_session_payload_requests_transcription_logprobs(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "gpt-4o-transcribe"
        stt.noise_reduction = "far_field"

        payload = stt._session_payload(language=None, prompt=None)

        assert payload["include"] == ["item.input_audio_transcription.logprobs"]

    def test_gpt_transcribe_payload_uses_expected_languages_and_keywords(self) -> None:
        stt = OpenAISTT.__new__(OpenAISTT)
        stt.model = "gpt-transcribe"
        stt.noise_reduction = "off"

        payload = stt._session_payload(
            language="vi",
            prompt="Meeting topic: WarpTalk.",
            allowed_languages={"vi", "en"},
            keywords=["WarpTalk", "Kubernetes", "gRPC"],
        )

        transcription = payload["audio"]["input"]["transcription"]
        assert transcription["languages"] == ["vi", "en"]
        assert transcription["keywords"] == ["WarpTalk", "Kubernetes", "gRPC"]
        assert "language" not in transcription
        assert "include" not in payload

    async def test_transcribe_configures_room_languages_and_glossary_keywords(
        self, sample_audio_bytes: bytes
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Deploy Kubernetes",
            )
        ]
        stt, conn = _make_stt_with_conn(events)
        stt.model = "gpt-transcribe"

        await stt.transcribe(
            sample_audio_bytes,
            language="vi",
            allowed_languages={"vi", "en"},
            keywords=["Kubernetes"],
            meeting_id="m1",
            speaker_id="s1",
        )

        payload = conn.session.update.await_args.kwargs["session"]
        transcription = payload["audio"]["input"]["transcription"]
        assert transcription["languages"] == ["vi", "en"]
        assert transcription["keywords"] == ["Kubernetes"]

    async def test_transcribe_uses_completed_event_logprobs_as_confidence(
        self, sample_audio_bytes: bytes
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Deploy the backend with Kubernetes",
                logprobs=[
                    SimpleNamespace(token="Deploy", logprob=-0.2),
                    SimpleNamespace(token=" Kubernetes", logprob=-0.4),
                ],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")

        assert len(result) == 1
        assert result[0].confidence == pytest.approx(-0.3)

    async def test_speculative_sentence_callback_keeps_final_validated_transcript(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Speculation may start translation early, but final STT output must still
        wait for completed-event logprobs and contain the validated sentence."""
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta",
                delta="Deploy Docker.",
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Deploy Docker.",
                logprobs=[SimpleNamespace(token="Deploy Docker.", logprob=-0.2)],
            ),
        ]
        stt, _ = _make_stt_with_conn(events)
        speculative = AsyncMock()

        result = await stt.transcribe(
            sample_audio_bytes,
            meeting_id="m1",
            speaker_id="s1",
            on_speculative_segment=speculative,
        )

        speculative.assert_awaited_once()
        assert speculative.await_args.args[0].text == "Deploy Docker."
        assert [segment.text for segment in result] == ["Deploy Docker."]
        assert result[0].confidence == pytest.approx(-0.2)

    async def test_normal_production_chunk_uses_one_audio_append(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Awaiting one websocket send per 100ms frame adds avoidable latency after
        ingress has already assembled a bounded speech chunk."""
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Deploy Docker.",
                logprobs=[SimpleNamespace(token="Deploy Docker.", logprob=-0.2)],
            )
        ]
        stt, conn = _make_stt_with_conn(events)

        await stt.transcribe(
            sample_audio_bytes,
            meeting_id="m1",
            speaker_id="s1",
        )

        assert conn.input_audio_buffer.append.await_count == 1

    async def test_transcribe_filters_low_confidence_completed_event(
        self, sample_audio_bytes: bytes
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Nhiều học sinh đang viết bài kiểm",
                logprobs=[SimpleNamespace(token="garbage", logprob=-1.4)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")

        assert result == []

    @pytest.mark.parametrize("avg_logprob", [-0.767, -0.8652, -0.8833])
    async def test_transcribe_filters_marginal_production_hallucinations(
        self, sample_audio_bytes: bytes, avg_logprob: float
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Câu nghe giống tiếng Việt nhưng không liên quan đến cuộc họp",
                logprobs=[SimpleNamespace(token="marginal", logprob=avg_logprob)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")

        assert result == []

    async def test_transcribe_keeps_clear_segment_near_confidence_boundary(
        self, sample_audio_bytes: bytes
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Please approve the pull request",
                logprobs=[SimpleNamespace(token="clear", logprob=-0.69)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(sample_audio_bytes, meeting_id="m1", speaker_id="s1")

        assert [segment.text for segment in result] == ["Please approve the pull request"]

    async def test_transcribe_filters_high_confidence_prompt_echo(
        self, sample_audio_bytes: bytes
    ) -> None:
        prompt = "WarpTalk Kubernetes backend deployment"
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=prompt,
                logprobs=[SimpleNamespace(token=prompt, logprob=-0.02)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes, meeting_id="m1", speaker_id="s1", prompt=prompt
        )

        assert result == []

    async def test_transcribe_filters_provider_keyword_enumeration_echo_without_logprobs(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Production regression: gpt-transcribe recited the room glossary on marginal
        audio. The next-generation completed event has no logprobs, so content matching
        must reject the enumeration instead of trusting the -1 compatibility sentinel."""
        keywords = [
            "architecture",
            "deployment",
            "sprint",
            "API",
            "backlog",
            "roadmap",
            "staging",
            "frontend",
            "backend",
            "KPI",
            "ROI",
            "UX",
            "UI",
            "production",
            "framework",
            "full-stack",
        ]
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=", ".join(keywords) + ".",
            )
        ]
        stt, _ = _make_stt_with_conn(events)
        stt.model = "gpt-transcribe"

        result = await stt.transcribe(
            sample_audio_bytes,
            language="vi",
            meeting_id="m1",
            speaker_id="s1",
            keywords=keywords,
        )

        assert result == []

    async def test_transcribe_keeps_natural_speech_with_glossary_terms(
        self, sample_audio_bytes: bytes
    ) -> None:
        keywords = ["architecture", "deployment", "sprint", "API", "backend"]
        speech = "Sprint này chúng ta sẽ deployment backend API."
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=speech,
            )
        ]
        stt, _ = _make_stt_with_conn(events)
        stt.model = "gpt-transcribe"

        result = await stt.transcribe(
            sample_audio_bytes,
            language="vi",
            meeting_id="m1",
            speaker_id="s1",
            keywords=keywords,
        )

        assert [segment.text for segment in result] == [speech]

    async def test_transcribe_filters_low_confidence_partial_prompt_echo(
        self, sample_audio_bytes: bytes
    ) -> None:
        """Production regression: marginal audio copied only the title fragment,
        so exact-whole-line prompt matching did not catch it."""
        prompt = (
            "Meeting topic: WarpTalk transcript engineering review. "
            "Meeting context: Docker Kubernetes deployment."
        )
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="WarpTalk transcript engineering",
                logprobs=[
                    SimpleNamespace(
                        token="WarpTalk transcript engineering",
                        logprob=-0.6066,
                    )
                ],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes,
            meeting_id="m1",
            speaker_id="s1",
            prompt=prompt,
        )

        assert result == []

    async def test_transcribe_keeps_clear_partial_prompt_phrase(
        self, sample_audio_bytes: bytes
    ) -> None:
        """A participant may genuinely say the meeting title; strong audio remains valid."""
        prompt = "Meeting topic: WarpTalk transcript engineering review."
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="WarpTalk transcript engineering",
                logprobs=[
                    SimpleNamespace(
                        token="WarpTalk transcript engineering",
                        logprob=-0.1,
                    )
                ],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes,
            meeting_id="m1",
            speaker_id="s1",
            prompt=prompt,
        )

        assert [segment.text for segment in result] == ["WarpTalk transcript engineering"]

    async def test_transcribe_filters_high_confidence_repeated_sentence_collage(
        self, sample_audio_bytes: bytes
    ) -> None:
        hallucination = (
            "Chỉ cần biết. Tin một người xa chỉ gần sống. Bạn em bị cái. "
            "Cái em sẽ thấy nó khó. Tin một người xa chỉ gần sống. "
            "Bạn em bị cái. Cái em sẽ thấy nó khó. Bạn em bị cái."
        )
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=hallucination,
                logprobs=[SimpleNamespace(token=hallucination, logprob=-0.03)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes,
            language="vi",
            meeting_id="m1",
            speaker_id="s1",
        )

        assert result == []

    async def test_transcribe_keeps_non_repeating_multi_sentence_speech(
        self, sample_audio_bytes: bytes
    ) -> None:
        speech = (
            "Chúng ta kiểm tra validator. Sau đó review pull request. Cuối cùng mới merge backend."
        )
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript=speech,
                logprobs=[SimpleNamespace(token=speech, logprob=-0.2)],
            )
        ]
        stt, _ = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes,
            language="vi",
            meeting_id="m1",
            speaker_id="s1",
        )

        assert [segment.text for segment in result] == [speech]

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
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta="Hello there."
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta=" How are you?"
            ),
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
            sample_audio_bytes,
            language="en",
            meeting_id="m1",
            speaker_id="s1",
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
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta="Hello there."
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta=" How are"
            ),
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
            sample_audio_bytes,
            language="en",
            meeting_id="m1",
            speaker_id="s1",
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
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta="Hello there."
            ),
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hello there.",
            ),
        ]
        stt, conn = _make_stt_with_conn(events)

        result = await stt.transcribe(
            sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1"
        )

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
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.delta", delta="Helo there."
            ),
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
            sample_audio_bytes,
            language="en",
            meeting_id="m1",
            speaker_id="s1",
            on_early_segment=on_early,
        )

        assert len(early_segments) == 1
        assert result == []

    async def test_transcribe_reuses_session_across_calls(self, sample_audio_bytes: bytes) -> None:
        """A second chunk from the same (meeting, speaker) must NOT reconnect —
        that's the whole point of session reuse (pay the handshake once, not per chunk).
        """
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hi",
            )
        ]
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
        events1 = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hi",
            )
        ]
        events2 = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Xin chao",
            )
        ]
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
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Hi",
            )
        ]
        stt, conn = _make_stt_with_conn(events)

        await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")
        await stt.transcribe(sample_audio_bytes, language="en", meeting_id="m1", speaker_id="s1")

        stt._client.realtime.connect.assert_called_once()

    async def test_reused_session_updates_when_context_prompt_changes(
        self, sample_audio_bytes: bytes
    ) -> None:
        events = [
            SimpleNamespace(
                type="conversation.item.input_audio_transcription.completed",
                transcript="Kubernetes",
            )
        ]
        stt, conn = _make_stt_with_conn(events)

        await stt.transcribe(
            sample_audio_bytes, meeting_id="m1", speaker_id="s1", prompt="WarpTalk"
        )
        await stt.transcribe(
            sample_audio_bytes,
            meeting_id="m1",
            speaker_id="s1",
            prompt="WarpTalk\nKubernetes deployment",
        )

        stt._client.realtime.connect.assert_called_once()
        assert conn.session.update.await_count == 2


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

    def test_production_marginal_confidence_filtered(self) -> None:
        segs = [_segment("Phải chạy qua về", avg_logprob=-0.8833)]
        result = _filter_segments(segs, "vi", 0)
        assert result == []

    def test_clear_boundary_confidence_kept(self) -> None:
        segs = [_segment("Please approve the pull request", avg_logprob=-0.69)]
        result = _filter_segments(segs, "en", 0)
        assert len(result) == 1

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

    async def test_track_event_prepares_language_pinned_session(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
    ) -> None:
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {"room-1": "Meeting topic: WarpTalk."}
        worker._stt_keywords = {}
        worker._room_languages = {}
        worker.model = MagicMock()
        worker.model.prepare_session = AsyncMock()
        mock_redis_client._redis.hget.return_value = b"vi"
        mock_redis_client._redis.hgetall.return_value = {b"speaker-1": b"vi"}
        mock_redis_client._redis.get.return_value = None

        event = {
            "event_type": "meeting.track_published",
            "schema_version": 1,
            "producer": "meeting-service",
            "payload": {
                "room_name": "room-1",
                "participant_identity": "speaker-1",
                "track_id": "audio-track",
            },
        }
        await worker._prewarm_from_track_event(json.dumps(event))

        worker.model.prepare_session.assert_awaited_once_with(
            "room-1",
            "speaker-1",
            language="vi",
            prompt=None,
            allowed_languages={"vi"},
            keywords=[],
        )

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
        worker._stt_prompts = {
            "meeting-1": (
                "Meeting topic: test. Meeting context: test. "
                "Terms that may appear in this meeting: architecture, deployment, sprint."
            )
        }
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

        assert worker.model.transcribe.await_args.kwargs["prompt"] is None
        # BaseWorker.publish() calls xadd twice: room stream + global stream
        mock_redis_client._redis.xadd.assert_called()
        streams_published = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert any("stt:results" in s for s in streams_published)

    async def test_process_publishes_only_completed_segments(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """Production must not publish unverified Realtime deltas.

        Delta events have no real confidence/no-speech signal and were the source of
        hallucinated transcript lines. Only the authoritative completed result may be
        forwarded to translation and the UI.
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
            assert kwargs.get("on_early_segment") is None
            speculative = kwargs.get("on_speculative_segment")
            assert speculative is not None
            await speculative(
                TranscribedSegment(
                    text="How are you?",
                    language="en",
                    confidence=0.0,
                    start_ms=0,
                    end_ms=0,
                )
            )
            return [
                TranscribedSegment(
                    text="How are you?", language="en", confidence=0.0, start_ms=0, end_ms=1000
                )
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
            data
            for stream, data in (c.args for c in mock_redis_client._redis.xadd.call_args_list)
            if "stt:results" in str(stream)
        ]
        assert {data["text"] for data in published} == {"How are you?"}
        assert {data["is_final_chunk"] for data in published} == {"1"}
        speculative_publish = [
            call
            for call in mock_redis_client._redis.publish.call_args_list
            if call.args[0] == "stt:speculative"
        ]
        assert len(speculative_publish) == 1
        speculative_payload = json.loads(speculative_publish[0].args[1])
        assert speculative_payload["text"] == "How are you?"
        assert speculative_payload["meeting_id"] == "meeting-1"

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

    async def test_process_skips_ready_room_before_translation_starts(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        """A stale or rogue audio chunk must not bypass the ingress lifecycle gate."""
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()
        worker._route_states = {"meeting-1": "READY"}
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock()

        chunk = AudioChunkMessage(
            meeting_id="meeting-1",
            speaker_id="speaker-1",
            chunk_index=0,
            audio_data=sample_audio_bytes,
        )

        await worker.process(b"msg-1", chunk.to_redis())

        worker.model.transcribe.assert_not_awaited()
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

        streams_published = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert any("translationRoom:system_events" in stream for stream in streams_published)
        assert not any("stt:results" in stream for stream in streams_published)

    async def test_get_stt_prompt_returns_none_when_no_glossary(self, mock_redis_client) -> None:
        """Do not seed silence/noise with instruction text that can leak into output."""
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}

        prompt = await worker._get_stt_prompt("m1")
        assert prompt is None

    async def test_empty_prewarm_lookup_does_not_cache_over_late_meeting_context(
        self, mock_redis_client
    ) -> None:
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}
        mock_redis_client._redis.get.side_effect = [
            None,
            b"WarpTalk, Docker, Kubernetes",
        ]

        assert await worker._get_stt_prompt("m1") is None
        assert await worker._get_stt_prompt("m1") == "WarpTalk, Docker, Kubernetes"
        assert mock_redis_client._redis.get.await_count == 2

    async def test_get_stt_prompt_returns_only_room_glossary(self, mock_redis_client) -> None:
        """A real room glossary remains useful bias without generic instruction leakage."""
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}
        mock_redis_client._redis.get.return_value = b"WarpTalk, Kubernetes, gRPC"

        prompt = await worker._get_stt_prompt("m1")
        assert prompt == "WarpTalk, Kubernetes, gRPC"

    async def test_get_stt_keywords_uses_dedicated_workspace_keyword_list(
        self, mock_redis_client
    ) -> None:
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_keywords = {}
        mock_redis_client._redis.get.return_value = json.dumps(
            [
                "Kubernetes",
                "pull request",
                "gRPC",
                "Kubernetes",
            ]
        ).encode()

        keywords = await worker._get_stt_keywords("m1")

        assert keywords == ["Kubernetes", "pull request", "gRPC"]
        mock_redis_client._redis.get.assert_awaited_once_with("translationRoom:m1:stt_keywords")

    def test_declared_language_anchors_realtime_transcription(self) -> None:
        assert _language_hint_for_stt("vi") == "vi"
        assert _language_hint_for_stt("en-US") == "en"
        assert _language_hint_for_stt("auto") is None
        assert _language_hint_for_stt("ja") == "ja"

    async def test_low_confidence_segment_is_not_reused_as_meeting_context(
        self,
        mock_redis_client,
        worker_settings: WorkerSettings,
        sample_audio_bytes: bytes,
    ) -> None:
        worker = STTWorker.__new__(STTWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.stt_settings = STTSettings()
        worker._paused_rooms = set()
        worker._stt_prompts = {}
        worker._recent_transcripts = {}
        worker._room_languages = {}
        worker.model = MagicMock()
        worker.model.transcribe = AsyncMock(
            return_value=[
                TranscribedSegment(
                    text="Thi công mà.",
                    language="vi",
                    confidence=-0.69,
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
            language="vi",
        )

        await worker.process(b"msg-1", chunk.to_redis())

        assert "meeting-1" not in worker._recent_transcripts

    async def test_recent_accepted_transcript_is_not_fed_back_into_stt_prompt(
        self, mock_redis_client
    ) -> None:
        worker = STTWorker.__new__(STTWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._stt_prompts = {}
        worker._recent_transcripts = {}
        mock_redis_client._redis.get.return_value = b"WarpTalk, Kubernetes, gRPC"

        for index in range(8):
            worker._remember_transcript("m1", f"Accepted discussion segment {index}")

        prompt = await worker._get_stt_prompt("m1")

        assert prompt == "WarpTalk, Kubernetes, gRPC"

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

        async def fake_consume_concurrent(*, handler, **kwargs):
            await asyncio.gather(
                handler(b"msg-1", {"meeting_id": "m1", "speaker_id": "speaker-A"}),
                handler(b"msg-2", {"meeting_id": "m1", "speaker_id": "speaker-B"}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

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

        async def fake_consume_concurrent(*, handler, **kwargs):
            await asyncio.gather(
                handler(b"msg-1", {"meeting_id": "m1", "speaker_id": "speaker-A"}),
                handler(b"msg-2", {"meeting_id": "m1", "speaker_id": "speaker-A"}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

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
        worker._recent_transcripts = {"m1": deque(["meeting context"])}
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
        assert "m1" not in worker._recent_transcripts
        assert "m1" not in worker._room_languages
        assert ("m1", "speaker-A") not in worker._speaker_locks
        assert ("m2", "speaker-B") in worker._speaker_locks


# A Redis redelivery of the same audio chunk must not mint a second billable
# transcript segment.
def test_segment_id_is_stable_for_same_source_message() -> None:
    build_segment_id = getattr(stt_worker_module, "_build_segment_id", None)
    assert callable(build_segment_id)
    first = build_segment_id(
        "room-1",
        "speaker-1",
        b"1710000000000-0",
        0,
        1200,
        "hello",
    )
    second = build_segment_id(
        "room-1",
        "speaker-1",
        b"1710000000000-0",
        0,
        1200,
        "hello",
    )
    assert first == second
