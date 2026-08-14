"""Tests for Translation Worker — mock translator, verify passthrough and routing logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TranslationSettings, WorkerSettings
from shared.schemas import STT_UNKNOWN_CONFIDENCE, ProsodyEnvelope, STTResultMessage
from translation_worker.translator import (
    OpenAITranslator,
    _build_glossary_block,
    _exception_clause,
    _lang_name,
)
from translation_worker.worker import TranslationWorker, _select_relevance_context


class TestLangName:
    """_lang_name helper tests."""

    def test_known_codes(self) -> None:
        assert _lang_name("en") == "English"
        assert _lang_name("vi") == "Vietnamese"
        assert _lang_name("zh") == "Chinese (Simplified)"
        assert _lang_name("ja") == "Japanese"

    def test_unknown_code_returns_code(self) -> None:
        assert _lang_name("xx") == "xx"

    def test_hyphenated_code_uses_base(self) -> None:
        # "en-US" → "en" → "English"
        assert _lang_name("en-US") == "English"


class TestRelevanceContextSelection:
    context = [
        "Meeting topic: WarpTalk transcript engineering review. "
        "Meeting context: Docker, Kubernetes, Redis, LiveKit, pull requests."
    ]

    def test_clear_topic_overlap_uses_fast_translation_path(self) -> None:
        assert (
            _select_relevance_context(
                "Hôm nay chúng ta review pull request và deploy Docker lên Kubernetes.",
                self.context,
            )
            == []
        )

    def test_short_acknowledgement_uses_fast_translation_path(self) -> None:
        assert _select_relevance_context("Được rồi.", self.context) == []

    def test_unrelated_long_utterance_keeps_context_for_ai_relevance_check(self) -> None:
        assert (
            _select_relevance_context(
                "Trận bóng tối qua thật hay và khán giả cổ vũ rất đông.",
                self.context,
            )
            == self.context
        )


class TestGlossaryBlock:
    """_build_glossary_block / _exception_clause — code-switching glossary support
    (see docs/code-switching-research.md).
    """

    def test_empty_returns_empty_string(self) -> None:
        assert _build_glossary_block(None) == ""
        assert _build_glossary_block([]) == ""

    def test_keep_verbatim_when_target_equals_source(self) -> None:
        block = _build_glossary_block([{"source": "architect", "target": "architect"}])
        assert "architect" in block
        assert "do not translate" in block.lower()

    def test_keep_verbatim_when_target_missing(self) -> None:
        block = _build_glossary_block([{"source": "sprint", "target": ""}])
        assert "sprint" in block
        assert "do not translate" in block.lower()

    def test_exact_mapping_when_target_differs(self) -> None:
        block = _build_glossary_block(
            [{"source": "marketing plan", "target": "kế hoạch marketing"}]
        )
        assert "marketing plan" in block
        assert "kế hoạch marketing" in block
        assert "exact translations" in block.lower()

    def test_skips_entries_without_source(self) -> None:
        block = _build_glossary_block([{"source": "", "target": "x"}])
        assert block == ""

    def test_exception_clause_mentions_glossary_only_when_present(self) -> None:
        assert "glossary" not in _exception_clause(None)
        assert "glossary" not in _exception_clause([])
        assert "glossary" in _exception_clause([{"source": "a", "target": "a"}])


class TestOpenAITranslator:
    """OpenAITranslator unit tests."""

    def test_latency_optimized_model_is_the_default(self) -> None:
        assert TranslationSettings().model == "gpt-5.4-nano"
        assert TranslationSettings().realtime_model == "gpt-realtime-2.1-mini"
        assert TranslationSettings().realtime_max_output_tokens == 128

    async def test_translate_calls_openai(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-4.1-mini"
        translator.max_tokens = 512
        translator.temperature = 0.1

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Xin chào"

        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await translator.translate("Hello", "en", "vi")

        assert result == "Xin chào"
        translator._client.chat.completions.create.assert_called_once()

    async def test_gpt5_uses_supported_low_latency_completion_parameters(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-5.4-nano"
        translator.max_tokens = 128
        translator.temperature = 0.0

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Confirm the validator."
        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await translator.translate("Confirm cái validator.", "vi", "en")

        _, kwargs = translator._client.chat.completions.create.call_args
        assert kwargs["max_completion_tokens"] == 128
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    async def test_warm_up_primes_the_real_model_connection(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-5.4-nano"
        translator.max_tokens = 128
        translator.temperature = 0.0
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OK"
        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await translator.warm_up()

        _, kwargs = translator._client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-5.4-nano"
        assert kwargs["max_completion_tokens"] == 8
        assert "temperature" not in kwargs

    async def test_translate_retries_on_transient_error(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-4.1-mini"
        translator.max_tokens = 512
        translator.temperature = 0.1

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Xin chào"

        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(
            side_effect=[RuntimeError("API transient error"), mock_response]
        )

        result = await translator.translate("Hello", "en", "vi")

        assert result == "Xin chào"
        assert translator._client.chat.completions.create.call_count == 2

    async def test_translate_empty_returns_empty(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator._client = MagicMock()

        result = await translator.translate("   ", "en", "vi")
        assert result == ""
        translator._client.chat.completions.create.assert_not_called()

    async def test_translate_same_language_passthrough(self) -> None:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator._client = MagicMock()

        result = await translator.translate("Hello", "en", "en")
        assert result == "Hello"
        translator._client.chat.completions.create.assert_not_called()

    async def test_translate_forwards_glossary_into_prompt(self) -> None:
        """The glossary must actually reach the OpenAI call, not just be accepted as a
        parameter — this is the fix for "architect" being over-translated/mistranslated
        (see docs/code-switching-research.md).
        """
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-4.1-mini"
        translator.max_tokens = 512
        translator.temperature = 0.1

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Cái architect hôm bữa không được ổn."
        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await translator.translate(
            "The architect thing wasn't good the other day.",
            "en",
            "vi",
            glossary_terms=[{"source": "architect", "target": "architect"}],
        )

        _, kwargs = translator._client.chat.completions.create.call_args
        user_message = kwargs["messages"][1]["content"]
        assert "architect" in user_message
        assert "do not translate" in user_message.lower()

    async def test_translate_prefers_realtime_and_includes_meeting_context(self) -> None:
        translator = OpenAITranslator(
            api_key="test",
            model="gpt-5.4-nano",
            realtime_model="gpt-realtime-2.1-mini",
        )
        translator._translate_realtime = AsyncMock(return_value="That's a validator.")
        translator._create_with_retry = AsyncMock()

        result = await translator.translate(
            "Đó là validator.",
            "vi",
            "en",
            glossary_terms=[{"source": "validator", "target": "validator"}],
            meeting_context=["Chúng ta đang review pull request."],
        )

        assert result == "That's a validator."
        translator._create_with_retry.assert_not_awaited()
        realtime_prompt = translator._translate_realtime.await_args.args[0]
        assert "Đó là validator." in realtime_prompt
        assert "Chúng ta đang review pull request." in realtime_prompt
        assert "validator" in realtime_prompt
        assert "[OUT_OF_MEETING_SCOPE]" in realtime_prompt
        assert "short acknowledgements" in realtime_prompt
        realtime_instructions = translator._translate_realtime.await_args.args[1]
        assert "[OUT_OF_MEETING_SCOPE]" in realtime_instructions
        assert "decide whether" in realtime_instructions

    async def test_translate_only_sends_glossary_terms_present_in_current_utterance(
        self,
    ) -> None:
        translator = OpenAITranslator(
            api_key="test",
            model="gpt-5.4-nano",
            realtime_model="gpt-realtime-2.1-mini",
        )
        translator._translate_realtime = AsyncMock(return_value="Deploy Docker.")
        translator._create_with_retry = AsyncMock()

        await translator.translate(
            "Deploy Docker.",
            "en",
            "vi",
            glossary_terms=[
                {"source": "Docker", "target": "Docker"},
                {"source": "marketing plan", "target": "kế hoạch marketing"},
                {"source": "UI", "target": "UI"},
            ],
        )

        realtime_prompt = translator._translate_realtime.await_args.args[0]
        assert '"Docker"' in realtime_prompt
        assert "marketing plan" not in realtime_prompt
        assert '"UI"' not in realtime_prompt

    async def test_translate_falls_back_to_chat_when_realtime_fails(self) -> None:
        translator = OpenAITranslator(
            api_key="test",
            model="gpt-5.4-nano",
            realtime_model="gpt-realtime-2.1-mini",
        )
        translator._translate_realtime = AsyncMock(side_effect=TimeoutError("slow"))
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "That's a validator."
        translator._create_with_retry = AsyncMock(return_value=mock_response)

        result = await translator.translate("Đó là validator.", "vi", "en")

        assert result == "That's a validator."
        translator._create_with_retry.assert_awaited_once()

    async def test_realtime_request_is_out_of_band_and_text_only(self) -> None:
        translator = OpenAITranslator(
            api_key="test",
            realtime_model="gpt-realtime-2.1-mini",
        )
        fake_connection = MagicMock()
        fake_connection.response.create = AsyncMock()
        created = MagicMock()
        created.type = "response.created"
        created.response.id = "resp-1"
        created.response.metadata = {"request_id": "request-1"}
        delta = MagicMock()
        delta.type = "response.output_text.delta"
        delta.response_id = "resp-1"
        delta.delta = "Hello"
        done = MagicMock()
        done.type = "response.done"
        done.response.id = "resp-1"
        done.response.status = "completed"
        fake_connection.recv = AsyncMock(side_effect=[created, delta, done])
        translator._acquire_realtime_connection = AsyncMock(return_value=(0, fake_connection))
        translator._release_realtime_connection = AsyncMock()

        result = await translator._translate_realtime(
            "Translate this", "System instructions", request_id="request-1"
        )

        assert result == "Hello"
        payload = fake_connection.response.create.await_args.kwargs["response"]
        assert payload["conversation"] == "none"
        assert payload["output_modalities"] == ["text"]
        assert payload["max_output_tokens"] == 128
        assert payload["reasoning"] == {"effort": "minimal"}
        assert payload["metadata"] == {"request_id": "request-1"}
        assert payload["input"][0]["content"][0]["text"] == "Translate this"
        translator._release_realtime_connection.assert_awaited_once_with(0, healthy=True)

    async def test_close_releases_all_realtime_connections(self) -> None:
        translator = OpenAITranslator(api_key="test", realtime_pool_size=2)
        first = MagicMock()
        first.close = AsyncMock()
        second = MagicMock()
        second.close = AsyncMock()
        translator._realtime_connections = [first, second]

        await translator.close()

        first.close.assert_awaited_once()
        second.close.assert_awaited_once()


class TestTranslateBatch:
    """OpenAITranslator.translate_batch() unit tests."""

    def _make_translator(self) -> OpenAITranslator:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-4.1-mini"
        translator.max_tokens = 512
        translator.temperature = 0.1
        translator._client = MagicMock()
        return translator

    async def test_empty_list_returns_empty(self) -> None:
        translator = self._make_translator()
        result = await translator.translate_batch([], "en", "vi")
        assert result == []
        translator._client.chat.completions.create.assert_not_called()

    async def test_same_language_passthrough(self) -> None:
        translator = self._make_translator()
        texts = ["Hello", "World"]
        result = await translator.translate_batch(texts, "en", "en")
        assert result == texts
        translator._client.chat.completions.create.assert_not_called()

    async def test_parses_numbered_response_in_order(self) -> None:
        translator = self._make_translator()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "[1] Xin chào\n[2] Thế giới"
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await translator.translate_batch(["Hello", "World"], "en", "vi")

        assert result == ["Xin chào", "Thế giới"]
        translator._client.chat.completions.create.assert_called_once()

    async def test_falls_back_to_concurrent_single_calls_on_parse_mismatch(self) -> None:
        translator = self._make_translator()
        mock_response = MagicMock()
        # Model only returned 1 line for 2 inputs — malformed batch response.
        mock_response.choices[0].message.content = "[1] Xin chào"
        translator._client.chat.completions.create = AsyncMock(return_value=mock_response)
        translator.translate_with_valence = AsyncMock(
            side_effect=[("Xin chào", None), ("Thế giới", None)]
        )

        result = await translator.translate_batch(["Hello", "World"], "en", "vi")

        assert result == ["Xin chào", "Thế giới"]
        assert translator.translate_with_valence.await_count == 2


class TestTranslationWorker:
    """TranslationWorker process() tests."""

    def _make_worker(self, mock_redis_client, worker_settings):
        worker = TranslationWorker.__new__(TranslationWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.translation_settings = TranslationSettings()
        worker._paused_rooms = set()
        # Translation is opt-in now: process() drops a segment unless the room has reported
        # translation active. These tests are about what translation DOES once started, so
        # they declare it started. The gate itself has its own tests.
        worker._route_states = {}
        worker._is_translation_active = lambda _room: True  # type: ignore[method-assign]
        worker._mt_glossaries = {}
        worker._recent_source_contexts = {}
        worker.worker_name = "translation"
        mock_translator = MagicMock()
        mock_translator.model = "gpt-4.1-mini"
        mock_translator.translate_with_valence = AsyncMock(return_value=("Xin chào", None))
        mock_translator.translate_batch = AsyncMock(return_value=[])
        worker.translator = mock_translator
        return worker

    def _make_stt_msg(self, language="en", text="Hello world"):
        return STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text=text,
            language=language,
            confidence=0.95,
        )

    async def test_delivery_is_carried_to_every_translation_of_a_segment(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """This worker is the courier for prosody, not a source of it.

        How something was said is settled at the audio, one stage upstream, and translating
        the words cannot change it. If it is dropped here the measurement is dead — the TTS
        worker reads this message and nothing else.
        """
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        stt = self._make_stt_msg(language="en").model_copy(
            update={
                "prosody": ProsodyEnvelope(
                    pitch_lift=1.3, pitch_variation=1.5, energy_ratio=1.4, arousal="high"
                )
            }
        )

        await worker.process(b"msg-1", stt.to_redis())

        published = [
            c.args[1]
            for c in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(c.args[0])
        ]
        assert published
        for payload in published:
            envelope = ProsodyEnvelope.from_wire(payload.get("prosody"))
            assert envelope is not None
            assert envelope.arousal == "high"
            assert envelope.pitch_lift == pytest.approx(1.3)

    async def test_same_language_listener_gets_nothing_published(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """S6 — this used to publish, and the publish is the bug.

        The old behaviour ("passthrough": skip the LLM but publish anyway) sent a
        TranslationResultMessage whose translated_text was the speaker's own words in the
        speaker's own language. TTSWorker had no language guard, so it synthesized that and
        published an ai-interpreter LiveKit track — and the listener, who is already
        subscribed to that speaker's raw mic, heard the real voice and a synthetic echo of
        the same sentence at once.
        """
        worker = self._make_worker(mock_redis_client, worker_settings)

        # The only other participant listens in the language the speaker is speaking.
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"en"}

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        worker.translator.translate.assert_not_called()
        streams = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert not any("translate:results" in s for s in streams)

    async def test_same_language_does_not_suppress_the_other_listeners(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Only the echoing target is dropped — a real listener must still be served."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hgetall.return_value = {
            b"listener-1": b"en",  # same as the speaker: echo, must be dropped
            b"listener-2": b"vi",  # a real translation target
        }

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        published = [
            c.kwargs.get("target_lang")
            for c in worker.translator.translate_with_valence.call_args_list
        ]
        assert published == ["vi"]

    async def test_regional_variant_of_the_speakers_language_is_still_an_echo(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """en-US -> en-GB is an echo: translator.translate returns the text verbatim for
        matching BASE tags, so an exact-match test would let a perfect echo through."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"en-GB"}

        await worker.process(b"msg-1", self._make_stt_msg(language="en-US").to_redis())

        streams = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert not any("translate:results" in s for s in streams)

    async def test_lone_english_speaker_does_not_get_an_echo_of_themselves(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """The `targets or {"en"}` fallback made a solo English speaker their own target."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hgetall.return_value = {}  # nobody else registered yet

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        streams = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert not any("translate:results" in s for s in streams)

    async def test_calls_translator_for_different_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should call translator when source != target language."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # Target lang = "vi", source lang = "en"
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        worker.translator.translate_with_valence.assert_called()

    async def test_default_fallback_language_is_en(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """When no per-speaker language set, fallback should be 'en' not 'vi'."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # No participants registered in the languages hash at all
        mock_redis_client._redis.hgetall.return_value = {}

        await worker.process(b"msg-1", self._make_stt_msg(language="vi").to_redis())

        # Translator called with target_lang="en"
        call_kwargs = worker.translator.translate_with_valence.call_args
        target = call_kwargs.kwargs.get("target_lang") or call_kwargs[1].get("target_lang")
        assert target == "en"

    async def test_publishes_translation_result(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """process() should publish TranslationResultMessage to translate:results."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        # Verify publish to translate:results stream
        streams_published = [str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list]
        assert any("translate:results" in s for s in streams_published)

    async def test_clearly_out_of_scope_background_speech_is_not_published_or_remembered(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"en"}

        async def get_value(key: str):
            if key.endswith(":meeting_context"):
                return b"Meeting topic: WarpTalk Docker deployment review."
            return None

        mock_redis_client._redis.get.side_effect = get_value
        worker.translator.translate_with_valence = AsyncMock(
            return_value=("[OUT_OF_MEETING_SCOPE]", None)
        )

        message = self._make_stt_msg(
            language="vi",
            text="Trận bóng tối qua thật hay.",
        )
        await worker.process(b"msg-background", message.to_redis())

        translated_stream_writes = [
            call
            for call in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(call.args[0])
        ]
        assert translated_stream_writes == []
        assert list(worker._recent_source_contexts.get("m1", ())) == []

    async def test_validated_final_uses_matching_speculative_translation_without_second_call(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._speculative_translations = {}
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"en"}
        mock_redis_client._redis.get.return_value = None
        worker.translator.translate_with_valence = AsyncMock(
            return_value=("Today we deploy Docker on Kubernetes.", None)
        )

        await worker._prefetch_from_event(
            {
                "meeting_id": "m1",
                "speaker_id": "s1",
                "text": "Hôm nay deploy Docker lên Kubernetes.",
                "language": "vi",
            }
        )
        worker.translator.translate_with_valence.assert_awaited_once()
        worker.translator.translate_with_valence.reset_mock()

        message = self._make_stt_msg(
            language="vi",
            text="Hôm nay deploy Docker lên Kubernetes.",
        )
        await worker.process(b"msg-final", message.to_redis())

        worker.translator.translate_with_valence.assert_not_awaited()
        published = [
            call.args[1]
            for call in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(call.args[0])
        ]
        assert published
        assert {item["translated_text"] for item in published} == {
            "Today we deploy Docker on Kubernetes."
        }

    def test_speculation_timeout_allows_observed_warm_provider_tail(self) -> None:
        # A real warm Realtime translation completed at 1.015s. Cancelling at 1.0s
        # discarded useful work and forced a second 2.3s translation on the final path.
        assert TranslationWorker._SPECULATIVE_TIMEOUT_SECONDS >= 1.25

    async def test_multi_sentence_uses_translate_for_first_and_batch_for_rest(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """A 2-sentence segment should call translate() once (sentence 0) and
        translate_batch() once (sentences 1..N-1) — not translate() in a per-sentence
        loop — and each sentence must still be published as its own chunk with the
        correct per-chunk text, since billing_worker charges credits per published
        translate:results message (keyed by chunk_segment_id + text length).
        """
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        worker.translator.translate_with_valence = AsyncMock(return_value=("Xin chào", None))
        worker.translator.translate_batch = AsyncMock(return_value=["Bạn khỏe không"])

        msg = self._make_stt_msg(language="en", text="Hello there. How are you?")
        await worker.process(b"msg-1", msg.to_redis())

        worker.translator.translate_with_valence.assert_awaited_once_with(
            "Hello there.",
            source_lang="en",
            target_lang="vi",
            glossary_terms=[],
            meeting_context=[],
        )
        worker.translator.translate_batch.assert_awaited_once_with(
            ["How are you?"],
            source_lang="en",
            target_lang="vi",
            glossary_terms=[],
            meeting_context=[],
        )

        published = [
            c.args
            for c in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(c.args[0])
        ]
        # BaseWorker.publish() dual-writes (per-room + flat global stream) per chunk,
        # so 2 sentences -> 4 xadd calls. Each chunk's own translated_text must appear
        # (not merged/duplicated across chunks) — this is what billing_worker's
        # per-chunk character-count charge depends on.
        assert len(published) == 4
        chunk_ids = {data["segment_id"] for _stream, data in published}
        # segment_id carries target_lang (f"{stt_segment_id}-{target_lang}-c{idx}") so
        # concurrent translations of the same STT segment into different listener
        # languages don't collide on the same chunk id — see the comment in
        # _translate_and_publish. billing_worker's _extract_underlying_segment_id()
        # only reads the first 36 chars (the GUID), so it is unaffected by this suffix.
        assert chunk_ids == {f"{msg.segment_id}-vi-c0", f"{msg.segment_id}-vi-c1"}
        texts_by_chunk = {
            data["segment_id"]: data["translated_text"] for _stream, data in published
        }
        assert texts_by_chunk[f"{msg.segment_id}-vi-c0"] == "Xin chào"
        assert texts_by_chunk[f"{msg.segment_id}-vi-c1"] == "Bạn khỏe không"

    async def test_preserves_source_chunk_index_in_published_results(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        msg = self._make_stt_msg(language="en", text="Hello world")
        msg.chunk_index = 37

        await worker.process(b"msg-1", msg.to_redis())

        published = [
            c.args[1]
            for c in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(c.args[0])
        ]
        assert published
        assert {int(data["chunk_index"]) for data in published} == {37}

    async def test_published_translation_carries_no_field_named_confidence(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """WT-278: the translator emits no quality score. What used to be published as
        `confidence` was the SOURCE segment's STT avg_logprob — a measurement of the audio —
        and TranscriptService persisted it into translation_contents.confidence as if it
        described the translation. It is now named for what it is, and nothing on a
        translation is called `confidence`."""
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        msg = self._make_stt_msg(language="en", text="Hello world")
        msg.confidence = -0.3421

        await worker.process(b"msg-1", msg.to_redis())

        published = [
            c.args[1]
            for c in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(c.args[0])
        ]
        assert published
        for data in published:
            assert "confidence" not in data
            assert float(data["source_stt_confidence"]) == pytest.approx(-0.3421)

    async def test_unknown_source_confidence_is_not_published_as_a_number(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """WT-277: stt_worker's -1.0 "this event exposed no token logprobs" sentinel is not a
        measurement. It must not travel onwards as data — the field is omitted so consumers
        store NULL instead of a fabricated score."""
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        msg = self._make_stt_msg(language="en", text="Hello world")
        msg.confidence = STT_UNKNOWN_CONFIDENCE

        await worker.process(b"msg-1", msg.to_redis())

        published = [
            c.args[1]
            for c in mock_redis_client._redis.xadd.call_args_list
            if "translate:results" in str(c.args[0])
        ]
        assert published
        for data in published:
            assert "confidence" not in data
            assert "source_stt_confidence" not in data

    async def test_skips_paused_room(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """process() should skip messages for paused rooms."""
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._paused_rooms = {"m1"}

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        worker.translator.translate.assert_not_called()
        mock_redis_client._redis.xadd.assert_not_called()

    async def test_process_forwards_workspace_glossary_to_translator(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """The glossary published by GlossaryStartedEventConsumer at
        `translationRoom:{meeting_id}:mt_glossary` must reach translate()/translate_batch()
        — this is the wiring that fixes over-translation of workspace terms.
        """
        import json

        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        glossary_value = json.dumps([{"source": "architect", "target": "architect"}]).encode()
        mock_redis_client._redis.get.side_effect = lambda key: (
            glossary_value if key.endswith(":mt_glossary") else None
        )

        msg = self._make_stt_msg(language="en", text="Hello there.")
        await worker.process(b"msg-1", msg.to_redis())

        worker.translator.translate_with_valence.assert_awaited_once_with(
            "Hello there.",
            source_lang="en",
            target_lang="vi",
            glossary_terms=[{"source": "architect", "target": "architect"}],
            meeting_context=[],
        )

    async def test_get_mt_glossary_caches_per_meeting(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        import json

        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = json.dumps(
            [{"source": "sprint", "target": "sprint"}]
        ).encode()

        first = await worker._get_mt_glossary("m1")
        mock_redis_client._redis.get.return_value = b"[]"
        second = await worker._get_mt_glossary("m1")

        assert first == [{"source": "sprint", "target": "sprint"}]
        assert second == first  # served from cache, not re-fetched
        mock_redis_client._redis.get.assert_called_once()

    async def test_get_mt_glossary_returns_empty_when_unset(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = None

        result = await worker._get_mt_glossary("m1")

        assert result == []

    async def test_get_mt_glossary_fails_open_on_malformed_json(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = b"not json"

        result = await worker._get_mt_glossary("m1")

        assert result == []

    async def test_get_meeting_context_caches_bounded_context(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.get.return_value = (
            b"Meeting topic: Sprint planning. Meeting context: Review WarpTalk."
        )

        first = await worker._get_meeting_context("m1")
        mock_redis_client._redis.get.return_value = b"changed"
        second = await worker._get_meeting_context("m1")

        assert first == ["Meeting topic: Sprint planning. Meeting context: Review WarpTalk."]
        assert second == first
        mock_redis_client._redis.get.assert_called_once_with("translationRoom:m1:meeting_context")

    async def test_process_includes_static_meeting_context_before_recent_utterances(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._recent_source_contexts = {"m1": ["Review the validator."]}
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        async def get_value(key: str):
            if key.endswith(":mt_glossary"):
                return None
            if key.endswith(":meeting_context"):
                return b"Meeting topic: WarpTalk code review."
            return None

        mock_redis_client._redis.get.side_effect = get_value

        message = self._make_stt_msg(language="en", text="Merge it.")
        message.meeting_id = "m1"
        await worker.process(b"msg-1", message.to_redis())

        assert worker.translator.translate_with_valence.await_args.kwargs["meeting_context"] == [
            "Review the validator.",
            "Meeting topic: WarpTalk code review.",
        ]

    def test_cleanup_room_clears_mt_glossary_cache(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._route_states = {}
        worker._translation_active = {}
        worker._room_routes = {}
        worker._mt_glossaries = {"m1": [{"source": "a", "target": "a"}]}
        worker._meeting_contexts = {"m1": ["Meeting topic: A."]}
        worker._recent_source_contexts = {"m1": ["previous"]}

        worker._cleanup_room("m1")

        assert "m1" not in worker._mt_glossaries
        assert "m1" not in worker._meeting_contexts
        assert "m1" not in worker._recent_source_contexts

    async def test_context_is_bounded_and_never_leaks_between_meetings(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = self._make_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        first = self._make_stt_msg(language="en", text="Review the validator.")
        first.meeting_id = "meeting-a"
        await worker.process(b"msg-a", first.to_redis())

        second = self._make_stt_msg(language="en", text="Approve the pull request.")
        second.meeting_id = "meeting-b"
        await worker.process(b"msg-b", second.to_redis())

        calls = worker.translator.translate_with_valence.await_args_list
        assert calls[0].kwargs["meeting_context"] == []
        assert calls[1].kwargs["meeting_context"] == []

        follow_up = self._make_stt_msg(language="en", text="Merge it.")
        follow_up.meeting_id = "meeting-a"
        await worker.process(b"msg-c", follow_up.to_redis())

        context = worker.translator.translate_with_valence.await_args_list[-1].kwargs[
            "meeting_context"
        ]
        assert context == ["Review the validator."]
        assert "Approve the pull request." not in context


class TestConsumeLoopConcurrency:
    """_consume_loop() must dispatch process() concurrently, not one-at-a-time.

    stt_worker's early-sentence pipelining means a single utterance's sentences now
    arrive as separate stt:results messages — if this worker's consume loop awaited
    process() for message 1 before even reading message 2, translation would be
    re-serialized right back to the sequential bottleneck the per-sentence
    asyncio.gather() fix (translation_worker.process()) was meant to remove.
    """

    async def test_dispatches_messages_concurrently(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        worker = TranslationWorker.__new__(TranslationWorker)
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker._shutdown_event = asyncio.Event()
        worker._consumer_name = "test-consumer"
        worker.input_stream = "stt:results"
        worker.consumer_group = "translate-workers"

        started: list[bytes] = []
        both_started = asyncio.Event()

        async def fake_process(message_id: bytes, data: dict) -> None:
            started.append(message_id)
            if len(started) == 2:
                both_started.set()
            # Message 1 can only reach here if message 2 has ALREADY started —
            # impossible unless both are running concurrently, not sequentially.
            await asyncio.wait_for(both_started.wait(), timeout=1.0)

        worker.process = fake_process

        async def fake_consume_concurrent(*, handler, **kwargs):
            await asyncio.gather(
                handler(b"msg-1", {}),
                handler(b"msg-2", {}),
            )
            worker._shutdown_event.set()

        worker.redis.consume_concurrent = fake_consume_concurrent

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)

        assert started == [b"msg-1", b"msg-2"]
