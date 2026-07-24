"""Tests for Translation Worker — mock translator, verify passthrough and routing logic."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from shared.config import TranslationSettings, WorkerSettings
from shared.schemas import STTResultMessage
from translation_worker.translator import OpenAITranslator, _lang_name
from translation_worker.worker import TranslationWorker


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


class TestOpenAITranslator:
    """OpenAITranslator unit tests."""

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
        translator.translate = AsyncMock(side_effect=["Xin chào", "Thế giới"])

        result = await translator.translate_batch(["Hello", "World"], "en", "vi")

        assert result == ["Xin chào", "Thế giới"]
        assert translator.translate.await_count == 2


class TestTranslationWorker:
    """TranslationWorker process() tests."""

    def _make_worker(self, mock_redis_client, worker_settings):
        worker = TranslationWorker.__new__(TranslationWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.translation_settings = TranslationSettings()
        worker._paused_rooms = set()
        worker.worker_name = "translation"
        mock_translator = MagicMock()
        mock_translator.model = "gpt-4.1-mini"
        mock_translator.translate = AsyncMock(return_value="Xin chào")
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

    async def test_passthrough_same_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should forward text unchanged if source == target language."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # Target lang = source lang (both "en") — some other participant listens in "en"
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"en"}

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        # Translator should NOT be called (passthrough)
        worker.translator.translate.assert_not_called()
        # Result should still be published (BaseWorker.publish calls xadd twice)
        mock_redis_client._redis.xadd.assert_called()

    async def test_calls_translator_for_different_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should call translator when source != target language."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # Target lang = "vi", source lang = "en"
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        worker.translator.translate.assert_called()

    async def test_default_fallback_language_is_en(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """When no per-speaker language set, fallback should be 'en' not 'vi'."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # No participants registered in the languages hash at all
        mock_redis_client._redis.hgetall.return_value = {}

        await worker.process(b"msg-1", self._make_stt_msg(language="vi").to_redis())

        # Translator called with target_lang="en"
        call_kwargs = worker.translator.translate.call_args
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
        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("translate:results" in s for s in streams_published)

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
        worker.translator.translate = AsyncMock(return_value="Xin chào")
        worker.translator.translate_batch = AsyncMock(return_value=["Bạn khỏe không"])

        msg = self._make_stt_msg(language="en", text="Hello there. How are you?")
        await worker.process(b"msg-1", msg.to_redis())

        worker.translator.translate.assert_awaited_once_with(
            "Hello there.", source_lang="en", target_lang="vi"
        )
        worker.translator.translate_batch.assert_awaited_once_with(
            ["How are you?"], source_lang="en", target_lang="vi"
        )

        published = [
            c.args for c in mock_redis_client._redis.xadd.call_args_list
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
        texts_by_chunk = {data["segment_id"]: data["translated_text"] for _stream, data in published}
        assert texts_by_chunk[f"{msg.segment_id}-vi-c0"] == "Xin chào"
        assert texts_by_chunk[f"{msg.segment_id}-vi-c1"] == "Bạn khỏe không"

    async def test_skips_paused_room(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """process() should skip messages for paused rooms."""
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._paused_rooms = {"m1"}

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        worker.translator.translate.assert_not_called()
        mock_redis_client._redis.xadd.assert_not_called()


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

        async def fake_consume(**kwargs):
            yield b"msg-1", {}
            yield b"msg-2", {}
            worker._shutdown_event.set()

        worker.redis.consume = fake_consume

        await asyncio.wait_for(worker._consume_loop(), timeout=2.0)
        await asyncio.sleep(0.05)  # let the dispatched create_task()s finish

        assert started == [b"msg-1", b"msg-2"]
