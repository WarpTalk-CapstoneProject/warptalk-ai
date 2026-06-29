"""Tests for Translation Worker — mock translator, verify passthrough and routing logic."""

from __future__ import annotations

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
        mock_translator.translate = AsyncMock(return_value="Xin chào")
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

        # Target lang = source lang (both "en")
        mock_redis_client._redis.hget.return_value = b"en"

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
        mock_redis_client._redis.hget.return_value = b"vi"

        await worker.process(b"msg-1", self._make_stt_msg(language="en").to_redis())

        worker.translator.translate.assert_called()

    async def test_default_fallback_language_is_en(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """When no per-speaker language set, fallback should be 'en' not 'vi'."""
        worker = self._make_worker(mock_redis_client, worker_settings)

        # No language configured for this speaker
        mock_redis_client._redis.hget.return_value = None

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

        mock_redis_client._redis.hget.return_value = b"vi"

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        # Verify publish to translate:results stream
        streams_published = [
            str(c.args[0]) for c in mock_redis_client._redis.xadd.call_args_list
        ]
        assert any("translate:results" in s for s in streams_published)

    async def test_skips_paused_room(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """process() should skip messages for paused rooms."""
        worker = self._make_worker(mock_redis_client, worker_settings)
        worker._paused_rooms = {"m1"}

        await worker.process(b"msg-1", self._make_stt_msg().to_redis())

        worker.translator.translate.assert_not_called()
        mock_redis_client._redis.xadd.assert_not_called()
