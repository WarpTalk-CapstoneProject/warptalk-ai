"""Tests for Translation Worker — mock translator, verify passthrough logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TranslationSettings, WorkerSettings
from shared.schemas import STTResultMessage

from translation_worker.translator import (
    LANG_CODE_MAP,
    NLLBTranslator,
    TranslatorWithFallback,
    to_flores_code,
)
from translation_worker.worker import TranslationWorker


class TestLangCodeMapping:
    """Language code conversion tests."""

    @pytest.mark.parametrize(
        "iso,flores",
        [
            ("en", "eng_Latn"),
            ("vi", "vie_Latn"),
            ("zh", "zho_Hans"),
            ("ja", "jpn_Jpan"),
        ],
    )
    def test_known_codes(self, iso: str, flores: str) -> None:
        assert to_flores_code(iso) == flores

    def test_unknown_code_passes_through(self) -> None:
        assert to_flores_code("xxx_Yyyy") == "xxx_Yyyy"


class TestTranslatorWithFallback:
    """TranslatorWithFallback tests."""

    async def test_uses_primary_on_success(self) -> None:
        primary = MagicMock()
        primary.translate = AsyncMock(return_value="translated")
        primary.load = AsyncMock()

        translator = TranslatorWithFallback(primary, fallback=None)
        result = await translator.translate("hello", "en", "vi")

        assert result == "translated"
        primary.translate.assert_called_once()

    async def test_falls_back_on_primary_error(self) -> None:
        primary = MagicMock()
        primary.translate = AsyncMock(side_effect=RuntimeError("GPU OOM"))
        primary.load = AsyncMock()

        fallback = MagicMock()
        fallback.translate = AsyncMock(return_value="fallback_result")
        fallback.load = AsyncMock()

        translator = TranslatorWithFallback(primary, fallback)
        result = await translator.translate("hello", "en", "vi")

        assert result == "fallback_result"
        fallback.translate.assert_called_once()


class TestTranslationWorker:
    """Translation Worker process() tests."""

    async def test_passthrough_same_language(
        self, mock_redis_client, worker_settings: WorkerSettings
    ) -> None:
        """Should forward text unchanged if source == target language."""
        worker = TranslationWorker.__new__(TranslationWorker)
        worker.settings = worker_settings
        worker.redis = mock_redis_client
        worker.logger = MagicMock()
        worker.translation_settings = TranslationSettings()

        mock_translator = MagicMock()
        mock_translator.translate = AsyncMock()
        worker.translator = mock_translator

        # Mock target language = same as source
        mock_redis_client._redis.hget = AsyncMock(return_value=b"en")

        stt_result = STTResultMessage(
            meeting_id="m1",
            speaker_id="s1",
            text="Hello world",
            language="en",
            confidence=0.95,
        )

        await worker.process(b"msg-1", stt_result.to_redis())

        # Translator should NOT be called
        mock_translator.translate.assert_not_called()

        # But publish should still be called (passthrough)
        mock_redis_client._redis.xadd.assert_called_once()
