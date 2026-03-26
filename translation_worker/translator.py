"""Translation backends — NLLB (local GPU) and Google Translate (API fallback).

All translators expose an async `translate()` method.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from shared.logger import get_logger

logger = get_logger(__name__)


class Translator(ABC):
    """Abstract translation backend."""

    @abstractmethod
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages.

        Args:
            text: Input text
            source_lang: Source language code (e.g. 'eng_Latn')
            target_lang: Target language code (e.g. 'vie_Latn')

        Returns:
            Translated text
        """

    @abstractmethod
    async def load(self) -> None:
        """Load model or initialize API client."""


# ---------------------------------------------------------------------------
# NLLB-200 Distilled (local GPU, asyncio.to_thread for blocking inference)
# ---------------------------------------------------------------------------

# NLLB uses Flores-200 language codes: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
# Common mappings from ISO 639-1 to Flores-200:
LANG_CODE_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "msa_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "pt": "por_Latn",
    "it": "ita_Latn",
}


def to_flores_code(iso_code: str) -> str:
    """Convert ISO 639-1 code to Flores-200 code."""
    return LANG_CODE_MAP.get(iso_code, iso_code)


class NLLBTranslator(Translator):
    """Facebook NLLB-200 Distilled translation model (local GPU).

    Uses asyncio.to_thread to run blocking inference without
    blocking the event loop.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: str = "cuda",
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    async def load(self) -> None:
        """Load NLLB model and tokenizer in a thread."""
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("loading_nllb_model", model=self.model_name, device=self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        logger.info("nllb_model_loaded")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""
        return await asyncio.to_thread(
            self._translate_sync, text, source_lang, target_lang
        )

    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> str:
        src_code = to_flores_code(source_lang)
        tgt_code = to_flores_code(target_lang)

        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self.device)

        tgt_lang_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        generated = self._model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=self.max_length,
        )

        return self._tokenizer.decode(generated[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Google Translate API fallback
# ---------------------------------------------------------------------------


class GoogleTranslator(Translator):
    """Google Translate API fallback (no GPU required)."""

    def __init__(self) -> None:
        self._translator = None

    async def load(self) -> None:
        """Initialize googletrans client."""
        from googletrans import Translator as GTranslator

        self._translator = GTranslator()
        logger.info("google_translator_loaded")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""

        result = await asyncio.to_thread(
            self._translator.translate,
            text,
            src=source_lang,
            dest=target_lang,
        )
        return result.text


# ---------------------------------------------------------------------------
# Factory with automatic fallback
# ---------------------------------------------------------------------------


class TranslatorWithFallback:
    """Primary translator with automatic fallback on error."""

    def __init__(self, primary: Translator, fallback: Translator | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    async def load(self) -> None:
        await self.primary.load()
        if self.fallback:
            await self.fallback.load()

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        try:
            return await self.primary.translate(text, source_lang, target_lang)
        except Exception:
            if self.fallback:
                logger.warning("primary_translator_failed_using_fallback")
                return await self.fallback.translate(text, source_lang, target_lang)
            raise
