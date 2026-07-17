"""Translation backend — OpenAI gpt-4.1-mini.

Single provider, no fallback. Exposes async `translate()` and `translate_batch()`.
"""

from __future__ import annotations

import asyncio
import re

from shared.logger import get_logger

logger = get_logger(__name__)

# ISO 639-1 → human-readable name for system prompt clarity
_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
}

_SYSTEM_PROMPT = (
    "You are a professional real-time interpreter in a multilingual business meeting. "
    "Translate the user's message accurately and naturally. "
    "Preserve tone, technical terms, and speaker intent. "
    "Output ONLY the translation — no explanations, no notes, no alternatives."
)

_BATCH_SYSTEM_PROMPT = (
    "You are a professional real-time interpreter in a multilingual business meeting. "
    "You will receive several numbered sentences, one per line, in the form '[n] text'. "
    "Translate each sentence accurately and naturally, preserving tone, technical terms, "
    "and speaker intent. Reply with exactly one line per input sentence, in the same "
    "order, each formatted as '[n] translation' using the same number n as the input. "
    "Output ONLY those numbered lines — no explanations, no notes, no alternatives."
)

_BATCH_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _lang_name(iso_code: str) -> str:
    return _LANG_NAMES.get(iso_code.split("-")[0], iso_code)


class OpenAITranslator:
    """OpenAI gpt-4.1-mini translation backend.

    Uses asyncio-native openai client — no to_thread() needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    async def load(self) -> None:
        """Initialize OpenAI async client."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI translation")

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_translator_loaded", model=self.model)

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using OpenAI chat completion.

        Args:
            text: Input text to translate
            source_lang: Source language ISO 639-1 code (e.g. 'vi', 'en')
            target_lang: Target language ISO 639-1 code (e.g. 'en', 'ja')

        Returns:
            Translated text string
        """
        if not text.strip():
            return ""

        # Skip if same language
        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return text

        src_name = _lang_name(src)
        tgt_name = _lang_name(tgt)
        user_message = f"Translate from {src_name} to {tgt_name}:\n{text}"

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        result = response.choices[0].message.content or ""
        result = result.strip()

        logger.debug(
            "translation_complete",
            src=src_name,
            tgt=tgt_name,
            input_chars=len(text),
            output_chars=len(result),
        )
        return result

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate several sentences in a single OpenAI call.

        Cuts N sequential API round-trips down to 1, which is where most of the
        per-sentence latency in translation_worker.process() came from (each call is
        a real network round-trip, not just model inference time). Falls back to
        concurrent per-sentence translate() calls — never to a sequential loop — if the
        model's numbered-line response can't be parsed back into exactly len(texts)
        entries, so a billing_worker charge (computed per translated_text length) is
        never silently mismatched to the wrong sentence.

        Returns a list the same length and order as `texts`.
        """
        if not texts:
            return []

        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return list(texts)

        src_name = _lang_name(src)
        tgt_name = _lang_name(tgt)
        numbered_input = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        user_message = f"Translate from {src_name} to {tgt_name}:\n{numbered_input}"

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens * len(texts),
            temperature=self.temperature,
        )

        raw = (response.choices[0].message.content or "").strip()
        parsed: dict[int, str] = {}
        for line in raw.splitlines():
            m = _BATCH_LINE_RE.match(line)
            if not m:
                continue
            idx = int(m.group(1))
            if 1 <= idx <= len(texts):
                parsed[idx] = m.group(2).strip()

        if len(parsed) != len(texts):
            logger.warning(
                "batch_translation_parse_mismatch",
                expected=len(texts),
                parsed=len(parsed),
                src=src_name,
                tgt=tgt_name,
            )
            return list(
                await asyncio.gather(
                    *(self.translate(t, source_lang, target_lang) for t in texts)
                )
            )

        results = [parsed[i + 1] for i in range(len(texts))]
        logger.debug(
            "batch_translation_complete",
            src=src_name,
            tgt=tgt_name,
            count=len(texts),
            total_input_chars=sum(len(t) for t in texts),
            total_output_chars=sum(len(t) for t in results),
        )
        return results
