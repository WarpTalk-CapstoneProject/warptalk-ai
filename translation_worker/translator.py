"""Translation backend — OpenAI gpt-4.1-mini.

Single provider, no fallback. Exposes async `translate()`.
"""

from __future__ import annotations

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
