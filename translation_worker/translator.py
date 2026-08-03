"""Translation backend — OpenAI gpt-4.1-mini.

Single provider, no fallback. Exposes async `translate()` and `translate_batch()`.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from shared.config import TranslationSettings
from shared.logger import get_logger
from shared.openai_usage import TokenUsage

logger = get_logger(__name__)
OUT_OF_MEETING_SCOPE = "[OUT_OF_MEETING_SCOPE]"

# Constructor defaults below mirror TranslationSettings — the values production code
# actually runs with (translation_worker/worker.py always passes them explicitly).
# Sourcing the defaults from here instead of a second hardcoded literal keeps direct/test
# instantiation in sync with config.py without anyone having to remember both places.
_DEFAULTS = TranslationSettings()

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

_CONTEXT_RELEVANCE_INSTRUCTION = (
    " Before translating, decide whether the current utterance has any plausible "
    "connection to the supplied meeting topic or accepted meeting utterances. If it is "
    "clearly unrelated background conversation, entertainment, or household speech, "
    f"output exactly {OUT_OF_MEETING_SCOPE} instead of a translation. If it is a short "
    "acknowledgement, question, correction, name, technical term, reasonable tangent, "
    "or is ambiguous, translate it normally. Suppress only when clearly unrelated."
)


def _build_system_prompt(base_prompt: str, glossary_terms: list[dict[str, str]] | None) -> str:
    glossary_block = _build_glossary_block(glossary_terms)
    return f"{base_prompt}{glossary_block}" if glossary_block else base_prompt


def _build_glossary_block(glossary_terms: list[dict[str, str]] | None) -> str:
    """Render this workspace's active glossary terms (see GlossaryStartedEventConsumer,
    published to `translationRoom:{meeting_id}:mt_glossary`) as a system-prompt addendum.

    Two shapes come out of the same glossary data: a term whose SourceTerm equals its
    TargetTerm (case-insensitively) is an admin's explicit "don't translate this" —
    slang, a brand name, or a term the team just says in English (e.g. "marketing plan",
    "sprint"). A term where they differ is an exact required mapping, overriding whatever
    generic translation the model would otherwise pick. Without this, the base prompt's
    blanket "translate everything" instruction has no way to know either exists — see
    docs/code-switching-research.md §1.2/§2.3 for why that over-translates jargon by default.
    """
    if not glossary_terms:
        return ""

    keep_lines: list[str] = []
    map_lines: list[str] = []
    for term in glossary_terms:
        source = (term.get("source") or "").strip()
        target = (term.get("target") or "").strip()
        if not source:
            continue
        if not target or target.lower() == source.lower():
            keep_lines.append(f'- "{source}"')
        else:
            map_lines.append(f'- "{source}" → "{target}"')

    if not keep_lines and not map_lines:
        return ""

    sections = [
        "\n\nThis workspace has a glossary. Apply it exactly, overriding your own "
        "default translation choice whenever one of these terms appears:"
    ]
    if keep_lines:
        sections.append(
            "Keep these terms exactly as written in the source — do not translate them:\n"
            + "\n".join(keep_lines)
        )
    if map_lines:
        sections.append(
            "Use these exact translations instead of a generic one:\n" + "\n".join(map_lines)
        )
    return "\n\n".join(sections)


def _exception_clause(glossary_terms: list[dict[str, str]] | None) -> str:
    """Return the proper-noun exception clause, mentioning glossary only when present."""
    base = "except for proper nouns/brand names with no natural translation"
    return f"{base}, or terms covered by the workspace glossary" if glossary_terms else base


def _select_relevant_glossary_terms(
    text: str,
    glossary_terms: list[dict[str, str]] | None,
    *,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Keep only mappings whose source term is present in the current utterance.

    Sending the room's entire glossary on every realtime request adds tokens and can
    steer short ambiguous speech toward unrelated terms. Word boundaries prevent a
    short term such as ``UI`` from matching the middle of ``build``; spaces and
    hyphens are treated equivalently for code-switched technical phrases.
    """
    if not glossary_terms or not text.strip():
        return []

    haystack = text.casefold()
    selected: list[dict[str, str]] = []
    for term in glossary_terms:
        source = (term.get("source") or "").strip()
        if not source:
            continue
        pieces = [piece for piece in re.split(r"[\s_-]+", source.casefold()) if piece]
        if not pieces:
            continue
        pattern = r"(?<!\w)" + r"[\s_-]+".join(map(re.escape, pieces)) + r"(?!\w)"
        if re.search(pattern, haystack):
            selected.append(term)
            if len(selected) >= limit:
                break
    return selected


_BATCH_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _lang_name(iso_code: str) -> str:
    return _LANG_NAMES.get(iso_code.split("-")[0], iso_code)


@dataclass(frozen=True)
class TranslationWithUsage:
    text: str
    usage: TokenUsage = TokenUsage()


@dataclass(frozen=True)
class TranslationBatchWithUsage:
    texts: list[str]
    usage: TokenUsage = TokenUsage()


class OpenAITranslator:
    """OpenAI low-latency translation backend.

    Uses asyncio-native openai client — no to_thread() needed.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULTS.model,
        realtime_model: str = _DEFAULTS.realtime_model,
        realtime_pool_size: int = _DEFAULTS.realtime_pool_size,
        realtime_timeout_seconds: float = _DEFAULTS.realtime_timeout_seconds,
        realtime_max_output_tokens: int = _DEFAULTS.realtime_max_output_tokens,
        max_tokens: int = _DEFAULTS.max_tokens,
        # 0.0, not 0.1: measured via real pipeline benchmark that temperature=0.1 let
        # identical repeated input sentences translate to different (equally valid)
        # phrasings across separate calls (e.g. "15%" vs "mười lăm phần trăm"), which
        # breaks tts_worker's text-based synthesis cache — a real repeated meeting
        # phrase misses the cache and pays a full ~1s Cartesia call instead of a ~2ms
        # cache hit. Determinism isn't perfectly guaranteed even at 0.0 (OpenAI's own
        # infra has some residual nondeterminism), but it meaningfully raises the hit rate.
        temperature: float = _DEFAULTS.temperature,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.realtime_model = realtime_model
        self.realtime_pool_size = max(1, realtime_pool_size)
        self.realtime_timeout_seconds = realtime_timeout_seconds
        self.realtime_max_output_tokens = realtime_max_output_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: AsyncOpenAI | None = None
        self._realtime_connections: list[Any | None] = [None] * self.realtime_pool_size
        self._realtime_available: asyncio.Queue[int] = asyncio.Queue()
        for index in range(self.realtime_pool_size):
            self._realtime_available.put_nowait(index)
        self._realtime_connect_lock = asyncio.Lock()

    async def load(self) -> None:
        """Initialize OpenAI async client."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI translation")

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info(
            "openai_translator_loaded",
            model=self.model,
            realtime_model=self.realtime_model or None,
        )

    async def warm_up(self) -> None:
        """Prime DNS/TLS/model routing before the first participant utterance.

        A cold first request measured above three seconds while subsequent calls were
        below one second. Spending one tiny request during worker startup keeps that
        one-time setup cost out of the meeting path.
        """
        try:
            await self._create_with_retry(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with exactly OK.",
                    },
                    {"role": "user", "content": "OK"},
                ],
                **self._completion_options(8),
            )
            logger.info("openai_translator_warmed", model=self.model)
        except Exception as exc:
            # A transient provider problem during startup must not stop the worker from
            # consuming once the provider recovers.
            logger.warning("openai_translator_warmup_failed", model=self.model, error=str(exc))

        if not getattr(self, "realtime_model", ""):
            return
        try:
            await self._ensure_realtime_pool()
            # Prime inference on every slot so the first participant utterance does not
            # pay a WebSocket handshake or cold model-routing penalty.
            await asyncio.gather(
                *(
                    self._translate_realtime(
                        "Reply with exactly OK.",
                        "Reply with exactly OK.",
                    )
                    for _ in range(self.realtime_pool_size)
                )
            )
            logger.info(
                "openai_realtime_translator_warmed",
                model=self.realtime_model,
                pool_size=self.realtime_pool_size,
            )
        except Exception as exc:
            # Chat Completions remains available; a later request will reconnect the
            # affected slot on demand.
            logger.warning(
                "openai_realtime_translator_warmup_failed",
                model=self.realtime_model,
                error=str(exc),
            )

    async def close(self) -> None:
        """Close persistent Realtime sockets and the HTTP client."""
        connections = list(getattr(self, "_realtime_connections", []))
        self._realtime_connections = [None] * len(connections)
        await asyncio.gather(
            *(connection.close() for connection in connections if connection is not None),
            return_exceptions=True,
        )
        client = getattr(self, "_client", None)
        if client is not None:
            await client.close()

    async def _connect_realtime(self) -> Any:
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI translator is not loaded")
        return await client.realtime.connect(model=self.realtime_model).enter()

    async def _ensure_realtime_pool(self) -> None:
        async with self._realtime_connect_lock:
            missing = [
                index
                for index, connection in enumerate(self._realtime_connections)
                if connection is None
            ]
            if not missing:
                return
            opened = await asyncio.gather(
                *(self._connect_realtime() for _ in missing),
                return_exceptions=True,
            )
            first_error: Exception | None = None
            for index, result in zip(missing, opened, strict=True):
                if isinstance(result, Exception):
                    first_error = first_error or result
                else:
                    self._realtime_connections[index] = result
            if first_error is not None and all(
                connection is None for connection in self._realtime_connections
            ):
                raise first_error

    async def _acquire_realtime_connection(self) -> tuple[int, Any]:
        await self._ensure_realtime_pool()
        index = await self._realtime_available.get()
        connection = self._realtime_connections[index]
        if connection is None:
            try:
                connection = await self._connect_realtime()
                self._realtime_connections[index] = connection
            except Exception:
                self._realtime_available.put_nowait(index)
                raise
        return index, connection

    async def _release_realtime_connection(self, index: int, *, healthy: bool) -> None:
        if not healthy:
            connection = self._realtime_connections[index]
            self._realtime_connections[index] = None
            if connection is not None:
                try:
                    await connection.close()
                except Exception:
                    logger.debug("openai_realtime_close_failed", exc_info=True)
        self._realtime_available.put_nowait(index)

    async def _translate_realtime(
        self,
        user_message: str,
        instructions: str,
        *,
        request_id: str | None = None,
    ) -> str:
        result = await self._translate_realtime_with_usage(
            user_message,
            instructions,
            request_id=request_id,
        )
        return result.text

    async def _translate_realtime_with_usage(
        self,
        user_message: str,
        instructions: str,
        *,
        request_id: str | None = None,
    ) -> TranslationWithUsage:
        """Run one isolated text response on an exclusive hot Realtime connection."""
        request_id = request_id or uuid.uuid4().hex
        index, connection = await self._acquire_realtime_connection()
        healthy = False
        response_id: str | None = None
        chunks: list[str] = []
        try:
            async with asyncio.timeout(self.realtime_timeout_seconds):
                await connection.response.create(
                    response={
                        "conversation": "none",
                        "input": [
                            {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": user_message}],
                            }
                        ],
                        "instructions": instructions,
                        "max_output_tokens": self.realtime_max_output_tokens,
                        "metadata": {"request_id": request_id},
                        "output_modalities": ["text"],
                        "reasoning": {"effort": "minimal"},
                    }
                )
                while True:
                    event = await connection.recv()
                    if event.type == "error":
                        raise RuntimeError(str(event.error))
                    if event.type == "response.created":
                        metadata = event.response.metadata or {}
                        if metadata.get("request_id") == request_id:
                            response_id = event.response.id
                    elif (
                        event.type == "response.output_text.delta"
                        and response_id
                        and event.response_id == response_id
                    ):
                        chunks.append(event.delta)
                    elif (
                        event.type == "response.done"
                        and response_id
                        and event.response.id == response_id
                    ):
                        if event.response.status != "completed":
                            raise RuntimeError(
                                f"Realtime translation ended with {event.response.status}"
                            )
                        result = "".join(chunks).strip()
                        if not result:
                            raise RuntimeError("Realtime translation returned empty text")
                        usage = TokenUsage.from_openai_usage(getattr(event.response, "usage", None))
                        healthy = True
                        return TranslationWithUsage(result, usage)
        finally:
            await self._release_realtime_connection(index, healthy=healthy)

    async def _create_with_retry(self, **kwargs: Any) -> Any:
        """Call OpenAI chat completions with transient error retries (up to 2 retries)."""
        retries = 2
        delay = 0.5
        for attempt in range(retries + 1):
            try:
                client = self._client
                if client is None:
                    raise RuntimeError("OpenAI translator is not loaded")
                return await client.chat.completions.create(**kwargs)
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        "openai_translation_transient_error",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    raise

    def _completion_options(self, token_limit: int) -> dict[str, Any]:
        """Return model-compatible generation controls.

        GPT-5 models reject the legacy ``max_tokens`` parameter and only support
        their default temperature. Keeping this in one helper prevents single and
        batch translation paths from drifting apart.
        """
        if self.model.startswith("gpt-5"):
            return {"max_completion_tokens": token_limit}
        return {
            "max_tokens": token_limit,
            "temperature": self.temperature,
        }

    async def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict[str, str]] | None = None,
        meeting_context: list[str] | None = None,
    ) -> str:
        """Translate text using OpenAI chat completion.

        Args:
            text: Input text to translate
            source_lang: Source language ISO 639-1 code (e.g. 'vi', 'en')
            target_lang: Target language ISO 639-1 code (e.g. 'en', 'ja')
            glossary_terms: This meeting's workspace glossary, as [{"source": ..., "target":
                ...}, ...] — see _build_glossary_block. None/empty when the workspace has no
                active glossary; translation falls back to the plain proper-nouns exception.

        Returns:
            Translated text string
        """
        result = await self.translate_with_usage(
            text,
            source_lang,
            target_lang,
            glossary_terms,
            meeting_context,
        )
        return result.text

    async def translate_with_usage(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None = None,
        meeting_context: list[str] | None = None,
    ) -> TranslationWithUsage:
        if not text.strip():
            return TranslationWithUsage("")

        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return TranslationWithUsage(text)

        src_name = _lang_name(src)
        tgt_name = _lang_name(tgt)
        relevant_glossary = _select_relevant_glossary_terms(text, glossary_terms)
        context_block = ""
        if meeting_context:
            bounded_context = meeting_context[-4:]
            context_block = (
                "\n\nEarlier accepted utterances from this same meeting are provided only "
                "to resolve pronouns, terminology, and topic. Never translate, repeat, or "
                "invent content from them:\n"
                + "\n".join(f"- {line}" for line in bounded_context)
                + "\n\nOnly when the current utterance is clearly background speech with no "
                "plausible connection to this meeting context, respond exactly "
                f"{OUT_OF_MEETING_SCOPE}. Never suppress short acknowledgements, questions, "
                "corrections, names, technical terms, reasonable tangents, or ambiguous "
                "utterances. If uncertain, translate normally."
            )
        user_message = (
            f"Translate from {src_name} to {tgt_name}.\n"
            f"Current utterance (translate only this):\n{text}\n\n"
            f"Respond entirely in {tgt_name} — never leave any word in {src_name} or "
            f"switch to a third language, {_exception_clause(relevant_glossary)}."
            f"{_build_glossary_block(relevant_glossary)}"
            f"{context_block}"
        )

        result = ""
        usage = TokenUsage()
        system_prompt = _build_system_prompt(_SYSTEM_PROMPT, relevant_glossary)
        if meeting_context:
            system_prompt += _CONTEXT_RELEVANCE_INSTRUCTION
        if getattr(self, "realtime_model", ""):
            try:
                realtime_result = await self._translate_realtime_with_usage(
                    user_message,
                    system_prompt,
                )
                result = realtime_result.text
                usage = realtime_result.usage
            except Exception as exc:
                logger.warning(
                    "openai_realtime_translation_failed_using_chat_fallback",
                    model=self.realtime_model,
                    error=repr(exc),
                )

        if not result:
            response = await self._create_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                **self._completion_options(self.max_tokens),
            )
            result = (response.choices[0].message.content or "").strip()
            usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))

        logger.debug(
            "translation_complete",
            src=src_name,
            tgt=tgt_name,
            input_chars=len(text),
            output_chars=len(result),
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return TranslationWithUsage(result, usage)

    async def translate_batch(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict[str, str]] | None = None,
        meeting_context: list[str] | None = None,
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
        result = await self.translate_batch_with_usage(
            texts,
            source_lang,
            target_lang,
            glossary_terms,
            meeting_context,
            fallback_uses_legacy_translate=True,
        )
        return result.texts

    async def translate_batch_with_usage(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None = None,
        meeting_context: list[str] | None = None,
        *,
        fallback_uses_legacy_translate: bool = False,
    ) -> TranslationBatchWithUsage:
        if not texts:
            return TranslationBatchWithUsage([])

        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return TranslationBatchWithUsage(list(texts))

        src_name = _lang_name(src)
        tgt_name = _lang_name(tgt)
        numbered_input = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
        relevant_glossary = _select_relevant_glossary_terms(
            "\n".join(texts),
            glossary_terms,
        )
        context_block = ""
        if meeting_context:
            context_block = (
                "\n\nEarlier accepted utterances from this same meeting are context only; "
                "do not translate or repeat them:\n"
                + "\n".join(f"- {line}" for line in meeting_context[-4:])
                + "\n\nFor each numbered input independently, only when it is clearly "
                "background speech with no plausible connection to this meeting context, "
                f"translate it as exactly {OUT_OF_MEETING_SCOPE}. Never suppress short "
                "acknowledgements, questions, corrections, names, technical terms, "
                "reasonable tangents, or ambiguous utterances. If uncertain, translate."
            )
        user_message = (
            f"Translate from {src_name} to {tgt_name}:\n{numbered_input}\n\n"
            f"Respond entirely in {tgt_name} — never leave any word in {src_name} or "
            f"switch to a third language, {_exception_clause(relevant_glossary)}."
            f"{_build_glossary_block(relevant_glossary)}"
            f"{context_block}"
        )

        system_prompt = _build_system_prompt(_BATCH_SYSTEM_PROMPT, relevant_glossary)
        if meeting_context:
            system_prompt += _CONTEXT_RELEVANCE_INSTRUCTION
        response = await self._create_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            **self._completion_options(self.max_tokens * len(texts)),
        )

        raw = (response.choices[0].message.content or "").strip()
        usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
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
            if fallback_uses_legacy_translate:
                fallback_texts = list(
                    await asyncio.gather(
                        *(
                            self.translate(
                                t,
                                source_lang,
                                target_lang,
                                glossary_terms,
                                meeting_context,
                            )
                            for t in texts
                        )
                    )
                )
                return TranslationBatchWithUsage(fallback_texts, usage)

            fallback_results = await asyncio.gather(
                *(
                    self.translate_with_usage(
                        t,
                        source_lang,
                        target_lang,
                        glossary_terms,
                        meeting_context,
                    )
                    for t in texts
                )
            )
            fallback_usage = usage
            for item in fallback_results:
                fallback_usage += item.usage
            return TranslationBatchWithUsage(
                [item.text for item in fallback_results],
                fallback_usage,
            )

        results = [parsed[i + 1] for i in range(len(texts))]
        logger.debug(
            "batch_translation_complete",
            src=src_name,
            tgt=tgt_name,
            count=len(texts),
            total_input_chars=sum(len(t) for t in texts),
            total_output_chars=sum(len(t) for t in results),
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
        )
        return TranslationBatchWithUsage(results, usage)
