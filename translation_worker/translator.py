"""Translation backend — OpenAI gpt-4.1-mini.

Single provider, no fallback. Exposes async `translate()` and `translate_batch()`.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
import uuid
from difflib import SequenceMatcher
from typing import Any

from openai import AsyncOpenAI

from shared.config import TranslationSettings
from shared.logger import get_logger
from shared.openai_options import completion_options

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

# The input is speech recognition output, and saying so is most of the work.
#
# Without this paragraph the translator treats what it receives as authoritative text and
# faithfully translates a mishearing into a confident wrong sentence: "Codex" heard as
# "cô đích" arrives as those two Vietnamese words and leaves as their literal English.
#
# The permission is deliberately narrow. A model told simply to "fix ASR errors" rewrites
# sentences that were already correct, and a fluent invention is WORSE than a visible
# mistranscription — a reader spots "cô đích" as nonsense instantly, but cannot tell a
# confidently repaired sentence from a real one. So repair is allowed only where the
# glossary or the meeting context supplies the evidence, and literal translation is the
# stated default everywhere else.
_ASR_REPAIR_INSTRUCTION = (
    " Your input is automatic speech recognition output, so it may contain words that were "
    "misheard — most often a name or technical term rendered as similar-sounding syllables "
    "in the speaker's own language. When the glossary or the meeting context makes it clear "
    "what was really said, translate what the speaker MEANT. When you have no such evidence, "
    "translate exactly what is written and invent nothing. Never add, drop, or 'complete' "
    "content that is not there."
)

_SYSTEM_PROMPT = (
    "You are a professional real-time interpreter in a multilingual business meeting. "
    "Translate the user's message accurately and naturally. "
    "Preserve tone, technical terms, and speaker intent. "
    "Output ONLY the translation — no explanations, no notes, no alternatives."
    + _ASR_REPAIR_INSTRUCTION
)

_BATCH_SYSTEM_PROMPT = (
    "You are a professional real-time interpreter in a multilingual business meeting. "
    "You will receive several numbered sentences, one per line, in the form '[n] text'. "
    "Translate each sentence accurately and naturally, preserving tone, technical terms, "
    "and speaker intent. Reply with exactly one line per input sentence, in the same "
    "order, each formatted as '[n] translation' using the same number n as the input. "
    "Output ONLY those numbered lines — no explanations, no notes, no alternatives."
    + _ASR_REPAIR_INSTRUCTION
)

_CONTEXT_RELEVANCE_INSTRUCTION = (
    " Before translating, decide whether the current utterance has any plausible "
    "connection to the supplied meeting topic or accepted meeting utterances. If it is "
    "clearly unrelated background conversation, entertainment, or household speech, "
    f"output exactly {OUT_OF_MEETING_SCOPE} instead of a translation. If it is a short "
    "acknowledgement, question, correction, name, technical term, reasonable tangent, "
    "or is ambiguous, translate it normally. Suppress only when clearly unrelated."
)


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
    misheard_lines: list[str] = []
    for term in glossary_terms:
        source = (term.get("source") or "").strip()
        target = (term.get("target") or "").strip()
        if not source:
            continue
        # A suspicion, not a fact — see _select_relevant_glossary_terms. It is listed
        # separately so the model can weigh it rather than apply it blindly.
        if term.get("match") == "possible":
            rendered = (
                f'"{source}"'
                if not target or target.lower() == source.lower()
                else (f'"{source}" (translate as "{target}")')
            )
            misheard_lines.append(f"- {rendered}")
        elif not target or target.lower() == source.lower():
            keep_lines.append(f'- "{source}"')
        else:
            map_lines.append(f'- "{source}" → "{target}"')

    if not keep_lines and not map_lines and not misheard_lines:
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
    if misheard_lines:
        sections.append(
            "None of these next terms appear literally, but part of the utterance SOUNDS "
            "like one of them, which usually means speech recognition mangled it. Read it "
            "as the term ONLY if that makes the sentence coherent — a meeting about "
            "deployment really can contain the word it sounds like. If reading it that way "
            "would not make sense here, translate what was actually said and change "
            "nothing:\n" + "\n".join(misheard_lines)
        )
    return "\n\n".join(sections)


def _exception_clause(glossary_terms: list[dict[str, str]] | None) -> str:
    """The "never leave any word..." instruction's exception clause — only mentions the
    glossary when there actually is one, so the sentence doesn't dangle a reference to
    "the glossary below" when _build_glossary_block returned nothing.
    """
    base = "except for proper nouns/brand names with no natural translation"
    return f"{base}, or terms covered by the glossary below" if glossary_terms else base


def _match_skeleton(value: str) -> str:
    """Fold a string to what it roughly SOUNDS like, for comparing across a mishearing.

    Diacritics and separators are dropped and everything is lowercased, so a Vietnamese
    rendering of an English word lands near the word itself: "cô đích" and "Codex" become
    "codich" and "codex". Vietnamese đ has no canonical decomposition, so it is mapped by
    hand — without that, every đ survives and the two skeletons never converge.
    """
    decomposed = unicodedata.normalize("NFD", value.casefold())
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[\s_\-.]+", "", stripped.replace("đ", "d"))


# Both numbers come from measuring real Vietnamese sentences, not from taste.
#
# Best-window similarity, glossary term against unrelated meeting speech:
#
#     term         len   mishearing   unrelated speech
#     WarpTalk       8       1.000          0.353
#     staging        7       0.923          0.533
#     Kubernetes    10       0.667          0.333
#     Codex          5       0.667          0.667   <-- indistinguishable
#
# 0.65 sits in the gap for every term of seven characters or more. Below that length the
# gap closes completely: a five-character skeleton scores the same against a genuine
# mishearing as against "Hôm nay trời đẹp quá", because any four Vietnamese letters share
# enough with it by chance. No threshold can separate those, so short terms are excluded
# rather than guessed at.
#
# That means this does NOT rescue "Codex" -> "cô đích", the case that prompted it. That
# one has to be fixed upstream, by getting the term into the recogniser's keyword list, and
# a matcher that fired at random on short terms would be worse than admitting the limit.
_MISHEARD_SIMILARITY = 0.65
_MIN_MISHEARD_SKELETON = 7


def _sounds_like(source: str, text: str) -> bool:
    """Whether `text` contains something close enough to `source` to be a mishearing."""
    needle = _match_skeleton(source)
    if len(needle) < _MIN_MISHEARD_SKELETON:
        return False
    haystack = _match_skeleton(text)
    if not haystack:
        return False
    # Diacritic-insensitive containment first: exact once the accents are gone, which is
    # both the cheapest case and the safest.
    if needle in haystack:
        return True

    # Otherwise slide a window of roughly the term's length across the utterance. A
    # mishearing changes length a little, so widths either side of the term are tried.
    for width in range(max(_MIN_MISHEARD_SKELETON, len(needle) - 2), len(needle) + 3):
        for start in range(0, len(haystack) - width + 1):
            if SequenceMatcher(None, needle, haystack[start : start + width]).ratio() >= (
                _MISHEARD_SIMILARITY
            ):
                return True
    return False


def _select_relevant_glossary_terms(
    text: str,
    glossary_terms: list[dict[str, str]] | None,
    *,
    limit: int = 12,
) -> list[dict[str, str]]:
    """Glossary entries worth sending with this utterance, tagged by how they matched.

    Two kinds, and the distinction is the whole point:

    ``match="exact"``
        The term is literally in the text. Apply it.

    ``match="possible"``
        Nothing matched literally, but part of the utterance SOUNDS like the term. This
        is the case the guardian behaviour exists for: STT hears "Codex" as "cô đích", and
        the literal matcher below then finds nothing — so the one glossary entry that
        could have repaired the sentence was dropped exactly when it was needed. A term is
        offered as a suspicion, never as a fact; _build_glossary_block words it that way
        and the system prompt tells the model to translate literally when unsure.

    Sending the room's whole glossary on every realtime request would add tokens and steer
    short ambiguous speech toward unrelated terms, so this stays capped at `limit`. Exact
    matches are collected first and therefore never lose their slot to a suspicion.
    """
    if not glossary_terms or not text.strip():
        return []

    haystack = text.casefold()
    exact: list[dict[str, str]] = []
    possible: list[dict[str, str]] = []

    for term in glossary_terms:
        source = (term.get("source") or "").strip()
        if not source:
            continue
        pieces = [piece for piece in re.split(r"[\s_-]+", source.casefold()) if piece]
        if not pieces:
            continue
        # Word boundaries keep a short term such as "UI" out of the middle of "build";
        # spaces and hyphens are equivalent for code-switched technical phrases.
        pattern = r"(?<!\w)" + r"[\s_-]+".join(map(re.escape, pieces)) + r"(?!\w)"
        if re.search(pattern, haystack):
            exact.append({**term, "match": "exact"})
        elif _sounds_like(source, text):
            possible.append({**term, "match": "possible"})

    return (exact + possible)[:limit]


_BATCH_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")


def _lang_name(iso_code: str) -> str:
    return _LANG_NAMES.get(iso_code.split("-")[0], iso_code)


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
                        healthy = True
                        return result
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

        The rule now lives in shared/openai_options.py so the assistant, chat-tool,
        suggestion and security workers obey it too — they each used to build this
        dict by hand and would fail outright against a gpt-5 model.
        """
        return completion_options(self.model, token_limit, self.temperature)

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
        if not text.strip():
            return ""

        # Skip if same language
        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return text

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
        system_prompt = _SYSTEM_PROMPT
        if meeting_context:
            system_prompt += _CONTEXT_RELEVANCE_INSTRUCTION
        if getattr(self, "realtime_model", ""):
            try:
                result = await self._translate_realtime(user_message, system_prompt)
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

        logger.debug(
            "translation_complete",
            src=src_name,
            tgt=tgt_name,
            input_chars=len(text),
            output_chars=len(result),
        )
        return result

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
        if not texts:
            return []

        src = source_lang.split("-")[0]
        tgt = target_lang.split("-")[0]
        if src == tgt:
            return list(texts)

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

        system_prompt = _BATCH_SYSTEM_PROMPT
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
