"""AI Assistant — OpenAI meeting summarization and action items.

Provides async methods for generating meeting summaries,
extracting action items, and answering questions about the meeting.
"""

from __future__ import annotations

import json
from typing import Any, cast

from openai import AsyncOpenAI

from ai_assistant_worker.minutes_translation import collect_translatable, merge_translation
from ai_assistant_worker.summary_grounding import ground_summary
from ai_assistant_worker.summary_templates import (
    build_system_prompt,
    resolve_template,
    spoken_text_only,
)
from shared.config import AssistantSettings
from shared.languages import normalize_language_code
from shared.logger import get_logger
from shared.openai_options import completion_options

logger = get_logger(__name__)

# Constructor defaults below mirror AssistantSettings — the values production code
# actually runs with (ai_assistant_worker/worker.py always passes them explicitly).
# Sourcing the defaults from here instead of a second hardcoded literal keeps
# direct/test instantiation (e.g. tests/test_ai_assistant.py) in sync with config.py
# without anyone having to remember to update both places.
_DEFAULTS = AssistantSettings()


class MeetingAssistant:
    """OpenAI-powered meeting assistant.

    Accumulates transcript segments and generates summaries on demand.
    """

    SYSTEM_PROMPT = """You are a professional meeting assistant. Your task is to analyze
meeting transcripts and produce clear, concise outputs.

When summarizing:
- Highlight key decisions made
- List action items with assignees if mentioned
- Note any unresolved questions
- Use bullet points for readability
- Keep the summary under 500 words

When extracting action items:
- Format: "[ ] Action item - @assignee (if mentioned)"
- Only include explicit commitments, not vague suggestions
"""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULTS.model,
        max_tokens: int = _DEFAULTS.max_tokens,
        temperature: float = _DEFAULTS.temperature,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: AsyncOpenAI | None = None

    async def load(self) -> None:
        """Initialize the OpenAI async client."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for WarpBot assistant")

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_client_initialized", model=self.model)

    async def summarize(self, transcript: str, context_snapshot: str = "") -> str:
        """Generate a meeting summary from the full transcript.

        Args:
            transcript: Formatted meeting transcript
                (e.g. "[Speaker A] Hello everyone...")
            context_snapshot: Extracted text from RAG documents

        Returns:
            Summary text with key decisions, action items, etc.
        """
        # WT-478: spoken words, not the formatted string — see generate_structured_summary.
        if not spoken_text_only(transcript):
            return "No transcript content to summarize."

        system_content = self.SYSTEM_PROMPT
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        client = self._require_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"Please summarize this meeting transcript:\n\n{transcript}",
                },
            ],
            **completion_options(self.model, self.max_tokens, self.temperature),
        )

        return response.choices[0].message.content or ""

    async def extract_action_items(self, transcript: str, context_snapshot: str = "") -> str:
        """Extract action items from the transcript.

        Returns:
            Formatted action items list
        """
        # WT-478: spoken words, not the formatted string — see generate_structured_summary.
        if not spoken_text_only(transcript):
            return "No action items found."

        system_content = self.SYSTEM_PROMPT
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        client = self._require_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        "Extract all action items from this meeting transcript. "
                        f"Format each as a checkbox item:\n\n{transcript}"
                    ),
                },
            ],
            # 0.2, not self.temperature: action-item extraction is deliberately tighter
            # than summarization. Preserved as-is through the shared-options move.
            **completion_options(self.model, self.max_tokens, 0.2),
        )

        return response.choices[0].message.content or ""

    async def generate_structured_summary(
        self,
        transcript: str,
        target_languages: list[str] | None = None,
        context_snapshot: str = "",
        template_key: str | None = None,
        summary_language: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured {summary, decisions[], actionItems[]} JSON object.

        Args:
            transcript: Formatted meeting transcript.
            target_languages: The room's configured target language(s). When more than one is
                configured the response additionally includes a "translations" map keyed by
                language code, each value having the same shape translated into that language —
                this is what makes a biên bản bilingual. It is produced one of two ways, and
                which one is invisible to the caller: inline with the summary when nobody chose
                a language, and by a second pass afterwards when somebody did (WT-665). Both
                exist because asking one prompt to write in Japanese AND to translate itself is
                a contradiction; see minutes_translation.
            context_snapshot: Extracted text from RAG documents.
            summary_language: ISO 639-1 code the summary must be WRITTEN in. None means
                nobody chose, and the model follows the transcript — how every summary
                already in storage was produced.

        Returns:
            Parsed JSON dict. On any failure (empty transcript, malformed model output),
            returns a safe fallback dict with insufficientData=True instead of raising —
            callers should never have to special-case exceptions from this method.
        """
        # Normalised once, here, so `vi-VN` from a room and `vi` from a request mean the same
        # thing to the prompt and to the key written back into the content.
        language = normalize_language_code(summary_language)

        # WT-478: tested against the SPOKEN WORDS, not the formatted transcript. A transcript
        # of segments with empty text is a wall of "[t=0] [Nhi] " scaffolding — non-empty to
        # `.strip()`, empty to a reader. That gap is what sent a contentless transcript to the
        # model, which correctly reported it was empty; that report then came back as a
        # normal summary (insufficientData=False) and was rendered to the user as one.
        #
        # Emptiness is a decision this code owns. The model is never asked to make it, and the
        # prompt in build_system_prompt now tells it so — see spoken_text_only.
        if not spoken_text_only(transcript):
            return {
                "summary": "No transcript content to summarize.",
                "decisions": [],
                "actionItems": [],
                "citations": [],
                "templateKey": resolve_template(template_key).key,
                "summaryLanguage": language,
                "insufficientData": True,
            }

        # The shape comes from the template, not from a constant. An earlier hardcoded prompt
        # asked for a "concise overview paragraph" and got exactly that — three thin
        # sentences for every meeting, whatever kind of meeting it was.
        template = resolve_template(template_key)
        system_content = build_system_prompt(template, language)
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        # Only when nobody chose a language. Asking for the whole summary in Japanese and then
        # for a "translations" map beside it are contradictory instructions, and a model given
        # both answers one of them at random.
        #
        # WT-665: what that used to mean was that choosing a summary language ALSO cancelled the
        # biên bản's other languages — a minutes document silently lost half of itself because
        # of a choice made about a different artifact on a different tab. The contradiction is
        # real, so it is still avoided here; the translating just moved to its own pass below,
        # where it is a separate question asked separately. See minutes_translation.
        languages = [lang for lang in (target_languages or []) if lang]
        if not language and len(languages) > 1:
            system_content += (
                "\n\nThis meeting has multiple target languages: "
                f"{', '.join(languages)}. In addition to the top-level fields (in the "
                'meeting\'s primary/source language), include a "translations" object '
                "keyed by each of these language codes, each value having the same "
                "{summary, decisions, actionItems} shape translated into that language."
            )

        try:
            client = self._require_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {
                        "role": "user",
                        "content": f"Summarize this meeting transcript as JSON:\n\n{transcript}",
                    },
                ],
                **completion_options(self.model, self.max_tokens, self.temperature),
                response_format={"type": "json_object"},
            )
            choice = response.choices[0]
            # Nobody read `finish_reason`, and that left two very different failures wearing
            # the same face. A response that runs into `max_tokens` comes back as truncated
            # JSON; `json.loads` raises; the branch below reports `generationFailed` — so a
            # ceiling set too low for the meeting is recorded, in the only place anybody
            # looks, as a model that could not write a summary. Logging the reason is what
            # makes the two tellable apart. Read defensively: the real client always sets it,
            # test doubles standing in for a choice do not.
            finish_reason: str | None = getattr(choice, "finish_reason", None)
            raw = choice.message.content or "{}"
            parsed = cast(dict[str, Any], json.loads(raw))
            parsed.setdefault("summary", "")
            # Every section the template declared, so a consumer never has to guess whether
            # an absent key means "none of these" or "the model forgot".
            for section in template.sections:
                if section.kind != "paragraph":
                    parsed.setdefault(section.key, [])
            parsed.setdefault("decisions", [])
            parsed.setdefault("actionItems", [])
            parsed.setdefault("citations", [])
            parsed["templateKey"] = template.key
            # Recorded, never inferred. The web has to be able to show WHICH language this
            # summary is in, and asking a language detector afterwards would be answering a
            # question we already knew the answer to — and getting it wrong on short text.
            # Empty means nobody chose, which is honest: the model followed the transcript
            # and no code here can say what it landed on.
            parsed["summaryLanguage"] = language
            parsed["insufficientData"] = False

            # Last thing before the summary leaves this process, because this is the last
            # place that still knows which moments the model was shown. Downstream a cited
            # `atMs` is just a number, and the meeting page will happily scroll to a number
            # that came from nowhere — see summary_grounding.
            grounded = ground_summary(parsed, transcript)
            logger.info(
                "structured_summary_generated",
                template=template.key,
                finish_reason=finish_reason,
                moments_checked=grounded.moments_checked,
                moments_dropped=grounded.moments_dropped,
                items_uncited=grounded.items_uncited,
            )
            if finish_reason != "stop":
                # "length" here means the summary is short because the budget ran out, not
                # because the meeting was.
                logger.warning(
                    "structured_summary_finished_unexpectedly",
                    finish_reason=finish_reason,
                    template=template.key,
                )
            if grounded.moments_dropped:
                logger.warning(
                    "structured_summary_moments_unverifiable",
                    template=template.key,
                    moments_checked=grounded.moments_checked,
                    moments_dropped=grounded.moments_dropped,
                    items_uncited=grounded.items_uncited,
                )

            # WT-665: the biên bản's other languages, when the branch above could not ask for
            # them. Deliberately AFTER grounding — this translates the summary that survived the
            # citation check, so a moment dropped for being unverifiable is not reintroduced by
            # its own translation.
            summary = grounded.summary
            if language and len(languages) > 1:
                translations = await self._translate_for_minutes(summary, languages, language)
                if translations:
                    summary["translations"] = translations

            return summary
        except Exception:
            logger.exception("structured_summary_generation_failed")
            # WT-530. Two keys here are load-bearing, and their absence was the bug.
            #
            # `templateKey`: without it the web never learns which template this summary is, so a
            # rewrite request waits for a template that never arrives and gives up after 90s with
            # "The rewritten summary has not arrived" — and nothing in the console, because from
            # the browser's side nothing failed.
            #
            # `generationFailed`: this dict was previously indistinguishable from a real summary,
            # so the worker published it as status="completed" and the backend wrote it OVER a
            # perfectly good existing summary. A failed rewrite must not destroy the summary it
            # failed to replace. `insufficientData` cannot carry that meaning — it is already the
            # honest answer for a meeting that genuinely has too little to summarise.
            return {
                "summary": (
                    "The AI assistant could not generate a structured summary for this meeting."
                ),
                "decisions": [],
                "actionItems": [],
                "insufficientData": True,
                "generationFailed": True,
                "templateKey": template.key,
                "summaryLanguage": language,
            }

    async def _translate_for_minutes(
        self,
        summary: dict[str, Any],
        target_languages: list[str],
        summary_language: str,
    ) -> dict[str, dict[str, Any]] | None:
        """The same summary in the meeting's other languages, for the biên bản. WT-665.

        A separate call on purpose — see minutes_translation for why translating cannot share a
        prompt with summarising, and why the model is shown strings and never structure.

        Returns None whenever there is nothing to do or anything at all goes wrong. The minutes
        are then single-language, which is exactly what they are today: a degradation back to
        the current behaviour, never a half-built document.
        """
        wanted = [
            lang
            for lang in dict.fromkeys(normalize_language_code(lang) for lang in target_languages)
            # The summary is already written in this one. Asking for it again would put the same
            # text on both sides of a bilingual page.
            if lang and lang != summary_language
        ]
        if not wanted:
            return None

        payload = collect_translatable(summary)
        if not payload:
            return None

        try:
            client = self._require_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You translate a meeting summary that has already been written. "
                            f"The source is {summary_language}. Return a JSON object keyed by "
                            f"each of these language codes: {', '.join(wanted)}. Each value has "
                            "EXACTLY the same keys as the input, and every array has EXACTLY the "
                            "same number of elements in the same order — each element is the "
                            "translation of the element in that position. Translate the words "
                            "only: do not merge, split, reorder, add or drop anything, and do "
                            "not add commentary. Keep people's names as they are written."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                **completion_options(self.model, self.max_tokens, self.temperature),
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            answered = cast(dict[str, Any], json.loads(raw))
        except Exception:
            # Warning, not exception: the summary itself succeeded and is about to be published.
            # This is the biên bản losing its second language, which is worth knowing about and
            # is not a reason to fail the summary that already exists.
            logger.warning(
                "minutes_translation_failed",
                summary_language=summary_language,
                languages=wanted,
                exc_info=True,
            )
            return None

        translations: dict[str, dict[str, Any]] = {}
        for lang in wanted:
            merged = merge_translation(summary, answered.get(lang))
            if merged:
                translations[lang] = merged
            else:
                # Named per language rather than as one failure: a model that mangles Japanese
                # and gets Korean right should cost the minutes only its Japanese half.
                logger.warning(
                    "minutes_translation_unusable",
                    language=lang,
                    summary_language=summary_language,
                )

        if translations:
            logger.info(
                "minutes_translation_produced",
                summary_language=summary_language,
                languages=sorted(translations),
            )

        return translations or None

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise RuntimeError("Meeting assistant is not loaded")
        return self._client
