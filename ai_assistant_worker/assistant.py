"""AI Assistant — OpenAI meeting summarization and action items.

Provides async methods for generating meeting summaries,
extracting action items, and answering questions about the meeting.
"""

from __future__ import annotations

import json
from typing import Any, cast

from openai import AsyncOpenAI

from ai_assistant_worker.summary_templates import (
    build_system_prompt,
    resolve_template,
    spoken_text_only,
)
from shared.config import AssistantSettings
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
    ) -> dict[str, Any]:
        """Generate a structured {summary, decisions[], actionItems[]} JSON object.

        Args:
            transcript: Formatted meeting transcript.
            target_languages: The room's configured target language(s). When more than
                one is configured, the response additionally includes a "translations"
                map keyed by language code with the same {summary, decisions, actionItems}
                shape translated into that language — so a bilingual/multilingual room gets
                a bilingual summary instead of one arbitrarily-chosen language.
            context_snapshot: Extracted text from RAG documents.

        Returns:
            Parsed JSON dict. On any failure (empty transcript, malformed model output),
            returns a safe fallback dict with insufficientData=True instead of raising —
            callers should never have to special-case exceptions from this method.
        """
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
                "insufficientData": True,
            }

        # The shape comes from the template, not from a constant. An earlier hardcoded prompt
        # asked for a "concise overview paragraph" and got exactly that — three thin
        # sentences for every meeting, whatever kind of meeting it was.
        template = resolve_template(template_key)
        system_content = build_system_prompt(template)
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        languages = [lang for lang in (target_languages or []) if lang]
        if len(languages) > 1:
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
            raw = response.choices[0].message.content or "{}"
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
            parsed["insufficientData"] = False
            return parsed
        except Exception:
            logger.exception("structured_summary_generation_failed")
            return {
                "summary": (
                    "The AI assistant could not generate a structured summary for this meeting."
                ),
                "decisions": [],
                "actionItems": [],
                "insufficientData": True,
            }

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise RuntimeError("Meeting assistant is not loaded")
        return self._client
