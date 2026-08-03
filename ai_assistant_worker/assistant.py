"""AI Assistant — OpenAI meeting summarization and action items.

Provides async methods for generating meeting summaries,
extracting action items, and answering questions about the meeting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

from openai import AsyncOpenAI

from shared.config import AssistantSettings
from shared.logger import get_logger
from shared.openai_usage import TokenUsage

logger = get_logger(__name__)

# Constructor defaults below mirror AssistantSettings — the values production code
# actually runs with (ai_assistant_worker/worker.py always passes them explicitly).
# Sourcing the defaults from here instead of a second hardcoded literal keeps
# direct/test instantiation (e.g. tests/test_ai_assistant.py) in sync with config.py
# without anyone having to remember to update both places.
_DEFAULTS = AssistantSettings()


@dataclass(frozen=True)
class TextWithUsage:
    text: str
    usage: TokenUsage = TokenUsage()


@dataclass(frozen=True)
class DictWithUsage:
    data: dict[str, Any]
    usage: TokenUsage = TokenUsage()


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
        result = await self.summarize_with_usage(transcript, context_snapshot)
        return result.text

    async def summarize_with_usage(
        self, transcript: str, context_snapshot: str = ""
    ) -> TextWithUsage:
        if not transcript.strip():
            return TextWithUsage("No transcript content to summarize.")

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
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
        return TextWithUsage(response.choices[0].message.content or "", usage)

    async def extract_action_items(self, transcript: str, context_snapshot: str = "") -> str:
        """Extract action items from the transcript.

        Returns:
            Formatted action items list
        """
        result = await self.extract_action_items_with_usage(transcript, context_snapshot)
        return result.text

    async def extract_action_items_with_usage(
        self, transcript: str, context_snapshot: str = ""
    ) -> TextWithUsage:
        if not transcript.strip():
            return TextWithUsage("No action items found.")

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
            max_tokens=self.max_tokens,
            temperature=0.2,
        )

        usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
        return TextWithUsage(response.choices[0].message.content or "", usage)

    STRUCTURED_SYSTEM_PROMPT = """You are a professional meeting assistant.
Analyze the meeting transcript and \

respond with a single JSON object only (no markdown, no commentary) matching exactly this shape:
{
  "summary": "a concise overview paragraph of what the meeting covered",
  "decisions": ["one string per key decision that was made"],
  "actionItems": [{"owner": "assignee name, or empty string if unclear", "task": "the action item"}]
}
Only include explicit decisions and commitments ? do not invent content that
isn't in the transcript.
If the transcript is empty or has no substantive content, return a summary
describing that, and empty arrays for decisions and actionItems."""

    async def generate_structured_summary(
        self,
        transcript: str,
        target_languages: list[str] | None = None,
        context_snapshot: str = "",
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
        result = await self.generate_structured_summary_with_usage(
            transcript,
            target_languages=target_languages,
            context_snapshot=context_snapshot,
        )
        return result.data

    async def generate_structured_summary_with_usage(
        self,
        transcript: str,
        target_languages: list[str] | None = None,
        context_snapshot: str = "",
    ) -> DictWithUsage:
        if not transcript.strip():
            return DictWithUsage(
                {
                    "summary": "No transcript content to summarize.",
                    "decisions": [],
                    "actionItems": [],
                    "insufficientData": True,
                }
            )

        system_content = self.STRUCTURED_SYSTEM_PROMPT
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
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            usage = TokenUsage.from_openai_usage(getattr(response, "usage", None))
            parsed = cast(dict[str, Any], json.loads(raw))

            parsed.setdefault("summary", "")
            parsed.setdefault("decisions", [])
            parsed.setdefault("actionItems", [])
            parsed["insufficientData"] = False
            return DictWithUsage(parsed, usage)
        except Exception:
            logger.exception("structured_summary_generation_failed")
            return DictWithUsage(
                {
                    "summary": (
                        "The AI assistant could not generate a structured summary for this meeting."
                    ),
                    "decisions": [],
                    "actionItems": [],
                    "insufficientData": True,
                }
            )

    def _require_client(self) -> AsyncOpenAI:
        if self._client is None:
            raise RuntimeError("Meeting assistant is not loaded")
        return self._client
