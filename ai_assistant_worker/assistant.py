"""AI Assistant — OpenAI meeting summarization and action items.

Provides async methods for generating meeting summaries,
extracting action items, and answering questions about the meeting.
"""

from __future__ import annotations

from shared.logger import get_logger

logger = get_logger(__name__)


class MeetingAssistant:
    """OpenAI-powered meeting assistant.

    Accumulates transcript segments and generates summaries on demand.
    """

    SYSTEM_PROMPT = """You are a professional meeting assistant. Your task is to analyze meeting transcripts and produce clear, concise outputs.

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
        model: str = "gpt-4.1",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    async def load(self) -> None:
        """Initialize the OpenAI async client."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for WarpBot assistant")

        from openai import AsyncOpenAI

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
        if not transcript.strip():
            return "No transcript content to summarize."

        system_content = self.SYSTEM_PROMPT
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        response = await self._client.chat.completions.create(
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

        return response.choices[0].message.content or ""

    async def extract_action_items(self, transcript: str, context_snapshot: str = "") -> str:
        """Extract action items from the transcript.

        Returns:
            Formatted action items list
        """
        if not transcript.strip():
            return "No action items found."

        system_content = self.SYSTEM_PROMPT
        if context_snapshot:
            system_content += f"\n\nMeeting Context (Reference Documents):\n{context_snapshot}"

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"Extract all action items from this meeting transcript. Format each as a checkbox item:\n\n{transcript}",
                },
            ],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )

        return response.choices[0].message.content or ""
