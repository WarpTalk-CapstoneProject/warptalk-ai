"""AI Assistant Worker — Consumes STT results, generates meeting summaries.

Pipeline:
    Redis Stream (stt:results:{meetingId}) — separate consumer group
    → Accumulate transcript segments per meeting
    → GPT-4o summarization on meeting end or on demand

This worker does NOT block the main STT→Translation→TTS pipeline.
It consumes stt:results in its own consumer group (assistant-workers)
independently of the translation worker.
"""

from __future__ import annotations

from shared.base_worker import BaseWorker
from shared.config import AssistantSettings
from shared.schemas import STTResultMessage

from ai_assistant_worker.assistant import MeetingAssistant


class AIAssistantWorker(BaseWorker):
    """AI Assistant worker — non-realtime meeting summarization."""

    worker_name = "assistant"
    input_stream = "stt:results"
    consumer_group = "assistant-workers"  # Separate from translate-workers!

    def __init__(
        self,
        assistant_settings: AssistantSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.assistant_settings = assistant_settings or AssistantSettings()
        self.assistant: MeetingAssistant | None = None
        # In-memory transcript accumulator: {meeting_id: [(speaker, text, timestamp), ...]}
        self._transcripts: dict[str, list[tuple[str, str, int]]] = {}

    async def load_model(self) -> None:
        """Initialize OpenAI client."""
        self.assistant = MeetingAssistant(
            api_key=self.assistant_settings.api_key,
            model=self.assistant_settings.model,
            max_tokens=self.assistant_settings.max_tokens,
            temperature=self.assistant_settings.temperature,
        )
        await self.assistant.load()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Accumulate transcript segment and check for summary trigger."""
        stt_result = STTResultMessage.from_redis(data)

        # Accumulate transcript
        if stt_result.meeting_id not in self._transcripts:
            self._transcripts[stt_result.meeting_id] = []

        self._transcripts[stt_result.meeting_id].append(
            (stt_result.speaker_id, stt_result.text, stt_result.timestamp_ms)
        )

        self.logger.debug(
            "transcript_accumulated",
            meeting_id=stt_result.meeting_id,
            segments=len(self._transcripts[stt_result.meeting_id]),
        )

        # Check for summary trigger (e.g. meeting end signal)
        # The backend sends a special "meeting:end" message via Redis
        # For now, we also support manual trigger via a special text marker
        if stt_result.text.strip().upper() == "__MEETING_END__":
            await self._generate_summary(stt_result.meeting_id)

    async def _generate_summary(self, meeting_id: str) -> None:
        """Generate and publish meeting summary."""
        segments = self._transcripts.get(meeting_id, [])
        if not segments:
            return

        # Format transcript
        transcript_lines = [
            f"[{speaker}] {text}" for speaker, text, _ in segments
        ]
        transcript_text = "\n".join(transcript_lines)

        self.logger.info(
            "generating_summary",
            meeting_id=meeting_id,
            segment_count=len(segments),
        )

        # Generate summary
        summary = await self.assistant.summarize(transcript_text)

        # Store summary in Redis for the backend to retrieve
        await self.redis.hset(
            f"meeting:{meeting_id}:summary",
            "content",
            summary,
        )

        # Generate action items
        action_items = await self.assistant.extract_action_items(transcript_text)
        await self.redis.hset(
            f"meeting:{meeting_id}:summary",
            "action_items",
            action_items,
        )

        self.logger.info(
            "summary_generated",
            meeting_id=meeting_id,
            summary_length=len(summary),
        )

        # Cleanup
        del self._transcripts[meeting_id]
