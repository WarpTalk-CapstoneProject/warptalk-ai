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

import json
import time
from typing import Any, cast

from ai_assistant_worker.assistant import MeetingAssistant
from ai_assistant_worker.summary_templates import format_transcript_line
from shared.base_worker import BaseWorker
from shared.config import AssistantSettings, resolve_openai_api_key
from shared.control_markers import is_control_marker
from shared.schemas import STTResultMessage

#: One accumulated STT segment: (speaker, text, timestamp_ms).
TranscriptSegment = tuple[str, str, int]


def substantive_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    """The segments that are actually somebody speaking. WT-478.

    Two kinds of non-speech accumulate in this worker's transcript and both used to be
    formatted and sent to the model:

    1. **Segments with no text.** `format_transcript_line` turns one into ``"[t=0] [Nhi] "`` —
       non-empty to code, empty to a reader. A transcript of those is truthy to `.strip()`, so
       it passed the emptiness check, reached the model, and the model reported the transcript
       was empty. That report was then stored and rendered as the meeting's summary, which is
       the bug users saw: a transcript full of conversation beside a summary calling it empty.
       ``summary_template_worker._load_transcript`` has always filtered these when reading the
       SAVED transcript — the two summary paths disagreeing on what counts as a line is the
       underlying defect, so this is the same rule applied to the live path.
    2. **The control marker.** ``process`` appends every segment before it tests for the
       end-of-meeting marker, so the ``__MEETING_END__`` that triggers summarisation is sitting
       in the list being summarised. On a short meeting it can be most of what the model is
       shown, and summarising the sentinel that ends the meeting is never correct.

    Deliberately not a minimum length: the ticket asks for short meetings to be summarised too,
    so the question is whether anybody said anything, not whether they said enough.
    """
    return [
        (speaker, text, timestamp_ms)
        for speaker, text, timestamp_ms in segments
        if text and text.strip() and not is_control_marker(text)
    ]


class AIAssistantWorker(BaseWorker):
    """AI Assistant worker — non-realtime meeting summarization."""

    worker_name = "assistant"
    input_stream = "stt:results"
    consumer_group = "assistant-workers"  # Separate from translate-workers!

    def __init__(
        self,
        assistant_settings: AssistantSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.assistant_settings = assistant_settings or AssistantSettings()
        self.assistant: MeetingAssistant | None = None
        # In-memory transcript accumulator: {meeting_id: [(speaker, text, timestamp), ...]}
        self._transcripts: dict[str, list[tuple[str, str, int]]] = {}

    async def load_model(self) -> None:
        """Initialize OpenAI client."""
        self.assistant = MeetingAssistant(
            api_key=resolve_openai_api_key(self.assistant_settings.api_key),
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

        # The meeting-end trigger: MeetingRoomService.EndMeetingAsync publishes a synthetic STT
        # segment carrying this marker.
        #
        # Through the shared predicate rather than an inline comparison, because this worker was
        # the ONLY one that knew the sentinel was not speech — translation_worker translated it
        # and tts_worker sang it. Anything reading stt:results has to be able to ask the same
        # question, and the answer has to be in one place.
        if is_control_marker(stt_result.text):
            await self._generate_summary(stt_result.meeting_id)

    async def _generate_summary(self, meeting_id: str) -> None:
        """Generate and publish meeting summary."""
        segments = substantive_segments(self._transcripts.get(meeting_id, []))
        if not segments:
            # Dropped here as well as on the success path: this method runs once per meeting,
            # on the end-of-meeting marker, so nothing will ever add to this entry again and
            # leaving it behind grows the accumulator for the life of the process.
            self._transcripts.pop(meeting_id, None)
            return

        # Format transcript WITH the moment each line was spoken. The timestamp was
        # already here and was being discarded by that `_` — the model was asked not to
        # invent things while being given nothing it could point at. Offsets are relative
        # to the first segment so a cited atMs resolves against the stored transcript,
        # which is rendered the same way: a base time plus a per-segment offset.
        base_ms = min(ts for _, _, ts in segments)
        transcript_lines = [
            format_transcript_line(ts - base_ms, speaker, text.strip())
            for speaker, text, ts in segments
        ]
        transcript_text = "\n".join(transcript_lines)

        self.logger.info(
            "generating_summary",
            meeting_id=meeting_id,
            segment_count=len(segments),
        )

        # Fetch context snapshot from Redis
        context_snapshot_bytes = await self.redis.get(f"meeting:{meeting_id}:context_snapshot")
        context_snapshot = (
            context_snapshot_bytes.decode("utf-8")
            if isinstance(context_snapshot_bytes, bytes)
            else context_snapshot_bytes or ""
        )

        # Best-effort: the room's configured target language(s), written by
        # MeetingService.EndMeetingAsync (WarpTalk.MeetingService) as a JSON array string.
        # Missing/unparseable is fine — generate_structured_summary treats it as "no
        # translation needed" and just skips the bilingual section.
        target_languages: list[str] = []
        try:
            target_languages_bytes = await self.redis.get(f"meeting:{meeting_id}:target_languages")
            if target_languages_bytes:
                raw_languages = (
                    target_languages_bytes.decode("utf-8")
                    if isinstance(target_languages_bytes, bytes)
                    else target_languages_bytes
                )
                target_languages = cast(list[str], json.loads(raw_languages))
        except Exception:
            self.logger.warning("failed_to_read_target_languages", meeting_id=meeting_id)

        # Generate summary
        assistant = self._require_assistant()
        summary = await assistant.summarize(
            transcript_text,
            context_snapshot=context_snapshot,
        )

        # Store summary in Redis Hash for persistent retrieval
        await self.redis.hset(
            f"meeting:{meeting_id}:summary",
            "content",
            summary,
        )

        # Publish to stream for gateway to consume via SignalR
        await self.publish(
            "ai_assistant:results",
            meeting_id,
            {
                "type": "summary",
                "content": summary,
                "timestamp_ms": str(int(time.time() * 1000)),
            },
        )

        # Generate action items
        action_items = await assistant.extract_action_items(
            transcript_text,
            context_snapshot=context_snapshot,
        )
        await self.redis.hset(
            f"meeting:{meeting_id}:summary",
            "action_items",
            action_items,
        )

        # Publish action items to stream
        await self.publish(
            "ai_assistant:results",
            meeting_id,
            {
                "type": "action_items",
                "content": action_items,
                "timestamp_ms": str(int(time.time() * 1000)),
            },
        )

        # WT-13: also generate the structured {summary, decisions[], actionItems[]} JSON
        # that TranslationRoomService.ArtifactsFinalizer stores as the SUMMARY_EXPORT
        # artifact's inline Content, so the frontend can render an overview, a decisions
        # list, and an owner/task action-item checklist instead of parsing markdown.
        structured = await assistant.generate_structured_summary(
            transcript_text,
            target_languages=target_languages,
            context_snapshot=context_snapshot,
        )
        await self.redis.hset(
            f"meeting:{meeting_id}:summary",
            "structured_json",
            json.dumps(structured),
        )

        self.logger.info(
            "summary_generated",
            meeting_id=meeting_id,
            summary_length=len(summary),
        )

        # Cleanup
        del self._transcripts[meeting_id]

    def _require_assistant(self) -> MeetingAssistant:
        if self.assistant is None:
            raise RuntimeError("Meeting assistant is not loaded")
        return self.assistant
