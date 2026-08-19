"""Re-summarise a finished meeting under a different template, on request.

The default summary is written once, when the meeting ends, in the General shape. This
worker exists for the second look: somebody opens a finished meeting, decides it was really
a standup or an interview, and asks for that shape instead.

WHY IT REFETCHES THE TRANSCRIPT
    AIAssistantWorker summarises from an in-memory accumulator built out of the live STT
    stream. That memory is gone the moment the meeting ends, and gone again on every
    restart, so a request arriving minutes or days later has nothing to summarise from. It
    reads the SAVED transcript instead — which is also what makes the citations line up,
    because those are the exact segments the meeting page renders and scrolls to.

WHY IT CARRIES A BEARER TOKEN
    The same reason ChatAssistantWorker does: tool calls hit sibling services' existing
    authenticated endpoints as the person who asked, never through a privileged bypass. A
    regeneration can only read a transcript its requester could already read.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ai_assistant_worker.assistant import MeetingAssistant
from ai_assistant_worker.summary_templates import format_transcript_line, resolve_template
from shared.base_worker import BaseWorker
from shared.config import AssistantSettings, resolve_openai_api_key
from shared.schemas import SummaryRequestMessage, SummaryResultMessage

# One page is enough for any meeting this product records, and a bounded read means a
# pathological transcript cannot stall the worker for everyone else.
SEGMENT_LIMIT = 2000


class SummaryTemplateWorker(BaseWorker):
    """Consumes `assistant:summary_requests`, publishes `assistant:summary_results`."""

    worker_name = "summary-template"
    input_stream = "assistant:summary_requests"
    consumer_group = "summary-template-workers"

    def __init__(
        self,
        assistant_settings: AssistantSettings | None = None,
        transcript_base_url: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.assistant_settings = assistant_settings or AssistantSettings()
        self.transcript_base_url = transcript_base_url or getattr(
            self.settings, "transcript_service_url", ""
        )
        self.assistant: MeetingAssistant | None = None
        self._transcript_client: httpx.AsyncClient | None = None

    async def load_model(self) -> None:
        self.assistant = MeetingAssistant(
            api_key=resolve_openai_api_key(self.assistant_settings.api_key),
            model=self.assistant_settings.model,
            max_tokens=self.assistant_settings.max_tokens,
            temperature=self.assistant_settings.temperature,
        )
        await self.assistant.load()
        self._transcript_client = httpx.AsyncClient(base_url=self.transcript_base_url, timeout=30.0)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = SummaryRequestMessage.from_redis(data)
        template = resolve_template(request.template_key)

        try:
            transcript = await self._load_transcript(request)
        except Exception as exc:  # noqa: BLE001 — the reason is reported, not swallowed
            self.logger.warning(
                "summary_transcript_fetch_failed", room_id=request.room_id, error=str(exc)
            )
            await self._publish_failure(request, template.key, "Could not read the transcript.")
            return

        if not transcript.strip():
            # Not an error: a meeting with no saved transcript has nothing to re-summarise,
            # and saying so is more useful than a generic failure.
            await self._publish_failure(
                request, template.key, "This meeting has no saved transcript to summarise."
            )
            return

        assert self.assistant is not None, "load_model() must run before process()"
        try:
            target_languages = json.loads(request.target_languages_json or "[]")
        except json.JSONDecodeError:
            target_languages = []

        content = await self.assistant.generate_structured_summary(
            transcript,
            target_languages=target_languages,
            template_key=template.key,
        )

        # WT-530: a generation that failed is not a rewrite that succeeded.
        #
        # This published every result as status="completed", so when the model call threw, the
        # assistant's placeholder ("could not generate a structured summary") was written over
        # the meeting's existing summary — a rewrite that both failed AND destroyed what it was
        # replacing. The consumer already treats a failure correctly: it logs the reason and
        # leaves the current summary alone.
        if content.get("generationFailed"):
            await self._publish_failure(
                request,
                template.key,
                "The summary could not be generated. The previous one is unchanged.",
            )
            return

        await self.publish(
            "assistant:summary_results",
            request.room_id,
            SummaryResultMessage(
                request_id=request.request_id,
                room_id=request.room_id,
                template_key=template.key,
                status="completed",
                content_json=json.dumps(content, ensure_ascii=False),
            ).to_redis(),
        )
        self.logger.info("summary_regenerated", room_id=request.room_id, template=template.key)

    async def _load_transcript(self, request: SummaryRequestMessage) -> str:
        """The saved transcript, formatted with the moments the model must cite."""
        client = self._transcript_client
        assert client is not None, "load_model() must run before process()"
        headers = {"Authorization": request.bearer_token} if request.bearer_token else {}

        lookup = await client.get(f"/api/v1/transcripts/by-room/{request.room_id}", headers=headers)
        if lookup.status_code == 404:
            return ""
        lookup.raise_for_status()
        transcript_id = (lookup.json() or {}).get("id")
        if not transcript_id:
            return ""

        response = await client.get(
            f"/api/v1/transcripts/{transcript_id}/segments",
            params={"skip": 0, "take": SEGMENT_LIMIT},
            headers=headers,
        )
        response.raise_for_status()
        segments = (response.json() or {}).get("items") or []

        # Offsets are already relative to the meeting start in the stored transcript, which
        # is the same origin the live path uses — so a cited atMs means the same thing
        # whichever worker produced the summary.
        lines = [
            format_transcript_line(
                int(segment.get("startTimeMs") or 0),
                str(segment.get("speakerName") or "Unknown speaker"),
                str(segment.get("originalText") or "").strip(),
            )
            for segment in segments
            if str(segment.get("originalText") or "").strip()
        ]
        return "\n".join(lines)

    async def _publish_failure(
        self, request: SummaryRequestMessage, template_key: str, error: str
    ) -> None:
        await self.publish(
            "assistant:summary_results",
            request.room_id,
            SummaryResultMessage(
                request_id=request.request_id,
                room_id=request.room_id,
                template_key=template_key,
                status="failed",
                error=error,
            ).to_redis(),
        )

    async def _cleanup(self) -> None:
        # `_cleanup` is the hook BaseWorker actually calls on shutdown. Named `cleanup`,
        # this method would simply never run and the HTTP client would leak on every restart.
        if self._transcript_client is not None:
            await self._transcript_client.aclose()
            self._transcript_client = None
