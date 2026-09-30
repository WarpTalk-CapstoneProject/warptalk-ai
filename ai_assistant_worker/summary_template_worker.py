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

WHAT A REWRITE CAN CHANGE
    Two things, and they are independent: the SHAPE (`template_key` — is this a standup or an
    interview) and the LANGUAGE (`summary_language`). Both arrive on the request rather than
    being inferred, because both are the requester's decision and neither is recoverable from
    the transcript. An empty language means the request did not express one, and the model
    falls back to following the transcript exactly as it always did.

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
from shared.config import AssistantSettings, ChatAssistantSettings, resolve_openai_api_key
from shared.languages import known_language_code, language_name, normalize_language_code
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
        # WT-684: every template switch answered "Could not read the transcript." in prod.
        #
        # This fell back to `self.settings.transcript_service_url`, and WorkerSettings has no such
        # field — so __main__, which passes no URL, handed httpx an EMPTY base_url and every fetch
        # raised UnsupportedProtocol before reaching the network. General kept working only
        # because ArtifactsFinalizer sends the transcript with the request and never takes this
        # path. The tests all pass a URL explicitly, which is how none of them saw it.
        #
        # The transcript service address is already configured in production for the chat
        # worker in this same container (ASSISTANT_CHAT_TRANSCRIPT_SERVICE_URL), so it is read
        # from there rather than inventing a second variable that would have to be deployed.
        self.transcript_base_url = (
            transcript_base_url
            or getattr(self.settings, "transcript_service_url", "")
            or ChatAssistantSettings().transcript_service_url
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

        # A READER'S RENDERING MUST COME BACK IN THE LANGUAGE THEY PICKED, OR SAY WHY NOT.
        #
        # The backend files a rendering under the pair stamped in its content and looks it up
        # under the pair that was asked for. For a canonical rewrite, dropping an unrecognised
        # language to "as spoken" (WT-703, below) is a fair degradation — the host still gets a
        # summary. For a reader's rendering it is a trap: the summary is stamped "", filed under
        # "", never found under the language they chose, and the retry does exactly the same.
        # So that case is refused here, with the reason, before any work is spent.
        if (
            request.delivery == "variant"
            and request.summary_language.strip()
            and not known_language_code(request.summary_language)
        ):
            await self._publish_failure(
                request,
                template.key,
                f"Summaries cannot be written in '{request.summary_language.strip()[:16]}'.",
            )
            return

        if request.mode == "translate" and request.summary_language.strip():
            await self._translate(request, template.key)
            return

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

        # WT-703: the backend only publishes languages the room allows, and this does not rely
        # on it. The language is spliced into the summary prompt, into the minutes translation
        # prompt and into the stored content, so a value the map cannot name is dropped here —
        # the model follows the transcript — rather than trusted anywhere downstream.
        summary_language = known_language_code(request.summary_language)
        if request.summary_language.strip() and not summary_language:
            self.logger.warning(
                "summary_language_unrecognised",
                room_id=request.room_id,
                request_id=request.request_id,
                requested=repr(request.summary_language[:16]),
            )

        content = await self.assistant.generate_structured_summary(
            transcript,
            target_languages=target_languages,
            template_key=template.key,
            summary_language=summary_language,
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
                # Echoed, never decided here. Whether this answer replaces the room's summary or
                # fills one reader's cache is the requester's act, and the content cannot tell
                # the two apart — see SummaryRequestMessage.delivery.
                delivery=request.delivery,
                content_json=json.dumps(content, ensure_ascii=False),
                requested_template_key=_requested_template(request),
                summary_language=normalize_language_code(request.summary_language),
            ).to_redis(),
        )
        self.logger.info(
            "summary_regenerated",
            room_id=request.room_id,
            template=template.key,
            language=summary_language or "as-spoken",
        )

    async def _translate(self, request: SummaryRequestMessage, template_key: str) -> None:
        """Answer a language switch by translating the published summary it was sent with.

        No transcript is read: the summary being translated IS the meeting's record of what was
        said, and translating it is what keeps the reader's version the same document as
        everybody else's. This is also why it cannot fail with "Could not read the transcript."
        """
        language = known_language_code(request.summary_language)
        try:
            source = json.loads(request.source_content_json or "")
        except json.JSONDecodeError:
            source = None
        if not isinstance(source, dict):
            await self._publish_failure(
                request, template_key, "The published summary could not be read to translate it."
            )
            return

        assert self.assistant is not None, "load_model() must run before process()"
        rendered = await self.assistant.translate_summary(source, language)
        if rendered is None:
            await self._publish_failure(
                request,
                template_key,
                f"The summary could not be translated into {language_name(language)}. "
                "Please try again.",
            )
            return

        await self.publish(
            "assistant:summary_results",
            request.room_id,
            SummaryResultMessage(
                request_id=request.request_id,
                room_id=request.room_id,
                template_key=str(rendered.get("templateKey") or template_key),
                status="completed",
                delivery=request.delivery,
                content_json=json.dumps(rendered, ensure_ascii=False),
                requested_template_key=_requested_template(request),
                summary_language=normalize_language_code(request.summary_language),
            ).to_redis(),
        )
        self.logger.info(
            "summary_translated",
            room_id=request.room_id,
            template=rendered.get("templateKey") or template_key,
            language=language,
        )

    async def _load_transcript(self, request: SummaryRequestMessage) -> str:
        """The saved transcript, formatted with the moments the model must cite."""
        # A system-initiated request brings the transcript with it, because the publisher —
        # ArtifactsFinalizer, finalising a meeting that has just ended — already read those same
        # stored segments over the internal gRPC mesh and holds them. Using them here is what
        # lets a background finalization summarise with no user and no bearer token, without
        # anyone inventing a privileged HTTP path into the transcript service.
        #
        # Checked BEFORE the client assert on purpose: this path makes no HTTP call at all.
        #
        # WT-716 — THIS TEXT IS RAW, AND IT IS USED AS SENT. `ArtifactsFinalizer` formats the
        # segments it already holds with `CitedTranscriptFormatter` over `FinalizedSegment.Text`,
        # which is the raw wording; the clean column travels on the segment rows, not on this
        # pre-rendered blob. There is no clean version to prefer here, and manufacturing one —
        # by re-fetching the segments the publisher just read, with no requester and no token —
        # is exactly the privileged read this worker refuses to invent (see the class docstring).
        # So the finalizer's fallback summary reads the raw wording, the fetching path below
        # reads the clean one, and both cite the same moments.
        if request.transcript_text.strip():
            self.logger.info(
                "summary_transcript_supplied",
                room_id=request.room_id,
                chars=len(request.transcript_text),
            )
            return request.transcript_text

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
        #
        # WT-605 — THE TWO PATHS NOW AGREE ON CONTENT, NOT YET ON SHAPE.
        #     Since the live path started gating on the pause flag, both paths summarise the
        #     same words: the saved transcript never held what was said during a pause, and now
        #     neither does the accumulator. What only the live path can say is WHERE the gaps
        #     were — it watched them happen, and emits `format_pause_marker` for each one.
        #
        #     Here there is nothing to emit from. A pause arrives as a jump in startTimeMs and
        #     is indistinguishable from a room that simply went quiet, so a marker inferred from
        #     the gap would be a guess, and a wrong one every time a meeting paused for thought.
        #     Left unmarked deliberately rather than approximated.
        #
        #     To close it, the transcript service would have to return the pause windows it
        #     already knows about — it is the component that skips segments while paused (see
        #     TranscriptRedisConsumerService) — as e.g. `pauseWindows: [{startMs, endMs}]` on
        #     GET /api/v1/transcripts/by-room/{roomId}. Given that, this method feeds them
        #     through the same `format_pause_marker` the live path uses and the two paths
        #     produce identical transcripts. Nothing else here needs to change.
        lines = [
            format_transcript_line(
                int(segment.get("startTimeMs") or 0),
                str(segment.get("speakerName") or "Unknown speaker"),
                spoken,
            )
            for segment in segments
            if (spoken := _segment_text(segment))
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
                # Carried on the failure too. The backend drops a failed result without writing
                # anything, so this changes no behaviour today — but a result that omitted it
                # would read as canonical, and the next person to give the failure path a
                # side effect would inherit a silent misroute.
                delivery=request.delivery,
                error=error,
                requested_template_key=_requested_template(request),
                summary_language=normalize_language_code(request.summary_language),
            ).to_redis(),
        )

    async def _cleanup(self) -> None:
        # `_cleanup` is the hook BaseWorker actually calls on shutdown. Named `cleanup`,
        # this method would simply never run and the HTTP client would leak on every restart.
        if self._transcript_client is not None:
            await self._transcript_client.aclose()
            self._transcript_client = None


def _segment_text(segment: dict[str, Any]) -> str:
    """The wording to summarise: the clean line when the transcript has one, raw otherwise.

    WT-716. The stored segment carries both — `originalText` is the raw record billing,
    retranscribe and corrections work from, and `cleanText` is that line with its fillers and
    stutters deleted. The three states the backend distinguishes (TranscriptSegmentDto) are all
    meaningful and all answered here:

        null   NOT CLEANED — an older row, an older producer, or a line a human has corrected
               since. The raw wording is the only wording there is, exactly as before WT-716.
        ""     FILLER ONLY. The Clean view hides the line, and so does this: there is nothing in
               it for a summary to rest on, and a transcript of "Ummm" lines reads to a model
               like a meeting where people said nothing.
        text   The clean line.

    NO ANCHOR MOVES. A citation is the `startTimeMs` this row was stored with, printed by
    `format_transcript_line` and checked back by `summary_grounding` — read from the row, never
    recomputed from the text, so changing WHICH of a row's two texts is shown cannot move it. A
    skipped filler-only line simply takes its own moment out of the set the model may cite,
    which is the point: it was never shown that line.

    `.strip()` for the same reason the raw read always did — whitespace is not content.
    """
    clean = segment.get("cleanText")
    if isinstance(clean, str):
        return clean.strip()
    return str(segment.get("originalText") or "").strip()


def _requested_template(request: SummaryRequestMessage) -> str:
    """The template the request asked for, as the backend spells it. See SummaryResultMessage."""
    return (request.template_key or "general").strip().lower()
