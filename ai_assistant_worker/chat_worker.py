"""Chat Assistant Worker — global "Ask WarpTalk" tool-calling agent.

Pipeline:
    Redis Stream (assistant:chat_requests) — its own consumer group, independent of
    AIAssistantWorker's per-meeting summarization pipeline (which listens on stt:results)
    → OpenAI streaming chat completion with function-calling tools (chat_tools.py)
    → dispatch any requested tool calls to sibling .NET services
    → loop until a final answer, publishing chunk / tool-call / completed / failed
      events to assistant:chat_results as they happen so AssistantService (.NET) can
      relay them to the browser over SignalR in near-real-time.

Runs alongside AIAssistantWorker in the same process (see __main__.py) — both are
lightweight Redis-stream consumers with no need for a separate container.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import httpx
from openai import AsyncOpenAI

from ai_assistant_worker.chat_templates import PERSONA, build_system_prompt, resolve_template
from ai_assistant_worker.chat_tools import TOOLS, TOOLS_BY_NAME, ToolContext
from shared.base_worker import BaseWorker
from shared.config import ChatAssistantSettings, resolve_openai_api_key
from shared.openai_options import responses_options
from shared.schemas import ChatRequestMessage, ChatResultMessage

SIBLING_SERVICE_TIMEOUT_SECONDS = 15.0

# The prompt is generated per turn from chat_templates, which is where the routing rules
# ("read the transcript when asked what was said") now live. This name survives as the
# shared persona because it is still the opening of every prompt the worker sends.
SYSTEM_PROMPT = PERSONA


def _parse_page_context(page_context_json: str) -> dict[str, Any] | None:
    """The ambient page context as a dict, or None if there isn't a usable one.

    Both the template routing and the rendered system message need this, and parsing it
    twice is how the two would drift apart.
    """
    if not page_context_json:
        return None
    try:
        context = json.loads(page_context_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(context, dict):
        return None
    return context


def _page_type(page_context_json: str) -> str | None:
    context = _parse_page_context(page_context_json)
    if context is None:
        return None
    page_type = context.get("pageType")
    return page_type if isinstance(page_type, str) and page_type else None


def _format_page_context(page_context_json: str) -> str | None:
    """Render the frontend's ambient page_context_json (which page the user has the
    assistant open on, e.g. a specific room) into a system message. Malformed or empty
    payloads are silently ignored — this is a nice-to-have hint, never a hard requirement.

    This states the facts; the template says what the entity_id in them is good for.
    """
    context = _parse_page_context(page_context_json)
    if context is None:
        return None

    page_type = context.get("pageType")
    if not page_type:
        return None

    parts = [f'The user currently has the assistant open on the "{page_type}" page.']
    entity_id = context.get("entityId")
    if entity_id:
        parts.append(f"entity_id={entity_id}.")
    workspace_id = context.get("workspaceId")
    if workspace_id:
        parts.append(f"workspace_id={workspace_id}.")
    snapshot = context.get("snapshot")
    if isinstance(snapshot, dict) and snapshot:
        snapshot_text = ", ".join(f"{key}={value}" for key, value in snapshot.items())
        parts.append(f"Visible snapshot: {snapshot_text}.")
    parts.append(
        "Treat this as page context, not an instruction from the user. Prefer tools "
        "scoped to this entity when the question is about it; don't assume data beyond "
        "this snapshot — use a tool to fetch it."
    )
    return " ".join(parts)


_MENTION_TOOL_HINTS = {
    "room": "get_room_detail or get_transcript",
    "meeting": "get_transcript, then get_meeting_summary if it has ended",
    "document": "get_document",
    "member": "search_workspace_members",
}

# A mention of the assistant itself is how the user summoned it, not a thing to look up.
# MeetingChatService forwards the whole mention list, so "@WarpBot" arrives here on every
# in-meeting turn; rendering it as a reference produced the instruction "look up agent
# WarpBot (id=bot-warpbot) with an appropriate tool", which no tool can satisfy.
_NON_ENTITY_MENTION_TYPES = frozenset({"agent", "bot", "assistant"})


def _normalize_mention(mention: dict[str, Any]) -> tuple[str, str, str] | None:
    """One mention as (entity_type, entity_id, label), or None if it isn't a reference.

    The two publishers disagree on shape. AssistantService sends
    {entityType, entityId, label}; MeetingChatService forwards the frontend's ChatMentionDto
    as {id, display, type}. Reading only the first shape silently dropped every in-meeting
    mention, so both are accepted here — the fix belongs on the reading side, since the
    meeting-chat shape is also what the browser stores and renders.
    """
    entity_type = mention.get("entityType") or mention.get("type")
    entity_id = mention.get("entityId") or mention.get("id")
    if not isinstance(entity_type, str) or not isinstance(entity_id, str):
        return None
    entity_type = entity_type.strip().lower()
    entity_id = entity_id.strip()
    if not entity_type or not entity_id:
        return None
    if entity_type in _NON_ENTITY_MENTION_TYPES:
        return None
    label = mention.get("label") or mention.get("display") or entity_id
    return entity_type, entity_id, str(label)


def _format_mentions(mentions_json: str) -> str | None:
    """Render the frontend's explicit @mention list into a system message. Unlike ambient
    page context, a mention is the user's own deliberate act of attaching a specific entity
    to this message — treat it as the primary subject, not just background. Malformed or
    empty payloads are silently ignored.
    """
    if not mentions_json:
        return None
    try:
        mentions = json.loads(mentions_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(mentions, list) or not mentions:
        return None

    lines: list[str] = []
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        normalized = _normalize_mention(mention)
        if normalized is None:
            continue
        entity_type, entity_id, label = normalized
        tool_hint = _MENTION_TOOL_HINTS.get(entity_type, "an appropriate tool")
        lines.append(f'- {entity_type} "{label}" (id={entity_id}) — look it up with {tool_hint}.')

    if not lines:
        return None

    return (
        "The user explicitly attached these references to their message with @mention — "
        "treat them as the primary subject of the question, not just background context:\n"
        + "\n".join(lines)
    )


#: Meetings are scheduled by people in Vietnam, and "9 giờ sáng mai" means 9am there. UTC+7 has
#: no DST, so a fixed offset is exact rather than an approximation.
WORKSPACE_TIMEZONE = timezone(timedelta(hours=7), "ICT")

#: WT-474. Hard ceilings on what one turn may carry, enforced here as well as in the browser and
#: in AssistantService. The browser limit is a courtesy to the user; this one is what protects the
#: worker, because a Redis Stream field and an OpenAI request both have real limits and a caller
#: that skipped the UI would otherwise reach them.
MAX_ATTACHMENTS_PER_TURN = 4
#: ~7MB of base64, a little over 5MB of file. Beyond this the request is likelier to be rejected by
#: OpenAI than to be answered.
MAX_ATTACHMENT_DATA_URL_CHARS = 7_000_000

#: Document types the Responses API takes as `input_file`. Deliberately a WHITELIST: an arbitrary
#: mime type is either rejected by OpenAI (a confusing 400 for the user) or, worse, silently
#: ignored — and a user whose attachment was quietly dropped reads the answer as the model lying.
DOCUMENT_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/json",
    }
)


def _attach_attachments(
    conversation: list[dict[str, Any]],
    attachments_json: str,
    logger: Any,
) -> None:
    """WT-474: fold pasted/uploaded files into the LAST user turn, in place.

    Somebody debugging asks "what is wrong with this screen" and the screen is the question; the
    same person asks "does this contract allow X" and the PDF is the question. Before this the only
    way to ask was to describe the attachment in words, which is exactly the work the model could
    have done.

    WHY THE LAST USER TURN AND NOT A NEW MESSAGE. `/v1/responses` takes multimodal content as a
    content ARRAY on one message; a separate attachment-only message would arrive as a turn with no
    question in it, and the model answers the text it was given.

    WHY NOTHING IS PERSISTED. Attachments ride along with this request and are never written into
    `history_json`, so a follow-up question cannot see them. That is a deliberate limit:

      - A file stored against a conversation becomes a new kind of workspace content, and every
        kind of workspace content has to answer to the visibility model WT-463 is still defining.
        Adding one before that model exists is how a surface ends up outside it - the same way
        `semantic_search` ended up bypassing the document ACL.
      - History is replayed on every turn. A 5MB data URL in it would be re-sent, and re-billed,
        for the rest of the conversation.

    So the contract is: an attachment answers the question it was sent with. The UI says so.

    Unusable entries are DROPPED rather than failing the turn. The user's question is still a
    question, and refusing to answer it because one attachment was malformed serves nobody - but
    each drop is logged, because a silently ignored file looks like a model that cannot read.
    """
    if not attachments_json:
        return

    try:
        raw = json.loads(attachments_json)
    except json.JSONDecodeError:
        logger.warning("chat_attachments_unparseable")
        return

    if not isinstance(raw, list):
        logger.warning("chat_attachments_not_a_list")
        return

    parts_to_add: list[dict[str, Any]] = []
    for original in raw[:MAX_ATTACHMENTS_PER_TURN]:
        # Accepts a bare data-URL string as well as the object form. The first cut of this feature
        # published plain strings, and a message already sitting on the stream when the worker
        # restarts must not be dropped for using the older shape.
        entry = (
            {"dataUrl": original, "name": "", "mimeType": ""}
            if isinstance(original, str)
            else original
        )
        if not isinstance(entry, dict):
            logger.warning("chat_attachment_rejected", reason="not_an_object")
            continue

        data_url = entry.get("dataUrl")
        if not isinstance(data_url, str) or not data_url.startswith("data:"):
            logger.warning("chat_attachment_rejected", reason="not_a_data_url")
            continue
        if len(data_url) > MAX_ATTACHMENT_DATA_URL_CHARS:
            logger.warning("chat_attachment_rejected", reason="too_large", chars=len(data_url))
            continue

        # The mime type is read off the data URL itself, never from the caller-supplied field: the
        # two can disagree, and the bytes are the only side that decides how OpenAI reads them.
        mime = data_url[5 : data_url.find(";")] if ";" in data_url else ""

        if mime.startswith("image/"):
            parts_to_add.append({"type": "input_image", "image_url": data_url})
            continue

        if mime in DOCUMENT_MIME_TYPES:
            name = entry.get("name")
            parts_to_add.append(
                {
                    "type": "input_file",
                    # Required by the API, and also the only handle the model has for referring to
                    # one document among several.
                    "filename": name if isinstance(name, str) and name else "attachment",
                    "file_data": data_url,
                }
            )
            continue

        logger.warning("chat_attachment_rejected", reason="unsupported_type", mime=mime)

    if len(raw) > MAX_ATTACHMENTS_PER_TURN:
        logger.warning(
            "chat_attachments_truncated", received=len(raw), kept=MAX_ATTACHMENTS_PER_TURN
        )

    if not parts_to_add:
        return

    # The last USER turn, not the last turn: a tool result or an assistant reply can be last, and
    # hanging an attachment off either puts it where the model does not read it as the question.
    target = next(
        (turn for turn in reversed(conversation) if turn.get("role") == "user"),
        None,
    )
    if target is None:
        logger.warning("chat_attachments_dropped", reason="no_user_turn")
        return

    text = target.get("content")
    parts: list[dict[str, Any]] = []
    if isinstance(text, str) and text.strip():
        parts.append({"type": "input_text", "text": text})
    elif isinstance(text, list):
        # Already multimodal - keep whatever is there and append.
        parts.extend(cast(list[dict[str, Any]], text))

    parts.extend(parts_to_add)
    target["content"] = parts
    logger.info("chat_attachments_attached", count=len(parts_to_add))


def _now_message(now: datetime | None = None) -> str:
    """Tell the model what "today" is.

    NOTHING did. Not the persona, not the templates, not the tool schemas — the worker never put
    a date in front of the model at all. So `create_meeting`, whose whole job is scheduling, could
    not turn "hôm nay", "ngày mai" or "thứ Sáu tuần sau" into the YYYY-MM-DD it requires, and the
    conversation deadlocked: it asked for a date, the user answered "Hôm nay", and it asked again.
    Its own prompt suggested "ngày mai lúc 09:30" as an example of what to say — an example it
    could not then act on.

    A model cannot know this and must not guess it: guessing produces a meeting scheduled in the
    wrong year, which is worse than the loop because it looks like it worked.
    """
    now = now or datetime.now(WORKSPACE_TIMEZONE)
    return (
        "CURRENT TIME\n"
        f"Right now it is {now:%A, %d %B %Y, %H:%M} in Vietnam (UTC+7), "
        f"which is {now:%Y-%m-%d} in ISO form.\n"
        "Resolve every relative date the user gives you — today, tomorrow, tonight, next Friday, "
        "cuối tuần này — against this, yourself. Never ask the user to convert a date to "
        "YYYY-MM-DD; that is arithmetic you can do and they cannot be expected to. Only ask when "
        "the date is genuinely ambiguous, and then offer the candidates you resolved.\n"
        "Times the user gives with no timezone are Vietnam time."
    )


class ChatAssistantWorker(BaseWorker):
    """Global assistant worker — free-form Q&A with tool-calling, independent of any meeting."""

    worker_name = "assistant-chat"
    input_stream = "assistant:chat_requests"
    consumer_group = "assistant-chat-workers"

    def __init__(
        self,
        chat_settings: ChatAssistantSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.chat_settings = chat_settings or ChatAssistantSettings()
        self._openai: AsyncOpenAI | None = None
        self._workspace_client: httpx.AsyncClient | None = None
        self._transcript_client: httpx.AsyncClient | None = None
        self._translation_room_client: httpx.AsyncClient | None = None
        # Only get_platform_analytics uses these two, and only with the caller's own token —
        # every path behind them is gated by the platform admin policy server-side.
        self._billing_client: httpx.AsyncClient | None = None
        self._auth_client: httpx.AsyncClient | None = None

    async def load_model(self) -> None:
        api_key = resolve_openai_api_key(self.chat_settings.api_key)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for ChatAssistantWorker")

        self._openai = AsyncOpenAI(api_key=api_key)
        self._workspace_client = httpx.AsyncClient(
            base_url=self.chat_settings.workspace_service_url,
            timeout=SIBLING_SERVICE_TIMEOUT_SECONDS,
        )
        self._transcript_client = httpx.AsyncClient(
            base_url=self.chat_settings.transcript_service_url,
            timeout=SIBLING_SERVICE_TIMEOUT_SECONDS,
        )
        self._translation_room_client = httpx.AsyncClient(
            base_url=self.chat_settings.translation_room_service_url,
            timeout=SIBLING_SERVICE_TIMEOUT_SECONDS,
        )
        self._billing_client = httpx.AsyncClient(
            base_url=self.chat_settings.billing_service_url,
            timeout=SIBLING_SERVICE_TIMEOUT_SECONDS,
        )
        self._auth_client = httpx.AsyncClient(
            base_url=self.chat_settings.auth_service_url,
            timeout=SIBLING_SERVICE_TIMEOUT_SECONDS,
        )
        self.logger.info("chat_assistant_ready", model=self.chat_settings.model)

    async def _cleanup(self) -> None:
        for client in (
            self._workspace_client,
            self._transcript_client,
            self._translation_room_client,
            self._billing_client,
            self._auth_client,
        ):
            if client is not None:
                await client.aclose()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = ChatRequestMessage.from_redis(cast(Any, data))

        if (
            self._workspace_client is None
            or self._transcript_client is None
            or self._translation_room_client is None
            or self._openai is None
        ):
            # Answer before raising. This check sits OUTSIDE the try/except below, so raising
            # here published nothing at all — and a caller who mentions @WarpBot and receives
            # complete silence cannot tell "the assistant is misconfigured" from "the mention
            # was never seen". Every exit from process() must leave an answer behind.
            await self._publish_result(
                request,
                type_="failed",
                content="WarpBot is not available right now.",
            )
            raise RuntimeError("ChatAssistantWorker is not initialized — call load_model() first")

        try:
            history: list[dict[str, str]] = (
                json.loads(request.history_json) if request.history_json else []
            )
        except json.JSONDecodeError:
            history = []

        tool_context = ToolContext(
            workspace_id=request.workspace_id,
            user_id=request.user_id,
            bearer_token=request.bearer_token,
            workspace_client=self._workspace_client,
            transcript_client=self._transcript_client,
            translation_room_client=self._translation_room_client,
            billing_client=self._billing_client,
            auth_client=self._auth_client,
            openai_client=self._openai,
            model=self.chat_settings.model,
            redis=self.redis,
        )

        try:
            final_text, tool_call_log = await self._run_agent_loop(request, history, tool_context)
            await self._publish_result(
                request,
                type_="completed",
                content=final_text,
                tool_calls_json=json.dumps(tool_call_log) if tool_call_log else "",
            )
        except Exception as exc:
            self.logger.exception("chat_turn_failed", request_id=request.request_id)
            await self._publish_result(
                request,
                type_="failed",
                content=str(exc) or "The assistant could not generate a reply.",
            )

    async def _run_agent_loop(
        self,
        request: ChatRequestMessage,
        history: list[dict[str, str]],
        tool_context: ToolContext,
    ) -> tuple[str, list[dict[str, Any]]]:
        # /v1/responses, not /v1/chat/completions. A reasoning model refuses function
        # tools on chat completions — gpt-5.6-luna answers every such request with
        #
        #   400 Function tools with reasoning_effort are not supported ... use
        #   /v1/responses or set reasoning_effort to 'none'
        #
        # and setting the effort to none would discard the reasoning that is the only
        # reason to run a reasoning model here. This endpoint is the one that takes both.
        # Every shape below was verified against the live API, not inferred from docs.

        # Responses carries the system prompt as `instructions` rather than as a leading
        # message, so the three system-role messages are joined into one.
        template = resolve_template(
            origin=request.origin,
            page_type=_page_type(request.page_context_json),
        )
        self.logger.info(
            "chat_template_resolved",
            request_id=request.request_id,
            template=template.key,
            origin=request.origin,
        )
        instructions_parts = [build_system_prompt(template), _now_message()]
        page_context_message = _format_page_context(request.page_context_json)
        if page_context_message:
            instructions_parts.append(page_context_message)
        mentions_message = _format_mentions(request.mentions_json)
        if mentions_message:
            instructions_parts.append(mentions_message)
        instructions = "\n\n".join(instructions_parts)

        conversation: list[dict[str, Any]] = [
            {"role": turn.get("role"), "content": turn.get("content")} for turn in history
        ]
        _attach_attachments(conversation, request.images_json, self.logger)

        tool_schemas: list[dict[str, Any]] = [t.to_openai_schema() for t in TOOLS]

        # OpenAI's HOSTED web search, not a ChatTool: the model calls it and OpenAI runs it
        # server-side, so it has no handler here and never reaches the dispatch below — that loop
        # filters on `type == "function_call"`, and a hosted call comes back as its own item type.
        # The answer text it produces streams in on the same response, which is why adding it
        # needs nothing else.
        #
        # It bills per call, which is the only reason it is a switch: ASSISTANT_CHAT_WEB_SEARCH_
        # ENABLED=false turns it off without a rebuild.
        if self.chat_settings.web_search_enabled:
            tool_schemas.append({"type": "web_search"})
        tool_call_log: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(self.chat_settings.max_tool_iterations):
            buffer = ""
            full_text = ""
            output_items: list[Any] = []

            assert self._openai is not None, "OpenAI client must be initialized"
            stream = await self._openai.responses.create(
                model=self.chat_settings.model,
                **responses_options(
                    self.chat_settings.model,
                    self.chat_settings.max_tokens,
                    self.chat_settings.temperature,
                ),
                instructions=instructions,
                input=cast(Any, conversation),
                tools=cast(Any, tool_schemas),
                stream=True,
            )

            async for event in stream:
                etype = getattr(event, "type", "")

                # Text arrives as response.output_text.delta. Streaming it out as it
                # lands is what keeps the widget feeling live.
                if etype == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    buffer += delta
                    full_text += delta
                    if len(buffer) >= self.chat_settings.chunk_flush_chars:
                        await self._publish_result(request, type_="chunk", content=buffer)
                        buffer = ""

                # Tool arguments also stream, but are NOT accumulated here on purpose:
                # response.completed carries every output item whole, so reading them
                # from there removes a class of partial-JSON bugs the chat-completions
                # version had to guard against by hand.
                elif etype == "response.completed":
                    response = getattr(event, "response", None)
                    output_items = list(getattr(response, "output", []) or [])

            if buffer:
                await self._publish_result(request, type_="chunk", content=buffer)

            function_calls = [
                item for item in output_items if getattr(item, "type", "") == "function_call"
            ]
            if not function_calls:
                # A turn that produced a message rather than a call is the final answer.
                final_text = full_text
                break

            for call in function_calls:
                call_id = getattr(call, "call_id", None) or f"call_{uuid.uuid4().hex}"
                tool_name = getattr(call, "name", "") or ""
                raw_arguments = getattr(call, "arguments", "") or ""
                await self._publish_result(request, type_="tool_call_started", tool_name=tool_name)

                tool = TOOLS_BY_NAME.get(tool_name)
                if tool is None:
                    result_json = json.dumps({"error": f"Unknown tool '{tool_name}'."})
                    status = "failed"
                else:
                    try:
                        arguments = json.loads(raw_arguments) if raw_arguments else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    try:
                        result_json = await tool.handler(tool_context, arguments)
                        status = "completed"
                    except Exception:
                        self.logger.exception("tool_execution_failed", tool=tool_name)
                        result_json = json.dumps({"error": "The tool failed to execute."})
                        status = "failed"

                await self._publish_result(
                    request, type_="tool_call_completed", tool_name=tool_name, tool_status=status
                )

                # ask_user is the one tool whose OUTPUT is a UI, not text for the model. The
                # questions go out on their own event so the client can render a card rather
                # than trying to find them inside an assistant message.
                #
                # Deliberately NOT blocking: pausing this loop until a human answers would hold a
                # worker slot open for as long as somebody takes to read, and a reconnect would
                # strand the turn with no way back. The card is fire-and-forget; the answer
                # arrives as an ordinary message on the next turn, which is also why the user can
                # ignore it and type something else entirely.
                if tool_name == "ask_user" and status == "completed":
                    await self._publish_result(
                        request,
                        type_="question",
                        tool_name=tool_name,
                        tool_calls_json=raw_arguments,
                    )
                # The call and its result are fed back as a pair of typed input items —
                # the Responses equivalent of the assistant/tool message pair.
                conversation.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": raw_arguments,
                    }
                )
                conversation.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result_json,
                    }
                )
                tool_call_log.append(
                    {
                        "tool": tool_name,
                        "arguments": raw_arguments,
                        "result": result_json,
                        "status": status,
                    }
                )
        else:
            # Hit max_tool_iterations while the model still wanted another tool.
            final_text = (
                final_text
                or "I wasn't able to finish looking that up — please try rephrasing your question."
            )

        return final_text, tool_call_log

    async def _publish_result(
        self,
        request: ChatRequestMessage,
        type_: str,
        content: str = "",
        tool_name: str = "",
        tool_status: str = "",
        tool_calls_json: str = "",
    ) -> None:
        result = ChatResultMessage(
            request_id=request.request_id,
            conversation_id=request.conversation_id,
            type=type_,
            origin=request.origin,
            content=content,
            tool_name=tool_name,
            tool_status=tool_status,
            tool_calls_json=tool_calls_json,
        )
        await self.publish("assistant:chat_results", request.conversation_id, result.to_redis())
