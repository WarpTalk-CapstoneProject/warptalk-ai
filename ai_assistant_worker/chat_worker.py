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
from typing import Any

import httpx

from ai_assistant_worker.chat_tools import TOOLS, TOOLS_BY_NAME, ToolContext
from shared.base_worker import BaseWorker
from shared.config import ChatAssistantSettings, resolve_openai_api_key
from shared.schemas import ChatRequestMessage, ChatResultMessage

SIBLING_SERVICE_TIMEOUT_SECONDS = 15.0

SYSTEM_PROMPT = (
    "You are WarpTalk AI, the assistant embedded in the WarpTalk real-time speech "
    "translation platform. Answer clearly and concisely. Use the available tools to look "
    "up real workspace data (members, terminology, recent meetings) whenever the user asks "
    "about something tool-shaped rather than guessing. If a tool returns no results, say so "
    "honestly instead of making something up."
)


def _format_page_context(page_context_json: str) -> str | None:
    """Render the frontend's ambient page_context_json (which page the user has the
    assistant open on, e.g. a specific room) into a system message. Malformed or empty
    payloads are silently ignored — this is a nice-to-have hint, never a hard requirement.
    """
    if not page_context_json:
        return None
    try:
        context = json.loads(page_context_json)
    except json.JSONDecodeError:
        return None
    if not isinstance(context, dict):
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
    "document": "get_document",
    "member": "search_workspace_members",
}


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
        entity_type = mention.get("entityType")
        entity_id = mention.get("entityId")
        if not entity_type or not entity_id:
            continue
        label = mention.get("label") or entity_id
        tool_hint = _MENTION_TOOL_HINTS.get(entity_type, "an appropriate tool")
        lines.append(f'- {entity_type} "{label}" (id={entity_id}) — look it up with {tool_hint}.')

    if not lines:
        return None

    return (
        "The user explicitly attached these references to their message with @mention — "
        "treat them as the primary subject of the question, not just background context:\n"
        + "\n".join(lines)
    )


class ChatAssistantWorker(BaseWorker):
    """Global assistant worker — free-form Q&A with tool-calling, independent of any meeting."""

    worker_name = "assistant-chat"
    input_stream = "assistant:chat_requests"
    consumer_group = "assistant-chat-workers"

    def __init__(self, chat_settings: ChatAssistantSettings | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.chat_settings = chat_settings or ChatAssistantSettings()
        self._openai = None
        self._workspace_client: httpx.AsyncClient | None = None
        self._transcript_client: httpx.AsyncClient | None = None
        self._translation_room_client: httpx.AsyncClient | None = None

    async def load_model(self) -> None:
        from openai import AsyncOpenAI

        api_key = resolve_openai_api_key(self.chat_settings.api_key)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for ChatAssistantWorker")

        self._openai = AsyncOpenAI(api_key=api_key)
        self._workspace_client = httpx.AsyncClient(
            base_url=self.chat_settings.workspace_service_url, timeout=SIBLING_SERVICE_TIMEOUT_SECONDS
        )
        self._transcript_client = httpx.AsyncClient(
            base_url=self.chat_settings.transcript_service_url, timeout=SIBLING_SERVICE_TIMEOUT_SECONDS
        )
        self._translation_room_client = httpx.AsyncClient(
            base_url=self.chat_settings.translation_room_service_url, timeout=SIBLING_SERVICE_TIMEOUT_SECONDS
        )
        self.logger.info("chat_assistant_ready", model=self.chat_settings.model)

    async def _cleanup(self) -> None:
        for client in (self._workspace_client, self._transcript_client, self._translation_room_client):
            if client is not None:
                await client.aclose()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = ChatRequestMessage.from_redis(data)

        try:
            history: list[dict[str, str]] = json.loads(request.history_json) if request.history_json else []
        except json.JSONDecodeError:
            history = []

        tool_context = ToolContext(
            workspace_id=request.workspace_id,
            user_id=request.user_id,
            bearer_token=request.bearer_token,
            workspace_client=self._workspace_client,
            transcript_client=self._transcript_client,
            translation_room_client=self._translation_room_client,
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
                request, type_="failed", content=str(exc) or "The assistant could not generate a reply."
            )

    async def _run_agent_loop(
        self,
        request: ChatRequestMessage,
        history: list[dict[str, str]],
        tool_context: ToolContext,
    ) -> tuple[str, list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        page_context_message = _format_page_context(request.page_context_json)
        if page_context_message:
            messages.append({"role": "system", "content": page_context_message})
        mentions_message = _format_mentions(request.mentions_json)
        if mentions_message:
            messages.append({"role": "system", "content": mentions_message})
        messages.extend({"role": turn.get("role"), "content": turn.get("content")} for turn in history)

        tool_schemas = [t.to_openai_schema() for t in TOOLS]
        tool_call_log: list[dict[str, Any]] = []
        final_text = ""

        for _ in range(self.chat_settings.max_tool_iterations):
            buffer = ""
            full_text = ""
            tool_calls_acc: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None

            stream = await self._openai.chat.completions.create(
                model=self.chat_settings.model,
                temperature=self.chat_settings.temperature,
                max_tokens=self.chat_settings.max_tokens,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
                stream=True,
            )

            async for chunk in stream:
                choice = chunk.choices[0]
                delta = choice.delta

                if delta.content:
                    buffer += delta.content
                    full_text += delta.content
                    if len(buffer) >= self.chat_settings.chunk_flush_chars:
                        await self._publish_result(request, type_="chunk", content=buffer)
                        buffer = ""

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        acc = tool_calls_acc.setdefault(tc.index, {"id": None, "name": None, "arguments": ""})
                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

            if buffer:
                await self._publish_result(request, type_="chunk", content=buffer)

            if finish_reason != "tool_calls" or not tool_calls_acc:
                final_text = full_text
                break

            ordered_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"] or f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in ordered_calls
                ],
            })

            for call in ordered_calls:
                call_id = call["id"] or f"call_{uuid.uuid4().hex}"
                tool_name = call["name"] or ""
                await self._publish_result(request, type_="tool_call_started", tool_name=tool_name)

                tool = TOOLS_BY_NAME.get(tool_name)
                if tool is None:
                    result_json = json.dumps({"error": f"Unknown tool '{tool_name}'."})
                    status = "failed"
                else:
                    try:
                        arguments = json.loads(call["arguments"]) if call["arguments"] else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    try:
                        result_json = await tool.handler(tool_context, arguments)
                        status = "completed"
                    except Exception:
                        self.logger.exception("tool_execution_failed", tool=tool_name)
                        result_json = json.dumps({"error": "The tool failed to execute."})
                        status = "failed"

                await self._publish_result(request, type_="tool_call_completed", tool_name=tool_name, tool_status=status)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": result_json})
                tool_call_log.append({
                    "tool": tool_name,
                    "arguments": call["arguments"],
                    "result": result_json,
                    "status": status,
                })
        else:
            # Hit max_tool_iterations without a non-tool-call finish.
            final_text = final_text or "I wasn't able to finish looking that up — please try rephrasing your question."

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
            content=content,
            tool_name=tool_name,
            tool_status=tool_status,
            tool_calls_json=tool_calls_json,
        )
        await self.publish("assistant:chat_results", request.conversation_id, result.to_redis())
