"""WT-881: WarpBot could not look up the latest meeting, and could hang on "Running...".

TWO DEFECTS, ONE REPORT
    1. Every read of `/translation-rooms/history` omitted `workspaceId`, which
       GetTranslationRoomHistoryAsync requires — so the backend answered 400 and WarpBot said
       "could not look up recent meetings" to every question about a meeting. The same omission
       silently emptied the meeting allowlist semantic_search scopes transcripts to.
    2. The chat client ran on the OpenAI SDK's 600s read timeout. A stream that stopped sending
       mid-answer left `async for event in stream` waiting, with no `failed` ever published.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import AsyncOpenAI

from ai_assistant_worker.chat_tools import ToolContext, _list_recent_meetings, _visible_meeting_ids
from ai_assistant_worker.chat_worker import (
    STREAM_STALLED_MESSAGE,
    ChatAssistantWorker,
    openai_timeout,
)
from shared.config import ChatAssistantSettings
from shared.schemas import ChatRequestMessage

HISTORY_PATH = "/api/v1/translation-rooms/history"


def _history_client(status_code: int = 200, rooms: list[Any] | None = None) -> AsyncMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"rooms": rooms or []}
    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    return client


def _ctx(translation_room_client: AsyncMock, workspace_id: str = "ws-1") -> ToolContext:
    return ToolContext(
        workspace_id=workspace_id,
        user_id="user-1",
        bearer_token="Bearer test-token",
        workspace_client=AsyncMock(),
        transcript_client=AsyncMock(),
        translation_room_client=translation_room_client,
        openai_client=None,
        model="gpt-4.1",
        redis=MagicMock(),
    )


# ── 1. History is workspace-scoped ───────────────────────────────────────────────────────────


class TestHistoryCallsCarryTheWorkspace:
    async def test_list_recent_meetings_sends_workspace_id(self) -> None:
        client = _history_client(
            rooms=[{"room": {"id": "room-1", "title": "Weekly sync", "status": "ENDED"}}]
        )

        result = json.loads(await _list_recent_meetings(_ctx(client), {"query": "sync"}))

        assert result[0]["id"] == "room-1"
        path = client.get.await_args.args[0]
        params = client.get.await_args.kwargs["params"]
        assert path == HISTORY_PATH
        # The backend DTO field is WorkspaceId; ASP.NET binds the query key case-insensitively,
        # and camelCase is what every other caller of this route sends.
        assert params["workspaceId"] == "ws-1"
        assert params["search"] == "sync"
        assert params["pageSize"] == 5

    async def test_visible_meeting_ids_sends_workspace_id(self) -> None:
        client = _history_client(rooms=[{"room": {"id": "room-1"}}, {"room": {"id": "room-2"}}])

        ids = await _visible_meeting_ids(_ctx(client))

        assert ids == ["room-1", "room-2"]
        assert client.get.await_args.args[0] == HISTORY_PATH
        assert client.get.await_args.kwargs["params"]["workspaceId"] == "ws-1"

    async def test_no_workspace_is_explained_not_requested(self) -> None:
        """A platform-scope turn forces workspace_id to "". Calling the route with it would only
        buy a 400 that reads as an outage; the tool says what is actually true instead."""
        client = _history_client()

        result = json.loads(await _list_recent_meetings(_ctx(client, workspace_id=""), {}))

        client.get.assert_not_awaited()
        assert result["reason"] == "no_workspace"
        assert "workspace" in result["error"].lower()

    async def test_no_workspace_scopes_semantic_search_to_nothing(self) -> None:
        client = _history_client(rooms=[{"room": {"id": "room-1"}}])

        assert await _visible_meeting_ids(_ctx(client, workspace_id="")) == []
        client.get.assert_not_awaited()


# ── 2. A stalled stream fails the turn instead of hanging ────────────────────────────────────


def test_timeout_defaults_bound_connect_and_silence() -> None:
    timeout = openai_timeout(ChatAssistantSettings())

    assert timeout.connect == 10.0
    assert timeout.read == 90.0


def test_timeout_is_configurable_from_the_environment(monkeypatch: Any) -> None:
    monkeypatch.setenv("ASSISTANT_CHAT_OPENAI_READ_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ASSISTANT_CHAT_OPENAI_CONNECT_TIMEOUT_SECONDS", "5")

    timeout = openai_timeout(ChatAssistantSettings())

    assert (timeout.connect, timeout.read) == (5.0, 45.0)


async def test_load_model_gives_the_chat_client_the_explicit_timeout() -> None:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = ChatAssistantSettings(api_key="sk-test", openai_read_timeout_seconds=12)
    worker.logger = MagicMock()
    for name in (
        "_workspace_client",
        "_assistant_client",
        "_transcript_client",
        "_translation_room_client",
        "_billing_client",
        "_auth_client",
        "_manifest_task",
    ):
        setattr(worker, name, None)
    # load_model also starts the tool-manifest publisher on the worker's Redis client.
    worker.redis = MagicMock()
    worker.redis.set_with_ttl = AsyncMock()

    await worker.load_model()
    try:
        assert worker._openai is not None
        timeout = worker._openai.timeout
        assert isinstance(timeout, httpx.Timeout)
        assert (timeout.connect, timeout.read) == (10.0, 12.0)
    finally:
        await worker._cleanup()
        await worker._openai.close()


async def _stalling_server() -> tuple[asyncio.AbstractServer, int]:
    """An OpenAI look-alike that starts a Responses stream, sends one delta, and goes silent.

    A real socket rather than a fake iterator, because the thing under test is that httpx's read
    timeout actually reaches `async for event in stream` through the SDK.
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Drain the request head and body; the content is irrelevant.
        head = await reader.readuntil(b"\r\n\r\n")
        length = 0
        for line in head.decode("latin-1").split("\r\n"):
            if line.lower().startswith("content-length:"):
                length = int(line.split(":", 1)[1])
        if length:
            await reader.readexactly(length)

        delta = {
            "type": "response.output_text.delta",
            "delta": "Your latest meeting was",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": 1,
            "logprobs": [],
        }
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Connection: close\r\n\r\n"
            + f"event: response.output_text.delta\ndata: {json.dumps(delta)}\n\n".encode()
        )
        await writer.drain()
        # ...and then nothing: no further delta, no response.completed, no close.
        try:
            await asyncio.sleep(3600)
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def test_a_stalled_stream_publishes_failed_instead_of_hanging() -> None:
    server, port = await _stalling_server()
    settings = ChatAssistantSettings(
        model="gpt-4.1",
        chunk_flush_chars=1,
        openai_read_timeout_seconds=0.5,
        openai_connect_timeout_seconds=2,
    )
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = settings
    worker.logger = MagicMock()
    worker.redis = MagicMock()
    worker._openai = AsyncOpenAI(
        api_key="sk-test",
        base_url=f"http://127.0.0.1:{port}/v1",
        timeout=openai_timeout(settings),
        max_retries=0,
    )
    for name in (
        "_workspace_client",
        "_assistant_client",
        "_transcript_client",
        "_translation_room_client",
        "_billing_client",
        "_auth_client",
    ):
        setattr(worker, name, AsyncMock())

    published: list[dict[str, Any]] = []

    async def publish(request: Any, **kwargs: Any) -> None:
        published.append(kwargs)

    worker._publish_result = publish

    # Platform scope keeps the turn off web search, MCP discovery and platform settings, none of
    # which this is about — the stream is shared by both scopes.
    request = ChatRequestMessage(
        request_id="req-881",
        conversation_id="conv-1",
        workspace_id="",
        user_id="user-1",
        bearer_token="Bearer test-token",
        scope="platform",
    )
    data = {key.encode(): value.encode() for key, value in request.to_redis().items()}

    try:
        # The bound is the assertion: before WT-881 this sat for the SDK's 600s.
        await asyncio.wait_for(worker.process(b"1-0", data), timeout=15)
    finally:
        server.close()
        await worker._openai.close()

    types = [event["type_"] for event in published]
    assert "chunk" in types, "the delta before the stall still reached the reader"
    assert types[-1] == "failed"
    assert "completed" not in types
    assert published[-1]["content"] == STREAM_STALLED_MESSAGE
