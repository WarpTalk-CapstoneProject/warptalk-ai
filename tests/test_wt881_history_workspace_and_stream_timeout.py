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

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_tools import ToolContext, _list_recent_meetings, _visible_meeting_ids

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
