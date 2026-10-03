"""Optional string arguments the model filled with nothing never reach the plugin's server.

Prod, 2-3 Oct 2026: 12 of 20 Linear calls failed with tool_error. The model fills every property,
and Linear's list_issues refuses customView="" (minLength 1), so the model sent " ", then "all",
then "x" - each a failed round before an answer, if one came at all.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx

from ai_assistant_worker.chat_worker import ChatAssistantWorker
from ai_assistant_worker.mcp_tools import drop_blank_optional_arguments

LIST_ISSUES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "team": {"type": "string"},
        "cursor": {"type": "string"},
        "customView": {"type": "string", "minLength": 1},
        "priority": {"type": "number"},
        "includeArchived": {"type": "boolean"},
        "title": {"type": "string"},
    },
    "required": ["title"],
}


def test_blank_and_whitespace_optional_strings_are_dropped() -> None:
    kept = drop_blank_optional_arguments(
        {"team": "", "cursor": "   ", "customView": " ", "title": "x"}, LIST_ISSUES_SCHEMA
    )

    assert kept == {"title": "x"}


def test_zero_false_and_real_strings_are_values_not_filler() -> None:
    arguments = {"team": "FPT", "priority": 0, "includeArchived": False, "title": "x"}

    assert drop_blank_optional_arguments(arguments, LIST_ISSUES_SCHEMA) == arguments


def test_a_required_blank_is_left_for_the_server_to_judge() -> None:
    assert drop_blank_optional_arguments({"title": ""}, LIST_ISSUES_SCHEMA) == {"title": ""}


def test_no_schema_means_nothing_is_required() -> None:
    assert drop_blank_optional_arguments({"a": "", "b": "y"}, None) == {"b": "y"}
    assert drop_blank_optional_arguments({"a": ""}, {"required": "not-a-list"}) == {}


async def test_the_handler_sends_the_server_only_what_the_model_actually_meant() -> None:
    posted: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/mcp/tools"):
            return httpx.Response(
                200,
                json=[
                    {
                        "name": "list_issues",
                        "pluginKey": "linear",
                        "label": "List issues",
                        "description": "",
                        "effect": "read",
                        "parameters": LIST_ISSUES_SCHEMA,
                    }
                ],
            )
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success", "content": []})

    client = httpx.AsyncClient(base_url="http://assistant", transport=httpx.MockTransport(respond))
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.logger = MagicMock()
    worker._publish_result = AsyncMock()
    request = SimpleNamespace(
        request_id="req-1",
        conversation_id="conv-1",
        workspace_id="ws-1",
        bearer_token="Bearer t",
        disabled_plugin_keys_json="",
    )
    context = SimpleNamespace(assistant_client=client)

    tools = await worker._load_dynamic_mcp_tools(request, context)  # type: ignore[arg-type]
    tool = next(t for t in tools if t.name == "list_issues")
    await tool.handler(
        context,  # type: ignore[arg-type]
        {"team": "", "cursor": "", "customView": " ", "priority": 0, "title": "bug"},
    )

    assert posted[0]["arguments"] == {"priority": 0, "title": "bug"}
    await client.aclose()
