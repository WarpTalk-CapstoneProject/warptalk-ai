"""Contract tests for the chat assistant's Responses-API agent loop.

The loop had no test coverage at all when it was migrated off /v1/chat/completions,
which is how a model change reached production and broke every "Ask WarpTalk" message
with a 400. These lock the wire contract that migration depends on, so the next person
to touch it finds out from a test rather than from prod.

Every shape asserted here was observed against the live API before being written down —
event names, the flat tool schema, and the function_call / function_call_output pair.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_tools import ChatTool
from ai_assistant_worker.chat_worker import ChatAssistantWorker
from shared.config import ChatAssistantSettings
from shared.openai_options import responses_options


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def _completed(*items: Any) -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(output=list(items)),
    )


def _function_call(name: str, arguments: str, call_id: str = "call_abc") -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def _message_item() -> SimpleNamespace:
    return SimpleNamespace(type="message")


class _FakeStream:
    """One turn's worth of Responses events."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _build_worker(
    turns: list[list[Any]], model: str = "gpt-5.6-luna"
) -> tuple[Any, list[dict[str, Any]]]:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = ChatAssistantSettings(model=model, max_tokens=256, temperature=0.4)
    worker.logger = MagicMock()

    streams = [_FakeStream(events) for events in turns]
    worker._openai = MagicMock()
    worker._openai.responses.create = AsyncMock(side_effect=streams)

    published: list[dict[str, Any]] = []

    async def publish(request: Any, **kwargs: Any) -> None:
        published.append(kwargs)

    worker._publish_result = publish
    return worker, published


def _request(
    *,
    origin: str = "assistant",
    page_context_json: str = "",
    mentions_json: str = "",
    # WT-474. Present with its real default rather than left off: the worker reads
    # `request.images_json` directly, so a stand-in that omits it passes here and would hide the
    # day the field stops being optional. ChatRequestMessage defaults it to "".
    images_json: str = "",
) -> Any:
    return SimpleNamespace(
        request_id="req-1",
        origin=origin,
        page_context_json=page_context_json,
        mentions_json=mentions_json,
        images_json=images_json,
    )


@pytest.fixture
def counting_tool(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a single fake tool and record what it was called with."""
    calls: list[dict[str, Any]] = []

    async def handler(ctx: Any, arguments: dict[str, Any]) -> str:
        calls.append(arguments)
        return json.dumps({"active_meetings": 3})

    tool = ChatTool(
        name="get_active_meeting_count",
        description="Count active meetings.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=handler,
    )
    monkeypatch.setattr(chat_worker_module, "TOOLS", [tool])
    monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", {tool.name: tool})
    return calls


class TestAgentLoop:
    async def test_text_only_turn_streams_and_finishes(self) -> None:
        worker, published = _build_worker(
            [[_text_delta("Hello "), _text_delta("there"), _completed(_message_item())]]
        )

        text, tool_log = await worker._run_agent_loop(_request(), [], MagicMock())

        assert text == "Hello there"
        assert tool_log == []
        assert [p["type_"] for p in published] == ["chunk"]
        assert published[0]["content"] == "Hello there"

    async def test_tool_call_is_executed_and_fed_back(
        self, counting_tool: list[dict[str, Any]]
    ) -> None:
        """The two-item feedback pair is what makes a second turn possible."""
        worker, published = _build_worker(
            [
                [_completed(_function_call("get_active_meeting_count", '{"workspace_id":"ws-1"}'))],
                [_text_delta("There are 3."), _completed(_message_item())],
            ]
        )

        text, tool_log = await worker._run_agent_loop(_request(), [], MagicMock())

        assert text == "There are 3."
        assert counting_tool == [{"workspace_id": "ws-1"}]
        assert [entry["tool"] for entry in tool_log] == ["get_active_meeting_count"]
        assert [entry["status"] for entry in tool_log] == ["completed"]
        assert [p["type_"] for p in published] == [
            "tool_call_started",
            "tool_call_completed",
            "chunk",
        ]

        # The second request must carry the call AND its result, as separate typed items.
        second_call = worker._openai.responses.create.await_args_list[1]
        conversation = second_call.kwargs["input"]
        types = [item.get("type") for item in conversation if isinstance(item, dict)]
        assert types == ["function_call", "function_call_output"]
        call_item, output_item = conversation[-2], conversation[-1]
        assert call_item["call_id"] == output_item["call_id"], "result must match its call"
        assert json.loads(output_item["output"]) == {"active_meetings": 3}

    async def test_unknown_tool_is_reported_not_raised(self) -> None:
        worker, published = _build_worker(
            [
                [_completed(_function_call("no_such_tool", "{}"))],
                [_text_delta("Sorry."), _completed(_message_item())],
            ]
        )

        text, tool_log = await worker._run_agent_loop(_request(), [], MagicMock())

        assert text == "Sorry."
        assert tool_log[0]["status"] == "failed"
        assert "error" in json.loads(tool_log[0]["result"])
        assert any(p["type_"] == "tool_call_completed" for p in published)

    async def test_system_prompt_travels_as_instructions(self) -> None:
        """Responses carries the system prompt out of band, not as a leading message."""
        worker, _ = _build_worker([[_text_delta("hi"), _completed(_message_item())]])

        await worker._run_agent_loop(_request(), [{"role": "user", "content": "hi"}], MagicMock())

        kwargs = worker._openai.responses.create.await_args.kwargs
        assert chat_worker_module.SYSTEM_PROMPT in kwargs["instructions"]
        assert all(item.get("role") != "system" for item in kwargs["input"])

    async def test_loop_stops_at_max_iterations_with_an_answer(
        self, counting_tool: list[dict[str, Any]]
    ) -> None:
        """A model that only ever asks for tools must still leave the user something."""
        forever = [
            [_completed(_function_call("get_active_meeting_count", "{}"))] for _ in range(10)
        ]
        worker, _ = _build_worker(forever)
        worker.chat_settings = ChatAssistantSettings(model="gpt-4o-mini", max_tool_iterations=3)

        text, tool_log = await worker._run_agent_loop(_request(), [], MagicMock())

        assert text, "the loop must not return an empty answer"
        assert len(tool_log) == 3


class TestResponsesOptions:
    def test_reasoning_model_gets_no_temperature(self) -> None:
        """Verified against the live API: gpt-5.6-luna 400s on temperature here too."""
        assert responses_options("gpt-5.6-luna", 256, 0.4) == {"max_output_tokens": 256}

    def test_other_models_keep_temperature(self) -> None:
        assert responses_options("gpt-4o-mini", 256, 0.4) == {
            "max_output_tokens": 256,
            "temperature": 0.4,
        }

    def test_cap_is_named_for_this_endpoint(self) -> None:
        """max_output_tokens, never max_tokens or max_completion_tokens."""
        options = responses_options("gpt-4o-mini", 128)
        assert "max_tokens" not in options
        assert "max_completion_tokens" not in options


class TestToolSchema:
    def test_schema_is_flat_for_responses(self) -> None:
        """/v1/responses takes name/description/parameters at the top level."""
        tool = ChatTool(
            name="demo", description="d", parameters={"type": "object"}, handler=AsyncMock()
        )
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["name"] == "demo"
        assert "function" not in schema, "nested chat-completions shape is rejected here"
