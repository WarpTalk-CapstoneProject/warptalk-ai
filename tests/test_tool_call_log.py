"""Wave 4: every `tool_call_log` entry carries the metadata Insights records.

The pure mapping is tested directly; the agent-loop tests check that each source (built-in,
plugin, hosted web search) lands in the log with the old keys intact and no query text.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_tools import ChatTool
from ai_assistant_worker.tool_call_log import (
    classify_tool_outcome,
    function_call_entry,
    iso_utc,
    outcome_for_plugin_code,
    web_search_entry,
)
from tests.test_chat_agent_loop import (
    _build_worker,
    _completed,
    _function_call,
    _message_item,
    _request,
    _text_delta,
)

NEW_KEYS = {"source", "pluginKey", "outcome", "outcomeCode", "durationMs", "startedAt"}
OLD_KEYS = {"tool", "arguments", "result", "status"}


class TestPluginCodeBuckets:
    @pytest.mark.parametrize(
        ("code", "bucket"),
        [
            ("permission_denied", "blocked"),
            ("tool_blocked", "blocked"),
            ("workspace_tool_blocked", "blocked"),
            ("access_denied", "declined"),
            ("missing_scope", "needs_setup"),
            ("not_connected", "needs_setup"),
            ("reauth_required", "needs_setup"),
            ("api_key_required", "needs_setup"),
            ("invalid_api_key", "needs_setup"),
            ("plugin_not_installed", "needs_setup"),
            ("connection_required", "needs_setup"),
            ("provider_account_mismatch", "needs_setup"),
            ("confirmation_required", "confirmation_required"),
            ("provider_unavailable", "error"),
            ("tool_error", "error"),
            ("something_new", "error"),
            (None, "error"),
        ],
    )
    def test_code_maps_to_bucket(self, code: str | None, bucket: str) -> None:
        assert outcome_for_plugin_code(code) == bucket


class TestClassifyToolOutcome:
    def test_builtin_completed_is_ok(self) -> None:
        assert classify_tool_outcome(
            source="builtin", status="completed", result_json='{"active_meetings": 3}'
        ) == ("ok", None)

    def test_builtin_exception_carries_class_name(self) -> None:
        assert classify_tool_outcome(
            source="builtin",
            status="failed",
            result_json='{"error": "The tool failed to execute."}',
            error_code="TimeoutError",
        ) == ("error", "TimeoutError")

    def test_builtin_handler_failed_status(self) -> None:
        assert classify_tool_outcome(
            source="builtin", status="completed", result_json='{"status": "failed"}'
        ) == ("error", "failed")

    def test_builtin_bare_error_result(self) -> None:
        assert classify_tool_outcome(
            source="builtin", status="completed", result_json='{"error": "No meeting found."}'
        ) == ("error", "error")

    def test_builtin_non_failure_status_is_ok(self) -> None:
        assert classify_tool_outcome(
            source="builtin", status="completed", result_json='{"status": "needs_more_information"}'
        ) == ("ok", None)

    def test_plugin_success(self) -> None:
        assert classify_tool_outcome(
            source="plugin", status="completed", result_json='{"isSuccess": true, "data": {}}'
        ) == ("ok", None)

    @pytest.mark.parametrize(
        ("code", "bucket"),
        [
            ("workspace_tool_blocked", "blocked"),
            ("access_denied", "declined"),
            ("missing_scope", "needs_setup"),
            ("confirmation_required", "confirmation_required"),
            ("provider_unavailable", "error"),
        ],
    )
    def test_plugin_execute_result_code(self, code: str, bucket: str) -> None:
        result = json.dumps({"isSuccess": False, "errorCode": code, "message": "x"})
        assert classify_tool_outcome(source="plugin", status="completed", result_json=result) == (
            bucket,
            code,
        )

    def test_plugin_http_error_with_code(self) -> None:
        result = json.dumps({"error": "Nope", "status": 403, "code": "permission_denied"})
        assert classify_tool_outcome(source="plugin", status="completed", result_json=result) == (
            "blocked",
            "permission_denied",
        )

    def test_plugin_http_error_without_code(self) -> None:
        result = json.dumps({"error": "Plugin tool failed.", "status": 502, "code": None})
        assert classify_tool_outcome(source="plugin", status="completed", result_json=result) == (
            "error",
            "http_502",
        )

    def test_plugin_failure_without_code(self) -> None:
        result = json.dumps({"isSuccess": False, "error": "invalid response"})
        assert classify_tool_outcome(source="plugin", status="completed", result_json=result) == (
            "error",
            None,
        )


class TestEntries:
    def test_function_call_entry_keeps_old_keys_and_adds_new(self) -> None:
        entry = function_call_entry(
            tool="create_meeting",
            arguments='{"title":"x"}',
            result='{"status":"created"}',
            status="completed",
            source="builtin",
            plugin_key=None,
            started_at=datetime(2026, 10, 1, 10, 0, 0, 123456, tzinfo=UTC),
            duration_seconds=0.1234,
        )
        assert set(entry) == OLD_KEYS | NEW_KEYS
        assert entry["arguments"] == '{"title":"x"}'
        assert entry["result"] == '{"status":"created"}'
        assert entry["durationMs"] == 123
        assert isinstance(entry["durationMs"], int)
        assert entry["startedAt"] == "2026-10-01T10:00:00.123Z"
        assert entry["pluginKey"] is None

    def test_iso_utc_converts_offsets(self) -> None:
        from datetime import timedelta, timezone

        moment = datetime(2026, 10, 1, 17, 0, 0, 5000, tzinfo=timezone(timedelta(hours=7)))
        assert iso_utc(moment) == "2026-10-01T10:00:00.005Z"

    def test_web_search_entry_has_no_query(self) -> None:
        item = SimpleNamespace(
            type="web_search_call",
            id="ws_1",
            status="completed",
            action=SimpleNamespace(type="search", query="secret roadmap"),
        )
        entry = web_search_entry(item, started_at=datetime(2026, 10, 1, tzinfo=UTC))
        assert entry == {
            "tool": "web_search",
            "arguments": "",
            "result": "",
            "status": "completed",
            "source": "web_search",
            "pluginKey": None,
            "outcome": "ok",
            "outcomeCode": None,
            "durationMs": None,
            "startedAt": "2026-10-01T00:00:00.000Z",
        }
        assert "secret" not in json.dumps(entry)

    def test_web_search_failed(self) -> None:
        item = SimpleNamespace(type="web_search_call", id="ws_2", status="failed")
        entry = web_search_entry(item, started_at=datetime(2026, 10, 1, tzinfo=UTC))
        assert entry["status"] == "failed"
        assert entry["outcome"] == "error"


@pytest.fixture
def two_builtins(monkeypatch: pytest.MonkeyPatch) -> None:
    async def ok(ctx: Any, arguments: dict[str, Any]) -> str:
        return json.dumps({"active_meetings": 3})

    async def boom(ctx: Any, arguments: dict[str, Any]) -> str:
        raise TimeoutError("slow")

    tools = [
        ChatTool(
            name=name,
            description=name,
            parameters={"type": "object", "properties": {}, "additionalProperties": False},
            handler=handler,
        )
        for name, handler in (("count_meetings", ok), ("explode", boom))
    ]
    monkeypatch.setattr(chat_worker_module, "TOOLS", tools)
    monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", {t.name: t for t in tools})


class TestAgentLoopLog:
    async def test_builtin_ok_and_error_entries(self, two_builtins: None) -> None:
        worker, _ = _build_worker(
            [
                [
                    _completed(
                        _function_call("count_meetings", "{}", call_id="c1"),
                        _function_call("explode", "{}", call_id="c2"),
                    )
                ],
                [_text_delta("Done."), _completed(_message_item())],
            ]
        )
        worker.chat_settings.web_search_enabled = False

        _, log = await worker._run_agent_loop(_request(), [], MagicMock())

        ok, err = log
        assert set(ok) == OLD_KEYS | NEW_KEYS
        assert (ok["source"], ok["outcome"], ok["outcomeCode"], ok["status"]) == (
            "builtin",
            "ok",
            None,
            "completed",
        )
        assert (err["source"], err["outcome"], err["outcomeCode"], err["status"]) == (
            "builtin",
            "error",
            "TimeoutError",
            "failed",
        )
        for entry in log:
            assert entry["pluginKey"] is None
            assert isinstance(entry["durationMs"], int)
            assert entry["startedAt"].endswith("Z")

    async def test_unknown_tool_is_builtin_error(self) -> None:
        worker, _ = _build_worker(
            [
                [_completed(_function_call("no_such_tool", "{}"))],
                [_text_delta("Sorry."), _completed(_message_item())],
            ]
        )
        worker.chat_settings.web_search_enabled = False

        _, log = await worker._run_agent_loop(_request(), [], MagicMock())

        assert (log[0]["source"], log[0]["outcome"], log[0]["outcomeCode"]) == (
            "builtin",
            "error",
            "unknown_tool",
        )

    @pytest.mark.parametrize(
        ("response", "outcome", "code"),
        [
            (httpx.Response(200, json={"isSuccess": True, "data": {}}), "ok", None),
            (
                httpx.Response(200, json={"isSuccess": False, "errorCode": "missing_scope"}),
                "needs_setup",
                "missing_scope",
            ),
            (
                httpx.Response(
                    200, json={"isSuccess": False, "errorCode": "workspace_tool_blocked"}
                ),
                "blocked",
                "workspace_tool_blocked",
            ),
            (
                httpx.Response(
                    200,
                    json={
                        "isSuccess": False,
                        "errorCode": "confirmation_required",
                        "confirmationToken": "tok",
                    },
                ),
                "confirmation_required",
                "confirmation_required",
            ),
            (
                httpx.Response(403, json={"message": "no", "code": "access_denied"}),
                "declined",
                "access_denied",
            ),
            (
                httpx.Response(200, json={"isSuccess": False, "errorCode": "tool_error"}),
                "error",
                "tool_error",
            ),
        ],
    )
    async def test_plugin_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        response: httpx.Response,
        outcome: str,
        code: str | None,
    ) -> None:
        monkeypatch.setattr(chat_worker_module, "TOOLS", [])
        monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", {})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "name": "google_calendar_create_event",
                            "pluginKey": "google_calendar",
                            "label": "Create event",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                )
            return response

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://a")
        worker, _ = _build_worker(
            [
                [_completed(_function_call("google_calendar_create_event", "{}"))],
                [_text_delta("Ok."), _completed(_message_item())],
            ]
        )
        worker.chat_settings.web_search_enabled = False
        try:
            _, log = await worker._run_agent_loop(
                _request(), [], SimpleNamespace(assistant_client=client, citations=None)
            )
        finally:
            await client.aclose()

        (entry,) = log
        assert set(entry) == OLD_KEYS | NEW_KEYS
        assert entry["tool"] == "google_calendar_create_event"
        assert entry["status"] == "completed"
        assert entry["source"] == "plugin"
        assert entry["pluginKey"] == "google_calendar"
        assert (entry["outcome"], entry["outcomeCode"]) == (outcome, code)
        assert isinstance(entry["durationMs"], int)

    async def test_web_search_entries_carry_no_query(self, two_builtins: None) -> None:
        search_item = SimpleNamespace(
            type="web_search_call",
            id="ws_1",
            status="completed",
            action=SimpleNamespace(type="search", query="private launch plans"),
        )
        failed_item = SimpleNamespace(
            type="web_search_call",
            id="ws_2",
            status="failed",
            action=SimpleNamespace(type="search", query="private budget"),
        )
        worker, _ = _build_worker(
            [
                [
                    SimpleNamespace(type="response.output_item.added", item=search_item),
                    _completed(search_item, _function_call("count_meetings", "{}")),
                ],
                [
                    _text_delta("Found it."),
                    _completed(failed_item, _message_item()),
                ],
            ]
        )
        worker._web_search_enabled = _async_true

        text, log = await worker._run_agent_loop(_request(), [], MagicMock(citations=None))

        assert text == "Found it."
        assert [(e["tool"], e["source"]) for e in log] == [
            ("web_search", "web_search"),
            ("count_meetings", "builtin"),
            ("web_search", "web_search"),
        ]
        first, _, second = log
        assert (first["status"], first["outcome"]) == ("completed", "ok")
        assert (second["status"], second["outcome"]) == ("failed", "error")
        for entry in (first, second):
            assert entry["arguments"] == "" and entry["result"] == ""
            assert entry["durationMs"] is None
            assert entry["pluginKey"] is None
        assert "private" not in json.dumps(log)


async def _async_true(request: Any) -> bool:
    return True
