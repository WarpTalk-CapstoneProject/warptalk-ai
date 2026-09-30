"""The platform-scope WarpBot: its read-only admin tools, and the scope boundary around them.

Three properties matter more than any single figure:

1. SCOPE ISOLATION. A platform turn is offered the platform tools and nothing else — no
   semantic_search, no documents, no plugins, no web search — and a workspace turn is never
   offered a platform tool. That is what keeps one scope's retrieval out of the other's answer.
2. READ-ONLY, AS THE CALLER. Every call is a GET carrying the caller's own token; a 403 is
   reported as not_authorized, never as an empty result or a zero.
3. FILLER IS NOT A VALUE. The chat model fills every property ("" / 0 / [] / "none"), so each
   optional enum has a "none" member and filler must reach the API as "not given".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_tools import TOOLS, ToolContext
from ai_assistant_worker.chat_worker import ChatAssistantWorker
from ai_assistant_worker.citations import SourceRegistry
from ai_assistant_worker.platform_tools import (
    PLATFORM_TOOLS,
    PLATFORM_TOOLS_BY_NAME,
    build_platform_system_prompt,
    resolve_insights_period,
)
from shared.config import ChatAssistantSettings
from shared.schemas import ChatRequestMessage, ChatResultMessage

WORKSPACE_ID = "0198f0d0-0000-7000-8000-000000000001"

Routes = dict[str, Callable[[httpx.Request], httpx.Response]]


class _Recorder:
    """Every request any client made, across all services."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def client(self, base_url: str, routes: Routes) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            # Longest prefix first, so /workspaces/by-slug beats /workspaces.
            for path in sorted(routes, key=len, reverse=True):
                if request.url.path.startswith(path):
                    return routes[path](request)
            return httpx.Response(404, json={})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def _json(payload: Any, status: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: httpx.Response(status, json=payload)


def _ctx(recorder: _Recorder, **routes: Routes) -> ToolContext:
    return ToolContext(
        workspace_id="",
        user_id="admin-1",
        bearer_token="Bearer admin-token",
        workspace_client=recorder.client("http://workspace", routes.get("workspace", {})),
        transcript_client=recorder.client("http://transcript", {}),
        translation_room_client=recorder.client("http://room", routes.get("room", {})),
        billing_client=recorder.client("http://billing", routes.get("billing", {})),
        auth_client=recorder.client("http://auth", routes.get("auth", {})),
        assistant_client=recorder.client("http://assistant", routes.get("assistant", {})),
        openai_client=MagicMock(),
        model="gpt-5.6-luna",
        redis=MagicMock(),
        citations=SourceRegistry(),
    )


_WORKSPACE_ROW = {
    "id": WORKSPACE_ID,
    "name": "Acme",
    "slug": "acme",
    "status": "active",
    "memberCount": 12,
    "owner": {"fullName": "Ana Owner", "email": "ana@acme.test"},
    "createdAt": "2026-01-01T00:00:00Z",
    "lastActivityAt": "2026-09-20T00:00:00Z",
}

_WORKSPACE_DETAIL = {**_WORKSPACE_ROW, "internalMemberCount": 10, "externalMemberCount": 2}


# ── read-only, as the caller ──────────────────────────────────────────────────────────────────


def _filler_arguments(tool_name: str) -> dict[str, Any]:
    """What gpt-5.6-luna actually sends: every property present, blank-shaped."""
    arguments: dict[str, Any] = {}
    for name, schema in PLATFORM_TOOLS_BY_NAME[tool_name].parameters["properties"].items():
        if "enum" in schema:
            arguments[name] = "none" if "none" in schema["enum"] else schema["enum"][0]
        elif schema.get("type") == "integer":
            arguments[name] = 0
        elif schema.get("type") == "array":
            arguments[name] = []
        else:
            arguments[name] = ""
    if tool_name == "get_workspace_summary":
        arguments["workspace"] = "acme"
    return arguments


@pytest.mark.parametrize("tool_name", sorted(PLATFORM_TOOLS_BY_NAME))
async def test_every_platform_tool_only_reads_and_forwards_the_callers_token(
    tool_name: str,
) -> None:
    recorder = _Recorder()
    ok = _json({"items": [], "total": 0})
    everything: Routes = {"/": ok}
    ctx = _ctx(
        recorder,
        workspace={
            "/api/v1/admin/workspaces/by-slug/": _json(_WORKSPACE_DETAIL),
            "/api/v1/admin/platform-health": _json({"monitoringAvailable": False}),
            "/": ok,
        },
        room=everything,
        billing=everything,
        auth=everything,
        assistant={"/api/v1/assistant/plugins/catalog": _json([])},
    )

    result = await PLATFORM_TOOLS_BY_NAME[tool_name].handler(ctx, _filler_arguments(tool_name))

    json.loads(result)  # always JSON, never a bare string
    assert recorder.requests, "a platform tool that calls nothing answers from nothing"
    assert {r.method for r in recorder.requests} == {"GET"}
    assert all(r.headers.get("Authorization") == "Bearer admin-token" for r in recorder.requests)


async def test_a_403_is_not_authorized_never_an_empty_directory() -> None:
    recorder = _Recorder()
    ctx = _ctx(recorder, workspace={"/api/v1/admin/workspaces": _json({}, 403)})

    payload = json.loads(
        await PLATFORM_TOOLS_BY_NAME["search_workspaces"].handler(
            ctx, _filler_arguments("search_workspaces")
        )
    )

    assert payload["error"] == "not_authorized"
    assert "workspaces" not in payload


# ── filler is not a value ─────────────────────────────────────────────────────────────────────

#: Enums that are the whole point of the call, so they have no "not applicable" value.
_MANDATORY_ENUMS = {("get_platform_insights", "period"), ("lookup_billing", "view")}


def test_every_optional_enum_offers_none() -> None:
    """WT-399: an optional enum without "none" is an unwinnable loop for a model that fills."""
    for tool in PLATFORM_TOOLS:
        for name, schema in tool.parameters["properties"].items():
            if "enum" in schema and (tool.name, name) not in _MANDATORY_ENUMS:
                assert "none" in schema["enum"], f"{tool.name}.{name} has no 'none' member"


def test_schemas_are_strict_and_fully_required() -> None:
    for tool in PLATFORM_TOOLS:
        properties = set(tool.parameters["properties"])
        assert set(tool.parameters.get("required", [])) == properties, tool.name
        assert tool.parameters.get("additionalProperties") is False, tool.name


async def test_filler_reaches_the_api_as_not_given() -> None:
    recorder = _Recorder()
    ctx = _ctx(recorder, workspace={"/api/v1/admin/workspaces": _json({"items": [], "total": 0})})

    await PLATFORM_TOOLS_BY_NAME["search_workspaces"].handler(
        ctx, {"query": "", "status": "none", "sort": "none", "limit": 0}
    )

    params = recorder.requests[0].url.params
    assert "search" not in params
    assert "sort" not in params
    assert params["status"] == "all"
    assert params["pageSize"] == "10"


# ── citations point at the admin page ─────────────────────────────────────────────────────────


async def test_each_workspace_cites_its_admin_page() -> None:
    recorder = _Recorder()
    ctx = _ctx(
        recorder,
        workspace={"/api/v1/admin/workspaces": _json({"items": [_WORKSPACE_ROW], "total": 1})},
    )

    payload = json.loads(
        await PLATFORM_TOOLS_BY_NAME["search_workspaces"].handler(
            ctx, {"query": "acme", "status": "active", "sort": "none", "limit": 5}
        )
    )

    row = payload["workspaces"][0]
    assert row["admin_link"] == "/admin/workspaces/acme"
    assert ctx.citations is not None
    cited = ctx.citations.cited(f"Acme has 12 members [{row['marker']}].")
    assert [(s.kind, s.ref) for s in cited] == [("admin", "/admin/workspaces/acme")]
    assert payload["admin_link"] == "/admin/workspaces?q=acme&status=active"


async def test_low_credit_subscriptions_resolve_names_and_link_each_workspace() -> None:
    recorder = _Recorder()
    ctx = _ctx(
        recorder,
        billing={
            "/api/v1/admin/subscriptions": _json(
                {
                    "items": [
                        {
                            "workspaceId": WORKSPACE_ID,
                            "planName": "Pro",
                            "status": "active",
                            "serviceState": "low_balance",
                            "creditsRemaining": 1200,
                        }
                    ],
                    "total": 1,
                }
            )
        },
        workspace={f"/api/v1/admin/workspaces/{WORKSPACE_ID}": _json(_WORKSPACE_DETAIL)},
    )

    payload = json.loads(
        await PLATFORM_TOOLS_BY_NAME["lookup_billing"].handler(
            ctx,
            {
                "view": "subscriptions",
                "status": "active",
                "sort": "credits_asc",
                "plan_slug": "",
                "workspace": "",
                "limit": 0,
            },
        )
    )

    row = payload["subscriptions"][0]
    assert row["workspace"] == "Acme"
    assert row["low_balance"] is True
    assert row["admin_link"] == "/admin/workspaces/acme"
    sub_request = next(r for r in recorder.requests if r.url.path == "/api/v1/admin/subscriptions")
    assert sub_request.url.params["sort"] == "credits_asc"
    assert payload["admin_link"] == "/admin/subscriptions?status=active&sort=credits_asc"


async def test_an_ambiguous_workspace_name_is_asked_about_not_guessed() -> None:
    recorder = _Recorder()
    two = {"items": [_WORKSPACE_ROW, {**_WORKSPACE_ROW, "slug": "acme-2", "id": "x"}], "total": 2}
    ctx = _ctx(
        recorder,
        workspace={
            "/api/v1/admin/workspaces/by-slug/": _json({}, 404),
            "/api/v1/admin/workspaces": _json(two),
        },
    )

    payload = json.loads(
        await PLATFORM_TOOLS_BY_NAME["get_workspace_summary"].handler(ctx, {"workspace": "Acme"})
    )

    assert payload["error"] == "ambiguous"
    assert [c["slug"] for c in payload["candidates"]] == ["acme", "acme-2"]
    assert not any("/api/v1/subscriptions" in r.url.path for r in recorder.requests)


async def test_system_health_reports_meeting_success_and_failing_stages() -> None:
    recorder = _Recorder()
    health = {
        "monitoringAvailable": True,
        "alerts": [{"name": "SttErrors", "severity": "critical", "state": "firing"}],
        "stageOutcomes": [
            {"stage": "stt", "ok": 90, "failed": 10, "deadLettered": 2, "successRate": 0.9},
        ],
        "meetings": {"window": "24h", "successRate": 0.8, "ended": 10, "reachedLive": 8},
    }
    ctx = _ctx(
        recorder,
        workspace={"/api/v1/admin/platform-health": _json(health)},
        room={"/": _json({})},
    )

    payload = json.loads(await PLATFORM_TOOLS_BY_NAME["get_system_health"].handler(ctx, {}))

    assert payload["meeting_success"]["success_rate"] == 0.8
    assert payload["meeting_success"]["admin_link"] == "/admin/health"
    stage = payload["monitoring"]["pipeline_stage_outcomes_last_hour"][0]
    assert (stage["stage"], stage["failed"], stage["dead_lettered"]) == ("stt", 10, 2)
    assert payload["monitoring"]["firing_alerts"][0]["name"] == "SttErrors"


async def test_unreadable_monitoring_is_not_an_outage() -> None:
    recorder = _Recorder()
    ctx = _ctx(
        recorder,
        workspace={"/api/v1/admin/platform-health": _json({"monitoringAvailable": False})},
        room={"/": _json({})},
    )

    payload = json.loads(await PLATFORM_TOOLS_BY_NAME["get_system_health"].handler(ctx, {}))

    assert payload["monitoring"]["monitoring_available"] is False
    assert "NOTHING" in payload["monitoring"]["note"]
    assert "meeting_success" not in payload


# ── periods ───────────────────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 9, 24, 3, 0, tzinfo=UTC)  # 10:00 in Ho Chi Minh City


def test_this_month_starts_at_local_midnight_and_links_the_insights_page() -> None:
    window = resolve_insights_period("this_month", None, None, now=_NOW)
    # 1 Sep 00:00 +07:00 is 31 Aug 17:00Z.
    assert window["from"] == "2026-08-31T17:00:00Z"
    assert window["to"] == "2026-09-24T03:00:00Z"
    assert window["link"] == "/admin?period=month"


def test_last_month_is_the_whole_previous_calendar_month() -> None:
    window = resolve_insights_period("last_month", None, None, now=_NOW)
    assert (window["from"], window["to"]) == ("2026-07-31T17:00:00Z", "2026-08-31T17:00:00Z")
    assert window["link"] == "/admin?period=month&month=2026-08"


def test_an_unusable_custom_range_is_refused_not_guessed() -> None:
    assert resolve_insights_period("custom", "", "", now=_NOW)["error"] == "invalid_period"
    assert (
        resolve_insights_period("custom", "2026-09-10", "2026-09-01", now=_NOW)["error"]
        == "invalid_period"
    )


async def test_revenue_this_month_vs_last_asks_for_the_previous_month_comparison() -> None:
    recorder = _Recorder()
    metrics = {"metrics": [{"id": "revenue", "value": 10, "previous": 8, "unit": "vnd"}]}
    ctx = _ctx(recorder, billing={"/api/v1/admin/billing/insights": _json(metrics)})

    payload = json.loads(
        await PLATFORM_TOOLS_BY_NAME["get_platform_insights"].handler(
            ctx,
            {
                "period": "this_month",
                "from_date": "",
                "to_date": "",
                "compare": "previous_month",
                "sections": ["billing"],
            },
        )
    )

    params = recorder.requests[0].url.params
    assert params["compare"] == "previousMonth"
    assert params["tz"] == "Asia/Ho_Chi_Minh"
    assert payload["billing"]["metrics"][0]["previous"] == 8
    assert payload["admin_link"].endswith("compare=previousMonth")


# ── the scope boundary in the agent loop ──────────────────────────────────────────────────────


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _answer(text: str) -> list[Any]:
    return [
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(output=[SimpleNamespace(type="message")]),
        ),
    ]


def _worker() -> Any:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = ChatAssistantSettings(model="gpt-5.6-luna", max_tokens=256)
    worker.chat_settings.web_search_enabled = True
    worker.logger = MagicMock()
    worker._openai = MagicMock()
    worker._openai.responses.create = AsyncMock(side_effect=[_FakeStream(_answer("ok"))])
    worker._publish_result = AsyncMock()
    worker._load_dynamic_mcp_tools = AsyncMock(return_value=[])
    return worker


def _request(scope: str) -> Any:
    return SimpleNamespace(
        request_id="req-1",
        conversation_id="conv-1",
        workspace_id="" if scope == "platform" else WORKSPACE_ID,
        bearer_token="Bearer t",
        origin="assistant",
        scope=scope,
        page_context_json="",
        mentions_json="",
        images_json="",
        disabled_plugin_keys_json="",
    )


def _offered(worker: Any) -> set[str]:
    tools = worker._openai.responses.create.await_args_list[0].kwargs["tools"]
    return {tool.get("name") or tool["type"] for tool in tools}


async def test_a_platform_turn_is_offered_the_platform_tools_and_nothing_else() -> None:
    worker = _worker()

    await worker._run_agent_loop(_request("platform"), [], MagicMock())

    assert _offered(worker) == set(PLATFORM_TOOLS_BY_NAME)
    assert "semantic_search" not in _offered(worker)
    assert "web_search" not in _offered(worker)
    # Plugin discovery is workspace-scoped; a platform turn must not even ask.
    worker._load_dynamic_mcp_tools.assert_not_awaited()
    instructions = worker._openai.responses.create.await_args_list[0].kwargs["instructions"]
    assert "ADMIN PORTAL" in instructions


async def test_a_workspace_turn_is_never_offered_a_platform_tool() -> None:
    worker = _worker()

    await worker._run_agent_loop(_request("workspace"), [], MagicMock())

    offered = _offered(worker)
    assert not offered & set(PLATFORM_TOOLS_BY_NAME)
    assert "semantic_search" in offered


def test_the_two_tool_sets_share_no_name() -> None:
    assert not {tool.name for tool in TOOLS} & set(PLATFORM_TOOLS_BY_NAME)


async def test_a_platform_turn_runs_with_no_workspace_even_if_the_stream_names_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Belt and braces: a stray workspace_id on a platform request scopes nothing."""
    worker = _worker()
    for name in (
        "_workspace_client",
        "_assistant_client",
        "_transcript_client",
        "_translation_room_client",
        "_billing_client",
        "_auth_client",
    ):
        setattr(worker, name, MagicMock())
    worker.redis = MagicMock()
    seen: list[ToolContext] = []

    async def fake_loop(request: Any, history: Any, ctx: ToolContext) -> tuple[str, list[Any]]:
        seen.append(ctx)
        return "ok", []

    monkeypatch.setattr(worker, "_run_agent_loop", fake_loop)
    message = ChatRequestMessage(
        request_id="r",
        conversation_id="c",
        workspace_id=WORKSPACE_ID,
        user_id="u",
        scope="platform",
    )

    await worker.process(b"1-0", {k.encode(): v.encode() for k, v in message.to_redis().items()})

    assert seen[0].workspace_id == ""


async def test_the_result_echoes_the_scope_so_the_backend_finalizes_the_right_store() -> None:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.publish = AsyncMock()

    await ChatAssistantWorker._publish_result(worker, _request("platform"), type_="completed")

    fields = worker.publish.await_args.args[2]
    assert fields["scope"] == "platform"


def test_scope_round_trips_and_defaults_to_workspace() -> None:
    legacy = {
        "request_id": "r",
        "conversation_id": "c",
        "workspace_id": WORKSPACE_ID,
        "user_id": "u",
    }
    assert ChatRequestMessage.from_redis(legacy).scope == "workspace"
    assert ChatRequestMessage.from_redis(legacy).is_platform is False

    platform = ChatRequestMessage(
        request_id="r", conversation_id="c", workspace_id="", user_id="u", scope="platform"
    )
    assert ChatRequestMessage.from_redis(platform.to_redis()).is_platform is True
    result = ChatResultMessage(request_id="r", conversation_id="c", type="chunk", scope="platform")
    assert ChatResultMessage.from_redis(result.to_redis()).scope == "platform"


def test_the_platform_prompt_says_it_cannot_read_workspace_content_or_write() -> None:
    prompt = build_platform_system_prompt()
    assert "cannot read workspace content" in prompt
    assert "read-only" in prompt
    for tool in PLATFORM_TOOLS:
        assert tool.name in prompt


def test_chat_worker_imports_the_platform_set() -> None:
    # The wiring the scope boundary depends on; a rename here would silently offer nothing.
    assert chat_worker_module.PLATFORM_TOOLS is PLATFORM_TOOLS
