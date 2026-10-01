"""Metadata on each `tool_call_log` entry, which AssistantService records for Insights.

The log reaches the backend as `tool_calls_json` on the chat result. Its original keys (`tool`,
`arguments`, `result`, `status`) are stored verbatim for the UI and are left alone here. This
module adds what Insights counts: where the call came from, which plugin it belongs to, how it
ended, and how long it took.

Nothing here reads argument or result text into the new keys. Insights stores metadata only, and
a web search entry carries no query at all.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

ToolSource = Literal["builtin", "plugin", "web_search"]
ToolOutcome = Literal["ok", "error", "blocked", "needs_setup", "declined", "confirmation_required"]

#: `PluginConstants.ErrorCodes` grouped the way the web groups them (plugin-activity.ts) and as
#: the Wave 4 contract lists them. A code not named here is a plain error.
_BLOCKED_CODES = frozenset({"permission_denied", "tool_blocked", "workspace_tool_blocked"})
_DECLINED_CODES = frozenset({"access_denied"})
_NEEDS_SETUP_CODES = frozenset(
    {
        "missing_scope",
        "not_connected",
        "reauth_required",
        "api_key_required",
        "invalid_api_key",
        "plugin_not_installed",
        "connection_required",
        "provider_account_mismatch",
        "client_registration_unsupported",
    }
)
_CONFIRMATION_CODES = frozenset({"confirmation_required"})


def outcome_for_plugin_code(code: str | None) -> ToolOutcome:
    """The Insights bucket for a failed plugin call's error code."""
    normalized = (code or "").strip().lower()
    if normalized in _BLOCKED_CODES:
        return "blocked"
    if normalized in _DECLINED_CODES:
        return "declined"
    if normalized in _NEEDS_SETUP_CODES:
        return "needs_setup"
    if normalized in _CONFIRMATION_CODES:
        return "confirmation_required"
    return "error"


def _parse_object(result_json: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(result_json)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_tool_outcome(
    *,
    source: ToolSource,
    status: str,
    result_json: str,
    error_code: str | None = None,
) -> tuple[ToolOutcome, str | None]:
    """`(outcome, outcomeCode)` for one executed function call. Pure.

    ``status`` is the log's own `completed` / `failed`. ``error_code`` is what the loop knows
    when the handler never returned (the exception class, or ``unknown_tool``).

    A plugin call reads the execute result its handler returned: ``isSuccess: false`` with an
    ``errorCode`` from AssistantService, or the ``{"error", "status", "code"}`` the handler builds
    from an HTTP error. A built-in call is ``ok`` when its handler completed, unless the handler
    reported ``status: "failed"`` or answered with a bare ``error``.
    """
    if status != "completed":
        return "error", error_code or "failed"

    parsed = _parse_object(result_json)
    if parsed is None:
        return "ok", None

    if source == "plugin":
        if parsed.get("isSuccess") is False:
            code = parsed.get("errorCode")
            code = code if isinstance(code, str) and code.strip() else None
            return outcome_for_plugin_code(code), code
        if "error" in parsed and "isSuccess" not in parsed:
            code = parsed.get("code")
            if isinstance(code, str) and code.strip():
                return outcome_for_plugin_code(code), code
            http_status = parsed.get("status")
            return "error", f"http_{http_status}" if isinstance(http_status, int) else None
        return "ok", None

    handler_status = parsed.get("status")
    if handler_status == "failed":
        return "error", "failed"
    if "error" in parsed and not handler_status:
        return "error", "error"
    return "ok", None


def iso_utc(moment: datetime) -> str:
    """`2026-10-01T10:00:00.123Z`: UTC, milliseconds, a Z rather than an offset."""
    utc = moment.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"


def function_call_entry(
    *,
    tool: str,
    arguments: str,
    result: str,
    status: str,
    source: ToolSource,
    plugin_key: str | None,
    started_at: datetime,
    duration_seconds: float,
    error_code: str | None = None,
) -> dict[str, Any]:
    """One log entry for a function call the worker dispatched."""
    outcome, outcome_code = classify_tool_outcome(
        source=source, status=status, result_json=result, error_code=error_code
    )
    return {
        "tool": tool,
        "arguments": arguments,
        "result": result,
        "status": status,
        "source": source,
        "pluginKey": plugin_key if source == "plugin" else None,
        "outcome": outcome,
        "outcomeCode": outcome_code,
        "durationMs": max(0, int(round(duration_seconds * 1000))),
        "startedAt": iso_utc(started_at),
    }


def web_search_entry(item: Any, *, started_at: datetime) -> dict[str, Any]:
    """One log entry for an OpenAI-hosted `web_search_call` output item.

    The query is never copied: `arguments` and `result` stay empty. Its duration is not
    measurable from here, since OpenAI runs the search server-side.
    """
    item_status = str(getattr(item, "status", "") or "")
    failed = item_status == "failed"
    return {
        "tool": "web_search",
        "arguments": "",
        "result": "",
        "status": "failed" if failed else "completed",
        "source": "web_search",
        "pluginKey": None,
        "outcome": "error" if failed else "ok",
        "outcomeCode": "failed" if failed else None,
        "durationMs": None,
        "startedAt": iso_utc(started_at),
    }
