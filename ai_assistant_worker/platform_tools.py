"""The platform-scope WarpBot: read-only admin tools for a system administrator.

WHY A SEPARATE TOOL SET, NOT MORE TOOLS IN THE WORKSPACE ONE
    Every workspace tool is scoped to `ctx.workspace_id` — semantic_search queries that
    workspace's collection, search_documents lists its files, create_meeting writes into it. A
    platform turn has no workspace, and the one guarantee that matters is that nothing from any
    workspace's retrieval reaches it (and nothing from the platform reaches a workspace chat). So
    a platform turn is offered THIS list and nothing else — no semantic_search, no documents, no
    plugins, no web search — and a workspace turn is never offered this list. The chat worker
    picks one list or the other from the request's `scope`; it never merges them.

WHO MAY USE IT
    Three independent gates, each sufficient on its own:
    1. AssistantService serves platform conversations only behind the system-admin policy, so a
       non-admin cannot create one or send a turn into one.
    2. Every endpoint these tools call carries the same policy server-side, and the tools send
       the CALLER'S OWN bearer token — never a service credential. A token that lost the role
       mid-conversation gets a 403, which is reported as `not_authorized`, never as zero.
    3. Nothing here writes. Every call is a GET.

WRITE ACTIONS
    None, deliberately. A future write (suspend a workspace, adjust credits) must go through the
    explicit confirmation pattern the plugin tools use — the tool returns a confirmation card and
    only a second, user-confirmed call performs the write — never a direct call from the model.

CITATIONS
    Every fact carries a marker for the admin page that shows it, registered as kind "admin"
    with the page's path as the ref. The model is told to cite each figure; the chips link back to
    /admin/... so the admin can check the number on the page it came from.

TOOL ARGUMENTS ARE ALWAYS FULLY FILLED
    gpt-5.6-luna does not omit optional properties; it fills them with "" / [] / 0. So every
    optional enum below has an explicit "none" member, and `_blank` treats "", "none", 0 and [] as
    "not given". See the WT-399 note on create_meeting for what happens without that.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo

import httpx

from ai_assistant_worker.chat_tools import ChatTool, ToolContext, _auth_headers, _cite
from shared.logger import get_logger

logger = get_logger("platform_tools")

#: The admins' calendar. The insights endpoints take the same zone as `tz`, so "this month" means
#: the same days here, on the server and on the Insights page the answer links to.
PLATFORM_TIME_ZONE = "Asia/Ho_Chi_Minh"

#: The server's own cap on an insights window (AdminComparisonRange.MaxSpanDays).
MAX_INSIGHTS_SPAN_DAYS = 366

DEFAULT_LIMIT = 10
MAX_LIMIT = 25

#: How many subscription rows get their workspace name looked up. The directory row carries only
#: the id, and a name is what the admin asked about — but each lookup is a round trip.
MAX_NAME_LOOKUPS = 10

NONE = "none"

INSIGHT_PERIODS = (
    "this_month",
    "last_month",
    "today",
    "last_7_days",
    "last_30_days",
    "last_6_months",
    "custom",
)
INSIGHT_SECTIONS = ("billing", "users", "workspaces", "meetings")
INSIGHT_COMPARE = (NONE, "previous_month")
WORKSPACE_STATUSES = (NONE, "active", "suspended", "deleted")
WORKSPACE_SORTS = (NONE, "created_desc", "created_asc", "name_asc", "members_desc", "updated_desc")
ACCOUNT_STATUSES = (NONE, "active", "locked", "unverified", "deactivated", "deleted")
ACCOUNT_SORTS = (NONE, "created_desc", "created_asc", "name_asc", "last_login_desc")
BILLING_VIEWS = ("subscriptions", "invoices", "snapshot")
SUBSCRIPTION_STATUSES = (NONE, "pending", "active", "cancelled", "expired", "suspended")
SUBSCRIPTION_SORTS = (NONE, "credits_asc", "period_end_asc", "period_end_desc", "created_desc")
AUDIT_RESULTS = (NONE, "succeeded", "failed")
PLUGIN_STATUSES = (NONE, "active", "retired")

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ── argument hygiene ──────────────────────────────────────────────────────────────────────────


def _blank(value: Any) -> bool:
    """Whether a tool argument means "not given" — see TOOL ARGUMENTS ARE ALWAYS FULLY FILLED."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return value == 0
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() == NONE
    if isinstance(value, list | tuple | dict):
        return len(value) == 0
    return False


def _text(arguments: dict[str, Any] | None, key: str) -> str | None:
    value = (arguments or {}).get(key)
    return None if _blank(value) else str(value).strip()


def _choice(arguments: dict[str, Any] | None, key: str, allowed: tuple[str, ...]) -> str | None:
    """An enum argument, or None for "none"/filler/anything outside the list."""
    value = _text(arguments, key)
    if value is None:
        return None
    lowered = value.lower()
    return lowered if lowered in allowed and lowered != NONE else None


def _limit(arguments: dict[str, Any] | None) -> int:
    value: Any = (arguments or {}).get("limit")
    try:
        number = DEFAULT_LIMIT if value is None or _blank(value) else int(value)
    except (TypeError, ValueError):
        number = DEFAULT_LIMIT
    return max(1, min(number, MAX_LIMIT))


def _is_uuid(value: str | None) -> bool:
    return bool(value and _UUID.match(value))


# ── HTTP ──────────────────────────────────────────────────────────────────────────────────────


def _not_authorized() -> dict[str, Any]:
    return {
        "error": "not_authorized",
        "message": (
            "This needs the platform administrator role, and the caller's token does not carry "
            "it (or has expired). Say so plainly; do not estimate any figure."
        ),
    }


async def _get(
    client: httpx.AsyncClient | None,
    path: str,
    ctx: ToolContext,
    params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    """One admin GET with the caller's own token. Returns (body, None) or (None, error).

    Every failure is NAMED. An empty list is a result; "the service did not answer" is not an
    empty list, and the model must be able to tell the two apart.
    """
    if client is None:
        return None, {"error": "unavailable", "message": "This deployment has no client for it."}

    clean = {k: v for k, v in (params or {}).items() if v is not None and v != ""}
    try:
        response = await client.get(path, params=clean or None, headers=_auth_headers(ctx))
    except Exception:
        logger.exception("platform_tool_request_failed", path=path)
        return None, {"error": "unavailable", "message": "The service did not answer."}

    if response.status_code in (401, 403):
        return None, _not_authorized()
    if response.status_code == 404:
        return None, {"error": "not_found", "message": "Nothing matches that."}
    if response.status_code != 200:
        logger.warning("platform_tool_bad_status", path=path, status=response.status_code)
        return None, {
            "error": "unavailable",
            "status": response.status_code,
            "message": f"The service answered {response.status_code}.",
        }
    try:
        return response.json(), None
    except ValueError:
        return None, {"error": "unavailable", "message": "Unreadable response."}


def _items(body: Any) -> list[dict[str, Any]]:
    """Rows from any of the paged envelopes the admin APIs use, or from a bare list."""
    if isinstance(body, list):
        return [row for row in body if isinstance(row, dict)]
    if isinstance(body, dict):
        rows = body.get("items")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _total(body: Any) -> int | None:
    if isinstance(body, dict):
        for key in ("total", "totalCount"):
            value = body.get(key)
            if isinstance(value, int):
                return value
    if isinstance(body, list):
        return len(body)
    return None


def _cited(ctx: ToolContext, title: str, path: str, item: dict[str, Any]) -> dict[str, Any]:
    """The item, carrying the marker of the admin page that shows it, plus the link itself."""
    marker = _cite(ctx, "admin", title, path)
    item["admin_link"] = path
    if marker:
        item["marker"] = marker
    return item


def _path(base: str, params: dict[str, Any] | None = None) -> str:
    clean = {k: v for k, v in (params or {}).items() if v not in (None, "")}
    return f"{base}?{urlencode(clean)}" if clean else base


def workspace_admin_path(ref: str) -> str:
    """The admin page for one workspace. A slug is canonical; an id redirects to the slug."""
    return f"/admin/workspaces/{quote(ref, safe='')}"


# ── workspace resolution (shared by several tools) ────────────────────────────────────────────


async def _resolve_workspace(
    ctx: ToolContext, reference: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """A workspace's admin detail from an id, a slug or a unique name.

    A name that matches more than one workspace is NOT resolved to the first hit — the error
    carries the candidates so the model can ask which one was meant.
    """
    ref = reference.strip()
    if _is_uuid(ref):
        return await _get(ctx.workspace_client, f"/api/v1/admin/workspaces/{ref}", ctx)

    detail, error = await _get(
        ctx.workspace_client, f"/api/v1/admin/workspaces/by-slug/{quote(ref, safe='')}", ctx
    )
    if detail is not None or (error and error.get("error") != "not_found"):
        return detail, error

    body, error = await _get(
        ctx.workspace_client,
        "/api/v1/admin/workspaces",
        ctx,
        {"search": ref, "status": "all", "page": 1, "pageSize": 5},
    )
    if error:
        return None, error
    rows = _items(body)
    if len(rows) == 1 and rows[0].get("id"):
        return await _get(ctx.workspace_client, f"/api/v1/admin/workspaces/{rows[0]['id']}", ctx)
    if not rows:
        return None, {"error": "not_found", "message": f"No workspace matches '{ref}'."}
    return None, {
        "error": "ambiguous",
        "message": f"'{ref}' matches more than one workspace. Ask which one was meant.",
        "candidates": [
            {"name": row.get("name"), "slug": row.get("slug"), "status": row.get("status")}
            for row in rows
        ],
    }


async def _workspace_names(ctx: ToolContext, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Name and slug for up to MAX_NAME_LOOKUPS workspace ids. A failed lookup is simply absent."""
    unique = [wid for wid in dict.fromkeys(ids) if _is_uuid(wid)][:MAX_NAME_LOOKUPS]
    results = await asyncio.gather(
        *(_get(ctx.workspace_client, f"/api/v1/admin/workspaces/{wid}", ctx) for wid in unique),
        return_exceptions=True,
    )
    names: dict[str, dict[str, Any]] = {}
    for wid, result in zip(unique, results, strict=True):
        if isinstance(result, BaseException):
            continue
        body, error = result
        if error or not isinstance(body, dict):
            continue
        names[wid] = {"name": body.get("name"), "slug": body.get("slug")}
    return names


# ── periods ───────────────────────────────────────────────────────────────────────────────────


def _local_midnight(day: date, zone: ZoneInfo) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=zone)


def _month_start(day: date) -> date:
    return day.replace(day=1)


def _add_months(day: date, months: int) -> date:
    index = day.year * 12 + (day.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def _iso_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_insights_period(
    period: str,
    from_day: str | None,
    to_day: str | None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The window an insights question is about, and the Insights page URL that shows it.

    Built on the admins' calendar (PLATFORM_TIME_ZONE), in the same vocabulary as the page's own
    period bar (`?period=today|7d|month|6m|custom`), so the cited link opens on the same numbers.
    Returns {"error": ...} for an unusable custom range instead of guessing one.
    """
    zone = ZoneInfo(PLATFORM_TIME_ZONE)
    current = (now or datetime.now(UTC)).astimezone(zone)
    today = current.date()

    if period == "today":
        start, end = _local_midnight(today, zone), current
        label, link = "Today", _path("/admin", {"period": "today"})
    elif period == "last_7_days":
        start, end = _local_midnight(today - timedelta(days=6), zone), current
        label, link = "Last 7 days", _path("/admin", {"period": "7d"})
    elif period == "last_month":
        first = _add_months(_month_start(today), -1)
        start = _local_midnight(first, zone)
        end = _local_midnight(_month_start(today), zone)
        label = first.strftime("%B %Y")
        link = _path("/admin", {"period": "month", "month": first.strftime("%Y-%m")})
    elif period == "last_6_months":
        start = _local_midnight(_add_months(_month_start(today), -5), zone)
        end = current
        label, link = "Last 6 months", _path("/admin", {"period": "6m"})
    elif period in ("last_30_days", "custom"):
        if period == "last_30_days":
            first_day, last_day = today - timedelta(days=29), today
        else:
            if (
                not from_day
                or not to_day
                or not _ISO_DAY.match(from_day)
                or not _ISO_DAY.match(to_day)
            ):
                return {
                    "error": "invalid_period",
                    "message": "A custom period needs from_date and to_date as YYYY-MM-DD.",
                }
            first_day, last_day = date.fromisoformat(from_day), date.fromisoformat(to_day)
            if first_day > last_day:
                return {"error": "invalid_period", "message": "from_date is after to_date."}
            if first_day > today:
                return {"error": "invalid_period", "message": "That range has not happened yet."}
            if (last_day - first_day).days + 1 > MAX_INSIGHTS_SPAN_DAYS:
                return {
                    "error": "invalid_period",
                    "message": f"A range can be at most {MAX_INSIGHTS_SPAN_DAYS} days.",
                }
        start = _local_midnight(first_day, zone)
        end = min(_local_midnight(last_day + timedelta(days=1), zone), current)
        label = f"{first_day.isoformat()} – {last_day.isoformat()}"
        link = _path(
            "/admin",
            {"period": "custom", "from": first_day.isoformat(), "to": last_day.isoformat()},
        )
    else:  # this_month, and the fallback for anything unrecognised
        start = _local_midnight(_month_start(today), zone)
        end = current
        label = f"{today.strftime('%B %Y')} to date"
        link = _path("/admin", {"period": "month"})

    return {"from": _iso_utc(start), "to": _iso_utc(end), "label": label, "link": link}


# ── 1. insights ───────────────────────────────────────────────────────────────────────────────

_INSIGHT_ENDPOINTS: dict[str, tuple[str, str]] = {
    "billing": ("billing_client", "/api/v1/admin/billing/insights"),
    "users": ("auth_client", "/api/v1/admin/users/insights"),
    "workspaces": ("workspace_client", "/api/v1/admin/workspaces/insights"),
    "meetings": ("translation_room_client", "/api/v1/admin/meetings/insights"),
}


def _metrics(body: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for metric in body.get("metrics") or []:
        if not isinstance(metric, dict):
            continue
        row = {
            "id": metric.get("id"),
            "value": metric.get("value"),
            "previous": metric.get("previous"),
            "unit": metric.get("unit"),
            "higher_is_better": metric.get("higherIsBetter"),
        }
        if metric.get("note"):
            row["note"] = metric["note"]
        out.append(row)
    return out


async def _get_platform_insights(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    period = _choice(arguments, "period", INSIGHT_PERIODS) or "this_month"
    window = resolve_insights_period(
        period, _text(arguments, "from_date"), _text(arguments, "to_date")
    )
    if window.get("error"):
        return json.dumps(window)

    compare = _choice(arguments, "compare", INSIGHT_COMPARE)
    raw_sections = (arguments or {}).get("sections") or []
    if isinstance(raw_sections, str):
        raw_sections = [raw_sections]
    sections = [
        s
        for s in dict.fromkeys(str(x).strip().lower() for x in raw_sections)
        if s in INSIGHT_SECTIONS
    ] or list(INSIGHT_SECTIONS)

    params = {
        "from": window["from"],
        "to": window["to"],
        "tz": PLATFORM_TIME_ZONE,
        "compare": "previousMonth" if compare == "previous_month" else None,
    }
    link = window["link"]
    if compare == "previous_month":
        link = f"{link}&compare=previousMonth"

    results = await asyncio.gather(
        *(
            _get(
                getattr(ctx, _INSIGHT_ENDPOINTS[s][0], None), _INSIGHT_ENDPOINTS[s][1], ctx, params
            )
            for s in sections
        ),
        return_exceptions=True,
    )

    payload: dict[str, Any] = {
        "period": window["label"],
        "compared_with": (
            "the same days of the previous month"
            if compare == "previous_month"
            else "the equally long period just before"
        ),
        "time_zone": PLATFORM_TIME_ZONE,
    }
    payload = _cited(ctx, f"Insights · {window['label']}", link, payload)

    for section, result in zip(sections, results, strict=True):
        if isinstance(result, BaseException):
            payload[section] = {"error": "unavailable", "message": "Could not be produced."}
            continue
        body, error = result
        if error or not isinstance(body, dict):
            payload[section] = error or {"error": "unavailable"}
            continue
        entry: dict[str, Any] = {"metrics": _metrics(body)}
        if section == "billing":
            top = []
            for row in (body.get("topWorkspaces") or [])[:5]:
                wid = str(row.get("workspaceId") or "")
                name = row.get("workspaceName") or wid
                top.append(
                    _cited(
                        ctx,
                        f"Workspace · {name}",
                        workspace_admin_path(wid),
                        {"workspace": name, "credits": row.get("credits")},
                    )
                )
            entry["top_workspaces_by_credits"] = top
            entry["note"] = (
                "Money metrics are per the unit given; never add amounts in different currencies."
            )
        if section == "workspaces" and "suspendedNow" in body:
            entry["suspended_now"] = body.get("suspendedNow")
        if section == "meetings":
            entry["live_now"] = body.get("liveNow")
            entry["started_today"] = body.get("startedToday")
        payload[section] = entry

    return json.dumps(payload, ensure_ascii=False, default=str)


# ── 2. workspaces directory ───────────────────────────────────────────────────────────────────


async def _search_workspaces(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = _text(arguments, "query")
    status = _choice(arguments, "status", WORKSPACE_STATUSES)
    sort = _choice(arguments, "sort", WORKSPACE_SORTS)
    limit = _limit(arguments)

    body, error = await _get(
        ctx.workspace_client,
        "/api/v1/admin/workspaces",
        ctx,
        {
            "search": query,
            "status": status or "all",
            "sort": sort,
            "page": 1,
            "pageSize": limit,
        },
    )
    if error:
        return json.dumps(error)

    list_link = _path("/admin/workspaces", {"q": query, "status": status, "sort": sort})
    rows = []
    for row in _items(body):
        slug = str(row.get("slug") or row.get("id") or "")
        owner = row.get("owner") or {}
        rows.append(
            _cited(
                ctx,
                f"Workspace · {row.get('name') or slug}",
                workspace_admin_path(slug),
                {
                    "name": row.get("name"),
                    "slug": row.get("slug"),
                    "status": row.get("status"),
                    "members": row.get("memberCount"),
                    "owner": owner.get("fullName") or owner.get("email"),
                    "owner_email": owner.get("email"),
                    "created_at": row.get("createdAt"),
                    "last_activity_at": row.get("lastActivityAt"),
                },
            )
        )
    result = _cited(
        ctx,
        "Workspaces directory",
        list_link,
        {"total_matching": _total(body), "shown": len(rows), "workspaces": rows},
    )
    return json.dumps(result, ensure_ascii=False, default=str)


# ── 3. accounts directory ─────────────────────────────────────────────────────────────────────


async def _search_accounts(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = _text(arguments, "query")
    status = _choice(arguments, "status", ACCOUNT_STATUSES)
    sort = _choice(arguments, "sort", ACCOUNT_SORTS)
    limit = _limit(arguments)

    body, error = await _get(
        ctx.auth_client,
        "/api/v1/admin/users",
        ctx,
        {"search": query, "status": status or "all", "sort": sort, "page": 1, "pageSize": limit},
    )
    if error:
        return json.dumps(error)

    rows = []
    for row in _items(body):
        uid = str(row.get("id") or "")
        rows.append(
            _cited(
                ctx,
                f"Account · {row.get('fullName') or row.get('email') or uid}",
                f"/admin/users/{uid}",
                {
                    "name": row.get("fullName"),
                    "email": row.get("email"),
                    "status": row.get("status"),
                    "roles": row.get("roles"),
                    "active_sessions": row.get("activeSessionCount"),
                    "last_login_at": row.get("lastLoginAt"),
                    "created_at": row.get("createdAt"),
                },
            )
        )
    result = _cited(
        ctx,
        "Accounts directory",
        _path("/admin/users", {"q": query, "status": status, "sort": sort}),
        {"total_matching": _total(body), "shown": len(rows), "accounts": rows},
    )
    return json.dumps(result, ensure_ascii=False, default=str)


# ── 4. one workspace ──────────────────────────────────────────────────────────────────────────


async def _get_workspace_summary(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    reference = _text(arguments, "workspace")
    if not reference:
        return json.dumps(
            {"error": "missing_workspace", "message": "Name the workspace (slug, name or id)."}
        )

    detail, error = await _resolve_workspace(ctx, reference)
    if error or not isinstance(detail, dict):
        return json.dumps(error or {"error": "not_found"}, ensure_ascii=False)

    wid = str(detail.get("id") or "")
    slug = str(detail.get("slug") or wid)
    page = workspace_admin_path(slug)

    analytics_result, subscription_result = await asyncio.gather(
        _get(ctx.billing_client, f"/api/v1/admin/billing/workspaces/{wid}/analytics", ctx),
        _get(ctx.billing_client, f"/api/v1/subscriptions/workspace/{wid}", ctx),
    )
    analytics, analytics_error = analytics_result
    subscription, subscription_error = subscription_result

    owner = detail.get("owner") or {}
    summary: dict[str, Any] = {
        "name": detail.get("name"),
        "slug": detail.get("slug"),
        "status": detail.get("status"),
        "owner": owner.get("fullName") or owner.get("email"),
        "owner_email": owner.get("email"),
        "members": detail.get("memberCount"),
        "internal_members": detail.get("internalMemberCount"),
        "external_members": detail.get("externalMemberCount"),
        "pending_invitations": detail.get("pendingInvitationCount"),
        "documents": detail.get("documentCount"),
        "created_at": detail.get("createdAt"),
        "last_activity_at": detail.get("lastActivityAt"),
        "deleted_at": detail.get("deletedAt"),
    }
    suspension = detail.get("currentSuspension")
    if isinstance(suspension, dict):
        summary["suspended"] = {
            "reason": suspension.get("reason"),
            "since": suspension.get("performedAt"),
        }

    if isinstance(subscription, dict):
        summary["plan"] = {
            "name": subscription.get("planName"),
            "status": subscription.get("status"),
            "service_state": subscription.get("serviceState"),
            "credits_remaining": subscription.get("creditsRemaining"),
            "credits_used_this_cycle": subscription.get("creditsUsedThisCycle"),
            "credits_per_cycle": subscription.get("effectiveCreditsPerCycle"),
            "current_period_end": subscription.get("currentPeriodEnd"),
            "cancel_at_period_end": subscription.get("cancelAtPeriodEnd"),
        }
    else:
        summary["plan"] = subscription_error or {"error": "unavailable"}

    if isinstance(analytics, dict):
        credits = analytics.get("credits") or {}
        summary["usage_last_30_days"] = {
            "credits_consumed": analytics.get("creditsConsumedInPeriod"),
            "credits_topped_up": analytics.get("creditsToppedUpInPeriod"),
            "meetings_with_billable_usage": analytics.get("meetingsWithBillableUsage"),
            "distinct_users_billed": analytics.get("distinctUsersBilled"),
            "by_feature": [
                {"usage_type": f.get("usageType"), "credits": f.get("creditsConsumed")}
                for f in (analytics.get("featureBreakdown") or [])
                if isinstance(f, dict)
            ],
        }
        if credits and "credits_remaining" not in (summary.get("plan") or {}):
            summary["credits_remaining"] = credits.get("creditsRemaining")
    else:
        summary["usage_last_30_days"] = analytics_error or {"error": "unavailable"}

    return json.dumps(
        _cited(ctx, f"Workspace · {detail.get('name') or slug}", page, summary),
        ensure_ascii=False,
        default=str,
    )


# ── 5. subscriptions / invoices / billing snapshot ────────────────────────────────────────────


async def _billing_subscriptions(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    status = _choice(arguments, "status", SUBSCRIPTION_STATUSES)
    sort = _choice(arguments, "sort", SUBSCRIPTION_SORTS)
    plan_slug = _text(arguments, "plan_slug")
    limit = _limit(arguments)

    body, error = await _get(
        ctx.billing_client,
        "/api/v1/admin/subscriptions",
        ctx,
        {
            "status": status or "all",
            "sort": sort,
            "planSlug": plan_slug,
            "page": 1,
            "pageSize": limit,
        },
    )
    if error:
        return error

    rows = _items(body)
    names = await _workspace_names(ctx, [str(r.get("workspaceId") or "") for r in rows])
    out = []
    for row in rows:
        wid = str(row.get("workspaceId") or "")
        known = names.get(wid) or {}
        name = known.get("name") or wid
        out.append(
            _cited(
                ctx,
                f"Workspace · {name}",
                workspace_admin_path(known.get("slug") or wid),
                {
                    "workspace": name,
                    "plan": row.get("planName"),
                    "status": row.get("status"),
                    "service_state": row.get("serviceState"),
                    "low_balance": row.get("serviceState") == "low_balance",
                    "credits_remaining": row.get("creditsRemaining"),
                    "credits_used_this_cycle": row.get("creditsUsedThisCycle"),
                    "billing_cycle": row.get("billingCycle"),
                    "monthly_value": row.get("monthlyValue"),
                    "current_period_end": row.get("currentPeriodEnd"),
                    "auto_renew": row.get("autoRenew"),
                },
            )
        )
    return _cited(
        ctx,
        "Subscriptions",
        _path("/admin/subscriptions", {"status": status, "sort": sort}),
        {
            "total_matching": _total(body),
            "shown": len(out),
            "subscriptions": out,
            "note": (
                "low_balance is the billing service's own verdict against the plan's threshold. "
                "credits_remaining alone is not 'low' without it."
            ),
        },
    )


async def _billing_invoices(ctx: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    limit = _limit(arguments)
    workspace = _text(arguments, "workspace")
    workspace_label = None
    if workspace:
        detail, error = await _resolve_workspace(ctx, workspace)
        if error or not isinstance(detail, dict):
            return error or {"error": "not_found"}
        workspace_label = detail.get("name")
        path = f"/api/v1/invoices/workspace/{detail.get('id')}"
    else:
        path = "/api/v1/invoices/global"

    body, error = await _get(ctx.billing_client, path, ctx, {"PageNumber": 1, "PageSize": limit})
    if error:
        return error

    invoices = [
        {
            "number": row.get("invoiceNumber"),
            "workspace": row.get("workspaceName") or workspace_label,
            "total": row.get("total"),
            "currency": row.get("currency"),
            "status": row.get("status"),
            "issued_at": row.get("issuedAt"),
            "due_at": row.get("dueAt"),
            "paid_at": row.get("paidAt"),
        }
        for row in _items(body)
    ]
    return _cited(
        ctx,
        "Billing · Invoices",
        "/admin/billing",
        {
            "total_matching": _total(body),
            "shown": len(invoices),
            "invoices": invoices,
            "note": "Report totals per currency; never add amounts in different currencies.",
        },
    )


async def _billing_snapshot(ctx: ToolContext, _arguments: dict[str, Any]) -> dict[str, Any]:
    body, error = await _get(
        ctx.billing_client,
        "/api/v1/admin/billing/insights/snapshot",
        ctx,
        {"tz": PLATFORM_TIME_ZONE},
    )
    if error or not isinstance(body, dict):
        return error or {"error": "unavailable"}

    high_usage = [
        _cited(
            ctx,
            f"Workspace · {row.get('workspaceName') or row.get('workspaceId')}",
            workspace_admin_path(str(row.get("workspaceId") or "")),
            {"workspace": row.get("workspaceName"), "credits_last_24h": row.get("credits24h")},
        )
        for row in (body.get("highUsageAlerts") or [])
        if isinstance(row, dict)
    ]
    ending = [
        {
            "workspace": row.get("workspaceName"),
            "plan": row.get("planName"),
            "ends_at": row.get("endsAt"),
            "cancel_at_period_end": row.get("cancelAtPeriodEnd"),
        }
        for row in (body.get("endingSoon") or [])
        if isinstance(row, dict)
    ]
    return _cited(
        ctx,
        "Insights · Right now",
        "/admin",
        {
            "revenue_today": body.get("revenueToday"),
            "revenue_yesterday": body.get("revenueYesterday"),
            "mrr": body.get("mrr"),
            "mrr_note": body.get("mrrNote"),
            "active_subscriptions": body.get("activeSubscriptions"),
            "trials": body.get("trials"),
            "trials_ending_this_week": body.get("trialsEndingThisWeek"),
            "past_due": body.get("pastDue"),
            "suspended": body.get("suspended"),
            "active_workspaces": body.get("activeWorkspaces"),
            "platform_credit_balance": body.get("platformCreditBalance"),
            "outstanding_invoices": body.get("outstandingInvoices"),
            "churn_this_month": body.get("churnRateMonth"),
            "high_usage_last_24h": high_usage,
            "ending_soon": ending,
        },
    )


_BILLING_VIEWS: dict[str, Callable[[ToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]] = {
    "subscriptions": _billing_subscriptions,
    "invoices": _billing_invoices,
    "snapshot": _billing_snapshot,
}


async def _lookup_billing(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    view = _choice(arguments, "view", BILLING_VIEWS) or "subscriptions"
    # One workspace's subscription is a workspace summary, which already carries plan and credits.
    if view == "subscriptions" and _text(arguments, "workspace"):
        return await _get_workspace_summary(ctx, {"workspace": _text(arguments, "workspace")})
    result = await _BILLING_VIEWS[view](ctx, arguments or {})
    return json.dumps({"view": view, **result}, ensure_ascii=False, default=str)


# ── 6. system health ──────────────────────────────────────────────────────────────────────────


async def _get_system_health(ctx: ToolContext, _arguments: dict[str, Any]) -> str:
    today = resolve_insights_period("today", None, None)
    health_result, counts_result, meetings_result = await asyncio.gather(
        _get(ctx.workspace_client, "/api/v1/admin/platform-health", ctx),
        _get(ctx.translation_room_client, "/api/v1/admin/meetings/counts", ctx),
        _get(
            ctx.translation_room_client,
            "/api/v1/admin/meetings/insights",
            ctx,
            {"from": today["from"], "to": today["to"], "tz": PLATFORM_TIME_ZONE},
        ),
    )
    health, health_error = health_result
    payload: dict[str, Any] = {}

    if health_error or not isinstance(health, dict):
        payload["monitoring"] = health_error or {"error": "unavailable"}
    elif not health.get("monitoringAvailable"):
        payload["monitoring"] = _cited(
            ctx,
            "System health",
            "/admin/health",
            {
                "monitoring_available": False,
                "reason": health.get("monitoringUnavailableReason"),
                "note": (
                    "Monitoring could not be read. That says NOTHING about whether the platform "
                    "is healthy — report 'I cannot see the metrics', never an outage."
                ),
            },
        )
    else:
        groups = [g for g in (health.get("streamGroups") or []) if isinstance(g, dict)]
        payload["monitoring"] = _cited(
            ctx,
            "System health",
            "/admin/health",
            {
                "monitoring_available": True,
                "observed_at": health.get("observedAt"),
                "firing_alerts": [
                    {
                        "name": a.get("name"),
                        "severity": a.get("severity"),
                        "state": a.get("state"),
                        "summary": a.get("summary"),
                        "active_since": a.get("activeSince"),
                    }
                    for a in (health.get("alerts") or [])
                    if isinstance(a, dict)
                ],
                "targets_down": [
                    t.get("job") for t in (health.get("targets") or []) if not t.get("isUp")
                ],
                "workers_at_zero_replicas": [
                    w.get("worker") for w in (health.get("workers") or []) if not w.get("replicas")
                ],
                "pipeline_stages_backed_up": [
                    {
                        "stream": g.get("stream"),
                        "group": g.get("group"),
                        "lag": g.get("lag"),
                        "pending": g.get("pending"),
                        "consumers": g.get("consumers"),
                    }
                    for g in groups
                    if (g.get("lag") or 0) > 0
                    or (g.get("pending") or 0) > 0
                    or not g.get("consumers")
                ],
                "dead_letter_streams": [
                    {"stream": d.get("stream"), "length": d.get("length")}
                    for d in (health.get("deadLetters") or [])
                    if isinstance(d, dict) and d.get("length")
                ],
                "stage_latency_p95_ms": health.get("stageLatencies"),
                # STT / translation / TTS attempt outcomes over the last hour. A stage with
                # failures or parked (dead-lettered) attempts is the direct answer to "is a
                # pipeline stage failing"; success_rate null means no attempts, not 0%.
                "pipeline_stage_outcomes_last_hour": [
                    {
                        "stage": o.get("stage"),
                        "ok": o.get("ok"),
                        "failed": o.get("failed"),
                        "dead_lettered": o.get("deadLettered"),
                        "success_rate": o.get("successRate"),
                    }
                    for o in (health.get("stageOutcomes") or [])
                    if isinstance(o, dict)
                ],
                "outbox_dead_letters": health.get("outboxDeadLetters"),
                "warnings": health.get("warnings"),
            },
        )
        # The headline of the System health page: did meetings work? Absent on a backend that
        # predates it, and null when the counters have no series yet — neither is zero.
        outcomes = health.get("meetings")
        if isinstance(outcomes, dict):
            payload["meeting_success"] = _cited(
                ctx,
                "System health · Meetings",
                "/admin/health",
                {
                    "window": outcomes.get("window"),
                    "success_rate": outcomes.get("successRate"),
                    "started": outcomes.get("started"),
                    "ended": outcomes.get("ended"),
                    "reached_live": outcomes.get("reachedLive"),
                    "ended_normally": outcomes.get("endedNormally"),
                    "ended_abandoned_after_live": outcomes.get("endedAbandoned"),
                    "failed": outcomes.get("failed"),
                    "live_rooms": outcomes.get("liveRooms"),
                    "note": (
                        "success_rate = reached live (two people joined AND a caption was "
                        "delivered) / ended, 0..1. null means nothing ended in the window."
                    ),
                },
            )

    counts, counts_error = counts_result
    meetings, meetings_error = meetings_result
    meeting_facts: dict[str, Any] = {}
    if isinstance(counts, dict):
        meeting_facts["live_now"] = counts.get("liveNow")
        meeting_facts["started_today"] = counts.get("startedToday")
    if isinstance(meetings, dict):
        meeting_facts["today_vs_yesterday"] = _metrics(meetings)
    if meeting_facts:
        payload["meetings"] = _cited(ctx, "Insights · Today", today["link"], meeting_facts)
    else:
        payload["meetings"] = counts_error or meetings_error or {"error": "unavailable"}

    return json.dumps(payload, ensure_ascii=False, default=str)


# ── 7. audit log ──────────────────────────────────────────────────────────────────────────────


async def _search_audit_log(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    action = _text(arguments, "action")
    entity_type = _text(arguments, "entity_type")
    result_filter = _choice(arguments, "result", AUDIT_RESULTS)
    actor_id = _text(arguments, "actor_id")
    limit = _limit(arguments)
    from_day, to_day = _text(arguments, "from_date"), _text(arguments, "to_date")

    workspace_id = None
    workspace = _text(arguments, "workspace")
    if workspace:
        detail, error = await _resolve_workspace(ctx, workspace)
        if error or not isinstance(detail, dict):
            return json.dumps(error or {"error": "not_found"}, ensure_ascii=False)
        workspace_id = detail.get("id")

    zone = ZoneInfo(PLATFORM_TIME_ZONE)
    params: dict[str, Any] = {
        "action": action,
        "entityType": entity_type,
        "result": result_filter,
        "actorId": actor_id if _is_uuid(actor_id) else None,
        "workspaceId": workspace_id,
        "page": 1,
        "pageSize": limit,
    }
    if from_day and _ISO_DAY.match(from_day):
        params["from"] = _iso_utc(_local_midnight(date.fromisoformat(from_day), zone))
    if to_day and _ISO_DAY.match(to_day):
        params["to"] = _iso_utc(
            _local_midnight(date.fromisoformat(to_day) + timedelta(days=1), zone)
        )

    body, error = await _get(ctx.workspace_client, "/api/v1/admin/audit-log", ctx, params)
    if error:
        return json.dumps(error)

    entries = [
        {
            "performed_at": row.get("performedAt"),
            "action": row.get("action"),
            "entity_type": row.get("entityType"),
            "entity_id": row.get("entityId"),
            "workspace_id": row.get("workspaceId"),
            "actor_id": row.get("actorId"),
            "result": row.get("result"),
            "reason": row.get("reason"),
            "source_service": row.get("sourceService"),
            "before": row.get("beforeSummary"),
            "after": row.get("afterSummary"),
        }
        for row in _items(body)
    ]
    result = _cited(
        ctx,
        "Audit log",
        _path("/admin/audit", {"entityType": entity_type, "result": result_filter}),
        {"total_matching": _total(body), "shown": len(entries), "entries": entries},
    )
    return json.dumps(result, ensure_ascii=False, default=str)


# ── 8. plugin marketplace ─────────────────────────────────────────────────────────────────────


async def _lookup_plugins(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    plugin_key = _text(arguments, "plugin_key")
    if plugin_key:
        body, error = await _get(
            ctx.assistant_client,
            f"/api/v1/assistant/plugins/catalog/{quote(plugin_key, safe='')}",
            ctx,
        )
        if error or not isinstance(body, dict):
            return json.dumps(error or {"error": "not_found"})
        tools = body.get("tools") or body.get("toolManifest") or []
        detail = {
            "key": body.get("pluginKey") or plugin_key,
            "label": body.get("label"),
            "description": body.get("description"),
            "kind": body.get("kind"),
            "provider": body.get("provider"),
            "category": body.get("category"),
            "active": body.get("isActive"),
            "featured": body.get("isFeatured"),
            "installations": body.get("installationCount"),
            "workspaces": body.get("workspaceCount"),
            "tools": [t.get("name") for t in tools if isinstance(t, dict)][:40],
        }
        return json.dumps(
            _cited(
                ctx,
                f"Plugin · {detail['label'] or detail['key']}",
                f"/admin/plugins/{quote(str(detail['key']), safe='')}",
                detail,
            ),
            ensure_ascii=False,
            default=str,
        )

    body, error = await _get(ctx.assistant_client, "/api/v1/assistant/plugins/catalog", ctx)
    if error:
        return json.dumps(error)

    query = (_text(arguments, "query") or "").casefold()
    status = _choice(arguments, "status", PLUGIN_STATUSES)
    limit = _limit(arguments)
    rows = []
    for row in _items(body):
        haystack = " ".join(
            str(row.get(k) or "")
            for k in ("pluginKey", "label", "description", "category", "provider", "kind")
        ).casefold()
        if query and query not in haystack:
            continue
        if status == "active" and not row.get("isActive"):
            continue
        if status == "retired" and row.get("isActive"):
            continue
        rows.append(row)

    shown = [
        _cited(
            ctx,
            f"Plugin · {row.get('label') or row.get('pluginKey')}",
            f"/admin/plugins/{quote(str(row.get('pluginKey') or ''), safe='')}",
            {
                "key": row.get("pluginKey"),
                "label": row.get("label"),
                "kind": row.get("kind"),
                "provider": row.get("provider"),
                "category": row.get("category"),
                "active": row.get("isActive"),
                "featured": row.get("isFeatured"),
                "tool_count": row.get("toolCount"),
                "installations": row.get("installationCount"),
                "workspaces": row.get("workspaceCount"),
            },
        )
        for row in rows[:limit]
    ]
    result = _cited(
        ctx,
        "Plugin marketplace",
        "/admin/plugins",
        {"total_matching": len(rows), "shown": len(shown), "plugins": shown},
    )
    return json.dumps(result, ensure_ascii=False, default=str)


# ── declarations ──────────────────────────────────────────────────────────────────────────────

_LIMIT_PROPERTY = {
    "type": "integer",
    "description": (
        f"How many rows to return, 1-{MAX_LIMIT}. 0 means the default ({DEFAULT_LIMIT})."
    ),
}

_WORKSPACE_REF = {
    "type": "string",
    "description": (
        "A workspace's slug, exact name or id. Empty string when not about one workspace."
    ),
}


def _enum(values: tuple[str, ...], description: str) -> dict[str, Any]:
    return {"type": "string", "enum": list(values), "description": description}


PLATFORM_TOOLS: list[ChatTool] = [
    ChatTool(
        name="get_platform_insights",
        description=(
            "Platform KPIs for a period, each compared with an earlier one: revenue, MRR, "
            "credits used, new/active accounts, new/active workspaces, meetings held and hours "
            "translated, plus the top workspaces by credits. Use for any 'how are we doing', "
            "'revenue this month vs last', growth or trend question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "period": _enum(
                    INSIGHT_PERIODS,
                    "The window. 'custom' needs from_date and to_date.",
                ),
                "from_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD, only for period=custom. Empty string otherwise.",
                },
                "to_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD inclusive, only for period=custom. Empty otherwise.",
                },
                "compare": _enum(
                    INSIGHT_COMPARE,
                    "'previous_month' compares with the same days of the previous month (use it "
                    "for 'this month vs last'). 'none' compares with the equally long period "
                    "just before.",
                ),
                "sections": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(INSIGHT_SECTIONS)},
                    "description": "Which KPI groups. Empty array means all four.",
                },
            },
            "required": ["period", "from_date", "to_date", "compare", "sections"],
            "additionalProperties": False,
        },
        handler=_get_platform_insights,
    ),
    ChatTool(
        name="search_workspaces",
        description=(
            "List or search every workspace on the platform by name, slug or owner, with status, "
            "member count, owner and last activity."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text. Empty lists them all."},
                "status": _enum(WORKSPACE_STATUSES, "Filter by lifecycle status. 'none' = any."),
                "sort": _enum(WORKSPACE_SORTS, "Order. 'none' = newest first."),
                "limit": _LIMIT_PROPERTY,
            },
            "required": ["query", "status", "sort", "limit"],
            "additionalProperties": False,
        },
        handler=_search_workspaces,
    ),
    ChatTool(
        name="search_accounts",
        description=(
            "List or search user accounts on the platform by name or email, with status, roles, "
            "sessions and last login."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Name or email. Empty lists them all."},
                "status": _enum(ACCOUNT_STATUSES, "Filter by account status. 'none' = any."),
                "sort": _enum(ACCOUNT_SORTS, "Order. 'none' = newest first."),
                "limit": _LIMIT_PROPERTY,
            },
            "required": ["query", "status", "sort", "limit"],
            "additionalProperties": False,
        },
        handler=_search_accounts,
    ),
    ChatTool(
        name="get_workspace_summary",
        description=(
            "Everything about ONE workspace: status (and suspension), owner, members count, plan, "
            "subscription status, credits remaining and used, and the last 30 days of usage by "
            "feature."
        ),
        parameters={
            "type": "object",
            "properties": {"workspace": _WORKSPACE_REF},
            "required": ["workspace"],
            "additionalProperties": False,
        },
        handler=_get_workspace_summary,
    ),
    ChatTool(
        name="lookup_billing",
        description=(
            "Subscriptions, invoices, or the billing snapshot. view=subscriptions lists "
            "subscriptions with plan, credits remaining and a low_balance flag — sort=credits_asc "
            "with status=active answers 'who is running low on credits'. view=invoices lists "
            "invoices (for one workspace or all). view=snapshot gives today's revenue, MRR, past "
            "due, trials ending, outstanding invoices and high-usage workspaces."
        ),
        parameters={
            "type": "object",
            "properties": {
                "view": _enum(BILLING_VIEWS, "What to look up."),
                "status": _enum(SUBSCRIPTION_STATUSES, "Subscriptions only. 'none' = any."),
                "sort": _enum(SUBSCRIPTION_SORTS, "Subscriptions only. 'none' = soonest renewal."),
                "plan_slug": {
                    "type": "string",
                    "description": "Subscriptions only: a plan slug. Empty string for any plan.",
                },
                "workspace": _WORKSPACE_REF,
                "limit": _LIMIT_PROPERTY,
            },
            "required": ["view", "status", "sort", "plan_slug", "workspace", "limit"],
            "additionalProperties": False,
        },
        handler=_lookup_billing,
    ),
    ChatTool(
        name="get_system_health",
        description=(
            "System health right now: the meeting success rate (last 24h), STT / translation / "
            "TTS stage outcomes (last hour), firing alerts, services down, workers at zero "
            "replicas, backed-up stream groups, dead-letter streams, stage latency, and meetings "
            "live / held today. Use for 'is anything broken', 'failing pipeline stages', meeting "
            "success and outage questions."
        ),
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_get_system_health,
    ),
    ChatTool(
        name="search_audit_log",
        description=(
            "Search the platform audit log of admin and lifecycle actions: who did what to which "
            "entity, when, and whether it succeeded."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Action name. Empty for any."},
                "entity_type": {
                    "type": "string",
                    "description": "Entity type, e.g. workspace, user, plan. Empty for any.",
                },
                "result": _enum(AUDIT_RESULTS, "Outcome filter. 'none' = any."),
                "workspace": _WORKSPACE_REF,
                "actor_id": {
                    "type": "string",
                    "description": "The acting user's id. Empty string for anyone.",
                },
                "from_date": {"type": "string", "description": "YYYY-MM-DD, or empty string."},
                "to_date": {"type": "string", "description": "YYYY-MM-DD inclusive, or empty."},
                "limit": _LIMIT_PROPERTY,
            },
            "required": [
                "action",
                "entity_type",
                "result",
                "workspace",
                "actor_id",
                "from_date",
                "to_date",
                "limit",
            ],
            "additionalProperties": False,
        },
        handler=_search_audit_log,
    ),
    ChatTool(
        name="lookup_plugins",
        description=(
            "The plugin marketplace catalog: which plugins exist, whether active or retired, how "
            "many workspaces installed them, and one plugin's tools."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search text. Empty lists them all."},
                "status": _enum(PLUGIN_STATUSES, "Filter. 'none' = any."),
                "plugin_key": {
                    "type": "string",
                    "description": "One plugin's key for its detail. Empty string for a list.",
                },
                "limit": _LIMIT_PROPERTY,
            },
            "required": ["query", "status", "plugin_key", "limit"],
            "additionalProperties": False,
        },
        handler=_lookup_plugins,
    ),
]

PLATFORM_TOOLS_BY_NAME: dict[str, ChatTool] = {tool.name: tool for tool in PLATFORM_TOOLS}


def build_platform_system_prompt() -> str:
    """The platform WarpBot's instructions. Nothing in it mentions a workspace's own content."""
    tools = "\n".join(f"- {tool.name}: {tool.description}" for tool in PLATFORM_TOOLS)
    return "\n".join(
        [
            "You are WarpBot in the WarpTalk ADMIN PORTAL, answering a platform system "
            "administrator about the whole platform. Answer clearly and concisely, in the "
            "language the user wrote in.",
            "",
            "SCOPE — PLATFORM",
            "You have no access to any workspace's meetings, transcripts, documents, glossary or "
            "knowledge base, and must not claim to. If asked about what was SAID or WRITTEN inside "
            "a workspace, say that the platform assistant cannot read workspace content and that "
            "WarpBot inside that workspace can.",
            "",
            "TOOLS — all read-only:",
            tools,
            "",
            "GROUND RULES",
            "- Look it up, then answer. Never state a figure no tool returned this turn.",
            "- Cite every figure: each tool result carries a `marker` for the admin page that "
            "shows it. Put the marker after the statement it supports.",
            "- An error such as not_authorized or unavailable is a result: say which lookup "
            "failed. Never turn it into zero.",
            "- Money is per currency. Never add amounts in different currencies.",
            "- You cannot change anything. If asked to suspend, refund, adjust credits, cancel or "
            "edit, say that WarpBot is read-only here and point to the admin page (its link) where "
            "the admin can do it themselves.",
        ]
    )
