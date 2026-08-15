"""Tests for get_platform_analytics — WarpBot's read of the platform admin API.

The property that matters most here is not that the numbers arrive. It is that when they do not,
the tool says so in a way the model cannot mistake for a figure: a 403 is "you are not allowed
this", an unreachable metrics store is "I cannot see", and neither is a zero.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker.chat_tools import (
    PLATFORM_ANALYTICS_MAX_DAYS,
    TOOLS_BY_NAME,
    ToolContext,
    _get_platform_analytics,
)


def _response(status_code: int, payload: Any = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload if payload is not None else {}
    return response


def _client(
    routes: dict[str, MagicMock],
    recorder: list[tuple[str, Any]] | None = None,
) -> AsyncMock:
    client = AsyncMock()

    async def _get(url: str, **kwargs: Any) -> MagicMock:
        if recorder is not None:
            recorder.append((url, kwargs.get("params")))
        for path, response in routes.items():
            if url.startswith(path):
                return response
        raise AssertionError(f"Unexpected request: {url}")

    client.get.side_effect = _get
    return client


def _ctx(**clients: Any) -> ToolContext:
    empty = _client({})
    return ToolContext(
        workspace_id="ws-1",
        user_id="user-1",
        bearer_token="Bearer test-token",
        workspace_client=clients.get("workspace", empty),
        transcript_client=empty,
        translation_room_client=clients.get("translation_room", empty),
        billing_client=clients.get("billing"),
        auth_client=clients.get("auth"),
        openai_client=MagicMock(),
        model="gpt-4o-mini",
        redis=MagicMock(),
    )


async def test_a_403_is_reported_as_not_authorized_not_as_zero() -> None:
    """The whole reason this tool can be offered to every user.

    The platform admin policy is enforced server-side, so a normal user's assistant gets the same
    403 they would get in a browser. What must not happen is the tool turning that into an empty
    result the model then reads as "there are no workspaces".
    """
    ctx = _ctx(workspace=_client({"/api/v1/admin/workspaces": _response(403)}))

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["overview"]}))

    assert payload["overview"]["error"] == "not_authorized"
    assert "workspaces_total" not in payload["overview"]


async def test_an_expired_token_reads_the_same_way_as_a_missing_role() -> None:
    ctx = _ctx(auth=_client({"/api/v1/admin/users": _response(401)}))

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["users"]}))

    assert payload["users"]["error"] == "not_authorized"


async def test_the_callers_own_token_is_forwarded() -> None:
    """Never a service credential. There is no elevation here to get wrong."""
    client = AsyncMock()
    seen: dict[str, Any] = {}

    async def _get(url: str, **kwargs: Any) -> MagicMock:
        seen["headers"] = kwargs.get("headers")
        return _response(200, {"total": 3})

    client.get.side_effect = _get

    await _get_platform_analytics(_ctx(workspace=client), {"reports": ["overview"]})

    assert seen["headers"] == {"Authorization": "Bearer test-token"}


async def test_overview_asks_for_one_row_and_reads_the_total() -> None:
    calls: list[tuple[str, Any]] = []
    ctx = _ctx(
        workspace=_client({"/api/v1/admin/workspaces": _response(200, {"total": 42})}, calls),
        translation_room=_client(
            {"/api/v1/admin/meetings/counts": _response(200, {"liveNow": 2, "startedToday": 9})}
        ),
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["overview"]}))

    assert payload["overview"]["workspaces_total"] == 42
    assert payload["overview"]["meetings_live_now"] == 2
    # A page of rows would be tokens spent on data nobody asked for.
    assert calls[0][1] == {"page": 1, "pageSize": 1}


async def test_one_failing_half_of_overview_does_not_erase_the_other() -> None:
    ctx = _ctx(
        workspace=_client({"/api/v1/admin/workspaces": _response(200, {"total": 42})}),
        translation_room=_client({"/api/v1/admin/meetings/counts": _response(500)}),
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["overview"]}))

    assert payload["overview"]["workspaces_total"] == 42
    assert "meetings_live_now" not in payload["overview"]
    assert payload["overview"]["meetings_note"]


async def test_revenue_keeps_one_figure_per_currency() -> None:
    """plans.currency defaults to VND and USD plans exist. Summing them invents a rate."""
    ctx = _ctx(
        billing=_client(
            {
                "/api/v1/admin/subscriptions/summary": _response(
                    200,
                    {
                        "monthlyRecurring": [
                            {"amount": 1_900_000, "currency": "VND"},
                            {"amount": 58, "currency": "USD"},
                        ],
                        "activeCount": 12,
                        "trialingCount": 3,
                        "cancellingCount": 1,
                    },
                )
            }
        )
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["revenue"]}))

    assert len(payload["revenue"]["monthly_recurring"]) == 2
    assert "Do not add them together." in payload["revenue"]["note"]


async def test_unreachable_monitoring_is_not_reported_as_an_outage() -> None:
    """The distinction the health endpoint exists to preserve, carried through to the model.

    Handed empty lists, a model would reasonably say every target is down. The note is what stops
    that, and it is worth a test because the failure is silent and confident.
    """
    ctx = _ctx(
        workspace=_client(
            {
                "/api/v1/admin/platform-health": _response(
                    200,
                    {
                        "monitoringAvailable": False,
                        "monitoringUnavailableReason": "The metrics store could not be reached.",
                        "targets": [],
                        "workers": [],
                        "alerts": [],
                    },
                )
            }
        )
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["health"]}))

    assert payload["health"]["monitoring_available"] is False
    assert "targets_down" not in payload["health"]
    assert "not as an outage" in payload["health"]["note"]


async def test_health_names_what_is_actually_down() -> None:
    ctx = _ctx(
        workspace=_client(
            {
                "/api/v1/admin/platform-health": _response(
                    200,
                    {
                        "monitoringAvailable": True,
                        "targets": [
                            {"job": "redis", "isUp": True},
                            {"job": "rabbitmq", "isUp": False},
                        ],
                        "workers": [
                            {"worker": "stt", "replicas": 2},
                            {"worker": "tts", "replicas": 0},
                        ],
                        "deadLetters": [
                            {"stream": "stt:dead-letter", "length": 4},
                            {"stream": "tts:dead-letter", "length": 0},
                        ],
                        "alerts": [
                            {
                                "name": "WarpTalkAiWorkerMissing",
                                "severity": "critical",
                                "summary": "tts heartbeat missing",
                            }
                        ],
                        "stageLatencies": [{"stage": "tts", "p95Ms": 2400}],
                    },
                )
            }
        )
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["health"]}))
    health = payload["health"]

    assert health["targets_down"] == ["rabbitmq"]
    assert health["targets_total"] == 2
    assert health["workers_at_zero"] == ["tts"]
    # An empty dead-letter stream is not a finding; only the non-empty one is carried.
    assert health["dead_letter_streams"] == [{"stream": "stt:dead-letter", "length": 4}]
    assert health["firing_alerts"][0]["severity"] == "critical"


async def test_feedback_carries_the_per_dimension_response_counts() -> None:
    """Four of the five dimensions are optional. An average without its n is not quotable."""
    ctx = _ctx(
        translation_room=_client(
            {
                "/api/v1/admin/feedback/summary": _response(
                    200,
                    {
                        "responseCount": 40,
                        "ratedMeetings": 31,
                        "endedMeetings": 88,
                        "responseRate": 0.352,
                        "dimensions": [
                            {
                                "dimension": "overallRating",
                                "responseCount": 40,
                                "averageRating": 4.2,
                                "distribution": [1, 2, 5, 12, 20],
                            },
                            {
                                "dimension": "voiceCloneQuality",
                                "responseCount": 3,
                                "averageRating": 4.8,
                                "distribution": [0, 0, 0, 1, 2],
                            },
                        ],
                    },
                )
            }
        )
    )

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["feedback"]}))
    feedback = payload["feedback"]

    assert feedback["dimensions"][1]["responseCount"] == 3
    assert "over its own responseCount" in feedback["note"]
    assert "not a score of zero" in feedback["note"]


async def test_the_feedback_window_is_clamped_rather_than_rejected() -> None:
    calls: list[tuple[str, Any]] = []
    ctx = _ctx(
        translation_room=_client(
            {"/api/v1/admin/feedback/summary": _response(200, {"responseCount": 0})}, calls
        )
    )

    payload = json.loads(
        await _get_platform_analytics(ctx, {"reports": ["feedback"], "days": 9999})
    )

    # An out-of-range window is a model slip, not a reason to answer nothing — and the response
    # states which window was actually used, so the model cannot report the one it asked for.
    assert payload["feedback"]["window_days"] == PLATFORM_ANALYTICS_MAX_DAYS
    assert calls[0][1] is not None and "from" in calls[0][1]


@pytest.mark.parametrize("days", [None, "not-a-number", 0, -5])
async def test_a_bad_window_still_produces_a_report(days: Any) -> None:
    ctx = _ctx(
        translation_room=_client(
            {"/api/v1/admin/feedback/summary": _response(200, {"responseCount": 0})}
        )
    )

    arguments: dict[str, Any] = {"reports": ["feedback"]}
    if days is not None:
        arguments["days"] = days

    payload = json.loads(await _get_platform_analytics(ctx, arguments))

    assert payload["feedback"]["window_days"] >= 1


async def test_several_reports_are_fetched_together() -> None:
    ctx = _ctx(
        workspace=_client({"/api/v1/admin/workspaces": _response(200, {"total": 5})}),
        translation_room=_client(
            {"/api/v1/admin/meetings/counts": _response(200, {"liveNow": 0, "startedToday": 0})}
        ),
        auth=_client({"/api/v1/admin/users": _response(200, {"total": 91})}),
    )

    payload = json.loads(
        await _get_platform_analytics(ctx, {"reports": ["overview", "users", "overview"]})
    )

    assert payload["overview"]["workspaces_total"] == 5
    assert payload["users"]["users_total"] == 91


async def test_an_unknown_report_is_named_rather_than_ignored() -> None:
    ctx = _ctx(auth=_client({"/api/v1/admin/users": _response(200, {"total": 1})}))

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["users", "churn"]}))

    assert payload["users"]["users_total"] == 1
    assert payload["unknown_reports"] == ["churn"]


async def test_asking_for_nothing_says_what_can_be_asked_for() -> None:
    """Never an empty answer the model has to guess its way out of."""
    payload = json.loads(await _get_platform_analytics(_ctx(), {"reports": []}))

    assert payload["error"] == "no_report_requested"
    assert "overview" in payload["message"]


async def test_a_missing_client_degrades_one_report_only() -> None:
    # billing_client is optional on ToolContext, so a deployment without it loses revenue and
    # keeps everything else — rather than the worker refusing to start.
    ctx = _ctx(auth=_client({"/api/v1/admin/users": _response(200, {"total": 7})}))

    payload = json.loads(await _get_platform_analytics(ctx, {"reports": ["revenue", "users"]}))

    assert payload["revenue"]["error"] == "unavailable"
    assert payload["users"]["users_total"] == 7


def test_the_declared_enum_matches_what_can_actually_be_fetched() -> None:
    """An enum member with no handler is a request the model can make and nothing can answer."""
    declared = TOOLS_BY_NAME["get_platform_analytics"].parameters["properties"]["reports"]["items"][
        "enum"
    ]

    from ai_assistant_worker.chat_tools import _ANALYTICS_HANDLERS

    assert sorted(declared) == sorted(_ANALYTICS_HANDLERS)


def test_the_description_forbids_inventing_a_figure() -> None:
    # The instruction that keeps a 403 from becoming a plausible number.
    description = TOOLS_BY_NAME["get_platform_analytics"].description

    assert "never estimate" in description.lower()
    assert "not_authorized" in description
