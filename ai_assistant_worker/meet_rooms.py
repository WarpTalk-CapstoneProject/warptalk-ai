"""Every Google Meet WarpBot creates lands on WarpTalk's own Calendar page (GMCAL1001).

WHY THE WORKER AND NOT THE MODEL
    The Calendar page lists WarpTalk rooms; a Google Meet meeting appears there as an
    EXTERNAL_BRIDGE room carrying the Meet link (drawn with the Meet logo). That room used to be a
    prompt rule - "after Google Meet, create an EXTERNAL_BRIDGE room" - which the model followed
    when it remembered to, asked about languages when it did not know them, and sometimes did
    twice. Filing it is bookkeeping, not a decision, so the worker does it once per Meet, after the
    turn's tool calls are done (so a Calendar event chained in the same turn is linked too).

WHAT IT SENDS
    POST /api/v1/translation-rooms as the user, the same call and the same token create_meeting
    uses, so the workspace's own create permission and language policy apply. Languages are left
    out on purpose: the room service falls back to the user's own speak/listen defaults (WT-65),
    which is what the desktop bridge would use too. ScheduledAt is sent only when it is in the
    future - the API refuses a past one - so "now" becomes a WAITING room the calendar places at
    its creation time. The API has no end/duration, so the end is not sent.

ONE ROOM PER MEET
    Within a turn: one room per Meet code, and none when create_meeting already bridged that code.
    Across the desktop bridge: POST /bridge/claim (#493) finds an open room by its stored Meet
    code, so the room service must stamp that code on a GOOGLE_MEET bridge room created here too
    (and answer the existing room on a conflict) for a later desktop join to land in this room.

NEVER BREAKS THE ANSWER
    The Meet already exists and the card already carries its link. A room that could not be filed
    is logged and forgotten; nothing here raises.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from ai_assistant_worker.meeting_draft import EXTERNAL_PROVIDER_GOOGLE_MEET
from ai_assistant_worker.meeting_links import (
    GOOGLE_MEET_URL_PREFIX,
    MEET_FALLBACK_TITLE,
    MeetingLink,
    meet_code_from_url,
)
from shared.logger import get_logger

logger = get_logger(__name__)

#: What a Meet created without a title is called on the WarpTalk calendar.
DEFAULT_ROOM_TITLE = "Google Meet meeting"

#: A bare date-time from the model is read in this zone when the tool call named none - the same
#: default create_meeting's recurrence uses.
DEFAULT_TIME_ZONE = "Asia/Ho_Chi_Minh"

#: A start this close to now is "now": the API refuses a ScheduledAt that is not in the future, and
#: one a few seconds ahead would be in the past by the time it is validated.
_NOW_TOLERANCE = timedelta(minutes=1)

#: The room is bookkeeping; the answer is already written. Never hold the turn longer than this.
_REQUEST_TIMEOUT_SECONDS = 10.0


def build_meet_room_payload(
    link: MeetingLink,
    workspace_id: str,
    *,
    time_zone: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """The create body for the room that puts this Meet on the WarpTalk calendar, or None."""
    if link.kind != "google_meet" or not link.url.startswith(GOOGLE_MEET_URL_PREFIX):
        return None
    code = link.code or meet_code_from_url(link.url)
    # The canonical form, the same one bridge claim stores (GoogleMeetCode.ToUrl).
    meet_url = f"{GOOGLE_MEET_URL_PREFIX}{code.lower()}" if code else link.url

    title = (link.title or "").strip()
    payload: dict[str, Any] = {
        "workspaceId": workspace_id,
        "title": title if title and title != MEET_FALLBACK_TITLE else DEFAULT_ROOM_TITLE,
        "translationRoomType": "EXTERNAL_BRIDGE",
        "externalProvider": EXTERNAL_PROVIDER_GOOGLE_MEET,
        "externalMeetingUrl": meet_url,
    }
    scheduled_at = scheduled_at_utc(link.start, time_zone, now=now)
    if scheduled_at:
        payload["scheduledAt"] = scheduled_at
    if link.calendar_event_id:
        payload["externalCalendarEventId"] = link.calendar_event_id
    if link.calendar_url:
        payload["externalCalendarEventUrl"] = link.calendar_url
    return payload


def scheduled_at_utc(
    start: str | None, time_zone: str | None = None, *, now: datetime | None = None
) -> str | None:
    """`start` as an ISO-8601 UTC instant, or None when absent, unreadable or not in the future."""
    text = (start or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_zone(time_zone))
    instant = parsed.astimezone(UTC)
    if instant <= (now or datetime.now(UTC)) + _NOW_TOLERANCE:
        return None
    return instant.strftime("%Y-%m-%dT%H:%M:%SZ")


async def file_meet_rooms(
    ctx: Any,
    links: Iterable[MeetingLink],
    *,
    already_bridged: set[str],
    time_zones: dict[str, str] | None = None,
) -> None:
    """Create one EXTERNAL_BRIDGE room per Google Meet created this turn. Never raises.

    `already_bridged` holds the Meet codes create_meeting already made a room for this turn.
    """
    client = getattr(ctx, "translation_room_client", None)
    workspace_id = getattr(ctx, "workspace_id", None)
    if not isinstance(client, httpx.AsyncClient) or not workspace_id:
        return

    done = {code.lower() for code in already_bridged}
    for link in links:
        if link.kind != "google_meet":
            continue
        code = (link.code or meet_code_from_url(link.url) or "").lower()
        if code and code in done:
            continue
        payload = build_meet_room_payload(
            link, workspace_id, time_zone=(time_zones or {}).get(code)
        )
        if payload is None:
            continue
        if code:
            done.add(code)
        await _create_room(client, ctx, payload, code)


async def _create_room(
    client: httpx.AsyncClient, ctx: Any, payload: dict[str, Any], code: str
) -> None:
    token = getattr(ctx, "bearer_token", None)
    headers = {"Authorization": token} if token else {}
    try:
        response = await asyncio.wait_for(
            client.post("/api/v1/translation-rooms", json=payload, headers=headers),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.exception("meet_room_create_error", meet_code=code)
        return
    if response.status_code not in (200, 201):
        logger.warning("meet_room_create_failed", status=response.status_code, meet_code=code)
        return
    room_id = None
    try:
        body = response.json()
        if isinstance(body, dict):
            room_id = body.get("id")
    except Exception:
        room_id = None
    logger.info("meet_room_created", meet_code=code, room_id=room_id)


def _zone(name: str | None) -> tzinfo:
    for candidate in (name, DEFAULT_TIME_ZONE):
        if candidate:
            try:
                return ZoneInfo(candidate)
            except Exception:
                continue
    return UTC
