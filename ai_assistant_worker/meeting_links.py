"""The meeting WarpBot just created, handed to the web as a card it cannot lose.

WHY THE WORKER AND NOT THE MODEL
    A meeting WarpBot creates is only useful if the user can get into it. The link came back from
    a tool, and whether it reached the answer was left to the model copying it out. Production,
    17 Sep: "tạo 1 cuộc họp bằng @Google Meet" ended on a sentence about a card and no link at
    all. A join link is not a matter of style, so once a tool has returned one the worker attaches
    it to the answer itself.

WHY A MARKER IN THE TEXT AND NOT A NEW FIELD
    The meeting travels as an HTML comment at the end of the answer. Markdown renders nothing for
    a comment, so the reader sees the prose and, under it, the card the web draws from the
    comment's JSON: link, meeting code, time, and Calendar event. It is stored with the message, so
    the card comes back when a conversation is reopened, on every surface that renders WarpBot's
    answers, with no new column.

    Cards come ONLY from these markers, never from links the model happened to write: a link in an
    answer about yesterday's meetings is not a meeting WarpBot just created.

TWO KINDS, NEVER CONFUSED
    A Google Meet meeting is hosted by Google; a WarpTalk room is hosted here. They are told apart
    by what the tool returned, not by which tool name the model picked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

GOOGLE_MEET_URL_PREFIX = "https://meet.google.com/"

#: Google's meeting code shape: three letter groups, e.g. abc-defg-hij.
_MEET_CODE = re.compile(r"^[a-z]{3,4}-[a-z]{3,4}-[a-z]{3,4}$")

#: The only hosts a Google Calendar event link may point at. It reaches the browser as a button.
_CALENDAR_URL_PREFIXES = ("https://www.google.com/calendar/", "https://calendar.google.com/")

#: Opens the machine-readable note appended under an answer (see lib/assistant/meeting-links.ts).
MEETING_MARKER_PREFIX = "<!-- warpbot:meeting "

#: ">" written as a JSON escape, so a title containing "-->" cannot close the comment early.
_ESCAPED_GT = "\\u003e"

MeetingKind = Literal["google_meet", "warptalk_room"]


@dataclass(frozen=True)
class MeetingLink:
    kind: MeetingKind
    url: str
    title: str
    code: str | None = None
    #: ISO 8601 as the service returned it. Google Meet: the event's start/end. Room: scheduled_at.
    start: str | None = None
    end: str | None = None
    #: Google Meet only: the Calendar event, for "Open in Calendar".
    calendar_url: str | None = None
    #: WarpTalk room only: EXTERNAL_BRIDGE is the room that translates a Google Meet meeting.
    room_type: str | None = None

    def marker(self) -> str:
        fields = {
            "kind": self.kind,
            "url": self.url,
            "title": self.title,
            "code": self.code,
            "start": self.start,
            "end": self.end,
            "calendarUrl": self.calendar_url,
            "roomType": self.room_type,
        }
        body = json.dumps(
            {key: value for key, value in fields.items() if value}, ensure_ascii=False
        )
        return f"{MEETING_MARKER_PREFIX}{body.replace('>', _ESCAPED_GT)} -->"


def room_url(room_id: str) -> str:
    """The slug-less address of a WarpTalk room.

    The worker is not told the workspace slug. ``/rooms/{id}`` is the address notifications
    already use, and the web redirects it to ``/{slug}/rooms/{id}`` for the open workspace.
    """
    return f"/rooms/{room_id.strip()}"


def meet_code_from_url(url: str) -> str | None:
    if not url.startswith(GOOGLE_MEET_URL_PREFIX):
        return None
    path = url[len(GOOGLE_MEET_URL_PREFIX) :].split("?", 1)[0].split("#", 1)[0].strip("/")
    return path if _MEET_CODE.fullmatch(path) else None


def meeting_link_from_tool_result(result_json: str) -> MeetingLink | None:
    """The meeting a successful tool call created, or None for anything else."""
    try:
        payload = json.loads(result_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    # AssistantService's plugin envelope: {isSuccess, data: {provider: "google_meet", ...}}.
    data = payload.get("data")
    if payload.get("isSuccess") is True and isinstance(data, dict):
        if data.get("provider") == "google_meet":
            link = _text(data.get("meetLink"))
            if link and link.startswith(GOOGLE_MEET_URL_PREFIX):
                calendar_url = _text(data.get("calendarEventLink"))
                return MeetingLink(
                    kind="google_meet",
                    url=link,
                    title=_text(data.get("summary")) or "Google Meet",
                    code=_text(data.get("meetingCode")) or meet_code_from_url(link),
                    start=_text(data.get("start")) or None,
                    end=_text(data.get("end")) or None,
                    calendar_url=calendar_url
                    if calendar_url.startswith(_CALENDAR_URL_PREFIXES)
                    else None,
                )
        return None

    # create_meeting's own result.
    if payload.get("status") == "created":
        url = _text(payload.get("room_url"))
        if url:
            return MeetingLink(
                kind="warptalk_room",
                url=url,
                title=_text(payload.get("title")) or "WarpTalk room",
                code=_text(payload.get("room_code")) or None,
                start=_text(payload.get("scheduled_at")) or None,
                room_type=_text(payload.get("room_type")) or None,
            )
    return None


def ensure_meeting_links(answer: str, links: list[MeetingLink]) -> str:
    """Append one marker per meeting created this turn.

    No visible link is added: the card under the answer carries the link and the code, and a
    second copy in the prose is what the mockups deliberately leave out.
    """
    markers: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        markers.append(link.marker())

    if not markers:
        return answer
    joined = "\n".join(markers)
    return f"{answer.rstrip()}\n\n{joined}" if answer.strip() else joined


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
