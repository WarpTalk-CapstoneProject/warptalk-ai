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

ONE MEETING, ONE CARD (GMCAL1001)
    Each Google plugin calls only its own API. The Meet tool creates the meeting (Meet REST, no
    Calendar event); when Calendar is connected the model chains Calendar's create_event with the
    Meet link. Those are two tool results about ONE meeting, so the Calendar result is folded onto
    the Meet card - its "Open in Calendar" link and its time - instead of becoming a second card.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any, Literal

GOOGLE_MEET_URL_PREFIX = "https://meet.google.com/"

#: A Meet card's title when the tool returned no summary.
MEET_FALLBACK_TITLE = "Google Meet"

#: Google's meeting code shape: three letter groups, e.g. abc-defg-hij.
_MEET_CODE = re.compile(r"^[a-z]{3,4}-[a-z]{3,4}-[a-z]{3,4}$")

#: The only hosts a Google Calendar event link may point at. It reaches the browser as a button.
_CALENDAR_URL_PREFIXES = ("https://www.google.com/calendar/", "https://calendar.google.com/")

#: Opens the machine-readable note appended under an answer (see lib/assistant/meeting-links.ts).
MEETING_MARKER_PREFIX = "<!-- warpbot:meeting "

#: ">" written as a JSON escape, so a title containing "-->" cannot close the comment early.
_ESCAPED_GT = "\\u003e"

#: A marker on its own line, with the newline that carried it.
_MARKER_LINE = re.compile(r"[ \t]*<!-- warpbot:meeting .*?-->[ \t]*\n?")

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
    #: Google Meet only, NOT in the marker: the Calendar event id, kept so the WarpTalk room the
    #: worker files for this meeting can point back at it (externalCalendarEventId).
    calendar_event_id: str | None = None

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


def room_url(room_id: str, slug: str | None = None) -> str:
    """The address of a WarpTalk room, slug-qualified when known.

    If a workspace slug is available, produces ``/{slug}/rooms/{id}`` so the browser opens
    the canonical room detail page directly. Falls back to ``/rooms/{id}`` when unknown.
    """
    clean_id = room_id.strip()
    if slug and slug.strip():
        return f"/{slug.strip()}/rooms/{clean_id}"
    return f"/rooms/{clean_id}"


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
    # GMCAL1001: the Meet tool now answers from the Meet REST API - link and code, with start, end
    # and summary only echoed when the model sent them - and no Calendar event. A card is still
    # owed for that: the link is the meeting. calendarEventLink / eventId are read only for a
    # service still on the old Calendar-backed shape.
    data = payload.get("data")
    if payload.get("isSuccess") is True and isinstance(data, dict):
        if data.get("provider") == "google_meet":
            link = _text(data.get("meetLink"))
            if link and link.startswith(GOOGLE_MEET_URL_PREFIX):
                return MeetingLink(
                    kind="google_meet",
                    url=link,
                    title=_text(data.get("summary")) or MEET_FALLBACK_TITLE,
                    code=_text(data.get("meetingCode")) or meet_code_from_url(link),
                    start=_text(data.get("start")) or None,
                    end=_text(data.get("end")) or None,
                    calendar_url=_calendar_url(data.get("calendarEventLink")),
                    calendar_event_id=_text(data.get("eventId")) or None,
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


@dataclass(frozen=True)
class CalendarEvent:
    """A Google Calendar event the Calendar plugin created this turn for a Google Meet meeting."""

    meet_url: str
    meet_code: str | None
    event_id: str | None = None
    #: The event's htmlLink, already checked against the Calendar host allow-list.
    html_link: str | None = None
    start: str | None = None
    end: str | None = None


def calendar_event_from_tool_result(result_json: str) -> CalendarEvent | None:
    """The Calendar event a successful google_calendar create_event made FOR a Meet link.

    An event with no Meet link on it is not about a meeting WarpBot created, so it is None here:
    it gets no card, and nothing to fold into one.
    """
    try:
        payload = json.loads(result_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("isSuccess") is not True:
        return None
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("provider") != "google_calendar":
        return None

    meet_url = ""
    for key in ("meetLink", "hangoutLink"):
        candidate = _text(data.get(key))
        if candidate.startswith(GOOGLE_MEET_URL_PREFIX):
            meet_url = candidate
            break
    if not meet_url:
        return None
    return CalendarEvent(
        meet_url=meet_url,
        meet_code=meet_code_from_url(meet_url),
        event_id=_text(data.get("eventId")) or None,
        html_link=_calendar_url(data.get("htmlLink")),
        start=_event_time(data.get("start")),
        end=_event_time(data.get("end")),
    )


def bridged_meet_code_from_tool_result(result_json: str) -> str | None:
    """The Meet code of an EXTERNAL_BRIDGE room create_meeting made this turn, if any.

    The worker files a WarpTalk room for every Meet it creates; when the model already did it
    through create_meeting, filing another would put the same meeting on the calendar twice.
    """
    try:
        payload = json.loads(result_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("status") != "created":
        return None
    return meet_code_from_url(_text(payload.get("external_meeting_url")))


def merge_calendar_events(
    links: list[MeetingLink], events: list[CalendarEvent]
) -> list[MeetingLink]:
    """Fold each Calendar event onto the Meet card it was made for, matched by Meet code or URL.

    The Calendar's own time wins over the Meet tool's echo: it is what Google stored, with an
    offset, where the echo is whatever the model typed. An event that matches no card this turn
    is dropped rather than drawn - a card is a meeting WarpBot just created.
    """
    if not events:
        return list(links)
    merged: list[MeetingLink] = []
    for link in links:
        event = (
            next((e for e in events if _same_meeting(link, e)), None)
            if link.kind == "google_meet"
            else None
        )
        if event is None:
            merged.append(link)
            continue
        merged.append(
            replace(
                link,
                calendar_url=event.html_link or link.calendar_url,
                calendar_event_id=event.event_id or link.calendar_event_id,
                start=event.start or link.start,
                end=event.end or link.end,
            )
        )
    return merged


def _same_meeting(link: MeetingLink, event: CalendarEvent) -> bool:
    link_code = link.code or meet_code_from_url(link.url)
    if link_code and event.meet_code:
        return link_code.lower() == event.meet_code.lower()
    return _bare_url(link.url) == _bare_url(event.meet_url)


def _bare_url(url: str) -> str:
    return url.split("?", 1)[0].split("#", 1)[0].rstrip("/").lower()


def _calendar_url(value: Any) -> str | None:
    url = _text(value)
    return url if url.startswith(_CALENDAR_URL_PREFIXES) else None


def _event_time(value: Any) -> str | None:
    """A Calendar time as sent: a string, or Google's own {dateTime, timeZone} object."""
    if isinstance(value, dict):
        value = value.get("dateTime") or value.get("date")
    return _text(value) or None


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


def strip_meeting_markers(text: str) -> str:
    """The answer without its markers — what history hands back to the model.

    A model that sees markers in its own past answers writes them, and a marker it wrote is a
    card, with a join link and a code, under an answer that only talks about a meeting.
    """
    if not text or MEETING_MARKER_PREFIX not in text:
        return text
    without = _MARKER_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", without).strip()


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
