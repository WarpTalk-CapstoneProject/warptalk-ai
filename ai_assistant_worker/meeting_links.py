"""The join link of a meeting WarpBot just created, guaranteed to reach the answer.

WHY THE WORKER AND NOT THE MODEL
    A meeting WarpBot creates is only useful if the user can get into it. The link came back from
    a tool, and whether it reached the answer was left to the model copying it out. Production,
    17 Sep: "tạo 1 cuộc họp bằng @Google Meet" ended on a sentence about a card and no link at
    all. A join link is not a matter of style, so once a tool has returned one the worker makes
    sure the answer carries it.

WHY A LINK IN THE TEXT AND NOT A NEW FIELD
    The web draws a meeting card from the link itself: a Google Meet URL carries its own meeting
    code (``meet.google.com/abc-defg-hij``), and ``/rooms/{id}`` resolves to the room. So the card
    survives a reload and a reopened conversation with no new column, and every surface that
    renders WarpBot's markdown gets it — the widget, the AI chat page, the in-meeting panel.

TWO KINDS, NEVER CONFUSED
    A Google Meet meeting is hosted by Google; a WarpTalk room is hosted here. They are told
    apart by what the tool returned, not by which tool name the model picked.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

GOOGLE_MEET_URL_PREFIX = "https://meet.google.com/"

#: Google's meeting code shape: three letter groups, e.g. abc-defg-hij.
_MEET_CODE = re.compile(r"^[a-z]{3,4}-[a-z]{3,4}-[a-z]{3,4}$")

MeetingKind = Literal["google_meet", "warptalk_room"]


@dataclass(frozen=True)
class MeetingLink:
    kind: MeetingKind
    url: str
    title: str
    code: str | None = None


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
                return MeetingLink(
                    kind="google_meet",
                    url=link,
                    title=_text(data.get("summary")) or "Google Meet",
                    code=_text(data.get("meetingCode")) or meet_code_from_url(link),
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
            )
    return None


def ensure_meeting_links(answer: str, links: list[MeetingLink]) -> str:
    """Append every created meeting's link the answer does not already carry.

    Language-neutral on purpose: a markdown link titled with the meeting's own name, no framing
    sentence, because the answer is in whatever language the user wrote and a worker-written
    sentence would be in the wrong one half the time.
    """
    missing: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link.url in seen:
            continue
        seen.add(link.url)
        if _mentions(answer, link):
            continue
        title = link.title.replace("[", "(").replace("]", ")")
        missing.append(f"[{title}]({link.url})")

    if not missing:
        return answer
    joined = "\n".join(missing)
    return f"{answer.rstrip()}\n\n{joined}" if answer.strip() else joined


def _mentions(answer: str, link: MeetingLink) -> bool:
    if link.url in answer:
        return True
    # A model often drops the scheme: "meet.google.com/abc-defg-hij".
    bare = link.url.split("://", 1)[-1]
    return bare in answer


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
