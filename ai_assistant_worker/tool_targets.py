"""What a tool call is ABOUT, in a few words a reader recognises.

WHY A STEP NEEDS A TARGET
    "Searching documents…" answers a question nobody asked. Every call is about something —
    a phrase, a file, a room, a term — and that is the half a person reads to decide whether
    WarpBot understood them. Without it, a trail of four steps reports only that four tools
    ran, which is the same amount of information as one spinner.

    It is also the only way a wrong turn is visible while it is happening: "Searching
    documents · Q3 budget" when you asked about hiring is a mistake you can interrupt.

WHAT IS DELIBERATELY NOT SHOWN
    Identifiers. A document_id is a UUID to the model and noise to a reader, so a value that
    is only an id is dropped rather than printed — a step reading "Reading the document ·
    0f2c…" is worse than one that simply says it is reading a document.

    Long free text. The target is a label beside a label, not the payload: it is collapsed to
    one line and cut at MAX_TARGET_CHARS with an ellipsis, because a step that wraps to three
    lines pushes the answer off screen while it is being written.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

#: Long enough for a real query ("the onboarding doc for new engineers"), short enough that a
#: step stays one line beside its label in a 320px-wide widget.
MAX_TARGET_CHARS = 56

#: Which argument names carry the point of each call, best first. The first non-empty, non-id
#: value wins; a tool absent from this map falls through to _FALLBACK_KEYS, so a tool added in
#: warptalk-ai without touching this file still names its target when it uses ordinary
#: argument names — the failure mode is a step with no detail, never a wrong one.
_TARGET_KEYS: dict[str, tuple[str, ...]] = {
    "search_workspace_members": ("query", "name", "email"),
    "search_terminology": ("term", "query", "text"),
    "semantic_search": ("query",),
    "search_documents": ("query", "title", "name"),
    "search_facts": ("query", "subject"),
    "translate_text": ("target_language", "target_lang", "text"),
    "list_recent_meetings": ("query",),
    "get_meeting_summary": ("title", "room_code"),
    "get_room_detail": ("title", "room_code"),
    "get_transcript": ("title", "room_code"),
    "get_document": ("title", "name", "file_name"),
    "get_platform_analytics": ("metric", "range", "period"),
    "create_meeting": ("title",),
    # Its target is the question itself, and the question is already rendered as a card.
    "ask_user": (),
}

_FALLBACK_KEYS = ("query", "title", "name", "term", "text", "topic")

_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _clean(value: Any) -> str:
    """One line, trimmed, cut — or empty when the value is not worth showing."""
    if not isinstance(value, str):
        # Numbers and booleans are settings, not subjects: "get_platform_analytics · 30" reads
        # as a target and is not one.
        return ""
    text = " ".join(value.split())
    if not text or _UUID.match(text):
        return ""
    if len(text) > MAX_TARGET_CHARS:
        return text[: MAX_TARGET_CHARS - 1].rstrip() + "…"
    return text


def describe_tool_target(tool_name: str, arguments: Any) -> str:
    """The subject of this call, or "" when the call has no subject worth naming.

    Empty is a normal answer, not a failure: list_recent_meetings with no query is genuinely
    about nothing in particular, and the label alone already says what is happening.
    """
    if not isinstance(arguments, dict):
        return ""

    keys = _TARGET_KEYS.get(tool_name, _FALLBACK_KEYS)
    for key in keys:
        detail = _clean(arguments.get(key))
        if detail:
            return detail
    return ""


def _attr(source: Any, name: str) -> Any:
    """Read a field from an SDK object or from the dict the same payload arrives as.

    The Responses stream hands back model objects, the tests and any replayed payload hand
    back dicts, and a reader written for one shape silently returns nothing for the other.
    """
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def describe_web_search_target(action: Any) -> str:
    """What the hosted web search is doing: the phrase it searched, or the site it opened.

    THE TWO ACTIONS ARE DIFFERENT FACTS
        A `search` action carries the query the model wrote. An `open_page` action carries a
        URL, and the useful part of a URL in a one-line step is the SITE — "Searching the web ·
        openai.com" is what somebody asked for, where the full URL would push the step off the
        widget.

    Returns "" for an action shape this does not recognise, which is the honest answer for a
    vendor event that changed: the step still names the tool, just not its target.
    """
    if action is None:
        return ""

    kind = _attr(action, "type") or ""
    if kind == "search":
        return _clean(_attr(action, "query"))

    url = _attr(action, "url")
    if isinstance(url, str) and url:
        host = urlsplit(url).hostname or ""
        if host:
            return _clean(host[4:] if host.startswith("www.") else host)
        return _clean(url)

    # Some shapes carry a query without naming the action type.
    return _clean(_attr(action, "query"))
