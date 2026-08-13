"""Turning what a user said into a meeting the API will actually accept.

WHY THIS IS A MODULE AND NOT A PROMPT
    "Create a meeting" is the assistant's first tool that WRITES. Every other tool answers a
    question and the worst case is a wrong answer; this one puts a row in the database, mails
    invitations to real colleagues, and — for a recurring booking — does it repeatedly. The rules
    about what may be sent are therefore not left to the model to remember between turns. They
    live here, they are the same rules on every call, and they are tested.

    A model can be told "ScheduledAt and Recurrence are mutually exclusive" and still send both
    on the turn where the user says "daily at 9, starting Monday". The server rejects that pair
    rather than resolving it, which is correct of the server and useless to the user, who sees an
    error for a sentence that made perfect sense.

WHAT THE API ACTUALLY HAS
    Checked against CreateTranslationRoomRequest, not assumed: Title, Description,
    TranslationRoomType, MaxParticipants, SourceLanguage, TargetLanguages, Settings, ScheduledAt,
    InvitedEmails, Recurrence.

    There is NO agenda field, so an agenda is folded into the description under its own heading
    rather than silently dropped — the user asked for one and must be able to see it on the
    meeting afterwards.

    There is NO document attachment, on this endpoint or any other: no room↔document link exists
    in the schema. `missing_fields` therefore never asks for documents, because collecting an
    answer nothing can store would be a worse lie than saying it is not supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# From TranslationRoomTypes. Sending anything else is a 400 the user cannot act on.
MEETING_TYPES: tuple[str, ...] = (
    "EVENT",
    "CHANNEL_MEETING",
    "WEBINAR",
    "COMPANY_MEETING",
    "VIRTUAL_APPOINTMENT",
    "LIVE_EVENT",
)

RECURRENCE_TYPES: tuple[str, ...] = ("DAILY", "WEEKLY", "MONTHLY")

# The four the server cannot default. Everything else has a sensible fallback, and asking about
# it would turn a one-sentence request into an interrogation.
REQUIRED_FIELDS: tuple[str, ...] = (
    "title",
    "translation_room_type",
    "source_language",
    "target_languages",
)

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_TIME_24H = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_DATE_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class MeetingDraft:
    """Everything gathered so far. Every field optional — a draft is allowed to be incomplete."""

    title: str | None = None
    description: str | None = None
    agenda: list[str] = field(default_factory=list)
    translation_room_type: str | None = None
    source_language: str | None = None
    target_languages: list[str] = field(default_factory=list)
    scheduled_at: str | None = None
    invited_emails: list[str] = field(default_factory=list)
    recurrence_type: str | None = None
    recurrence_start_time_local: str | None = None
    recurrence_time_zone: str | None = None
    recurrence_start_date_local: str | None = None
    recurrence_end_date_local: str | None = None
    max_participants: int | None = None


def missing_fields(draft: MeetingDraft) -> list[str]:
    """Which required answers are still absent, in the order worth asking about them.

    Order is deliberate: a person can answer "what is it called" without thinking, and answering
    "which languages" is easier once they have said what the meeting is. Asking the hard one first
    is how a clarifying question becomes a form.
    """
    absent: list[str] = []
    if not (draft.title or "").strip():
        absent.append("title")
    if not draft.translation_room_type:
        absent.append("translation_room_type")
    if not (draft.source_language or "").strip():
        absent.append("source_language")
    if not [t for t in draft.target_languages if (t or "").strip()]:
        absent.append("target_languages")

    # Recurrence is not required — but a HALF-specified one is worse than none, because the
    # server would either reject it or invent the missing half.
    if draft.recurrence_type and not draft.recurrence_start_time_local:
        absent.append("recurrence_start_time_local")
    return absent


def validate(draft: MeetingDraft) -> list[str]:
    """Problems that would come back as a 400, phrased so the model can fix them itself."""
    problems: list[str] = []

    if draft.translation_room_type and draft.translation_room_type not in MEETING_TYPES:
        problems.append(
            f"translation_room_type must be one of {', '.join(MEETING_TYPES)}; "
            f"got '{draft.translation_room_type}'."
        )

    if draft.recurrence_type and draft.recurrence_type not in RECURRENCE_TYPES:
        problems.append(
            f"recurrence_type must be one of {', '.join(RECURRENCE_TYPES)}; "
            f"got '{draft.recurrence_type}'."
        )

    # The trap this module exists for. "Daily at 9 starting Monday" reads as both a schedule and
    # a rule, and the server refuses the pair rather than picking one.
    if draft.recurrence_type and draft.scheduled_at:
        problems.append(
            "A meeting is either a single scheduled_at or a recurrence rule, never both. "
            "For a repeating meeting, put the first date in recurrence_start_date_local and "
            "leave scheduled_at empty."
        )

    if draft.recurrence_start_time_local and not _TIME_24H.match(draft.recurrence_start_time_local):
        problems.append("recurrence_start_time_local must be 24-hour HH:mm, e.g. '09:00'.")

    for label, value in (
        ("recurrence_start_date_local", draft.recurrence_start_date_local),
        ("recurrence_end_date_local", draft.recurrence_end_date_local),
    ):
        if value and not _DATE_ISO.match(value):
            problems.append(f"{label} must be yyyy-MM-dd, e.g. '2026-08-20'.")

    bad_emails = [e for e in draft.invited_emails if e and not _EMAIL.match(e.strip())]
    if bad_emails:
        # Named rather than silently dropped: an invitation the user believes they sent and
        # nobody received is the failure people notice a week later.
        problems.append(f"These do not look like email addresses: {', '.join(bad_emails)}.")

    if draft.max_participants is not None and draft.max_participants < 2:
        problems.append("max_participants must be at least 2 — a meeting needs two people.")

    return problems


def compose_description(draft: MeetingDraft) -> str | None:
    """Description and agenda in one field, because the API has only one.

    The agenda is appended under a heading rather than merged into the prose: the user asked for
    an agenda and has to be able to find it on the meeting afterwards, and a numbered list that
    survives as a numbered list is the whole point of having asked.
    """
    parts: list[str] = []
    description = (draft.description or "").strip()
    if description:
        parts.append(description)

    items = [line.strip() for line in draft.agenda if (line or "").strip()]
    if items:
        parts.append("Agenda\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(items, 1)))

    return "\n\n".join(parts) if parts else None


def build_payload(draft: MeetingDraft, workspace_id: str) -> dict[str, Any]:
    """The request body, with absent fields OMITTED rather than sent as null.

    The meeting type seeds whatever is left out — settings, seat count — so sending an explicit
    null is not the same as sending nothing, and would overwrite a sensible default with an empty
    one.
    """
    payload: dict[str, Any] = {
        "workspaceId": workspace_id,
        "title": (draft.title or "").strip(),
        "translationRoomType": draft.translation_room_type,
        "sourceLanguage": draft.source_language,
        "targetLanguages": [t.strip() for t in draft.target_languages if (t or "").strip()],
    }

    description = compose_description(draft)
    if description:
        payload["description"] = description

    if draft.max_participants is not None:
        payload["maxParticipants"] = draft.max_participants

    emails = [e.strip() for e in draft.invited_emails if (e or "").strip()]
    if emails:
        payload["invitedEmails"] = emails

    if draft.recurrence_type:
        recurrence: dict[str, Any] = {
            "type": draft.recurrence_type,
            "startTimeLocal": draft.recurrence_start_time_local,
            # A rule without a zone is a rule that means different things to different readers.
            "timeZone": draft.recurrence_time_zone or "Asia/Ho_Chi_Minh",
        }
        if draft.recurrence_start_date_local:
            recurrence["startDateLocal"] = draft.recurrence_start_date_local
        if draft.recurrence_end_date_local:
            recurrence["endDateLocal"] = draft.recurrence_end_date_local
        payload["recurrence"] = recurrence
    elif draft.scheduled_at:
        payload["scheduledAt"] = draft.scheduled_at

    return payload


def draft_from_arguments(arguments: dict[str, Any]) -> MeetingDraft:
    """Build a draft from tool arguments, tolerating the shapes a model actually emits."""
    args = arguments or {}

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            # Models hand back "vi, en" as often as ["vi", "en"].
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, list):
            return [str(part).strip() for part in value if str(part).strip()]
        return []

    def as_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    room_type = as_text(args.get("translation_room_type"))
    recurrence_type = as_text(args.get("recurrence_type"))

    return MeetingDraft(
        title=as_text(args.get("title")),
        description=as_text(args.get("description")),
        agenda=as_list(args.get("agenda")),
        # Upper-cased so "webinar" from a model that did not read the enum still lands.
        translation_room_type=room_type.upper() if room_type else None,
        source_language=as_text(args.get("source_language")),
        target_languages=as_list(args.get("target_languages")),
        scheduled_at=as_text(args.get("scheduled_at")),
        invited_emails=as_list(args.get("invited_emails")),
        recurrence_type=recurrence_type.upper() if recurrence_type else None,
        recurrence_start_time_local=as_text(args.get("recurrence_start_time_local")),
        recurrence_time_zone=as_text(args.get("recurrence_time_zone")),
        recurrence_start_date_local=as_text(args.get("recurrence_start_date_local")),
        recurrence_end_date_local=as_text(args.get("recurrence_end_date_local")),
        max_participants=(
            int(args["max_participants"])
            if str(args.get("max_participants") or "").strip().isdigit()
            else None
        ),
    )
