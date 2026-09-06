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

    There is NO document attachment table either — no room↔document link exists in the schema.
    Documents are therefore LINKED from the description rather than joined to the room: the
    description is a TipTap field with the Markdown extension enabled (`Markdown.configure({
    html: true })` on the room page), so a markdown list of links renders as links, and clicking
    one opens the document. That is the whole feature without a migration, and it degrades
    honestly — a reader with no access to a linked document sees a link, not a broken join.
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
    "EXTERNAL_BRIDGE",
)

EXTERNAL_PROVIDER_GOOGLE_MEET = "GOOGLE_MEET"

# "This meeting is not hosted anywhere else" - the WT-399 trap, a third time.
#
# external_provider was offered to the model as a single-member enum, ["GOOGLE_MEET"], whose
# only legal value asserts the meeting IS an externally hosted bridge. A model fills every
# property it is offered, so an ordinary internal request came back tagged GOOGLE_MEET,
# validate() refused it for not being an EXTERNAL_BRIDGE, and the one argument the model could
# change to escape was the room type - which passes validate() here AND the room API, and
# produces a two-seat bridge room needing the desktop shell and virtual audio drivers for a
# meeting that was never external at all.
#
# Same fix as NO_RECURRENCE and NO_TRANSLATION above: give the enum the member that means
# "none of these", and strip it before the payload is built.
NO_EXTERNAL_PROVIDER = "NONE"
EXTERNAL_PROVIDER_CHOICES: tuple[str, ...] = (
    NO_EXTERNAL_PROVIDER,
    EXTERNAL_PROVIDER_GOOGLE_MEET,
)

# Other spellings of "not an external meeting" a model reaches for.
_NO_EXTERNAL_PROVIDER_ALIASES = frozenset(
    {NO_EXTERNAL_PROVIDER, "NO", "INTERNAL", "WARPTALK", "NOT_EXTERNAL", "NA"}
)

#: The only host a Google Meet join link may use. Mirrors the room API's own allow-list so a
#: plausible-looking link the model composed itself is refused here, before it is stored and
#: handed to every invitee as the Join button.
GOOGLE_MEET_URL_PREFIX = "https://meet.google.com/"

RECURRENCE_TYPES: tuple[str, ...] = ("DAILY", "WEEKLY", "MONTHLY")

# What the TOOL offers the model, which is not what the server accepts.
#
# WT-399. A model does not omit optional properties — it fills every one of them, with "" for a
# string, [] for an array, 0 for a number. `recurrence_type` was the one property with no filler
# available: an enum of three repeating rules and no member meaning "this does not repeat". So a
# one-off meeting arrived tagged DAILY, `missing_fields` then demanded the start time a DAILY
# rule needs, the model asked the user for a time they had explicitly said they did not want,
# and there was no argument it could send that escaped — every route led back to the same two
# errors until the turn ran out of tool iterations.
#
# NONE is that missing member. It is stripped in draft_from_arguments and never reaches the
# server, which knows only the three above.
NO_RECURRENCE = "NONE"
RECURRENCE_CHOICES: tuple[str, ...] = (NO_RECURRENCE, *RECURRENCE_TYPES)

# Other spellings of "not repeating" a model reaches for when it has not read the enum.
_NO_RECURRENCE_ALIASES = frozenset({NO_RECURRENCE, "NEVER", "ONCE", "NO", "ONE_OFF", "SINGLE"})

# "This meeting does not need translating" — the answer that had no way to be given.
#
# Exactly the WT-399 trap above, one field over, and it survived because the fix was applied to
# recurrence_type alone. `target_languages` is required, and a model fills every property it is
# offered, so a monolingual meeting had no argument it could send: an empty list reads as "not
# answered yet" and re-asks, and any real language code claims a translation the user refused.
#
# Production, 15 Aug. The user opened with "tạo meeting ngay bây giờ để nói về starlink, mời ngô
# xuân hạnh nhi vào, ngôn ngữ tiếng việt" — the language was in the FIRST sentence — and was then
# asked for it eight times, each answer rejected and re-asked in different words. It ended by
# accepting "Tiếng Việt, có dịch sang tiếng Việt": a vi→vi meeting, which generates no audio route
# at all. Eight turns to arrive at a configuration that translates nothing, having started from a
# request that wanted exactly that.
#
# Resolved to [source_language] in draft_from_arguments — the server has no "monolingual" flag,
# and a room whose source and targets match is precisely how it represents one.
NO_TRANSLATION = "NONE"

# What a model reaches for when it means "no translation" and has not read the enum. Includes the
# words a Vietnamese-speaking user's answer gets paraphrased into.
_NO_TRANSLATION_ALIASES = frozenset(
    {NO_TRANSLATION, "NO", "NONE", "SAME", "SAME_AS_SOURCE", "NO_TRANSLATION", "MONOLINGUAL"}
)

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
    # True when the user has said this meeting needs no translation. Distinct from an empty
    # target_languages, which means "not asked yet" — the two were the same value, and that is
    # why the question could not be answered. Never sent to the server; see build_payload.
    no_translation: bool = False
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
    #: (title, document_id) pairs. Linked from the description — see the module docstring.
    documents: list[tuple[str, str]] = field(default_factory=list)
    #: Needed to build a document URL. Absent means documents are listed by name only.
    workspace_slug: str | None = None
    external_provider: str | None = None
    external_meeting_url: str | None = None
    external_calendar_event_id: str | None = None
    external_calendar_event_url: str | None = None


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
    # A declared "no translation" counts as ANSWERED. Treating it as absent is what made the
    # question unanswerable: the user says they want one language, the draft records nothing, and
    # the model asks again — forever. See NO_TRANSLATION.
    targets = [t for t in draft.target_languages if (t or "").strip()]
    if not targets and not draft.no_translation:
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

    if draft.external_provider and draft.external_provider != EXTERNAL_PROVIDER_GOOGLE_MEET:
        problems.append(
            "external_provider must be GOOGLE_MEET when external meeting metadata is sent."
        )

    # A provider with no link is a bridge room nobody can reach: the schedule only renders the
    # join affordance when both are present, so it fails silently rather than visibly.
    if draft.external_provider and not draft.external_meeting_url:
        problems.append("external_meeting_url is required when external_provider is set.")

    # The four values are supposed to come from the Google Calendar plugin's own response. That
    # tool runs in AssistantService, not here, so nothing about their provenance can be checked -
    # a fabricated https://meet.google.com/abc-defg-hij would otherwise be persisted and echoed
    # back as "created", and the user would find a Meet chip pointing at a room that never existed.
    # The host check is the one thing that can be enforced from here.
    if draft.external_meeting_url and not draft.external_meeting_url.startswith(
        GOOGLE_MEET_URL_PREFIX
    ):
        problems.append(
            f"external_meeting_url must start with {GOOGLE_MEET_URL_PREFIX} - send the link the "
            "Google Calendar tool returned, never one you composed."
        )

    if (
        any(
            (
                draft.external_provider,
                draft.external_meeting_url,
                draft.external_calendar_event_id,
                draft.external_calendar_event_url,
            )
        )
        and draft.translation_room_type != "EXTERNAL_BRIDGE"
    ):
        problems.append(
            "External meeting metadata is only valid for translation_room_type EXTERNAL_BRIDGE."
        )

    return problems


def compose_description(draft: MeetingDraft) -> str | None:
    """Description, agenda and document links in one markdown field.

    MARKDOWN, not plain text and not HTML. The room page edits this with TipTap under
    `Markdown.configure({ html: true })` and styles `[&_h2]`, so `## Agenda` renders as a heading
    and `1.` renders as a list. Writing prose here would lose the structure the user asked for;
    writing HTML would fight the editor that owns the field.

    The agenda gets its own heading rather than being merged into the prose: an agenda that
    survives as a numbered list is the whole point of having asked for one.
    """
    parts: list[str] = []
    description = (draft.description or "").strip()
    if description:
        parts.append(description)

    items = [line.strip() for line in draft.agenda if (line or "").strip()]
    if items:
        parts.append("## Agenda\n" + "\n".join(f"{i}. {line}" for i, line in enumerate(items, 1)))

    links = [
        _document_link(title, document_id, draft.workspace_slug)
        for title, document_id in draft.documents
        if (title or "").strip() and (document_id or "").strip()
    ]
    if links:
        # Linked, not attached. There is no room↔document table, and a link that opens the
        # document is the thing a reader actually wanted from an attachment.
        parts.append("## Documents\n" + "\n".join(f"- {link}" for link in links))

    return "\n\n".join(parts) if parts else None


def _document_link(title: str, document_id: str, workspace_slug: str | None) -> str:
    """A markdown link, or a bare name when there is no slug to build a URL from.

    Degrading to the name is deliberate: a link to `/undefined/documents/...` is worse than no
    link, because it looks clickable and goes nowhere.
    """
    clean_title = title.strip()
    if not workspace_slug:
        return clean_title
    return f"[{clean_title}](/{workspace_slug}/documents/{document_id.strip()})"


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

    if draft.external_provider:
        payload["externalProvider"] = draft.external_provider
    if draft.external_meeting_url:
        payload["externalMeetingUrl"] = draft.external_meeting_url
    if draft.external_calendar_event_id:
        payload["externalCalendarEventId"] = draft.external_calendar_event_id
    if draft.external_calendar_event_url:
        payload["externalCalendarEventUrl"] = draft.external_calendar_event_url

    return payload


def _external_provider(value: str | None) -> str | None:
    """The provider the server should see, or None when the model meant "not external".

    Upper-cased because the two ends disagree and nothing else reconciles them: the Google
    Workspace plugin gateway returns ``google_meet`` in lower case, while the room API compares
    against ``GOOGLE_MEET`` with an ordinal comparison. Do not simplify this away.
    """
    text = (value or "").strip().upper()
    if not text or text in _NO_EXTERNAL_PROVIDER_ALIASES:
        return None
    return text


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

    source_language = as_text(args.get("source_language"))

    # "NONE" and its cousins are the answer "this meeting needs no translation". Resolved here so
    # the sentinel never reaches the draft, the payload or the server — the same treatment
    # recurrence_type's NONE gets a few lines below.
    raw_targets = as_list(args.get("target_languages"))
    declared_no_translation = any(t.strip().upper() in _NO_TRANSLATION_ALIASES for t in raw_targets)
    target_languages = [t for t in raw_targets if t.strip().upper() not in _NO_TRANSLATION_ALIASES]

    # A monolingual meeting IS a room whose targets match its source — that is how the server
    # represents one, and it is what produces no audio route between same-language participants.
    # Saying it explicitly beats leaving the list empty, which reads downstream as "unanswered".
    if declared_no_translation and not target_languages and source_language:
        target_languages = [source_language]

    # The user naming one language for the whole meeting is the same statement. It arrived as
    # source=vi, targets=[vi] and was re-asked as though nothing had been said.
    if (
        source_language
        and target_languages
        and all(t.strip().lower() == source_language.strip().lower() for t in target_languages)
    ):
        declared_no_translation = True

    room_type = as_text(args.get("translation_room_type"))

    scheduled_at = as_text(args.get("scheduled_at"))
    recurrence_type = as_text(args.get("recurrence_type"))
    recurrence_type = recurrence_type.upper() if recurrence_type else None
    recurrence_start_time_local = as_text(args.get("recurrence_start_time_local"))
    recurrence_start_date_local = as_text(args.get("recurrence_start_date_local"))

    if recurrence_type in _NO_RECURRENCE_ALIASES:
        recurrence_type = None
    elif (
        recurrence_type
        and scheduled_at
        and not recurrence_start_time_local
        and not recurrence_start_date_local
    ):
        # WT-399. A rule with a concrete one-off start AND nothing that makes it a rule — no
        # time of day, no first date — is the enum's filler value showing through, not a request
        # to repeat anything. Read it as the one-off the scheduled_at already describes.
        #
        # Narrow on purpose. A recurrence carrying ANY detail of its own is taken at face value
        # and, if it also carries a scheduled_at, still fails `validate` as a contradiction — a
        # user who asked for a weekly series must never be given one meeting instead, which is a
        # far worse failure than the error this replaces.
        recurrence_type = None

    # 0 is the number-shaped filler, the same way "" is the string one — see RECURRENCE_CHOICES.
    # Left as a real value it trips the "at least 2" rule in validate() and becomes the next dead
    # end in the same loop. A genuine 1 is still an error: somebody meant it.
    #
    # Written out rather than left to `str(args.get(...) or "").strip().isdigit()`, which dropped
    # the integer 0 only because 0 is falsy — while letting the STRING "0" through to fail
    # validation. Which of those a model sends is not something to leave to chance.
    max_participants_raw = args.get("max_participants")
    max_participants: int | None = None
    if max_participants_raw is not None:
        try:
            parsed = int(str(max_participants_raw).strip())
        except (TypeError, ValueError):
            parsed = 0
        if parsed > 0:
            max_participants = parsed

    documents: list[tuple[str, str]] = []
    raw_documents = args.get("documents")
    if isinstance(raw_documents, list):
        for entry in raw_documents:
            if isinstance(entry, dict):
                doc_title = str(entry.get("title") or "").strip()
                doc_id = str(entry.get("id") or "").strip()
                if doc_title and doc_id:
                    documents.append((doc_title, doc_id))

    return MeetingDraft(
        documents=documents,
        workspace_slug=as_text(args.get("workspace_slug")),
        title=as_text(args.get("title")),
        description=as_text(args.get("description")),
        agenda=as_list(args.get("agenda")),
        # Upper-cased so "webinar" from a model that did not read the enum still lands.
        translation_room_type=room_type.upper() if room_type else None,
        source_language=source_language,
        target_languages=target_languages,
        no_translation=declared_no_translation,
        scheduled_at=scheduled_at,
        invited_emails=as_list(args.get("invited_emails")),
        recurrence_type=recurrence_type,
        recurrence_start_time_local=recurrence_start_time_local,
        recurrence_time_zone=as_text(args.get("recurrence_time_zone")),
        recurrence_start_date_local=recurrence_start_date_local,
        recurrence_end_date_local=as_text(args.get("recurrence_end_date_local")),
        max_participants=max_participants,
        external_provider=_external_provider(as_text(args.get("external_provider"))),
        external_meeting_url=as_text(args.get("external_meeting_url")),
        external_calendar_event_id=as_text(args.get("external_calendar_event_id")),
        external_calendar_event_url=as_text(args.get("external_calendar_event_url")),
    )
