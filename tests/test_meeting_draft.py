"""The rules a WarpBot-created meeting has to obey before it reaches the API.

create_meeting is the assistant's first tool that WRITES. Every other tool answers a question and
the worst case is a wrong answer; this one creates a row, mails invitations to real colleagues,
and for a recurring booking does it repeatedly. So the rules live in code and are tested, rather
than being something the model is asked to remember between turns.
"""

from __future__ import annotations

import pytest

from ai_assistant_worker.meeting_draft import (
    NO_RECURRENCE,
    RECURRENCE_CHOICES,
    RECURRENCE_TYPES,
    MeetingDraft,
    build_payload,
    compose_description,
    draft_from_arguments,
    missing_fields,
    validate,
)

WORKSPACE = "11111111-1111-1111-1111-111111111111"


def _complete() -> MeetingDraft:
    return MeetingDraft(
        title="Sprint review",
        translation_room_type="CHANNEL_MEETING",
        source_language="vi",
        target_languages=["en"],
    )


# ── what still has to be asked ────────────────────────────────────────────────────────────────


def test_a_bare_request_asks_for_the_four_the_server_cannot_default() -> None:
    assert missing_fields(MeetingDraft()) == [
        "title",
        "translation_room_type",
        "source_language",
        "target_languages",
    ]


def test_a_complete_draft_asks_for_nothing() -> None:
    assert missing_fields(_complete()) == []


def test_optional_fields_are_never_asked_about() -> None:
    # Asking about description, agenda or invitees would turn a one-sentence request into a form.
    assert missing_fields(_complete()) == []


def test_a_half_specified_recurrence_is_asked_to_completion() -> None:
    # Worse than no recurrence: the server would either reject it or invent the missing half.
    draft = _complete()
    draft.recurrence_type = "DAILY"
    assert "recurrence_start_time_local" in missing_fields(draft)


# ── what would come back as a 400 ─────────────────────────────────────────────────────────────


def test_scheduled_at_and_recurrence_together_are_refused_here_not_by_the_server() -> None:
    # "Daily at 9 starting Monday" reads as both a time and a rule. The server rejects the pair
    # rather than resolving it — correct of the server, and useless to a user whose sentence made
    # perfect sense. Caught here, the model can fix it without the user seeing anything.
    draft = _complete()
    draft.recurrence_type = "DAILY"
    draft.recurrence_start_time_local = "09:00"
    draft.scheduled_at = "2026-08-20T09:00:00Z"

    problems = validate(draft)
    assert any("never both" in p for p in problems), problems


def test_an_unknown_meeting_type_is_named_with_the_valid_ones() -> None:
    draft = _complete()
    draft.translation_room_type = "STANDUP"
    problems = validate(draft)
    assert any("CHANNEL_MEETING" in p for p in problems), problems


@pytest.mark.parametrize("bad_time", ["9:00", "0900", "25:00", "09:60", "9am"])
def test_a_recurrence_time_that_is_not_24_hour_hhmm_is_refused(bad_time: str) -> None:
    draft = _complete()
    draft.recurrence_type = "DAILY"
    draft.recurrence_start_time_local = bad_time
    assert any("HH:mm" in p for p in validate(draft)), bad_time


def test_a_malformed_invitee_is_named_rather_than_silently_dropped() -> None:
    # An invitation the user believes they sent and nobody received is the failure people notice
    # a week later, when the meeting happens without them.
    draft = _complete()
    draft.invited_emails = ["real@example.com", "not-an-email"]
    problems = validate(draft)
    assert any("not-an-email" in p for p in problems), problems


def test_a_valid_draft_has_no_complaints() -> None:
    assert validate(_complete()) == []


# ── the payload ───────────────────────────────────────────────────────────────────────────────


def test_absent_fields_are_omitted_rather_than_sent_as_null() -> None:
    # The meeting type seeds whatever is left out, so an explicit null would overwrite a sensible
    # default with an empty one.
    payload = build_payload(_complete(), WORKSPACE)
    for absent in ("description", "scheduledAt", "recurrence", "invitedEmails", "maxParticipants"):
        assert absent not in payload, absent


def test_a_recurring_booking_never_carries_scheduled_at() -> None:
    draft = _complete()
    draft.recurrence_type = "DAILY"
    draft.recurrence_start_time_local = "09:00"
    draft.scheduled_at = "2026-08-20T09:00:00Z"  # would be a 400 if it survived

    payload = build_payload(draft, WORKSPACE)
    assert "scheduledAt" not in payload
    assert payload["recurrence"]["type"] == "DAILY"


def test_a_recurrence_always_carries_a_time_zone() -> None:
    # A rule without a zone is a rule that means different things to different readers.
    draft = _complete()
    draft.recurrence_type = "DAILY"
    draft.recurrence_start_time_local = "09:00"
    assert build_payload(draft, WORKSPACE)["recurrence"]["timeZone"]


def test_the_agenda_survives_as_a_numbered_list() -> None:
    # There is no agenda field on the API. Folding it into the description under a heading is what
    # lets the user find it on the meeting afterwards — which is why they asked for one.
    draft = _complete()
    draft.description = "Weekly check-in."
    draft.agenda = ["Demo", "Blockers", "Next sprint"]

    description = compose_description(draft)
    assert description is not None
    assert "Weekly check-in." in description
    assert "1. Demo" in description
    assert "3. Next sprint" in description


def test_an_agenda_with_no_description_still_renders() -> None:
    draft = _complete()
    draft.agenda = ["Only item"]
    assert "1. Only item" in (compose_description(draft) or "")


# ── the shapes models actually emit ───────────────────────────────────────────────────────────


def test_a_comma_separated_string_is_accepted_where_a_list_is_expected() -> None:
    draft = draft_from_arguments({"target_languages": "en, ja , ko"})
    assert draft.target_languages == ["en", "ja", "ko"]


def test_a_lowercase_meeting_type_is_accepted() -> None:
    assert (
        draft_from_arguments({"translation_room_type": "webinar"}).translation_room_type
        == "WEBINAR"
    )


def test_empty_arguments_produce_an_empty_draft_rather_than_raising() -> None:
    draft = draft_from_arguments({})
    assert draft.title is None
    assert missing_fields(draft) == list(missing_fields(MeetingDraft()))


# ── markdown, because the description field parses it ─────────────────────────────────────────
#
# The room page edits this with TipTap under Markdown.configure({ html: true }) and styles
# [&_h2], so a heading is a heading and a numbered list is a list. Plain prose would lose the
# structure the user asked for; HTML would fight the editor that owns the field.


def test_the_agenda_heading_is_markdown_not_a_bare_word() -> None:
    draft = _complete()
    draft.agenda = ["Demo", "Blockers"]
    assert "## Agenda" in (compose_description(draft) or "")


def test_documents_are_linked_from_the_description() -> None:
    # There is no room↔document table. A markdown link is the whole feature without a migration,
    # and clicking it opens the document — which is what an attachment was wanted for.
    draft = _complete()
    draft.workspace_slug = "warptalk-demo"
    draft.documents = [("Onboarding spec", "doc-123")]

    description = compose_description(draft) or ""
    assert "## Documents" in description
    assert "[Onboarding spec](/warptalk-demo/documents/doc-123)" in description


def test_a_document_without_a_workspace_slug_degrades_to_its_name() -> None:
    # A link to /undefined/documents/... is worse than no link: it looks clickable and goes
    # nowhere.
    draft = _complete()
    draft.documents = [("Onboarding spec", "doc-123")]

    description = compose_description(draft) or ""
    assert "Onboarding spec" in description
    assert "](/" not in description


def test_a_document_missing_an_id_is_skipped_rather_than_linked_to_nothing() -> None:
    draft = _complete()
    draft.workspace_slug = "warptalk-demo"
    draft.documents = [("Real doc", "doc-1"), ("Half a doc", "")]

    description = compose_description(draft) or ""
    assert "Real doc" in description
    assert "Half a doc" not in description


def test_documents_arrive_from_tool_arguments_as_title_id_pairs() -> None:
    draft = draft_from_arguments(
        {"documents": [{"title": "Spec", "id": "doc-9"}, {"title": "", "id": "doc-8"}]}
    )
    assert draft.documents == [("Spec", "doc-9")]


# ── WT-399: the filler values a model sends for properties it does not mean ──────────────────
#
# Captured verbatim from `assistant:chat_results` on production, 2026-08-14 15:55 UTC. The user
# asked for one meeting, now, and said explicitly they did not want it scheduled. `create_meeting`
# was called TWELVE times across three turns, never once reached the meeting service, and the
# conversation ended with an English error in a Vietnamese chat.
#
# Every property is filled — "" for strings, [] for arrays, 0 for the number. That is what a model
# does with an optional property. `recurrence_type` was the one that had no filler to reach for,
# because its enum listed three ways to repeat and no way not to.

PROD_ARGUMENTS = {
    "title": "Thảo luận tin tức AI",
    "description": "Cuộc họp thảo luận các tin tức AI gần đây.",
    "agenda": ["Tóm tắt các tin tức AI gần đây"],
    "translation_room_type": "CHANNEL_MEETING",
    "source_language": "vi",
    "target_languages": ["en"],
    "scheduled_at": "2026-08-14T16:10:00Z",
    "invited_emails": [],
    "recurrence_type": "DAILY",
    "recurrence_start_time_local": "",
    "recurrence_time_zone": "",
    "recurrence_start_date_local": "",
    "recurrence_end_date_local": "",
    "max_participants": 0,
    "documents": [],
}


def test_the_exact_production_arguments_now_reach_the_meeting_service() -> None:
    draft = draft_from_arguments(PROD_ARGUMENTS)

    assert missing_fields(draft) == [], "still asking for something the user already answered"
    assert validate(draft) == [], "still refusing a request the user made correctly"

    payload = build_payload(draft, WORKSPACE)
    assert "recurrence" not in payload, "a one-off meeting was booked as a repeating series"
    assert payload["scheduledAt"] == "2026-08-14T16:10:00Z"
    assert "maxParticipants" not in payload, "0 was sent as a real seat cap"


def test_none_is_a_thing_the_model_can_actually_say() -> None:
    # The root cause: the enum had no member meaning "this does not repeat", so a model that
    # fills every property had to pick a repeating rule for a one-off meeting.
    assert NO_RECURRENCE in RECURRENCE_CHOICES
    assert NO_RECURRENCE not in RECURRENCE_TYPES, "NONE must never be sent to the server"

    draft = draft_from_arguments({**PROD_ARGUMENTS, "recurrence_type": NO_RECURRENCE})

    assert draft.recurrence_type is None
    assert "recurrence" not in build_payload(draft, WORKSPACE)


@pytest.mark.parametrize("alias", ["NEVER", "ONCE", "one_off", "single", "no"])
def test_other_ways_of_saying_not_repeating_land_too(alias: str) -> None:
    draft = draft_from_arguments({**PROD_ARGUMENTS, "recurrence_type": alias})
    assert draft.recurrence_type is None


def test_a_real_recurrence_is_never_quietly_turned_into_one_meeting() -> None:
    """The failure the fix must not introduce.

    Reading every recurrence_type as filler would give a user who asked for a weekly series a
    single meeting and no error — worse than the loop being fixed here, because nothing tells
    them. A rule carrying ANY detail of its own is taken at face value.
    """
    draft = draft_from_arguments(
        {
            **PROD_ARGUMENTS,
            "scheduled_at": "",
            "recurrence_type": "WEEKLY",
            "recurrence_start_time_local": "09:00",
            "recurrence_time_zone": "Asia/Ho_Chi_Minh",
        }
    )

    assert draft.recurrence_type == "WEEKLY"
    payload = build_payload(draft, WORKSPACE)
    assert payload["recurrence"]["type"] == "WEEKLY"
    assert payload["recurrence"]["startTimeLocal"] == "09:00"
    assert "scheduledAt" not in payload


def test_a_recurrence_with_only_a_first_date_is_still_a_recurrence() -> None:
    # startDateLocal alone is enough detail to mean it: the model chose a date, it did not have
    # one handed to it by the schema.
    draft = draft_from_arguments(
        {
            **PROD_ARGUMENTS,
            "recurrence_type": "MONTHLY",
            "recurrence_start_date_local": "2026-09-01",
        }
    )

    assert draft.recurrence_type == "MONTHLY"
    assert validate(draft), "a complete recurrence beside a scheduled_at is still a contradiction"


def test_a_genuine_contradiction_is_still_refused() -> None:
    # The "never both" rule is not weakened — a rule with real detail AND a one-off start is two
    # different meetings described at once, and guessing which one was meant is not this code's
    # decision to make.
    draft = draft_from_arguments(
        {
            **PROD_ARGUMENTS,
            "recurrence_type": "WEEKLY",
            "recurrence_start_time_local": "09:00",
        }
    )

    assert any("never both" in p for p in validate(draft))


@pytest.mark.parametrize("filler", [0, "0", -1, "", "  ", "none"])
def test_no_spelling_of_an_empty_seat_cap_becomes_a_real_one(filler: object) -> None:
    """Both shapes, deliberately.

    The integer 0 used to be dropped only because `0 or ""` is falsy — an accident, not a rule.
    The STRING "0" went straight through it and failed validation as a seat cap of zero. Which
    of the two a model sends is not something worth leaving to chance.
    """
    draft = draft_from_arguments({**PROD_ARGUMENTS, "max_participants": filler})

    assert draft.max_participants is None
    assert validate(draft) == []
    assert "maxParticipants" not in build_payload(draft, WORKSPACE)


def test_a_seat_cap_somebody_actually_typed_is_still_checked() -> None:
    # 1 is a number a person chose, and a meeting for one person is still an error worth showing
    # them — the filler handling above must not swallow it.
    assert validate(draft_from_arguments({**PROD_ARGUMENTS, "max_participants": 1}))
    assert draft_from_arguments({**PROD_ARGUMENTS, "max_participants": 8}).max_participants == 8
    assert draft_from_arguments({**PROD_ARGUMENTS, "max_participants": "8"}).max_participants == 8
