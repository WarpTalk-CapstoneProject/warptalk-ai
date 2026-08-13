"""The rules a WarpBot-created meeting has to obey before it reaches the API.

create_meeting is the assistant's first tool that WRITES. Every other tool answers a question and
the worst case is a wrong answer; this one creates a row, mails invitations to real colleagues,
and for a recurring booking does it repeatedly. So the rules live in code and are tested, rather
than being something the model is asked to remember between turns.
"""

from __future__ import annotations

import pytest

from ai_assistant_worker.meeting_draft import (
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
    assert draft_from_arguments({"translation_room_type": "webinar"}).translation_room_type == "WEBINAR"


def test_empty_arguments_produce_an_empty_draft_rather_than_raising() -> None:
    draft = draft_from_arguments({})
    assert draft.title is None
    assert missing_fields(draft) == list(missing_fields(MeetingDraft()))
