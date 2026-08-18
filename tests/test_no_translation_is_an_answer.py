"""WT-399, one field over: "this meeting needs no translation" had no way to be said.

`target_languages` is required, and a model fills every property it is offered. A monolingual
meeting therefore had no argument it could send: an empty list reads as "not answered yet" and
re-asks, and any real language code claims a translation the user refused.

Production, 15 Aug. The user opened with the language already in it —

    "tạo meeting ngay bây giờ để nói về starlink, mời ngô xuân hạnh nhi vào, ngôn ngữ tiếng việt"

— and was asked for it eight times, each answer rejected and re-asked in different words. It ended
by accepting "Tiếng Việt, có dịch sang tiếng Việt": a vi→vi meeting, which generates no audio route
at all. Eight turns to reach a configuration that translates nothing, from a request that wanted
exactly that.
"""

from __future__ import annotations

from ai_assistant_worker.meeting_draft import (
    draft_from_arguments,
    missing_fields,
)

BASE = {
    "title": "Trao đổi về Starlink",
    "translation_room_type": "CHANNEL_MEETING",
    "source_language": "vi",
}


def test_declining_translation_is_an_answer_not_a_silence() -> None:
    """The loop. 'NONE' used to leave target_languages empty, which reads as unanswered."""
    draft = draft_from_arguments({**BASE, "target_languages": ["NONE"]})

    assert "target_languages" not in missing_fields(draft)


def test_an_empty_list_is_still_unanswered() -> None:
    # The negative control. Declining must be a POSITIVE statement — if silence also counted,
    # the assistant would stop asking a question it genuinely needs answered.
    draft = draft_from_arguments({**BASE, "target_languages": []})

    assert "target_languages" in missing_fields(draft)


def test_declining_resolves_to_the_source_language() -> None:
    # A monolingual meeting IS a room whose targets match its source — that is how the server
    # represents one, and it is what produces no route between same-language participants.
    draft = draft_from_arguments({**BASE, "target_languages": ["NONE"]})

    assert draft.target_languages == ["vi"]
    assert draft.no_translation is True


def test_naming_one_language_for_the_whole_meeting_is_the_same_statement() -> None:
    # What the user actually said in the first sentence: "ngôn ngữ tiếng việt". It arrived as
    # source=vi, targets=[vi] and was re-asked as though nothing had been said.
    draft = draft_from_arguments({**BASE, "target_languages": ["vi"]})

    assert draft.no_translation is True
    assert "target_languages" not in missing_fields(draft)


def test_the_aliases_a_model_reaches_for_all_work() -> None:
    # Every one of these was tried in that conversation, in some paraphrase, and rejected.
    for alias in ("NONE", "no", "same", "SAME_AS_SOURCE", "no_translation", "MONOLINGUAL"):
        draft = draft_from_arguments({**BASE, "target_languages": [alias]})
        assert "target_languages" not in missing_fields(draft), alias


def test_the_sentinel_never_reaches_the_draft() -> None:
    # It is a token for the model, not a language. The server knows only real codes.
    draft = draft_from_arguments({**BASE, "target_languages": ["NONE"]})

    assert "NONE" not in draft.target_languages


def test_a_real_translation_request_is_untouched() -> None:
    # The other negative control: fixing the monolingual case must not quietly disable
    # translation for everyone who does want it.
    draft = draft_from_arguments({**BASE, "target_languages": ["en", "ja"]})

    assert draft.no_translation is False
    assert draft.target_languages == ["en", "ja"]
    assert "target_languages" not in missing_fields(draft)


def test_a_sentinel_mixed_with_real_languages_keeps_the_languages() -> None:
    # A model that hedges by sending both. The real request is the specific one.
    draft = draft_from_arguments({**BASE, "target_languages": ["NONE", "en"]})

    assert draft.target_languages == ["en"]
