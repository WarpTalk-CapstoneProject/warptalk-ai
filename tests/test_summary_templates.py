"""The summary's shape is data, and every claim in it must be citable.

The old prompt asked for "a concise overview paragraph" in one fixed shape, and that is what
every meeting got — three thin sentences whether it was a standup, an interview or a demo.
Worse, the transcript handed to the model carried no timestamps, so "do not invent content"
was an instruction the model had no way to obey and nobody had any way to check.
"""

from __future__ import annotations

import pytest

from ai_assistant_worker.summary_templates import (
    GENERAL,
    TEMPLATES,
    TRACEABLE,
    build_system_prompt,
    format_transcript_line,
    resolve_template,
    spoken_text_only,
    transcript_offsets,
)


def _shape_line(prompt: str, key: str) -> str:
    """The one line of the JSON skeleton that declares `key`."""
    return next(row for row in prompt.splitlines() if row.strip().startswith(f'"{key}"'))


def test_an_unknown_template_falls_back_rather_than_failing() -> None:
    # A meeting that ends with no summary because somebody sent a typo is worse than a
    # meeting summarised in the general shape.
    assert resolve_template("does-not-exist") is GENERAL
    assert resolve_template(None) is GENERAL
    assert resolve_template("") is GENERAL


def test_template_keys_are_matched_case_and_space_insensitively() -> None:
    assert resolve_template("  StandUp ").key == "standup"


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_demands_a_citation_for_every_item(key: str) -> None:
    prompt = build_system_prompt(TEMPLATES[key])

    for section in TEMPLATES[key].sections:
        if section.kind == "paragraph":
            continue
        line = next(
            row for row in prompt.splitlines() if row.strip().startswith(f'"{section.key}"')
        )
        assert '"atMs": <number>' in line, f"{key}.{section.key} can be written uncited"


@pytest.mark.parametrize("key", sorted(TEMPLATES))
def test_every_template_explains_each_of_its_sections(key: str) -> None:
    prompt = build_system_prompt(TEMPLATES[key])
    for section in TEMPLATES[key].sections:
        assert section.guidance in prompt, f"{key}.{section.key} has a shape but no meaning"


def test_the_prompt_forbids_the_uncitable_claim_outright() -> None:
    prompt = build_system_prompt(GENERAL)
    assert "uncitable claim is a fabricated claim" in prompt
    # And it must say what to do instead, or the model just drops the citation.
    assert "do not make the statement" in prompt


def test_the_overview_is_not_asked_to_be_concise() -> None:
    # The single word that produced the summary the owner complained about.
    prompt = build_system_prompt(GENERAL)
    assert "concise" not in prompt.lower()
    assert "3–6 sentences" in prompt


def test_templates_differ_from_one_another() -> None:
    shapes = {key: tuple(s.key for s in t.sections) for key, t in TEMPLATES.items()}
    assert len(set(shapes.values())) == len(shapes), "two templates produce the same shape"


def test_a_transcript_line_carries_the_moment_it_was_spoken() -> None:
    assert format_transcript_line(90210, "Tu", "cap it at 500") == "[t=90210] [Tu] cap it at 500"


def test_a_negative_offset_never_reaches_the_model() -> None:
    # Clock skew between segments is real; a negative citation anchor would resolve to
    # nothing on the meeting page.
    assert format_transcript_line(-5, "Tu", "hello").startswith("[t=0]")


# WT-478 — a transcript of scaffolding is empty, and the model must never be asked to say so.


def test_a_transcript_of_empty_segments_reads_as_empty() -> None:
    # The exact shape that produced the bug: real timestamps, real speakers, nothing said.
    # `.strip()` on this string is truthy, which is how it reached the model at all.
    formatted = "\n".join(
        [
            format_transcript_line(0, "Nhi", ""),
            format_transcript_line(1200, "Ky", "   "),
        ]
    )
    assert formatted.strip(), "precondition: the formatted transcript is not blank"
    assert spoken_text_only(formatted) == ""


def test_the_words_survive_without_their_labels() -> None:
    formatted = "\n".join(
        [
            format_transcript_line(0, "Nhi", "chốt công nợ quý ba"),
            format_transcript_line(4000, "Ky", ""),
            format_transcript_line(9000, "Tuan", "gửi hợp đồng chiều nay"),
        ]
    )
    assert spoken_text_only(formatted) == "chốt công nợ quý ba\ngửi hợp đồng chiều nay"


def test_a_speaker_named_with_brackets_does_not_eat_the_line() -> None:
    # The prefix pattern stops at the first "]", so a bracketed display name must not
    # swallow what was said.
    assert spoken_text_only("[t=0] [Nhi] xin chào [nội bộ]") == "xin chào [nội bộ]"


def test_the_prompt_never_asks_the_model_to_declare_the_transcript_empty() -> None:
    # This instruction is what let a refusal come back as a summary: the model wrote
    # "the transcript is empty and contains no substantive meeting content", the call
    # succeeded, insufficientData stayed False, and the UI rendered it as prose.
    prompt = build_system_prompt(GENERAL)
    assert "no substantive content, say so" not in prompt
    assert "Never claim the transcript is empty" in prompt


# WT-663 — the traceable template: no overview paragraph, every sentence carrying its moments.


def test_the_traceable_template_resolves_without_disturbing_the_fallback() -> None:
    assert resolve_template("traceable") is TRACEABLE
    assert resolve_template("does-not-exist") is GENERAL


def test_the_traceable_template_has_no_overview_to_hide_a_claim_in() -> None:
    # The absence is the design, not an omission: an overview paragraph is the one part of
    # a summary that points at the meeting in general and so at nothing in particular.
    assert "summary" not in {section.key for section in TRACEABLE.sections}
    assert build_system_prompt(TRACEABLE).count('"summary": "<text>"') == 0


def test_every_narrative_sentence_may_rest_on_more_than_one_moment() -> None:
    line = _shape_line(build_system_prompt(TRACEABLE), "narrative")
    assert '"text": "<text>"' in line
    assert '"atMs": <number>' in line
    assert '"alsoAtMs": [<number>, ...]' in line


@pytest.mark.parametrize("key", ["decisions", "actionItems", "openQuestions"])
def test_a_cited_list_stays_one_claim_per_moment(key: str) -> None:
    # `alsoAtMs` belongs to the sentence kind alone. Widening the list items to accept it
    # would let a decision quietly cite three places at once — deliberately out of scope.
    assert "alsoAtMs" not in _shape_line(build_system_prompt(TRACEABLE), key)


def test_a_template_without_a_paragraph_is_not_asked_for_the_paragraph_citations() -> None:
    # "citations" is the overview's evidence. With no overview it names a field the model
    # was never told to write, which is an invitation to invent one.
    prompt = build_system_prompt(TRACEABLE)
    assert "citations" not in prompt


def test_dropping_the_citations_line_leaves_the_skeleton_well_formed() -> None:
    # The last entry of a JSON object carries no comma, and citations used to be it.
    prompt = build_system_prompt(TRACEABLE)
    shape = prompt.splitlines()
    closing = shape.index("}")
    assert not shape[closing - 1].endswith(","), shape[closing - 1]
    assert all(row.endswith(",") for row in shape[closing - len(TRACEABLE.sections) : closing - 1])


def test_the_general_prompt_is_left_exactly_as_it_was() -> None:
    prompt = build_system_prompt(GENERAL)
    assert '  "summary": "<text>",' in prompt
    assert '  "citations": [{"key": "summary", "atMs": <number>}]' in prompt
    assert "the moments the overview paragraph draws on" in prompt
    assert "alsoAtMs" not in prompt


# The set a cited moment has to be a member of, exactly.


def test_transcript_offsets_returns_the_moments_the_model_was_handed() -> None:
    transcript = "\n".join(
        [
            format_transcript_line(0, "Nhi", "mở đầu"),
            format_transcript_line(4200, "Ky", "chốt ngân sách"),
            format_transcript_line(90210, "Tu", "cap it at 500"),
        ]
    )
    assert transcript_offsets(transcript) == {0, 4200, 90210}


def test_a_timestamp_somebody_said_out_loud_is_not_a_moment() -> None:
    # Only the prefix marks a moment. If spoken text could mint one, a model could cite a
    # number it had itself just read back out of the words — the fabrication, laundered.
    transcript = format_transcript_line(700, "Tu", "xem lại đoạn [t=999999] nhé")
    assert transcript_offsets(transcript) == {700}


def test_transcript_offsets_of_nothing_is_the_empty_set() -> None:
    assert transcript_offsets("") == set()
    assert transcript_offsets("\n\n") == set()


class TestTheLanguageTheSummaryIsWrittenIn:
    """Which language a summary comes out in is a choice, not a guess.

    It used to be neither: one sentence told the model to follow the meeting, nothing recorded
    what it landed on, and nobody could ask for anything else. These tests pin the two halves
    of the replacement — that a chosen language is stated unmissably, and that choosing nothing
    still means exactly what it used to.
    """

    def test_choosing_nothing_keeps_the_original_instruction(self) -> None:
        # Every summary already in storage was written under this sentence. A default that
        # quietly changed would rewrite the meaning of documents nobody asked to touch.
        for language in (None, "", "   "):
            prompt = build_system_prompt(GENERAL, language)
            assert "Write in the language the meeting was held in." in prompt

    def test_a_chosen_language_is_named_not_coded(self) -> None:
        prompt = build_system_prompt(GENERAL, "ja")

        # The model is given a language, not a tag to echo back into its output.
        assert "JAPANESE" in prompt
        assert "Write in the language the meeting was held in." not in prompt

    def test_a_locale_tag_means_the_same_as_its_bare_code(self) -> None:
        # Rooms store `vi-VN`; requests carry `vi`. A summary must not depend on which
        # spelling happened to reach it.
        assert build_system_prompt(GENERAL, "vi-VN") == build_system_prompt(GENERAL, "vi")

    def test_an_unknown_code_still_produces_an_instruction(self) -> None:
        # Falling back to the code is a worse prompt; raising would be a missing summary.
        prompt = build_system_prompt(GENERAL, "xx")
        assert "XX" in prompt

    def test_the_rule_covers_every_string_not_only_the_prose(self) -> None:
        # The failure this guards against is a half-translated document: prose in Japanese,
        # owner labels left in the transcript's language, and no stated original.
        prompt = build_system_prompt(GENERAL, "ja").lower()
        assert "owner label" in prompt
        assert "not a bilingual one" in prompt
