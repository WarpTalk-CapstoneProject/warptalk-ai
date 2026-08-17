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
    build_system_prompt,
    format_transcript_line,
    resolve_template,
    spoken_text_only,
)


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
