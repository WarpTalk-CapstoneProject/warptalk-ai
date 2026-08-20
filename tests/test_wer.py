"""Tests for the transcript accuracy measurement.

This module exists to produce a number that will be quoted in a report and defended in a room, so
the cases here are mostly about the ways a WER implementation flatters itself: folding diacritics,
counting punctuation as speech, dividing by an empty reference, and collapsing deletions and
insertions into one figure that hides which of the two happened.
"""

from __future__ import annotations

from shared.wer import (
    Op,
    character_error_rate,
    compare,
    normalise,
    word_error_rate,
    words,
)


def test_an_exact_transcript_scores_zero():
    result = word_error_rate("phát hành vào thứ sáu", "phát hành vào thứ sáu")

    assert result.rate == 0.0
    assert result.errors == 0


def test_case_and_punctuation_are_not_transcription_errors():
    # A punctuation-adding post-process must not read as a regression.
    result = word_error_rate("phát hành vào thứ sáu", "Phát hành, vào thứ Sáu.")

    assert result.rate == 0.0


def test_diacritics_are_never_folded_away():
    # In Vietnamese they are the difference between words. Folding them would flatter every
    # Vietnamese measurement — which is the half of the traffic the complaint is about.
    result = word_error_rate("anh Tú", "anh Tu")

    assert result.substitutions == 1
    assert result.rate == 0.5


def test_decomposed_and_composed_diacritics_are_the_same_word():
    # Same rendered text from two producers must not score an error.
    composed = "ế"  # ế
    decomposed = "ế"

    assert normalise(composed) == normalise(decomposed)
    assert word_error_rate(composed, decomposed).rate == 0.0


def test_a_dropped_word_is_a_deletion_and_an_added_one_is_an_insertion():
    # The direction is the point: a deletion loses content silently, an insertion puts words in
    # somebody's mouth. A single WER figure hides which happened.
    dropped = word_error_rate("phát hành vào thứ sáu", "phát hành thứ sáu")
    added = word_error_rate("phát hành thứ sáu", "phát hành vào thứ sáu")

    assert dropped.deletions == 1 and dropped.insertions == 0
    assert added.insertions == 1 and added.deletions == 0


def test_the_alignment_shows_where_it_went_wrong():
    result = word_error_rate("ship on friday", "ship on monday")

    substitutions = [token for token in result.alignment if token.op is Op.SUBSTITUTION]
    assert len(substitutions) == 1
    assert substitutions[0].reference == "friday"
    assert substitutions[0].hypothesis == "monday"


def test_the_alignment_covers_every_reference_and_hypothesis_token():
    result = word_error_rate("một hai ba bốn", "một ba bốn năm")

    from_reference = [t.reference for t in result.alignment if t.reference is not None]
    from_hypothesis = [t.hypothesis for t in result.alignment if t.hypothesis is not None]

    assert from_reference == words("một hai ba bốn")
    assert from_hypothesis == words("một ba bốn năm")


def test_saying_nothing_where_nothing_was_said_is_not_an_error():
    assert word_error_rate("", "").rate == 0.0


def test_inventing_words_where_nothing_was_said_is_total_error():
    # Rather than a division by zero, and rather than 0.0 — which would score a hallucination
    # over silence as a perfect transcript.
    result = word_error_rate("", "xin chào các bạn")

    assert result.rate == 1.0
    assert result.insertions == 4


def test_transcribing_nothing_at_all_scores_one():
    result = word_error_rate("phát hành vào thứ sáu", "")

    assert result.rate == 1.0
    assert result.deletions == 5


def test_character_error_rate_is_kinder_to_a_syllable_boundary():
    # Vietnamese is written in syllables, and word-level WER punishes a boundary as hard as a
    # wrong word. Both are reported so the pair can be read together.
    wer = word_error_rate("thứ sáu", "thứsáu")
    cer = character_error_rate("thứ sáu", "thứsáu")

    assert wer.rate > 0
    assert cer.rate == 0.0


def test_a_comparison_reports_what_fraction_of_the_errors_went_away():
    reference = "phát hành vào thứ sáu"

    result = compare(reference, first_pass="phát hành vào thứ bảy", second_pass=reference)

    assert result.first.rate > 0
    assert result.second.rate == 0.0
    assert result.relative_improvement == 1.0


def test_a_first_pass_that_was_already_perfect_reports_no_improvement_not_a_crash():
    reference = "phát hành vào thứ sáu"

    result = compare(reference, first_pass=reference, second_pass=reference)

    assert result.relative_improvement == 0.0
    assert result.absolute_improvement == 0.0


def test_a_second_pass_that_made_things_worse_reports_a_negative_improvement():
    # The measurement must be able to say the change was bad. One that can only report
    # improvement is not a measurement.
    reference = "phát hành vào thứ sáu"

    result = compare(reference, first_pass=reference, second_pass="phát hành vào chủ nhật")

    assert result.absolute_improvement < 0
    assert result.relative_improvement == 0.0  # first pass was perfect; nothing to improve on


def test_the_summary_carries_the_parts_not_just_the_rate():
    result = word_error_rate("một hai ba", "một bốn")

    summary = result.as_dict()
    assert summary["reference_length"] == 3
    assert (
        summary["errors"] == summary["substitutions"] + summary["deletions"] + summary["insertions"]
    )
