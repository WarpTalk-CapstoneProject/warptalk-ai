"""What the second pass may and may not change.

Every case here is a refusal or a bounded permission, because the dangerous direction is only one:
a fluent replacement for a passage nobody can check. The transcript feeds the summary, the minutes
somebody signs, and the knowledge index — a wrong line that reads well travels further than a
garbled one, and nobody downstream can tell it happened.
"""

from __future__ import annotations

from stt_worker.second_pass import decide_rewrite

CONFIDENT = -0.15
UNSURE = -1.2


def test_a_confident_segment_may_be_punctuated_but_not_reworded():
    original = "chúng ta sẽ phát hành vào thứ sáu"

    formatted = decide_rewrite(original, "Chúng ta sẽ phát hành vào thứ sáu.", CONFIDENT)
    reworded = decide_rewrite(original, "chúng ta sẽ phát hành vào thứ bảy", CONFIDENT)

    assert formatted.accepted
    assert reworded.accepted is False
    assert reworded.text == original
    assert reworded.reason == "substantive_rewrite_of_a_confident_segment"


def test_a_confident_segment_may_still_have_fillers_removed():
    # The guardian's rule: deletion is allowed, insertion and substitution are not.
    original = "ờ thì chúng ta sẽ phát hành vào thứ sáu"

    decision = decide_rewrite(original, "chúng ta sẽ phát hành vào thứ sáu", CONFIDENT)

    assert decision.accepted
    assert decision.reason == "formatting_only"


def test_a_low_confidence_segment_may_be_corrected():
    # This is the one thing a second pass exists to do, and the guardian's rule forbids it — which
    # is why the permission is bounded by evidence rather than granted or refused outright.
    original = "chúng ta sẽ phát hành vào thứ bay"

    decision = decide_rewrite(original, "chúng ta sẽ phát hành vào thứ bảy", UNSURE)

    assert decision.accepted
    assert decision.reason == "corrected_low_confidence"
    assert decision.text == "chúng ta sẽ phát hành vào thứ bảy"


def test_a_low_confidence_segment_may_be_corrected_but_not_replaced():
    # Past a bounded fraction the output has stopped being a correction of what was said and
    # started being a plausible sentence about the same topic.
    original = "chúng ta sẽ phát hành vào thứ bay"

    decision = decide_rewrite(
        original, "nhóm quyết định hoãn toàn bộ kế hoạch sang quý sau", UNSURE
    )

    assert decision.accepted is False
    assert decision.text == original
    assert decision.reason == "rewrote_too_much_of_a_low_confidence_segment"


def test_unknown_confidence_is_treated_as_confident():
    # WT-277 keeps NULL meaning "unknown". The safe reading of unknown is the one that changes
    # nothing: otherwise every segment from a producer reporting no confidence is freely
    # rewritable.
    original = "chúng ta sẽ phát hành vào thứ bay"

    decision = decide_rewrite(original, "chúng ta sẽ phát hành vào thứ bảy", None)

    assert decision.accepted is False
    assert decision.text == original


def test_an_empty_rewrite_never_replaces_anything():
    original = "chúng ta sẽ phát hành vào thứ sáu"

    for answer in ["", "   ", "\n"]:
        decision = decide_rewrite(original, answer, UNSURE)
        assert decision.accepted is False
        assert decision.text == original
        assert decision.reason == "empty_rewrite"


def test_an_identical_rewrite_is_accepted_and_reported_as_unchanged():
    original = "chúng ta sẽ phát hành vào thứ sáu"

    decision = decide_rewrite(original, original, UNSURE)

    assert decision.accepted
    assert decision.reason == "unchanged"
    assert decision.changed is False


def test_a_segment_with_no_words_has_nothing_to_correct():
    # WT-478's lesson: scaffolding is not speech, and a model asked to correct punctuation will
    # happily produce a sentence.
    decision = decide_rewrite("   ", "xin chào các bạn", UNSURE)

    assert decision.accepted is False
    assert decision.reason == "nothing_to_correct"


def test_the_threshold_is_a_setting_not_a_constant():
    # The right value is measurable, so it must be movable without editing this module. A long
    # fixture on purpose: a short one would be decided by the change budget instead and the test
    # would pass for the wrong reason.
    original = "chúng ta sẽ phát hành vào thứ bay"
    corrected = "chúng ta sẽ phát hành vào thứ bảy"

    strict = decide_rewrite(original, corrected, -0.6, low_confidence_below=-2.0)
    lenient = decide_rewrite(original, corrected, -0.6, low_confidence_below=-0.1)

    assert strict.accepted is False
    assert lenient.accepted


def test_a_short_utterance_can_still_be_corrected_by_one_word():
    # In a two-word segment any correction is 50%, so a bare fraction would make "thứ bay" ->
    # "thứ bảy" impossible — and short utterances are where mis-hearing costs most.
    decision = decide_rewrite("thứ bay", "thứ bảy", UNSURE)

    assert decision.accepted
    assert decision.reason == "corrected_low_confidence"


def test_the_floor_does_not_open_up_a_long_segment():
    # One token of slack, not one token per however many the model felt like changing.
    original = "một hai ba bốn năm sáu bảy tám chín mười"

    decision = decide_rewrite(original, "hoàn toàn khác hẳn nội dung của câu ban đầu", UNSURE)

    assert decision.accepted is False
    assert decision.text == original


def test_the_change_budget_is_a_setting_too():
    # Long enough that the fraction binds above the one-token floor, or the floor would decide
    # this and the test would pass without exercising the setting.
    original = "một hai ba bốn năm sáu bảy tám"
    corrected = "một hai ba bốn năm sáu bảy chín"

    tight = decide_rewrite(
        original, corrected, UNSURE, max_changed_fraction=0.0, min_absolute_changes=0
    )
    loose = decide_rewrite(original, corrected, UNSURE, max_changed_fraction=0.5)

    assert tight.accepted is False
    assert loose.accepted


def test_both_budgets_at_zero_refuses_every_correction():
    # A setting that cannot turn the behaviour off is not a setting.
    decision = decide_rewrite(
        "thứ bay", "thứ bảy", UNSURE, max_changed_fraction=0.0, min_absolute_changes=0
    )

    assert decision.accepted is False


def test_every_refusal_publishes_the_original_rather_than_nothing():
    # No caller can publish an unchecked model output by forgetting to read a boolean.
    original = "chúng ta sẽ phát hành vào thứ sáu"

    for rewritten, confidence in [
        ("", UNSURE),
        ("hoàn toàn khác hẳn nội dung ban đầu của câu này", UNSURE),
        ("chúng ta sẽ phát hành vào thứ bảy", CONFIDENT),
    ]:
        decision = decide_rewrite(original, rewritten, confidence)
        assert decision.accepted is False
        assert decision.text == original
