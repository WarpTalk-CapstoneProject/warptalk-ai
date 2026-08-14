"""The dub should take about as long as the thing it is dubbing.

A translated sentence is rarely the same length as its source — Vietnamese from English
routinely runs longer — so the dub finishes after the speaker has moved on and the listener
hears the answer to a question while the next one is being asked.

These pin the controller's SAFETY properties above its correction, because the failure modes
here are heard directly: a runaway ratio would speak everybody faster and faster, and a
mismeasured pair folded in once would poison every later utterance from that speaker.
"""

from __future__ import annotations

import pytest

from shared.isochrony import (
    MAX_SPEED_CENTER,
    MIN_SPEED_CENTER,
    NO_FIT,
    DubFit,
    observe,
    speed_center,
)


def _settle(ratio: float, rounds: int = 12, source_ms: int = 4000) -> DubFit:
    fit = NO_FIT
    for _ in range(rounds):
        fit = observe(fit, source_ms, int(source_ms * ratio))
    return fit


def test_an_unknown_speaker_changes_nothing() -> None:
    # The whole point of the guard. Until there is a fit, every call must be byte-for-byte what
    # the worker did before this module existed.
    assert speed_center(NO_FIT) == 1.0


def test_one_observation_is_not_yet_a_fit() -> None:
    # One utterance is one person saying one thing, not a description of the language pair.
    assert speed_center(observe(NO_FIT, 4000, 5000)) == 1.0


def test_a_dub_that_overruns_asks_for_more_speed() -> None:
    fit = _settle(1.2)

    assert speed_center(fit) > 1.0


def test_a_dub_that_finishes_early_asks_for_less() -> None:
    fit = _settle(0.85)

    assert speed_center(fit) < 1.0


def test_a_dub_that_already_fits_asks_for_nothing() -> None:
    fit = _settle(1.0)

    assert speed_center(fit) == pytest.approx(1.0, abs=0.02)


def test_the_correction_is_bounded_in_both_directions() -> None:
    """Two reasons, and the second is the one that bites.

    `sonic-3.5` damps `speed` to roughly a fifth of what is asked (measured, see
    shared/prosody.py). An unbounded integral controller against an actuator that delivers 20%
    of its command winds up forever. The clamp turns that into a bounded partial correction.
    """
    assert speed_center(_settle(4.0)) <= MAX_SPEED_CENTER
    assert speed_center(_settle(0.3)) >= MIN_SPEED_CENTER


def test_a_wildly_implausible_pair_is_not_a_slow_dub() -> None:
    # A truncated synthesis or an unstamped end_ms is a measurement fault, not evidence about
    # tempo. Folding it in would poison every later utterance from this speaker.
    established = _settle(1.1)

    assert observe(established, 4000, 400_000) == established
    assert observe(established, 4000, 10) == established


def test_pairs_too_short_to_time_are_skipped() -> None:
    # A one-word acknowledgement, where tens of milliseconds of leading silence swamp the
    # measurement.
    established = _settle(1.1)

    assert observe(established, 200, 240) == established


def test_a_missing_source_duration_is_skipped_rather_than_treated_as_instant() -> None:
    assert observe(NO_FIT, 0, 3000) == NO_FIT
    assert observe(NO_FIT, -100, 3000) == NO_FIT


def test_the_fit_follows_a_speaker_who_changes() -> None:
    # Somebody who starts terse and becomes discursive must be tracked, not averaged forever
    # against their first minute.
    fit = _settle(0.9)
    before = speed_center(fit)

    for _ in range(12):
        fit = observe(fit, 4000, 5200)

    assert speed_center(fit) > before


def test_one_unusual_sentence_does_not_lurch_the_tempo() -> None:
    # Tempo correction is heard directly, so it settles over a few utterances rather than
    # snapping after one.
    fit = _settle(1.0)
    steady = speed_center(fit)

    jolted = speed_center(observe(fit, 4000, 7000))

    assert abs(jolted - steady) < 0.2
