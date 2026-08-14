"""The contour operation a Cartesia-supplementing pitch pass would perform.

Cartesia accepts no pitch input, so the only way to move a dub's intonation without changing
TTS provider is to reshape the audio afterwards. The reshape cannot IMPOSE the source's contour
— that needs a source-to-target alignment nothing in this pipeline produces — so it scales the
dub's OWN contour deviation instead, by a factor measured from the source.

That distinction is what makes the approach safe for Vietnamese, and it is the property these
tests pin: the six lexical tones live in the dub's own contour, and scaling deviation around a
mean preserves their SHAPES while changing only how animated the delivery is. Replace the
contour instead and you change which words are heard.

No vocoder here — this is the pure half, and it is the half that can be wrong silently.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.probe_world_roundtrip import _expand_contour

UNVOICED = 0.0


def test_unvoiced_frames_stay_exactly_zero() -> None:
    # WORLD uses 0 as its unvoiced marker. Scaling it would invent pitch inside silence, which
    # resynthesises as a tone where the speaker was not speaking.
    f0 = np.array([UNVOICED, 120.0, 130.0, UNVOICED, 125.0, UNVOICED])

    out = _expand_contour(f0, 1.5)

    assert list(out[[0, 3, 5]]) == [0.0, 0.0, 0.0]


def test_expansion_keeps_the_mean_where_it_was() -> None:
    # Scaling deviation must not move the speaker's register. A shifted mean is heard as a
    # different person, not as a more animated one.
    f0 = np.array([100.0, 120.0, 140.0, 160.0])

    for factor in (0.5, 1.0, 1.6):
        out = _expand_contour(f0, factor)
        assert np.mean(out) == pytest.approx(np.mean(f0), abs=0.01)


def test_the_shape_of_the_contour_survives() -> None:
    """The property that keeps a tonal language intact.

    A Vietnamese syllable's tone is the DIRECTION its pitch moves. Scaling deviation around a
    mean preserves every rise and fall; only their depth changes. If any step reversed sign,
    the tone — and therefore the word — would have changed.
    """
    f0 = np.array([110.0, 135.0, 128.0, 150.0, 118.0, 105.0])
    reference_steps = np.sign(np.diff(f0))

    for factor in (0.4, 0.8, 1.3, 2.0):
        out = _expand_contour(f0, factor)
        assert list(np.sign(np.diff(out))) == list(reference_steps), f"tone shape moved at {factor}"


def test_a_larger_factor_makes_a_wider_contour() -> None:
    f0 = np.array([110.0, 140.0, 120.0, 155.0])

    narrow = _expand_contour(f0, 0.5)
    wide = _expand_contour(f0, 1.5)

    assert np.std(narrow) < np.std(f0) < np.std(wide)


def test_a_factor_of_one_changes_nothing() -> None:
    # The identity case has to be exact: it is what a speaker with no measured delivery gets,
    # and any drift there would reshape every dub in the product for no reason.
    f0 = np.array([UNVOICED, 118.0, 132.0, 127.0, UNVOICED])

    assert np.allclose(_expand_contour(f0, 1.0), f0)


def test_an_extreme_factor_cannot_drive_pitch_through_zero() -> None:
    # A large factor on a wide contour can push a low frame negative, which resynthesises as an
    # unvoiced frame — a hole punched in the middle of a word.
    f0 = np.array([80.0, 300.0])

    out = _expand_contour(f0, 8.0)

    assert np.all(out > 0.0)


def test_an_entirely_unvoiced_contour_is_returned_untouched() -> None:
    f0 = np.zeros(16)

    assert np.array_equal(_expand_contour(f0, 1.7), f0)


def test_the_input_array_is_not_mutated() -> None:
    # The caller keeps the analysed contour to compare against; mutating it in place would make
    # the probe measure the modification against itself and report no change.
    f0 = np.array([120.0, 140.0, 130.0])
    original = f0.copy()

    _expand_contour(f0, 1.5)

    assert np.array_equal(f0, original)
