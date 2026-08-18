"""A clone reference is judged on how much of the speaker's RANGE it covers (WT-371 #9).

Every existing check in clone_sample_quality measures ENERGY, so ten flat seconds into a
microphone passes all of them — and a clone built from one pitch is a clone of that pitch. The
report is what that sounds like from the other side: raise or crack your voice and the clone stops
being you.

These pin the ORDERING the score has to produce, not the weights that produce it. The weights are
a judgement and should be tunable; "a clip covering more of the speaker's range is a better
reference" is the property, and it must survive any retuning.
"""

from __future__ import annotations

import numpy as np
import pytest

from tts_worker.clone_sample_quality import (
    TARGET_PITCH_SEMITONES,
    assess_clone_sample,
    estimate_pitch_semitone_range,
)

SAMPLE_RATE = 16000
SECONDS = 12.0


def _voice(f0_hz: np.ndarray, amplitude: float = 0.25) -> bytes:
    """A voiced signal whose fundamental follows `f0_hz`, with syllable-rate amplitude.

    Harmonics matter: a pure sine autocorrelates just as well one octave down, and the energy
    envelope has to look like speech or the checks that run before pitch reject the clip first.
    """
    n = f0_hz.size
    t = np.arange(n) / SAMPLE_RATE
    phase = 2 * np.pi * np.cumsum(f0_hz) / SAMPLE_RATE

    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    # ~4Hz syllable envelope, never reaching zero, so frames stay above the activity floor while
    # still varying enough to read as speech rather than a tone.
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)
    signal = signal * envelope
    signal = signal / np.max(np.abs(signal)) * amplitude

    return (signal * 32767).astype(np.int16).tobytes()


def _flat(f0: float = 120.0) -> bytes:
    n = int(SAMPLE_RATE * SECONDS)
    return _voice(np.full(n, f0, dtype=np.float64))


def _varied(low: float = 110.0, high: float = 220.0) -> bytes:
    """A speaker whose pitch moves an octave — a question, an emphasis, a full stop."""
    n = int(SAMPLE_RATE * SECONDS)
    t = np.arange(n) / SAMPLE_RATE
    sweep = (np.sin(2 * np.pi * 0.4 * t) + 1) / 2
    return _voice(low + (high - low) * sweep)


def test_a_monotone_clip_reads_as_narrow() -> None:
    spread = estimate_pitch_semitone_range(
        np.frombuffer(_flat(), dtype=np.int16).astype(np.float32) / 32768.0,
        SAMPLE_RATE,
    )
    assert spread < 2.0, f"a constant fundamental should measure near zero; got {spread}"


def test_a_clip_that_moves_an_octave_reads_as_wide() -> None:
    spread = estimate_pitch_semitone_range(
        np.frombuffer(_varied(), dtype=np.int16).astype(np.float32) / 32768.0,
        SAMPLE_RATE,
    )
    # An octave is 12 semitones; the 10th/90th percentiles clip the extremes, so expect most of it.
    assert spread > TARGET_PITCH_SEMITONES, f"expected a wide spread, got {spread}"


def test_a_wider_clip_scores_higher_than_a_monotone_one() -> None:
    """The whole point. Without this the worker has no way to tell that a later clip is a better
    reference, which is why the first one was kept for the entire meeting."""
    monotone = assess_clone_sample(_flat(), SAMPLE_RATE)
    varied = assess_clone_sample(_varied(), SAMPLE_RATE)

    assert monotone.accepted, monotone.reason
    assert varied.accepted, varied.reason
    assert varied.score > monotone.score


def test_a_monotone_speaker_is_still_cloned() -> None:
    """Pitch coverage is a score, not a gate.

    Speaking flatly is a way of speaking, not a defect. Refusing to clone it would leave that
    person with a generic catalogue voice for the whole meeting — worse than a narrow clone of
    their own voice.
    """
    assessment = assess_clone_sample(_flat(), SAMPLE_RATE)

    assert assessment.accepted
    assert assessment.score > 0.0


def test_the_score_stays_inside_its_range() -> None:
    # The worker compares scores against a fixed upgrade margin, so an unbounded score would make
    # that margin meaningless.
    for clip in (_flat(), _varied()):
        assessment = assess_clone_sample(clip, SAMPLE_RATE)
        assert 0.0 <= assessment.score <= 1.0


def test_silence_reports_no_pitch_rather_than_guessing() -> None:
    quiet = (np.zeros(int(SAMPLE_RATE * 2), dtype=np.int16)).tobytes()
    assert estimate_pitch_semitone_range(
        np.frombuffer(quiet, dtype=np.int16).astype(np.float32) / 32768.0, SAMPLE_RATE
    ) == pytest.approx(0.0)


def test_a_rejected_clip_carries_no_score() -> None:
    # Scores are only comparable between clips worth cloning; a non-zero score on a rejected clip
    # could win an upgrade comparison it should never enter.
    quiet = (np.zeros(int(SAMPLE_RATE * 2), dtype=np.int16)).tobytes()
    assessment = assess_clone_sample(quiet, SAMPLE_RATE)

    assert not assessment.accepted
    assert assessment.score == 0.0
