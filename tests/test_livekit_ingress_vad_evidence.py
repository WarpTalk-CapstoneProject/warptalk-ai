"""The speech gate must not require every frame of a window to be speech.

WT-371 #7. `VAD_WINDOW_SAMPLES = 512 * MIN_VAD_SPEECH_FRAMES` and the check
`speech_frames >= MIN_VAD_SPEECH_FRAMES` were the same constant, so the window held exactly as
many frames as it demanded. The rule was never "some evidence" — it was UNANIMITY, and the knob
could not express anything else: raising it grew the window by exactly as much as it raised the
bar.

Unanimity fails hardest at the start of an utterance, which is where speech is least confident: a
breath before the first vowel, an unvoiced consonant, a word begun partway through a window. One
such frame rejected the whole 96ms. That is the reported symptom — speech registering late, and
registering better when background noise keeps every frame's probability up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from livekit_ingress_worker.worker import (
    MIN_VAD_SPEECH_FRAMES,
    VAD_WINDOW_FRAMES,
    VAD_WINDOW_SAMPLES,
    LiveKitIngressWorker,
)
from shared.config import LiveKitSettings, WorkerSettings

THRESHOLD = 0.35


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    return LiveKitIngressWorker(settings=settings)


class _ScriptedVad:
    """Returns a fixed probability per frame, in order, so a window's verdict is exact."""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._index = 0

    def __call__(self, _tensor: Any, _sample_rate: int) -> Any:
        value = self._probabilities[self._index]
        self._index += 1

        class _Scalar:
            def item(self) -> float:
                return value

        return _Scalar()


def _window() -> Any:
    return np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32)


def _verdict(probabilities: list[float]) -> float:
    return _worker()._run_vad_on_window(
        _window(),
        threshold=THRESHOLD,
        vad_model=_ScriptedVad(probabilities),
    )


def test_the_bar_is_not_the_window_length() -> None:
    # The whole defect in one assertion. While these are the same number, "how much evidence do we
    # need" cannot be tuned without also changing "how long do we wait", and the answer is always
    # unanimity.
    assert MIN_VAD_SPEECH_FRAMES < VAD_WINDOW_FRAMES, (
        "requiring every frame in the window makes the evidence bar untunable and rejects "
        "utterance onsets"
    )
    assert VAD_WINDOW_SAMPLES == 512 * VAD_WINDOW_FRAMES


def test_a_word_beginning_midway_through_a_window_is_speech() -> None:
    # Silence, then the speaker starts. Under unanimity this was silence and detection waited for
    # the next window — the missing first word.
    assert _verdict([0.05, 0.88, 0.91]) > 0


def test_a_soft_consonant_inside_a_word_does_not_erase_the_window() -> None:
    assert _verdict([0.93, 0.10, 0.87]) > 0


def test_an_isolated_spike_is_still_rejected() -> None:
    # The rule the original docstring was written for, and it still holds: one loud frame in an
    # otherwise quiet window is a door or a keyboard, not somebody talking.
    assert _verdict([0.97, 0.02, 0.03]) == 0.0


def test_silence_is_still_silence() -> None:
    assert _verdict([0.01, 0.02, 0.01]) == 0.0


def test_the_returned_probability_is_the_strongest_frame() -> None:
    # Callers compare the return value against the same threshold, so a window that qualifies must
    # report a value that passes it.
    verdict = _verdict([0.42, 0.05, 0.77])
    assert verdict == pytest.approx(0.77)
    assert verdict >= THRESHOLD
