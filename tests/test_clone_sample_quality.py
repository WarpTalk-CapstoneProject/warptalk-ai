"""The bar a clip must clear before it becomes somebody's voice for a whole meeting.

The in-meeting clone had no bar at all: it took the first N seconds off the stream and cloned
them, and `_get_voice_id` short-circuits, so a microphone check ("alo alo, nghe rõ không") became
the speaker's voice until the meeting ended. That is the whole of "voice clone chưa sát tiếng" —
the model was never the problem, the reference clip was.

These cases mirror warptalk-web/src/lib/voice/voice-sample-quality.ts, which has enforced exactly
this on deliberately uploaded samples for a long time. Two bars for one question would drift, and
the day they disagree is the day a clip the upload page rejects gets cloned anyway.
"""

from __future__ import annotations

import numpy as np
import pytest

from tts_worker.clone_sample_quality import (
    MIN_ACTIVE_SPEECH_RATIO,
    assess_clone_sample,
)

SAMPLE_RATE = 16_000


def _pcm(signal: np.ndarray) -> bytes:
    """float32 in [-1, 1] → 16-bit mono PCM, the shape the worker buffers."""
    return (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _speech_like(
    seconds: float, *, amplitude: float = 0.3, speech_ratio: float = 1.0
) -> np.ndarray:
    """A carrier with syllable-rate amplitude modulation and pauses — energy that varies."""
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    carrier = np.sin(2 * np.pi * 180 * t)
    # ~4 Hz envelope is roughly syllable rate; the offset keeps it from ever hitting pure zero.
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    signal = carrier * envelope * amplitude
    if speech_ratio < 1.0:
        silent_from = int(n * speech_ratio)
        signal[silent_from:] = 0.0
    return signal


def test_a_clip_of_actual_speech_is_accepted() -> None:
    verdict = assess_clone_sample(_pcm(_speech_like(20.0)), SAMPLE_RATE)
    assert verdict.accepted, verdict.reason


def test_silence_is_not_a_voice() -> None:
    # The old code would have cloned this happily and produced a voice built from room tone.
    verdict = assess_clone_sample(_pcm(np.zeros(SAMPLE_RATE * 20)), SAMPLE_RATE)
    assert not verdict.accepted
    assert verdict.reason == "too quiet"


def test_a_clipped_recording_is_refused() -> None:
    # Distortion in the reference is baked into every future utterance.
    loud = np.sign(np.sin(2 * np.pi * 180 * np.arange(SAMPLE_RATE * 20) / SAMPLE_RATE))
    verdict = assess_clone_sample(_pcm(loud), SAMPLE_RATE)
    assert not verdict.accepted
    assert verdict.reason == "clipped"


def test_a_steady_tone_is_not_speech() -> None:
    # A fan, a hum, a hold tone. Loud, unclipped, and completely flat — speech has syllables.
    t = np.arange(SAMPLE_RATE * 20) / SAMPLE_RATE
    verdict = assess_clone_sample(_pcm(0.3 * np.sin(2 * np.pi * 180 * t)), SAMPLE_RATE)
    assert not verdict.accepted
    assert verdict.reason == "no speech pattern"


def test_mostly_background_with_a_little_speech_is_refused() -> None:
    # The exact shape of the clip this gate exists for: someone says two words at the top of the
    # meeting and the rest is a quiet room.
    verdict = assess_clone_sample(
        _pcm(_speech_like(20.0, speech_ratio=MIN_ACTIVE_SPEECH_RATIO / 2)), SAMPLE_RATE
    )
    assert not verdict.accepted
    assert verdict.reason == "too little speech"


def test_a_very_quiet_speaker_is_refused_rather_than_cloned_badly() -> None:
    verdict = assess_clone_sample(_pcm(_speech_like(20.0, amplitude=0.002)), SAMPLE_RATE)
    assert not verdict.accepted
    assert verdict.reason == "too quiet"


@pytest.mark.parametrize("payload", [b"", b"\x00", b"\x01\x02\x03"])
def test_degenerate_input_never_raises_on_the_worker_hot_path(payload: bytes) -> None:
    # This runs inside the audio consumer loop. An exception here would take the loop down and
    # with it every dub for the meeting, so a malformed buffer must be a verdict, not a crash.
    verdict = assess_clone_sample(payload, SAMPLE_RATE)
    assert not verdict.accepted


def test_a_zero_sample_rate_is_a_verdict_not_a_division_by_zero() -> None:
    assert not assess_clone_sample(_pcm(_speech_like(20.0)), 0).accepted


def test_an_odd_trailing_byte_is_tolerated() -> None:
    # Chunk boundaries can split a 16-bit sample. Losing half a sample out of twenty seconds
    # cannot change any verdict, so it must not change the outcome either.
    good = _pcm(_speech_like(20.0))
    assert assess_clone_sample(good + b"\x7f", SAMPLE_RATE).accepted
