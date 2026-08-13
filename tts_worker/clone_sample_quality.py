"""Is this ten seconds of audio worth cloning a voice from?

WHY THIS EXISTS
    The in-meeting clone took the FIRST `voice_clone_min_seconds` of a speaker's audio and sent
    it straight to Cartesia, with no check of any kind, and never cloned again — `_get_voice_id`
    short-circuits, so that clip became the speaker's voice for the whole meeting. In practice
    the reference was "alo alo, mọi người nghe rõ không", a chair scraping, and whoever else was
    talking in the room. The report was "voice clone chưa sát tiếng"; the model was never the
    problem.

    The bar itself is not new. warptalk-web's `voice-sample-quality.ts` has enforced exactly
    these checks for years on deliberately uploaded voice-profile samples — the path a user takes
    when they sit down and read a paragraph. It was never applied to the path that produces the
    voice people actually hear in a meeting. This is that same bar, on that path.

WHY THE THRESHOLDS ARE COPIED RATHER THAN INVENTED
    Two different bars for "is this a usable voice sample" would drift, and the day they disagree
    is the day a clip the upload page rejects gets cloned anyway. The constants below are the
    web module's, value for value. Change them together or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mirrors warptalk-web/src/lib/voice/voice-sample-quality.ts. Keep in step.
MIN_RMS = 0.008
MIN_ACTIVE_SPEECH_RATIO = 0.18
MAX_CLIPPED_RATIO = 0.02
MIN_ACTIVE_ENERGY_VARIATION = 0.08
FRAME_SECONDS = 0.02

_INT16_FULL_SCALE = 32768.0
_CLIP_THRESHOLD = 0.98


@dataclass(frozen=True)
class CloneSampleAssessment:
    """Why a clip was accepted or turned away. `reason` is empty when accepted."""

    accepted: bool
    reason: str
    rms: float
    active_speech_ratio: float
    clipped_ratio: float
    energy_variation: float


def assess_clone_sample(pcm_bytes: bytes, sample_rate: int) -> CloneSampleAssessment:
    """Judge 16-bit mono PCM as a voice-cloning reference.

    Deliberately cheap — a few numpy passes over ten seconds of audio, no model. It runs on the
    worker's hot path once per speaker, and a gate that costs real time would be a gate somebody
    later removes.
    """
    if sample_rate <= 0 or len(pcm_bytes) < 2:
        return CloneSampleAssessment(False, "empty audio", 0.0, 0.0, 0.0, 0.0)

    # Truncate a trailing odd byte rather than raising: a chunk boundary can split a sample, and
    # losing half a sample out of ten seconds cannot change any verdict below.
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    samples = np.frombuffer(pcm_bytes[:usable], dtype=np.int16).astype(np.float32)
    samples /= _INT16_FULL_SCALE

    if samples.size == 0:
        return CloneSampleAssessment(False, "empty audio", 0.0, 0.0, 0.0, 0.0)

    rms = float(np.sqrt(np.mean(samples * samples)))
    clipped_ratio = float(np.mean(np.abs(samples) >= _CLIP_THRESHOLD))

    if rms < MIN_RMS:
        # Near-silence. Cloning this produces a voice built from room tone.
        return CloneSampleAssessment(False, "too quiet", rms, 0.0, clipped_ratio, 0.0)

    if clipped_ratio > MAX_CLIPPED_RATIO:
        # A distorted reference bakes the distortion into every future utterance.
        return CloneSampleAssessment(False, "clipped", rms, 0.0, clipped_ratio, 0.0)

    frame_size = max(1, round(sample_rate * FRAME_SECONDS))
    frame_count = samples.size // frame_size
    if frame_count == 0:
        return CloneSampleAssessment(False, "too short to frame", rms, 0.0, clipped_ratio, 0.0)

    frames = samples[: frame_count * frame_size].reshape(frame_count, frame_size)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))

    active = frame_rms[frame_rms >= MIN_RMS]
    active_speech_ratio = float(active.size / frame_count)

    if active_speech_ratio < MIN_ACTIVE_SPEECH_RATIO:
        # Mostly background with a little speech in it — the clip the old code cloned happily.
        return CloneSampleAssessment(
            False, "too little speech", rms, active_speech_ratio, clipped_ratio, 0.0
        )

    mean_active = float(np.mean(active))
    energy_variation = float(np.std(active) / mean_active) if mean_active > 0 else 0.0

    if energy_variation < MIN_ACTIVE_ENERGY_VARIATION:
        # Flat energy is a fan, a hum, or a hold tone. Speech has syllables, and syllables have
        # peaks and troughs — this is what separates "someone is talking" from "something is on".
        return CloneSampleAssessment(
            False, "no speech pattern", rms, active_speech_ratio, clipped_ratio, energy_variation
        )

    return CloneSampleAssessment(
        True, "", rms, active_speech_ratio, clipped_ratio, energy_variation
    )
