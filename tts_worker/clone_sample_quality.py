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

PITCH COVERAGE, AND WHY IT IS A SCORE AND NOT A GATE (WT-371 #9)
    Every check above measures ENERGY. A speaker who says ten flat seconds into their microphone
    passes all of them — and a clone built from ten seconds at one pitch is a clone of that pitch.
    The report is what that sounds like from the other side: "voice cloning không nhận dạng được
    khi nói thay đổi tông giọng". Raise or crack your voice and the clone stops being you.

    So the clip is also measured on how much of the speaker's RANGE it covers, and that is
    reported as a `score` rather than another reject. A monotone delivery is a way of speaking,
    not a defect; refusing to clone it would leave that person with a generic catalogue voice for
    the whole meeting, which is worse than a narrow clone of their own voice. The score exists so
    a LATER, wider clip can be recognised as better and replace the first one — see
    `TtsWorker._consume_audio_for_cloning`, which used to stop listening the moment it had any
    clone at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Mirrors warptalk-web/src/lib/voice/voice-sample-quality.ts. Keep in step.
MIN_RMS = 0.008
MIN_ACTIVE_SPEECH_RATIO = 0.18
MAX_CLIPPED_RATIO = 0.02
MIN_ACTIVE_ENERGY_VARIATION = 0.08
FRAME_SECONDS = 0.02

_INT16_FULL_SCALE = 32768.0
_CLIP_THRESHOLD = 0.98

# Human speaking fundamental, generously bounded: ~70Hz is a low male voice, ~400Hz a raised
# female one. Searching outside this invites octave errors on breath and consonant noise.
_MIN_F0_HZ = 70.0
_MAX_F0_HZ = 400.0
_PITCH_FRAME_SECONDS = 0.04
_PITCH_HOP_SECONDS = 0.02
# How periodic a frame must be before its pitch estimate is believed. Unvoiced sounds (s, f, sh)
# have no fundamental, and reading one anyway is how a pitch range becomes noise.
_MIN_VOICING_STRENGTH = 0.35
# Enough voiced frames that the percentile spread below means something.
_MIN_VOICED_FRAMES = 12

# A perfect fifth. Conversational speech that carries a question, an emphasis and a full stop
# covers roughly this much; a clip that does is a reference that survives the speaker raising
# their voice. Used to normalise the score, NOT as a threshold to reject below.
TARGET_PITCH_SEMITONES = 7.0

# What a "good" active-speech ratio looks like — half the clip being speech is plenty for a
# reference. Above this, more speech does not make a better clone.
_SATURATING_SPEECH_RATIO = 0.5

# The score's two halves. Pitch coverage weighs more because it is the property this score was
# added to capture; speech ratio is already gated above and only breaks ties. Both numbers are a
# judgement — the tests pin the ORDERING they must produce, not these values.
_PITCH_WEIGHT = 0.65
_SPEECH_WEIGHT = 0.35


@dataclass(frozen=True)
class CloneSampleAssessment:
    """Why a clip was accepted or turned away. `reason` is empty when accepted."""

    accepted: bool
    reason: str
    rms: float
    active_speech_ratio: float
    clipped_ratio: float
    energy_variation: float
    # How many semitones separate the 10th and 90th percentile of this clip's voiced pitch.
    # Percentiles rather than min/max: a single octave-halved frame at either end would otherwise
    # dominate the figure. 0.0 when too few frames were voiced to say anything.
    pitch_semitone_range: float = 0.0
    # 0..1, higher is better, comparable BETWEEN accepted clips of the same speaker. Meaningless
    # for a rejected clip, which is why it is 0.0 there.
    score: float = 0.0


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

    pitch_range = estimate_pitch_semitone_range(samples, sample_rate)
    pitch_component = min(1.0, pitch_range / TARGET_PITCH_SEMITONES)
    speech_component = min(1.0, active_speech_ratio / _SATURATING_SPEECH_RATIO)
    score = _PITCH_WEIGHT * pitch_component + _SPEECH_WEIGHT * speech_component

    return CloneSampleAssessment(
        True,
        "",
        rms,
        active_speech_ratio,
        clipped_ratio,
        energy_variation,
        pitch_semitone_range=pitch_range,
        score=score,
    )


def estimate_pitch_semitone_range(samples: npt.NDArray[np.float32], sample_rate: int) -> float:
    """How many semitones of pitch this clip actually covers.

    Autocorrelation per frame, which is the cheapest estimator that is honest about voicing: an
    unvoiced frame simply has no strong periodic peak, so it drops out instead of contributing a
    fabricated pitch. No model and no new dependency — this runs on the worker's hot path, and a
    gate that costs real time is a gate somebody later deletes.

    Returns 0.0 rather than guessing when too little of the clip was voiced.
    """
    frame_size = max(1, round(sample_rate * _PITCH_FRAME_SECONDS))
    hop = max(1, round(sample_rate * _PITCH_HOP_SECONDS))
    min_lag = max(1, int(sample_rate / _MAX_F0_HZ))
    max_lag = min(frame_size - 1, int(sample_rate / _MIN_F0_HZ))
    if max_lag <= min_lag or samples.size < frame_size:
        return 0.0

    frequencies: list[float] = []
    for start in range(0, samples.size - frame_size + 1, hop):
        frame = samples[start : start + frame_size]

        energy = float(np.dot(frame, frame))
        if energy <= 0:
            continue
        # Skip near-silence with the same bar the energy checks use, so a pause cannot be read as
        # a very low note.
        if np.sqrt(energy / frame.size) < MIN_RMS:
            continue

        frame = frame - float(np.mean(frame))

        # Autocorrelation via FFT. Zero-padded to avoid the circular wrap that would fold the
        # frame's tail onto its head and invent periodicity that is not there.
        spectrum = np.fft.rfft(frame, n=2 * frame_size)
        correlation = np.fft.irfft(spectrum * np.conjugate(spectrum))[:frame_size]
        if correlation[0] <= 0:
            continue

        window = correlation[min_lag : max_lag + 1]
        if window.size == 0:
            continue

        best = int(np.argmax(window))
        strength = float(window[best] / correlation[0])
        if strength < _MIN_VOICING_STRENGTH:
            continue

        lag = min_lag + best
        frequencies.append(sample_rate / lag)

    if len(frequencies) < _MIN_VOICED_FRAMES:
        return 0.0

    pitches = np.asarray(frequencies, dtype=np.float64)
    low = float(np.percentile(pitches, 10))
    high = float(np.percentile(pitches, 90))
    if low <= 0 or high <= 0:
        return 0.0

    return float(12.0 * np.log2(high / low))
