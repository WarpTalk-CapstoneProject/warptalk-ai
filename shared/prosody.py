"""How something was said, carried across the language boundary.

THE PROBLEM THIS SOLVES
    The pipeline is speech → STT → text → translation → text → TTS → speech. Only text crosses
    the middle, so everything about the delivery — a question's rising end, a whispered aside,
    the speed of someone in a hurry — is discarded at the STT boundary and the dub comes back in
    the target language's default reading voice. The speaker's IDENTITY survives (voice cloning
    reproduces their timbre); their DELIVERY does not.

WHAT IS AND IS NOT POSSIBLE
    Cartesia's synthesis API accepts exactly three delivery controls: an `emotion` label, a
    `speed` float in [0.6, 1.5], and a `volume` float. There is NO pitch-contour input. So the
    per-second pitch curve of the original can never be replayed — measuring it precisely would
    not help, because there is nowhere to put it. What this module does instead is measure the
    delivery, reduce it to those three controls, and pass them along.

EVERYTHING IS RELATIVE TO THE SPEAKER
    Adult male F0 typically sits around 85–180 Hz and adult female around 165–255 Hz. An
    absolute rule like "above 200 Hz means excited" therefore classifies most women as
    permanently excited and most men as never excited. Every feature here is expressed as a
    ratio against that speaker's own rolling baseline, which the system can afford because it
    already buffers their audio to clone them.

AROUSAL COMES FROM SOUND, VALENCE COMES FROM WORDS
    Pitch and energy tell you how ACTIVATED someone is. They do not tell you whether the feeling
    is positive or negative: anger and delight look nearly identical on both. That distinction
    lives in what was actually said, which the translation model has already read. So this module
    derives arousal from audio and takes valence as an argument — it never guesses valence from
    sound, and it never guesses arousal from text.

    Sarcasm is deliberately out of scope. It is usually delivered with FLAT prosody that
    contradicts the words, so acoustic features are not merely unhelpful there, they point the
    wrong way.

WRONG IS WORSE THAN NEUTRAL
    Dubbing a calm participant as angry in a meeting is a bigger failure than dubbing them
    plainly. Every path here falls back to neutral: too little baseline, too little voiced audio,
    or a measurement that lands between bands all produce "say it normally".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

# ── Cartesia's accepted ranges. Verified against the live API, which rejects anything
# outside them with `speed must be between 0.600000 and 1.500000`. ────────────────────
SPEED_MIN = 0.6
SPEED_MAX = 1.5
VOLUME_MIN = 0.5
VOLUME_MAX = 1.5

# Human speech only. Anything outside is a measurement artifact — a bad frame, a cough, or the
# autocorrelation locking onto a harmonic — and letting it into the median drags the baseline.
F0_MIN_HZ = 60.0
F0_MAX_HZ = 400.0

# Below this share of voiced frames the utterance is mostly silence or noise, and its pitch
# statistics describe the room rather than the person.
MIN_VOICED_RATIO = 0.25

# A baseline built from fewer utterances than this is not yet a description of how the speaker
# normally sounds, so nothing is inferred from it.
MIN_BASELINE_SAMPLES = 3

# How fast the baseline follows the speaker. Low enough that one animated sentence does not
# redefine "normal", high enough to track someone who settles down after the first minute.
BASELINE_EMA_ALPHA = 0.2

Arousal = Literal["low", "neutral", "high"]
Valence = Literal["negative", "neutral", "positive"]


@dataclass(frozen=True, slots=True)
class ProsodyFeatures:
    """What one utterance sounded like, before it is compared to anything."""

    pitch_median_hz: float
    """Centre of the voiced pitch. The speaker's register for this utterance."""

    pitch_iqr_hz: float
    """Spread of the voiced pitch — how much the voice moved. A monotone reading and an
    animated one can share a median and differ entirely here."""

    rms: float
    """Loudness, as root-mean-square amplitude in [0, 1]."""

    voiced_ratio: float
    """Share of frames carrying pitch. Below MIN_VOICED_RATIO nothing here is trustworthy."""

    speech_rate: float
    """Voiced seconds per wall-clock second. A proxy for tempo that needs no transcript, so it
    is available at the moment the audio arrives rather than after recognition."""

    duration_ms: int

    @property
    def is_usable(self) -> bool:
        return self.voiced_ratio >= MIN_VOICED_RATIO and self.pitch_median_hz > 0


@dataclass(frozen=True, slots=True)
class SpeakerBaseline:
    """How this speaker normally sounds, updated as they keep talking."""

    pitch_median_hz: float = 0.0
    pitch_iqr_hz: float = 0.0
    rms: float = 0.0
    speech_rate: float = 0.0
    sample_count: int = 0

    @property
    def is_established(self) -> bool:
        return self.sample_count >= MIN_BASELINE_SAMPLES and self.pitch_median_hz > 0


@dataclass(frozen=True, slots=True)
class Delivery:
    """The utterance expressed as departures from the speaker's own normal."""

    pitch_lift: float
    """Median pitch ÷ baseline median. Above 1 is higher than they usually speak."""

    pitch_variation: float
    """Pitch spread ÷ baseline spread. Above 1 is more animated than they usually are."""

    energy_ratio: float
    rate_ratio: float
    arousal: Arousal

    @property
    def is_measured(self) -> bool:
        """False when the speaker has no usable baseline yet — the ratios are all 1.0 and mean
        nothing, and callers must not read intent into them."""
        return self.arousal != "neutral" or self.pitch_lift != 1.0


NEUTRAL_DELIVERY = Delivery(
    pitch_lift=1.0,
    pitch_variation=1.0,
    energy_ratio=1.0,
    rate_ratio=1.0,
    arousal="neutral",
)


def measure(
    pcm: npt.NDArray[np.floating[Any]], sample_rate: int, frame_ms: int = 30
) -> ProsodyFeatures:
    """Measure one utterance's delivery from mono PCM in [-1, 1].

    Pitch is found by autocorrelation per frame rather than with a library: the whole
    measurement is four statistics, it has to run on every utterance inside the latency budget,
    and a dependency that loads a model to do it would cost more than it explains. Frames whose
    autocorrelation peak is weak are treated as unvoiced rather than assigned a confident wrong
    pitch, which is what keeps a median honest.
    """
    if pcm.size == 0 or sample_rate <= 0:
        return ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    audio = pcm.astype(np.float64, copy=False)
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = audio.size // frame_len
    if frame_count == 0:
        return ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    frames = audio[: frame_count * frame_len].reshape(frame_count, frame_len)

    pitches: list[float] = []
    voiced = 0
    for frame in frames:
        f0 = _frame_pitch(frame, sample_rate)
        if f0 > 0:
            pitches.append(f0)
            voiced += 1

    duration_ms = int(audio.size / sample_rate * 1000)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    voiced_ratio = voiced / frame_count

    if pitches:
        arr = np.asarray(pitches)
        pitch_median = float(np.median(arr))
        pitch_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    else:
        pitch_median = 0.0
        pitch_iqr = 0.0

    return ProsodyFeatures(
        pitch_median_hz=pitch_median,
        pitch_iqr_hz=pitch_iqr,
        rms=rms,
        voiced_ratio=voiced_ratio,
        # Voiced share IS the tempo proxy: someone speaking quickly leaves less silence per
        # second. It needs no transcript, so it is available before recognition returns.
        speech_rate=voiced_ratio,
        duration_ms=duration_ms,
    )


def _frame_pitch(frame: npt.NDArray[np.floating[Any]], sample_rate: int) -> float:
    """Autocorrelation pitch for one frame, or 0.0 when the frame is not voiced.

    The peak has to clear 30% of the zero-lag energy to count. Voiced speech is strongly
    periodic and clears it comfortably; breath, room tone and consonants do not, and admitting
    them is how a median ends up describing the air conditioning.
    """
    frame = frame - frame.mean()
    energy = float(np.dot(frame, frame))
    if energy <= 1e-9:
        return 0.0

    corr = np.correlate(frame, frame, mode="full")[frame.size - 1 :]

    min_lag = int(sample_rate / F0_MAX_HZ)
    max_lag = int(sample_rate / F0_MIN_HZ)
    if max_lag >= corr.size or min_lag >= max_lag:
        return 0.0

    window = corr[min_lag:max_lag]
    peak_index = int(np.argmax(window))
    peak = float(window[peak_index])
    if peak < 0.3 * energy:
        return 0.0

    lag = min_lag + peak_index
    return sample_rate / lag if lag > 0 else 0.0


def update_baseline(baseline: SpeakerBaseline, features: ProsodyFeatures) -> SpeakerBaseline:
    """Fold one utterance into the speaker's rolling normal.

    Unusable utterances are skipped rather than averaged in as zeros: a baseline that has been
    pulled toward silence would make every later utterance look loud and high.
    """
    if not features.is_usable:
        return baseline

    if baseline.sample_count == 0:
        return SpeakerBaseline(
            pitch_median_hz=features.pitch_median_hz,
            pitch_iqr_hz=features.pitch_iqr_hz,
            rms=features.rms,
            speech_rate=features.speech_rate,
            sample_count=1,
        )

    a = BASELINE_EMA_ALPHA
    return SpeakerBaseline(
        pitch_median_hz=(1 - a) * baseline.pitch_median_hz + a * features.pitch_median_hz,
        pitch_iqr_hz=(1 - a) * baseline.pitch_iqr_hz + a * features.pitch_iqr_hz,
        rms=(1 - a) * baseline.rms + a * features.rms,
        speech_rate=(1 - a) * baseline.speech_rate + a * features.speech_rate,
        sample_count=baseline.sample_count + 1,
    )


# Bands for calling an utterance activated or subdued.
#
# THESE ARE STARTING POINTS, NOT MEASUREMENTS. They are the shape of the rule — everything
# relative, two features required to agree — with plausible values. Before anyone claims
# accuracy for them they must be calibrated the way STT's per-language thresholds were: run a
# labelled set of real meeting utterances through tools/, sweep, and take the values where
# agreement stops improving. Guessing them and then quoting them as tuned is the failure this
# comment exists to prevent.
HIGH_PITCH_LIFT = 1.12
HIGH_ENERGY_RATIO = 1.25
HIGH_PITCH_VARIATION = 1.40
LOW_PITCH_LIFT = 0.93
LOW_ENERGY_RATIO = 0.80


def to_delivery(features: ProsodyFeatures, baseline: SpeakerBaseline) -> Delivery:
    """Compare an utterance to its speaker's normal.

    Two features must agree before an utterance is called activated or subdued. Pitch alone
    rises for a question as readily as for excitement, and energy alone rises when someone
    simply moves closer to the microphone; requiring both is what keeps ordinary speech
    ordinary.
    """
    if not features.is_usable or not baseline.is_established:
        return NEUTRAL_DELIVERY

    pitch_lift = _ratio(features.pitch_median_hz, baseline.pitch_median_hz)
    pitch_variation = _ratio(features.pitch_iqr_hz, baseline.pitch_iqr_hz)
    energy_ratio = _ratio(features.rms, baseline.rms)
    rate_ratio = _ratio(features.speech_rate, baseline.speech_rate)

    arousal: Arousal = "neutral"
    if (pitch_lift >= HIGH_PITCH_LIFT and energy_ratio >= HIGH_ENERGY_RATIO) or (
        pitch_variation >= HIGH_PITCH_VARIATION and energy_ratio >= HIGH_ENERGY_RATIO
    ):
        arousal = "high"
    elif pitch_lift <= LOW_PITCH_LIFT and energy_ratio <= LOW_ENERGY_RATIO:
        arousal = "low"

    return Delivery(
        pitch_lift=pitch_lift,
        pitch_variation=pitch_variation,
        energy_ratio=energy_ratio,
        rate_ratio=rate_ratio,
        arousal=arousal,
    )


def _ratio(value: float, reference: float) -> float:
    if reference <= 1e-9:
        return 1.0
    return value / reference


# Arousal decides how activated the delivery was; valence decides which side of activated it is
# on. Neither is enough alone, which is why this is a table rather than a threshold.
_EMOTION_TABLE: dict[tuple[Arousal, Valence], str] = {
    ("high", "positive"): "excited",
    ("high", "negative"): "frustrated",
    ("high", "neutral"): "surprised",
    ("low", "positive"): "content",
    ("low", "negative"): "sad",
    ("low", "neutral"): "calm",
}


def to_generation_config(
    delivery: Delivery,
    valence: Valence = "neutral",
    *,
    speed_center: float = 1.0,
) -> dict[str, float | str]:
    """Turn a measured delivery into Cartesia's three controls.

    Speed and volume are carried continuously — they are ratios of the speaker's own tempo and
    loudness, so they degrade gracefully: a small measurement error makes the dub slightly fast
    rather than wrongly angry. The emotion label is the only categorical judgement here, and it
    is omitted entirely when the delivery is ordinary, because "neutral" is what the model
    already does and sending it adds a claim without adding information.
    """
    speed = _clamp(speed_center * delivery.rate_ratio, SPEED_MIN, SPEED_MAX)
    volume = _clamp(delivery.energy_ratio, VOLUME_MIN, VOLUME_MAX)

    config: dict[str, float | str] = {
        "speed": round(speed, 3),
        "volume": round(volume, 3),
    }

    emotion = _EMOTION_TABLE.get((delivery.arousal, valence))
    if emotion is not None:
        config["emotion"] = emotion

    return config


def _clamp(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return 1.0
    return max(low, min(high, value))


def pcm16_to_float(raw: bytes) -> npt.NDArray[np.float32]:
    """Signed 16-bit little-endian PCM to float32 in [-1, 1] — the format every audio chunk in
    this pipeline arrives in (see AudioChunkMessage.audio_data)."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return samples / 32768.0


__all__ = [
    "Arousal",
    "Delivery",
    "NEUTRAL_DELIVERY",
    "ProsodyFeatures",
    "SpeakerBaseline",
    "Valence",
    "measure",
    "pcm16_to_float",
    "to_delivery",
    "to_generation_config",
    "update_baseline",
]
