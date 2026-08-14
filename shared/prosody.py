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

HOW MUCH OF IT SURVIVES ON THE MODEL THIS PRODUCT SHIPS
    Measured 2026-08-13 against the live API, same sentence, median of three renders:

        control          sonic-3 (en)              sonic-3.5 (en)      sonic-3.5 (vi)
        speed 0.6        9567 ms  (1.70x longer)   6800 ms  (1.18x)    4960 ms  (1.19x)
        speed 1.0        5619 ms                   5760 ms             4160 ms
        speed 1.5        3436 ms  (1.64x shorter)  5600 ms  (1.03x)    3680 ms  (1.13x)
        volume 0.6       rms 0.0564                rms 0.0884          rms 0.0887
        volume 2.0       rms 0.2124  (3.8x)        rms 0.2626  (3.0x)  rms 0.3050  (3.4x)

    So: volume is honoured almost literally everywhere. Speed is honoured almost literally on
    sonic-3 and heavily damped on sonic-3.5 — a request to slow down 40% buys 18%, and a request
    to speed up 50% buys almost nothing. TTS_MODEL is sonic-3.5 because it is the model with
    Vietnamese, so the tempo half of this module currently lands as a nudge rather than a match.
    That is a property of the model, not a defect here: the ratios sent are the true ones, and
    they will land in full the day sonic-3 (or its successor) covers the target languages.

    The older `speed` ENUM ("slow"/"normal"/"fast", TTSSettings.speed) does nothing at all on
    sonic-3.5 — four renders each gave medians of 6120/6200/5960 ms with a per-case spread of
    5680–6560, i.e. the setting is inside the noise. It is still sent, because it is not inert
    on every model and removing it would be a silent behaviour change on models where it works.

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
# 2.0, not 1.5. The 1.5 was SPEED_MAX copied one line down: the verified rejection quoted above
# is `speed must be between 0.600000 and 1.500000`, which says nothing about volume. Cartesia's
# accepted volume range is [0.5, 2.0] — the SDK's own GenerationConfig carries it, and the
# measured table at the top of this file records a successful render AT volume 2.0 on all three
# model/language combinations. So a third of the available dynamic range was being clamped away
# by a constant that contradicted this module's own measurements.
VOLUME_MAX = 2.0

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
#
# Each cell is a LADDER, mildest first, and the rung is chosen by how far past its band the
# delivery actually went — see `_intensity`. Two coarse three-way axes can only ever name six
# feelings, but the pipeline measures `pitch_lift` and `energy_ratio` continuously and then
# throws the magnitude away by bucketing; the ladders spend that magnitude instead of inventing
# a third axis to justify more labels.
#
# THE SAFETY PROPERTY THAT MAKES THIS WORTH DOING
#     Every rung within a cell is the SAME feeling at a different strength. So an error in the
#     rung is "excited" where "happy" was meant — one step along one scale. It can never be
#     "angry" where "happy" was meant, because that would take an error in valence, which no
#     amount of intensity can produce. The tier boundaries are exactly as uncalibrated as the
#     bands they sit on (see the comment above HIGH_PITCH_LIFT); ordering them this way is what
#     keeps that uncalibration cheap.
#
# Every name below is copied from cartesia.types.GenerationConfig's own Literal — the SDK is the
# authority, not the docs page. Cartesia states that an emotion outside its list is "not
# supported, and results are not guaranteed", so an invented name fails silently and strangely
# rather than loudly.
# Rung 0 of every ladder is EXACTLY the label this table produced before the ladders existed.
# That is not a coincidence to preserve tests — it is what makes this a strict extension: the
# ladder can only ever add a stronger word for a stronger delivery, and can never re-label an
# ordinary one. Reordering a rung 0 is a behaviour change and should be argued for on its own.
_EMOTION_LADDERS: dict[tuple[Arousal, Valence], tuple[str, ...]] = {
    ("high", "positive"): ("excited", "elated", "euphoric"),
    ("high", "negative"): ("frustrated", "angry", "outraged"),
    ("high", "neutral"): ("surprised", "amazed"),
    ("low", "positive"): ("content", "peaceful", "serene"),
    ("low", "negative"): ("sad", "dejected"),
    ("low", "neutral"): ("calm", "tired"),
}

# How far past its band a delivery has to go to reach full intensity. A ratio 40% beyond the
# threshold is emphatic by any reading; beyond that the ladder has nothing further to say.
_FULL_INTENSITY_MARGIN = 0.40

# Intensity needed to climb to rung 1, then rung 2. Deliberately NOT an even split of the range.
#
# This module's stated principle is that wrong is worse than neutral, and the bands these rungs
# sit on are uncalibrated. Even thirds would put an ordinary emphatic sentence — the common case —
# on the middle rung, so "angry" would become the routine label for anyone who raised their voice
# while saying something negative. Biasing the steps late keeps the mild rung as the default and
# reserves the strong ones for deliveries that are extreme by this speaker's own standard.
#
# The practical effect: a delivery around half-intensity keeps exactly the label this table
# produced before the ladders existed. The ladder only ever ADDS a stronger word for a stronger
# delivery; it never re-labels an ordinary one.
_LADDER_STEPS = (0.5, 0.9)


def _intensity(delivery: Delivery) -> float:
    """0..1 — how far past its own band this delivery went.

    Measured on the two features the bands are drawn from, taking the STRONGER of the two: a
    speaker can be emphatic mostly in pitch or mostly in loudness, and requiring both to agree
    is the job of `to_delivery`, which has already been done by the time this runs.

    Returns 0.0 for a delivery with no arousal, where no ladder is consulted anyway.
    """
    if delivery.arousal == "high":
        excess = max(
            delivery.pitch_lift - HIGH_PITCH_LIFT,
            delivery.energy_ratio - HIGH_ENERGY_RATIO,
        )
    elif delivery.arousal == "low":
        excess = max(
            LOW_PITCH_LIFT - delivery.pitch_lift,
            LOW_ENERGY_RATIO - delivery.energy_ratio,
        )
    else:
        return 0.0

    if math.isnan(excess):
        return 0.0
    return max(0.0, min(1.0, excess / _FULL_INTENSITY_MARGIN))


def _emotion_for(delivery: Delivery, valence: Valence) -> str | None:
    ladder = _EMOTION_LADDERS.get((delivery.arousal, valence))
    if ladder is None:
        return None

    intensity = _intensity(delivery)
    rung = 0
    for index, step in enumerate(_LADDER_STEPS[: len(ladder) - 1], start=1):
        if intensity >= step:
            rung = index
    return ladder[rung]


def to_generation_config(
    delivery: Delivery,
    valence: Valence | None = None,
    *,
    speed_center: float = 1.0,
) -> dict[str, float | str]:
    """Turn a measured delivery into Cartesia's three controls.

    Speed and volume are carried continuously — they are ratios of the speaker's own tempo and
    loudness, so they degrade gracefully: a small measurement error makes the dub slightly fast
    rather than wrongly angry. The emotion label is the only categorical judgement here, and it
    is omitted entirely when the delivery is ordinary, because "neutral" is what the model
    already does and sending it adds a claim without adding information.

    `valence=None` means NOT DETERMINED, which is the pipeline's actual state today: nothing
    upstream reads the words for sentiment yet. It is not the same as "neutral", and it must not
    collapse into it — ("high", "neutral") would label an emphatic speaker "surprised", which is
    a guess about their feelings made from loudness alone. Unknown valence yields no emotion at
    all, and the delivery still carries through speed and volume.
    """
    speed = _clamp(speed_center * delivery.rate_ratio, SPEED_MIN, SPEED_MAX)
    volume = _clamp(delivery.energy_ratio, VOLUME_MIN, VOLUME_MAX)

    config: dict[str, float | str] = {
        "speed": round(speed, 3),
        "volume": round(volume, 3),
    }

    if valence is not None:
        emotion = _emotion_for(delivery, valence)
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
