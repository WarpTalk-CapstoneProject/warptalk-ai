"""Make the dub take about as long as the thing it is dubbing.

THE PROBLEM
    A translated sentence is rarely the same length as its source. Vietnamese rendered from
    English routinely runs longer, and the dub then finishes well after the speaker has moved on
    — so the listener is hearing the answer to a question while the next one is already being
    asked. Level 4 calls this out as rhythm; dubbing calls it isochrony.

    The pipeline already knows the source duration: STT stamps `start_ms`/`end_ms` on every
    segment and they survive all the way to `TranslationResultMessage`. Nothing has ever used
    them for this.

WHY A CLOSED LOOP AND NOT AN ESTIMATE
    To make a dub fit you must know how long it WOULD be, and that is only known after
    synthesising it. The two obvious ways out are both bad: predicting duration from character
    count is a per-language, per-voice guess that would need its own model, and synthesising
    twice doubles the cost and the latency of the one thing the listener is waiting for.

    So this measures instead. Every completed dub reports how long it actually ran against how
    long the source ran, and the ratio feeds forward into the NEXT utterance from that speaker.
    It needs no model, costs nothing, and converges on whatever this speaker, this language pair
    and this voice actually do — which is more than an estimator could know anyway.

    Same shape as `SpeakerBaseline` in shared/prosody.py, and for the same reason: everything
    here is relative to the speaker rather than absolute.

WHY IT IS CLAMPED HARD
    Two independent reasons, and the second is the one that bites.

    A ratio built from a mismeasured pair could otherwise ask for speech at a tempo nobody can
    follow. That is the ordinary reason.

    The other is that `sonic-3.5` DAMPS `speed` to roughly a fifth of what is asked (measured;
    see shared/prosody.py). A naive integral controller against an actuator that only delivers
    20% of its command winds up: it keeps asking for more, gets a fifth of it, and pins itself at
    the limit forever. The clamp is what turns that runaway into a bounded, honest partial
    correction — and it is why "the dub still overruns a bit on sonic-3.5" is a property of the
    model, not a bug here.
"""

from __future__ import annotations

from dataclasses import dataclass

# How fast the fit follows the speaker. Deliberately slower than prosody's baseline EMA: tempo
# correction is heard directly, so it should settle over a few utterances rather than lurch
# after one unusual sentence.
FIT_EMA_ALPHA = 0.25

# Below this many observations the ratio is one speaker saying one thing, not a description of
# how this language pair behaves.
MIN_FIT_SAMPLES = 2

# Correction bounds. Never ask for more than a 25% tempo change on account of fit alone —
# beyond that the cure is worse than the overrun, and the delivery the speaker actually used
# (which also moves `speed`) stops being audible under the correction.
MIN_SPEED_CENTER = 0.8
MAX_SPEED_CENTER = 1.25

# Pairs outside this band are not a slow dub, they are a measurement fault: a truncated
# synthesis, a segment whose end_ms never got stamped, a chunk that was mostly silence.
# Folding them in would poison the ratio for every later utterance.
MIN_PLAUSIBLE_RATIO = 0.25
MAX_PLAUSIBLE_RATIO = 4.0

# Under this the pair is too short to time reliably — a one-word acknowledgement, where a few
# tens of milliseconds of leading silence swamps the measurement.
MIN_MEASURABLE_MS = 400


@dataclass(frozen=True, slots=True)
class DubFit:
    """How this speaker's dubs have been running against the clock, in this language pair."""

    ratio: float = 1.0
    """Dub duration ÷ source duration, rolling. Above 1 means the dub overruns."""

    sample_count: int = 0

    @property
    def is_established(self) -> bool:
        return self.sample_count >= MIN_FIT_SAMPLES and self.ratio > 0


NO_FIT = DubFit()


def observe(fit: DubFit, source_ms: int, dub_ms: int) -> DubFit:
    """Fold one completed dub into the rolling fit.

    Unusable pairs are SKIPPED rather than averaged in. A dub that came back empty, or a segment
    with no timing, would otherwise drag the ratio toward zero and make every later utterance
    ask to be spoken slower and slower.
    """
    if source_ms < MIN_MEASURABLE_MS or dub_ms < MIN_MEASURABLE_MS:
        return fit

    ratio = dub_ms / source_ms
    if not (MIN_PLAUSIBLE_RATIO <= ratio <= MAX_PLAUSIBLE_RATIO):
        return fit

    if fit.sample_count == 0:
        return DubFit(ratio=ratio, sample_count=1)

    a = FIT_EMA_ALPHA
    return DubFit(
        ratio=(1 - a) * fit.ratio + a * ratio,
        sample_count=fit.sample_count + 1,
    )


def speed_center(fit: DubFit) -> float:
    """The tempo this speaker's next dub should be centred on.

    A dub that has been running 20% long should be asked to speak 20% faster, so the correction
    is the ratio itself. Returns exactly 1.0 until the fit is established, which makes an
    un-established speaker byte-for-byte identical to the behaviour before this module existed.

    This is a CENTRE, not the final speed: `to_generation_config` multiplies the speaker's own
    measured `rate_ratio` through it, so someone who genuinely slowed down still sounds like they
    slowed down — just inside a slot that fits.
    """
    if not fit.is_established:
        return 1.0
    return max(MIN_SPEED_CENTER, min(MAX_SPEED_CENTER, fit.ratio))


__all__ = [
    "DubFit",
    "NO_FIT",
    "MAX_SPEED_CENTER",
    "MIN_SPEED_CENTER",
    "observe",
    "speed_center",
]
