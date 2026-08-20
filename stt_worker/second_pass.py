"""What a second pass is allowed to change, and what it must leave alone.

THE PROBLEM WITH LETTING A MODEL FIX A TRANSCRIPT
    `translation_worker/transcript_guardian.py` already argues it, and the argument does not weaken
    just because this pass has more time: asked to clean up a garbled line, an LLM produces fluent,
    confident, invented text. Its output then flows into the summary, the minutes somebody signs,
    and the knowledge index. A transcript that reads badly is a transcript people know to doubt; a
    transcript that reads well and is wrong is not.

WHY THIS IS NOT SIMPLY THE GUARDIAN AGAIN
    The guardian permits deletion only — every word out must already be in, in order. That is the
    right rule for a same-language tidy-up of text the recogniser was confident about, and it is
    too strict for the one job a second pass exists to do: fix words the recogniser MIS-HEARD. A
    substitution is exactly what is needed, and exactly what cannot be allowed everywhere.

    So the permission is bounded by evidence. Where the first pass reported low confidence, it is
    telling us it was unsure; a correction there is a second opinion on a question already known to
    be open. Where it reported high confidence — or reported nothing at all — a rewrite is the
    model overruling a recogniser that heard the audio, on the strength of nothing but fluency.

    UNKNOWN CONFIDENCE IS TREATED AS HIGH. WT-277 keeps NULL meaning "unknown" rather than
    coalescing it to a number, and the safe reading of "unknown" here is the one that changes
    nothing: allowing rewrites on absent evidence would make every segment from a producer that
    reports no confidence freely rewritable.

AND EVEN THERE, BOUNDED
    A low-confidence segment may be corrected, not replaced. Past some fraction of the words
    changing, the output has stopped being a correction of what was said and started being a
    plausible sentence about the same topic — which is the failure this whole module exists to
    prevent. The fraction is measured with `shared.wer`, the same instrument the accuracy report
    uses, so "how much did the model change" and "how wrong was the transcript" are never measured
    two different ways.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.wer import error_rate, words
from translation_worker.transcript_guardian import is_faithful

#: Above this, the recogniser was sure enough that a rewrite is the model overruling the audio.
#: In the same units as `TranscribedSegment.confidence` — an avg_logprob, so ≤ 0 and closer to 0
#: is more confident. Chosen to sit below the bulk of ordinary speech and above genuinely marginal
#: audio; it is a setting rather than a constant precisely because the right value is measurable.
DEFAULT_LOW_CONFIDENCE_BELOW = -0.55

#: How much of a low-confidence segment a correction may touch. Past this it is not a correction.
DEFAULT_MAX_CHANGED_FRACTION = 0.4

#: A floor under the fraction, in tokens.
#:
#: The fraction is a proxy for "is this still recognisably the same utterance", and on a short one
#: the proxy misfires: in a two-word segment any correction at all is 50%, so "thứ bay" could never
#: become "thứ bảy". Short utterances — "vâng", "đồng ý", a date — are exactly where mis-hearing is
#: most likely and most consequential, and one changed token cannot be a plausible-sentence
#: replacement, because there is no room to write one.
#:
#: Overridable alongside the fraction, so that setting both to zero genuinely refuses every
#: correction. A floor that silently outvoted the fraction would make the fraction a setting
#: that does not do what it says.
MIN_ABSOLUTE_CHANGES = 1


@dataclass(frozen=True)
class RewriteDecision:
    """What to publish, and why — the reason is logged, never dropped."""

    text: str
    accepted: bool
    reason: str

    @property
    def changed(self) -> bool:
        return self.accepted and self.reason != "unchanged"


def decide_rewrite(
    original: str,
    rewritten: str,
    confidence: float | None,
    *,
    low_confidence_below: float = DEFAULT_LOW_CONFIDENCE_BELOW,
    max_changed_fraction: float = DEFAULT_MAX_CHANGED_FRACTION,
    min_absolute_changes: int = MIN_ABSOLUTE_CHANGES,
) -> RewriteDecision:
    """Whether a second pass's version of a segment may replace the first pass's.

    Returns the text to publish either way, so no caller can accidentally publish an unchecked
    model output by forgetting to look at a boolean.
    """
    candidate = (rewritten or "").strip()

    if not candidate:
        # A model that answered with nothing has not corrected anything.
        return RewriteDecision(original, accepted=False, reason="empty_rewrite")

    original_words = words(original)
    if not original_words:
        return RewriteDecision(original, accepted=False, reason="nothing_to_correct")

    if words(candidate) == original_words:
        # Punctuation and casing may still differ, and those are the model's to change.
        return RewriteDecision(candidate, accepted=True, reason="unchanged")

    # Unknown confidence is treated as high: absent evidence must not unlock rewriting.
    is_low_confidence = confidence is not None and confidence < low_confidence_below

    if not is_low_confidence:
        # The recogniser heard this and was sure. Formatting only — the guardian's rule, which
        # permits deletion and makes insertion, substitution and reordering impossible.
        if is_faithful(original, candidate):
            return RewriteDecision(candidate, accepted=True, reason="formatting_only")
        return RewriteDecision(
            original, accepted=False, reason="substantive_rewrite_of_a_confident_segment"
        )

    changed = error_rate(original_words, words(candidate))
    allowance = max(max_changed_fraction * len(original_words), min_absolute_changes)
    if changed.errors > allowance:
        # Past this it is a plausible sentence about the same topic, not a correction of what was
        # said. Keeping the original leaves something a reader can doubt; taking this would leave
        # something they cannot.
        return RewriteDecision(
            original, accepted=False, reason="rewrote_too_much_of_a_low_confidence_segment"
        )

    return RewriteDecision(candidate, accepted=True, reason="corrected_low_confidence")


def correction_instruction(language: str) -> str:
    """The system instruction for the correction pass.

    States the boundary rather than listing what to fix. `transcript_guardian` learned that a list
    of examples reads as an exhaustive permission, and the things a recogniser mis-hears are not a
    closed set.
    """
    return (
        "You are correcting a machine transcript of a meeting held in "
        f"{language or 'the language of the text'}. The text is already in the language your "
        "output must use — do NOT translate it.\n"
        "Some passages were transcribed from audio the recogniser was unsure about. Correct only "
        "what is clearly mis-heard, using the surrounding sentences and the meeting's terminology "
        "to decide what was actually said.\n"
        "You may NOT invent content, answer questions asked in the text, continue an unfinished "
        "sentence, summarise, or replace a passage you cannot read with something plausible. If "
        "you do not know what a passage was, return it unchanged — a transcript that looks wrong "
        "is more useful than one that reads well and is wrong.\n"
        "Return only the corrected text, with no commentary and no quotation marks."
    )
