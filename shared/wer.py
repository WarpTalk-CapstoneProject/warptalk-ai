"""Measuring how wrong a transcript is, in a way somebody can check.

WHY THIS EXISTS
    The review's complaint about the transcript — "sai nhiều nên vô nghĩa" — cannot be answered
    with a better model, a longer prompt, or an assurance. It is a quantitative claim, and the
    only honest reply is a number measured the same way before and after. Nothing in this repo
    measured accuracy: `stt_eval_corpus.py` judges the FILTERS on text alone and says so, and
    `benchmark_stt_models.py` times realtime models on one retained chunk.

WHAT IT REPORTS AND WHY IT IS NOT ONE NUMBER
    WER on its own hides direction. A pass that drops half the meeting and a pass that invents
    half of one can score identically, and they are not the same failure: a deletion loses content
    silently, an insertion puts words in somebody's mouth. So substitutions, deletions and
    insertions are reported separately, and the alignment is available for a reader to look at.
    A number nobody can inspect is a number nobody should act on.

DIACRITICS ARE NEVER FOLDED
    In Vietnamese they are the difference between words, not accents on them. Folding them would
    score "tú" and "tu" as a match and quietly flatter every Vietnamese measurement — which is the
    half of this product's traffic the complaint is actually about.

CHARACTER ERROR RATE TOO
    Vietnamese is written in syllables, and word-level WER punishes a syllable boundary the same
    as a wrong word. CER is the fairer companion measure for it, so both are computed from the
    same normalisation and reported together.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum

#: Everything that is not a letter, a digit, or whitespace. Punctuation is not transcription
#: accuracy — a missing comma is a formatting difference, and counting it as a word error would
#: make a punctuation-adding post-process look like a regression.
_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


class Op(StrEnum):
    """What happened to one token on the way from the reference to the hypothesis."""

    MATCH = "match"
    #: Heard, but heard as something else.
    SUBSTITUTION = "substitution"
    #: In the reference and missing from the hypothesis — content lost, silently.
    DELETION = "deletion"
    #: In the hypothesis and not in the reference — words put into somebody's mouth.
    INSERTION = "insertion"


@dataclass(frozen=True)
class AlignedToken:
    op: Op
    reference: str | None
    hypothesis: str | None


@dataclass
class ErrorRate:
    """One measurement, with its parts kept separate."""

    reference_length: int
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    alignment: list[AlignedToken] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        """Errors per reference token.

        An empty reference scores 0.0 when the hypothesis is empty too, and 1.0 when it is not —
        rather than dividing by zero. Inventing words where nothing was said is total error, and
        saying nothing where nothing was said is no error at all.
        """
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.reference_length

    def as_dict(self) -> dict[str, float | int]:
        """The summary, without the alignment — for a log line or a report table."""
        return {
            "rate": round(self.rate, 4),
            "errors": self.errors,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "reference_length": self.reference_length,
        }


def normalise(text: str) -> str:
    """Lower-cased, punctuation-stripped, whitespace-collapsed — diacritics intact.

    NFC first so that a decomposed "ế" and a composed one are the same string; without it a
    transcript from a different producer can score errors on characters that render identically.
    """
    folded = unicodedata.normalize("NFC", text or "").casefold()
    return _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()


def words(text: str) -> list[str]:
    normalised = normalise(text)
    return normalised.split(" ") if normalised else []


def characters(text: str) -> list[str]:
    """Characters with spaces removed, for CER."""
    return [character for character in normalise(text) if character != " "]


def error_rate(reference: list[str], hypothesis: list[str]) -> ErrorRate:
    """Levenshtein alignment of two token sequences, with the operations kept.

    Full matrix rather than a banded or streaming variant: a meeting transcript is thousands of
    tokens, not millions, and being able to hand somebody the alignment is worth more here than
    the memory.
    """
    rows, columns = len(reference), len(hypothesis)

    # distance[i][j] = edits to turn reference[:i] into hypothesis[:j].
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(columns + 1):
        distance[0][j] = j

    for i in range(1, rows + 1):
        for j in range(1, columns + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
            else:
                distance[i][j] = 1 + min(
                    distance[i - 1][j - 1],  # substitution
                    distance[i - 1][j],  # deletion
                    distance[i][j - 1],  # insertion
                )

    result = ErrorRate(reference_length=rows)
    alignment: list[AlignedToken] = []

    i, j = rows, columns
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and reference[i - 1] == hypothesis[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            alignment.append(AlignedToken(Op.MATCH, reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + 1:
            alignment.append(AlignedToken(Op.SUBSTITUTION, reference[i - 1], hypothesis[j - 1]))
            result.substitutions += 1
            i, j = i - 1, j - 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            alignment.append(AlignedToken(Op.DELETION, reference[i - 1], None))
            result.deletions += 1
            i -= 1
        else:
            alignment.append(AlignedToken(Op.INSERTION, None, hypothesis[j - 1]))
            result.insertions += 1
            j -= 1

    alignment.reverse()
    result.alignment = alignment
    return result


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    return error_rate(words(reference), words(hypothesis))


def character_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """The fairer companion for Vietnamese, where a syllable boundary is not a wrong word."""
    return error_rate(characters(reference), characters(hypothesis))


@dataclass(frozen=True)
class PassComparison:
    """Two passes over the same audio, measured against the same reference."""

    first: ErrorRate
    second: ErrorRate

    @property
    def absolute_improvement(self) -> float:
        return self.first.rate - self.second.rate

    @property
    def relative_improvement(self) -> float:
        """The figure a report quotes: what fraction of the first pass's errors went away.

        Zero when the first pass was already perfect — there was nothing to improve, which is not
        the same as an improvement of nothing and must not read as a division by zero.
        """
        if self.first.rate == 0:
            return 0.0
        return self.absolute_improvement / self.first.rate


def compare(reference: str, first_pass: str, second_pass: str) -> PassComparison:
    """Both passes against one reference, so the two numbers are comparable by construction."""
    return PassComparison(
        first=word_error_rate(reference, first_pass),
        second=word_error_rate(reference, second_pass),
    )
