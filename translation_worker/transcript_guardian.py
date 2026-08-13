"""Tidying a same-language transcript without letting the model rewrite it.

WHAT THIS IS FOR
    When a listener's language matches the speaker's, the translation worker forwards the STT
    text unchanged (`passthrough`). That text is raw recogniser output: no sentence casing,
    little punctuation, and every "ờ", "à", "ừm" the speaker actually said. A same-language pass
    can make it read like writing.

WHAT THIS REFUSES TO BE
    A repair tool. If the recogniser mis-heard a word, that word is gone — the audio is not
    available here and no amount of language modelling can recover it. Asked to "clean up" a
    garbled line, an LLM produces fluent, confident, invented text, which then flows into the
    summary, the knowledge index and the meeting record. A transcript that reads badly is a
    transcript people know to doubt; a transcript that reads well and is wrong is not.

    So the model is instructed to change only formatting — and, because an instruction is not a
    guarantee, `is_faithful` checks mechanically that it did. Anything that fails the check is
    discarded and the original is kept. The worst outcome of this module is "no improvement".
    It is never "different words".
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

# Disfluencies are the one thing the guardian may DELETE outright, because a filler carries no
# information and every language has its own. Anything not on this list must survive.
_FILLERS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "vi": ("ờ", "à", "ừm", "ừ", "ơ", "ạ"),
    "en": ("uh", "um", "erm", "ah", "hmm"),
}

_PUNCTUATION_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

# Below this, the "polish" is a rewrite. Chosen high on purpose: legitimate work here is
# punctuation, casing and a handful of dropped fillers, all of which survive normalisation
# almost entirely. A model that paraphrases lands far below it.
_MIN_SIMILARITY = 0.90

# A tidy-up cannot need materially more characters than it was given. Real formatting adds
# punctuation and capitals — a few percent — while fabrication adds sentences.
_MAX_GROWTH_RATIO = 1.15


def guardian_instruction(language: str) -> str:
    """The system instruction for a same-language tidy-up.

    Deliberately narrow, and negative where it matters: the failure this guards against is a
    helpful model deciding a garbled sentence "obviously meant" something.
    """
    fillers = _FILLERS_BY_LANGUAGE.get(_base_language(language), ())
    filler_hint = (
        f" Filler sounds such as {', '.join(fillers)} may be removed."
        if fillers
        else ""
    )

    return (
        "You are formatting a live meeting transcript. The text is in the same language as your "
        "output must be — do NOT translate it.\n"
        "Do exactly this and nothing else: fix capitalisation, add sentence punctuation, and "
        "correct spacing." + filler_hint + "\n"
        "Never change, add, remove or reorder any word that carries meaning. Never complete an "
        "unfinished sentence. Never guess what a garbled passage was meant to say — leave it "
        "exactly as it is. If the text is already well formatted, return it unchanged.\n"
        "Return only the corrected text, with no commentary and no quotation marks."
    )


def _base_language(language: str) -> str:
    return (language or "").split("-", 1)[0].strip().lower()


def _skeleton(text: str) -> str:
    """The part of the text the guardian is not allowed to touch.

    Case, punctuation and spacing are stripped, because those are precisely what it MAY change.
    Diacritics are kept: in Vietnamese they are the difference between words, not decoration.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _WHITESPACE_RE.sub(" ", _PUNCTUATION_RE.sub(" ", normalized)).strip()


def is_faithful(original: str, polished: str, language: str = "") -> bool:
    """Whether `polished` is `original` with only formatting changed.

    Fillers are allowed to disappear, so they are removed from BOTH sides before comparing —
    otherwise every correctly-tidied Vietnamese sentence would look like a rewrite.
    """
    if not polished.strip():
        return False
    if len(polished) > len(original) * _MAX_GROWTH_RATIO + 8:
        return False

    fillers = set(_FILLERS_BY_LANGUAGE.get(_base_language(language), ()))

    def words(text: str) -> list[str]:
        return [word for word in _skeleton(text).split(" ") if word and word not in fillers]

    original_words = words(original)
    polished_words = words(polished)
    if not original_words:
        return False

    return SequenceMatcher(None, original_words, polished_words).ratio() >= _MIN_SIMILARITY


def choose_transcript(original: str, polished: str, language: str = "") -> str:
    """The text to publish: the tidy version when it is faithful, otherwise the original.

    The single place the decision is made, so no caller can accidentally publish an unchecked
    model output.
    """
    return polished.strip() if is_faithful(original, polished, language) else original
