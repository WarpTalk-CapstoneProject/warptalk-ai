"""The hard rules every clean transcript line must satisfy (WT-716).

These are checked on the prepass's own output (a violation there means a bug, and the raw text
is published instead) and — in a later task — on the LLM tier's output, where a violation means
the model rewrote something and its answer is discarded.

I1 deletion-only: after dropping punctuation and case, the clean tokens are an in-order
    subsequence of the raw tokens. Punctuation, sentence-initial capitals and 。/、 may change.
    For ja the morpheme sequence of the clean text is compared first; morpheme segmentation is
    context-sensitive, so when re-tokenizing the clean text splits differently, a character-level
    subsequence is accepted if every deleted run starts and ends on a raw morpheme boundary.
I2 counts preserved: negations, numbers, and a sentence-final question marker. Deleting a
    "not" or a digit changes what was said even when I1 holds.
"""

from __future__ import annotations

import re

from shared.disfluency import lexicon_en, lexicon_ja, lexicon_vi
from shared.disfluency.normalize import normalize_key, normalize_width_ja, resolve_language
from shared.disfluency.tokenize import Token, ja_morphology_available, tokenize_spans

I1_NOT_SUBSEQUENCE = "I1_not_subsequence"
I2_NEGATION_COUNT = "I2_negation_count"
I2_NUMBER_COUNT = "I2_number_count"
I2_QUESTION_MARKER = "I2_question_marker"

_JA_NEGATION_RE = re.compile(r"ない|ません|なかっ|ず(?=[、。，．,.\s!?！？]|$|に)")
_JA_NUMBER_RE = re.compile("[" + "".join(sorted(lexicon_ja.NUMERAL_CHARS)) + "]+")
_JA_QUESTION_ENDINGS = ("っけ", "かな", "よね", "か", "の")
_VI_QUESTION_FINALS = frozenset(
    normalize_key(w, "vi") for w in ("à", "hả", "hở", "chứ", "nhỉ", "không", "chưa")
)


def _lexical(text: str, lang: str) -> list[Token]:
    if lang == "ja":
        text, _ = normalize_width_ja(text)
    return [t for t in tokenize_spans(text, lang) if not t.punct]


def _is_subsequence(candidate: list[str], source: list[str]) -> bool:
    iterator = iter(source)
    return all(token in iterator for token in candidate)


def _ja_char_subsequence(raw_tokens: list[Token], clean_tokens: list[Token]) -> bool:
    raw = "".join(t.key for t in raw_tokens)
    clean = "".join(t.key for t in clean_tokens)
    boundaries = {0}
    pos = 0
    for t in raw_tokens:
        pos += len(t.key)
        boundaries.add(pos)
    i = 0
    gap_start: int | None = None
    for ch in clean:
        while i < len(raw) and raw[i] != ch:
            if gap_start is None:
                gap_start = i
            i += 1
        if i == len(raw):
            return False
        if gap_start is not None:
            if gap_start not in boundaries or i not in boundaries:
                return False
            gap_start = None
        i += 1
    return i == len(raw) or i in boundaries


def _negations(tokens: list[Token], text: str, lang: str) -> int:
    if lang == "en":
        return sum(1 for t in tokens if t.key in lexicon_en.NEGATIONS or t.key.endswith("n't"))
    if lang == "vi":
        return sum(1 for t in tokens if t.key in lexicon_vi.NEGATIONS)
    return len(_JA_NEGATION_RE.findall(text))


def _numbers(tokens: list[Token], text: str, lang: str) -> int:
    if lang == "ja":
        return len(_JA_NUMBER_RE.findall(text))
    words = lexicon_en.NUMBER_WORDS if lang == "en" else lexicon_vi.NUMBER_SYLLABLES
    return sum(1 for t in tokens if any(c.isdigit() for c in t.key) or t.key in words)


def _question_marker(tokens: list[Token], text: str, lang: str) -> tuple[bool, str]:
    """(ends with "?", sentence-final interrogative particle or "") — compared raw vs clean."""
    stripped = text.rstrip().rstrip("\"'”’」』)")
    has_mark = stripped.endswith(("?", "？"))
    body = stripped.rstrip("。.！!?？ ")
    if lang == "ja":
        for ending in _JA_QUESTION_ENDINGS:
            if body.endswith(ending):
                return has_mark, ending
        return has_mark, ""
    if lang == "vi" and tokens and tokens[-1].key in _VI_QUESTION_FINALS:
        return has_mark, tokens[-1].key
    return has_mark, ""


def check_invariants(raw: str, clean: str, language: str) -> list[str]:
    """The invariants `clean` violates relative to `raw`, as codes; empty means it is safe.

    An empty `clean` passes I1 trivially — it is how a filler-only turn is reported — but still
    has to pass I2, so a turn that was "No." can never be cleaned away.
    """
    lang = resolve_language(language, raw) or "en"
    raw_tokens = _lexical(raw, lang)
    clean_tokens = _lexical(clean, lang)
    violations: list[str] = []

    raw_keys = [t.key for t in raw_tokens]
    clean_keys = [t.key for t in clean_tokens]
    if not _is_subsequence(clean_keys, raw_keys):
        if not (lang == "ja" and _ja_char_subsequence(raw_tokens, clean_tokens)):
            violations.append(I1_NOT_SUBSEQUENCE)

    if _negations(raw_tokens, raw, lang) != _negations(clean_tokens, clean, lang):
        violations.append(I2_NEGATION_COUNT)
    if _numbers(raw_tokens, raw, lang) != _numbers(clean_tokens, clean, lang):
        violations.append(I2_NUMBER_COUNT)
    if clean.strip():
        raw_mark, raw_particle = _question_marker(raw_tokens, raw, lang)
        clean_mark, clean_particle = _question_marker(clean_tokens, clean, lang)
        # A question may gain a "?" (punctuation policy), never lose its marker or particle.
        if (raw_mark and not clean_mark) or (raw_particle and clean_particle != raw_particle):
            violations.append(I2_QUESTION_MARKER)
    return violations


def ja_tokenizer_is_fallback() -> bool:
    return not ja_morphology_available()


__all__ = [
    "I1_NOT_SUBSEQUENCE",
    "I2_NEGATION_COUNT",
    "I2_NUMBER_COUNT",
    "I2_QUESTION_MARKER",
    "check_invariants",
    "normalize_key",
]
