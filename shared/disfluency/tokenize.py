"""Tokenization for the disfluency prepass and the transcript guardian (WT-716).

en/vi: a regex over words, numbers and punctuation. A hyphenated word is ONE token, so fillers
are never found inside "uh-oh" or "mm-hmm", and a trailing dash stays on its word so a restart
fragment ("con-", "con—") is visible as such.

ja: morphemes from fugashi + unidic-lite, because Japanese has no spaces and "ええ"/"ええと" can
only be told apart on morpheme boundaries. UniDic splits several fillers ("えー|と", "その|ー"),
so consecutive morphemes whose concatenated key is a lexicon entry are glued back into one token.

WHY THE FALLBACK EXISTS. fugashi is a C extension with a ~50 MB dictionary; a worker image built
without it must still start. The fallback splits on character class and is flagged as low
confidence so the prepass only does the safest thing with it (see prepass.py). The Tagger is
created lazily, once per process, so importing this package costs nothing on en/vi paths.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

import structlog

from shared.disfluency import lexicon_ja
from shared.disfluency.normalize import katakana_to_hiragana, normalize_key

logger = structlog.get_logger(__name__)

COMMAS = frozenset({",", "、", "，"})
TERMINALS = frozenset({".", "?", "!", "。", "？", "！", "…"})


@dataclass(slots=True)
class Token:
    """One token of the working text. Offsets are into the string that was tokenized."""

    text: str
    start: int
    end: int
    punct: bool
    key: str
    pos1: str = ""
    pos2: str = ""
    reading: str = ""  # ja: hiragana reading when the dictionary knows one
    deleted: bool = False
    protected: bool = False


_LATIN_TOKEN_RE = re.compile(
    r"(?P<num>\d+(?:[.,:]\d+)+%?)"
    r"|(?P<word>[^\W_]+(?:['’][^\W_]+)*(?:[-‐][^\W_]+)*[-‐—–]?)"
    r"|(?P<punct>[^\w\s])",
    re.UNICODE,
)


def _latin_tokens(text: str, language: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _LATIN_TOKEN_RE.finditer(text):
        surface = m.group(0)
        is_punct = m.lastgroup == "punct"
        key = surface if is_punct else normalize_key(surface, language)
        tokens.append(Token(surface, m.start(), m.end(), is_punct, key))
    return tokens


# --- Japanese -------------------------------------------------------------------------------

_TAGGER: Any = None
_TAGGER_FAILED = False
_TAGGER_LOCK = threading.Lock()


def _tagger() -> Any:
    """The process-wide fugashi Tagger, or None when fugashi/unidic-lite is not installed."""
    global _TAGGER, _TAGGER_FAILED
    if _TAGGER is not None or _TAGGER_FAILED:
        return _TAGGER
    with _TAGGER_LOCK:
        if _TAGGER is None and not _TAGGER_FAILED:
            try:
                import fugashi

                _TAGGER = fugashi.Tagger()
            except Exception as exc:  # ImportError, or a dictionary that failed to load
                _TAGGER_FAILED = True
                logger.warning("disfluency_ja_tokenizer_fallback", error=str(exc))
    return _TAGGER


def ja_morphology_available() -> bool:
    return _tagger() is not None


_JA_PUNCT_RE = re.compile(r"[^\w぀-ヿ一-鿿㐀-䶿ー～〜]", re.UNICODE)


def _is_punct_surface(surface: str) -> bool:
    return bool(surface) and all(_JA_PUNCT_RE.fullmatch(c) for c in surface)


def _ja_raw_morphemes(text: str) -> list[Token]:
    tagger = _tagger()
    tokens: list[Token] = []
    if tagger is None:
        return _ja_fallback_tokens(text)
    cursor = 0
    for word in tagger(text):
        surface = word.surface
        start = text.find(surface, cursor)
        if start < 0:  # the tagger never rewrites text, but never trust an offset search blindly
            start = cursor
        end = start + len(surface)
        cursor = end
        feature = word.feature
        pos1 = getattr(feature, "pos1", "") or ""
        kana = getattr(feature, "kana", None) or ""
        is_punct = pos1 in ("補助記号", "空白") and _is_punct_surface(surface)
        tokens.append(
            Token(
                surface,
                start,
                end,
                is_punct,
                surface if is_punct else normalize_key(surface, "ja"),
                pos1=pos1,
                pos2=getattr(feature, "pos2", "") or "",
                reading=katakana_to_hiragana(kana),
            )
        )
    return tokens


_CHAR_CLASS_RE = re.compile(
    r"(?P<kana>[぀-ゟー～〜]+)"
    r"|(?P<kata>[゠-ヿー]+)"
    r"|(?P<kanji>[一-鿿㐀-䶿々]+)"
    r"|(?P<word>[^\W_]+)"
    r"|(?P<punct>[^\w\s])",
    re.UNICODE,
)


def _ja_fallback_tokens(text: str) -> list[Token]:
    """Character-class runs. Coarse on purpose: good enough to see a filler between commas."""
    tokens: list[Token] = []
    for m in _CHAR_CLASS_RE.finditer(text):
        surface = m.group(0)
        is_punct = m.lastgroup == "punct"
        key = surface if is_punct else normalize_key(surface, "ja")
        tokens.append(Token(surface, m.start(), m.end(), is_punct, key))
    return tokens


def _merge_lexicon_morphemes(tokens: list[Token]) -> list[Token]:
    """Glue split lexicon entries back together, longest match first."""
    merged: list[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if tok.punct:
            merged.append(tok)
            i += 1
            continue
        best = 0
        limit = min(n, i + lexicon_ja.MERGE_MAX_MORPHEMES)
        for j in range(limit, i + 1, -1):
            span = tokens[i:j]
            if any(t.punct for t in span):
                continue
            if any(span[k].end != span[k + 1].start for k in range(len(span) - 1)):
                continue
            if normalize_key("".join(t.text for t in span), "ja") in lexicon_ja.MERGE_KEYS:
                best = j
                break
        if best:
            span = tokens[i:best]
            surface = "".join(t.text for t in span)
            merged.append(
                Token(
                    surface,
                    span[0].start,
                    span[-1].end,
                    False,
                    normalize_key(surface, "ja"),
                    pos1=span[0].pos1,
                    pos2=span[0].pos2,
                    reading="".join(t.reading for t in span),
                )
            )
            i = best
        else:
            merged.append(tok)
            i += 1
    return merged


def tokenize_spans(text: str, language: str) -> list[Token]:
    """Tokens (words AND punctuation) with offsets into `text`.

    For ja, pass text that has already been width-normalised (normalize.normalize_width_ja) if
    the offsets are to be meaningful against it; keys are width-insensitive either way.
    """
    if language == "ja":
        return _merge_lexicon_morphemes(_ja_raw_morphemes(text))
    return _latin_tokens(text, language)


def tokenize(text: str, language: str) -> list[str]:
    """The lexical tokens of `text` (punctuation dropped), as surfaces, in order.

    `language` must already be a base tag ("en", "vi", "ja"); anything else tokenizes as Latin.
    This is the tokenization the invariants are defined over — callers comparing two texts
    should compare `normalize_key` of these, not the surfaces.
    """
    return [t.text for t in tokenize_spans(text, language) if not t.punct]
