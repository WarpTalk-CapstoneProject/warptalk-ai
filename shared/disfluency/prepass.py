"""Deterministic disfluency prepass: the rule tier of clean transcript (WT-716).

It runs on every final STT segment, before anything else sees the text, and does only what a
lexicon can do WITHOUT understanding the sentence: delete tokens that are never content (um, ừm,
えーと), collapse restarts on words people restart on (the the, tôi tôi, その、その), and drop
broken word fragments (con- contract). Everything that needs meaning — "like", "I mean", "à
không", "じゃなくて" — is left in place and reported through `escalate` for the LLM tier.

Pipeline (per segment, O(n) apart from a bounded phrase-repeat window):
  P0 lookup keys (normalize.normalize_key) — the text itself is never rewritten
  P1 protect: numbers, reduplications (very very, từ từ, まだまだ), backchannels, self-repairs
  P2 delete tier-A fillers (+ the comma that belongs to them)
  P3 collapse repeats / stutters
  P4 remove broken word fragments
  P5 punctuation, spacing, sentence-initial capital (en/vi), terminal punctuation
     (punctuation.normalize_terminal_punctuation — its docstring is the terminal policy)
  P6 flags, then the invariants (invariants.check_invariants). A violation means a rule here is
     wrong for this input; the raw text is returned untouched with `escalate`.

Rule of the whole package: when unsure, keep. A missed "um" costs a reader nothing; a deleted
"not" or "ba ba" costs them the meaning.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from shared.disfluency import lexicon_en, lexicon_ja, lexicon_vi
from shared.disfluency.invariants import check_invariants
from shared.disfluency.normalize import normalize_width_ja, resolve_language, squeeze_key
from shared.disfluency.punctuation import normalize_terminal_punctuation
from shared.disfluency.tokenize import (
    COMMAS,
    TERMINALS,
    Token,
    ja_morphology_available,
    tokenize_spans,
)

FLAG_FILLER_ONLY = "filler_only"
FLAG_FILLERS_REMOVED = "fillers_removed"
FLAG_STUTTER_REMOVED = "stutter_removed"
FLAG_ESCALATE = "escalate"

_OPENING = frozenset('([{“‘「『"')
_CLOSING = frozenset(",.?!;:)]}…”’、。？！」』%")
_FRAGMENT_DASHES = ("-", "‐", "—", "–")
_ALPHA_RE = re.compile(r"^[^\W\d_]+$", re.UNICODE)
_HIRAGANA_RE = re.compile(r"^[ぁ-ゟ]+$")


@dataclass(frozen=True, slots=True)
class PrepassResult:
    """What the prepass decided for one segment.

    `clean_text` is "" (never None) for a filler-only turn. `removed_spans` are (start, end)
    offsets into the text that was passed in. `escalate_reasons` explains every `escalate`.
    """

    clean_text: str
    flags: frozenset[str] = frozenset()
    removed_spans: list[tuple[int, int]] = field(default_factory=list)
    escalate_reasons: list[str] = field(default_factory=list)


class _Run:
    """Mutable state for one prepass run over a token list."""

    def __init__(self, toks: list[Token], lang: str) -> None:
        self.toks = toks
        self.lang = lang
        self.reasons: list[str] = []
        self.fillers_removed = False
        self.stutter_removed = False

    # -- navigation ----------------------------------------------------------------------

    def lex(self) -> list[int]:
        return [i for i, t in enumerate(self.toks) if not t.punct]

    def kept_lex(self) -> list[int]:
        return [i for i, t in enumerate(self.toks) if not t.punct and not t.deleted]

    def is_comma(self, i: int) -> bool:
        t = self.toks[i]
        return t.punct and t.text in COMMAS

    def soft_next(self, i: int) -> int | None:
        """Next kept word after i, crossing only commas and deleted tokens; None at hard punct."""
        for j in range(i + 1, len(self.toks)):
            t = self.toks[j]
            if t.deleted or (t.punct and t.text in COMMAS):
                continue
            return None if t.punct else j
        return None

    def soft_prev(self, i: int) -> int | None:
        for j in range(i - 1, -1, -1):
            t = self.toks[j]
            if t.deleted or (t.punct and t.text in COMMAS):
                continue
            return None if t.punct else j
        return None

    def has_kept_before(self, i: int) -> bool:
        return any(not t.punct and not t.deleted for t in self.toks[:i])

    def has_kept_after(self, i: int) -> bool:
        return any(not t.punct and not t.deleted for t in self.toks[i + 1 :])

    # -- edits ---------------------------------------------------------------------------

    def delete(self, i: int, *, kind: str) -> None:
        self.toks[i].deleted = True
        if kind == "filler":
            self.fillers_removed = True
        else:
            self.stutter_removed = True
        nxt = i + 1
        if nxt < len(self.toks) and self.is_comma(nxt) and not self.toks[nxt].deleted:
            self.toks[nxt].deleted = True

    def protect(self, indices: Iterable[int]) -> None:
        for i in indices:
            self.toks[i].protected = True

    def escalate(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    # -- phrase matching -----------------------------------------------------------------

    def match_phrases(
        self, phrases: Iterable[tuple[str, ...]], *, kept_only: bool = False
    ) -> list[tuple[int, list[int], tuple[str, ...]]]:
        """Every (start, indices, phrase) where a phrase occurs on contiguous words."""
        words = self.kept_lex() if kept_only else self.lex()
        found: list[tuple[int, list[int], tuple[str, ...]]] = []
        ordered = sorted(phrases, key=len, reverse=True)
        for pos in range(len(words)):
            for phrase in ordered:
                n = len(phrase)
                span = words[pos : pos + n]
                if len(span) < n:
                    continue
                if any(span[k + 1] != span[k] + 1 for k in range(n - 1)):
                    continue  # punctuation (or a deleted token) inside the phrase
                if tuple(self.toks[k].key for k in span) == phrase:
                    found.append((pos, span, phrase))
                    break
        return found


# --- shared passes ----------------------------------------------------------------------------


def _collapse_runs(run: _Run, decide: Callable[[list[int]], None]) -> None:
    """Group kept words into runs of equal key (crossing commas/deleted) and let `decide` act."""
    words = run.kept_lex()
    i = 0
    while i < len(words):
        group = [words[i]]
        j = i
        while j + 1 < len(words):
            nxt = run.soft_next(group[-1])
            if nxt != words[j + 1] or run.toks[nxt].key != run.toks[words[i]].key:
                break
            group.append(nxt)
            j += 1
        if len(group) >= 2:
            decide(group)
        i = j + 1


def _render(run: _Run, *, capitalize: bool) -> str:
    toks = run.toks
    n = len(toks)

    def neighbour_deleted(c: int) -> bool:
        return (c > 0 and toks[c - 1].deleted) or (c + 1 < n and toks[c + 1].deleted)

    # Comma cleanup — only where a deletion made the comma orphaned; STT's own commas stay.
    for c in range(n):
        t = toks[c]
        if t.deleted or not run.is_comma(c) or not neighbour_deleted(c):
            continue
        nxt = next((k for k in range(c + 1, n) if not toks[k].deleted), None)
        if not run.has_kept_before(c):
            t.deleted = True
        elif nxt is None or (toks[nxt].punct and toks[nxt].text in COMMAS | TERMINALS):
            t.deleted = True

    kept = [i for i in range(n) if not toks[i].deleted]
    parts: list[str] = []
    prev: int | None = None
    for i in kept:
        t = toks[i]
        text = t.text
        gap_deleted = prev is not None and any(toks[k].deleted for k in range(prev + 1, i))
        if capitalize and not t.punct and _starts_sentence(toks, prev, gap_deleted):
            if text[:1].islower() and not any(c.isupper() for c in text):
                text = text[:1].upper() + text[1:]
        if prev is None:
            parts.append(text)
        else:
            p = toks[prev]
            if not gap_deleted:
                sep = " " if t.start > p.end else ""
            elif run.lang == "ja":
                sep = ""
            elif (t.punct and t.text[:1] in _CLOSING) or (p.punct and p.text[-1:] in _OPENING):
                sep = ""
            else:
                sep = " "
            parts.append(sep + text)
        prev = i
    return "".join(parts).strip()


def _starts_sentence(toks: list[Token], prev: int | None, gap_deleted: bool) -> bool:
    """The first word of the line, or a word whose sentence lost its first words to deletion.

    Words that STT itself left lower-case after a full stop are not ours to change.
    """
    if prev is None:
        return True
    return gap_deleted and toks[prev].punct and toks[prev].text in TERMINALS


def _removed_spans(toks: list[Token], origin: list[int] | None) -> list[tuple[int, int]]:
    """Deleted tokens as (start, end) offsets into the caller's text, adjacent runs merged."""
    spans: list[tuple[int, int]] = []
    for t in toks:
        if not t.deleted:
            continue
        start, end = (origin[t.start], origin[t.end]) if origin else (t.start, t.end)
        if spans and start - spans[-1][1] <= 1:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


# --- English ----------------------------------------------------------------------------------


def _en(run: _Run, prev_q: bool, standalone: bool | None) -> None:
    toks = run.toks
    lex = run.lex()

    # P1 protect.
    for pos, i in enumerate(lex):
        t = toks[i]
        if any(c.isdigit() for c in t.key) or t.key in lexicon_en.NUMBER_WORDS:
            t.protected = True
            if pos + 1 < len(lex) and lex[pos + 1] == i + 1:
                toks[i + 1].protected = True  # a unit after a number: "5 mm"
        if t.key in lexicon_en.C_KEEP:
            t.protected = True
        if len(t.text) >= 2 and t.text.isupper():
            t.protected = True  # an acronym ("ER"), not a hesitation

    def protect_idiom(group: list[int]) -> None:
        if toks[group[0]].key in lexicon_en.PROTECTED_REPEAT_WORDS:
            run.protect(group)

    _collapse_runs(run, protect_idiom)

    # Self-repair after a comma.
    for _, span, _ in run.match_phrases(lexicon_en.SELF_REPAIR_AFTER_COMMA):
        if span[0] > 0 and run.is_comma(span[0] - 1):
            run.escalate("self_repair")

    # P2 fillers.
    def filler_tier(t: Token) -> str:
        if t.protected or not _ALPHA_RE.match(t.key):
            return ""
        sq = squeeze_key(t.key)
        if sq in lexicon_en.A1_FILLERS:
            return "A1"
        if sq in lexicon_en.A2_FILLERS:
            return "A2"
        return ""

    if standalone is None:
        standalone = all(filler_tier(toks[i]) or toks[i].key in lexicon_en.C_KEEP for i in lex)
    for i in lex:
        tier = filler_tier(toks[i])
        if tier == "A1":
            run.delete(i, kind="filler")
        elif tier == "A2" and not standalone:
            if not run.has_kept_before(i) and prev_q:
                continue
            run.delete(i, kind="filler")

    # P3a phrase repeats: "we need to, we need to go" → drop the first copy.
    _phrase_repeats(run)

    # P3b word stutter.
    def stutter(group: list[int]) -> None:
        if any(toks[i].protected for i in group):
            return
        key = toks[group[0]].key
        if key.endswith(("-", "—")) or any(toks[i].text.endswith(_FRAGMENT_DASHES) for i in group):
            return
        if key in lexicon_en.STUTTER_FUNCTION_WORDS or len(group) >= lexicon_en.CONTENT_REPEAT_MIN:
            for i in group[:-1]:
                run.delete(i, kind="repeat")

    _collapse_runs(run, stutter)

    # P4 fragments: "con- contract", "we- we".
    for i in run.kept_lex():
        t = toks[i]
        if not t.text.endswith(_FRAGMENT_DASHES) or t.protected or not t.key:
            continue
        nxt = run.soft_next(i)
        if nxt is None:
            continue
        nkey = toks[nxt].key
        if nkey in lexicon_en.SUSPENDED_HYPHEN_NEXT:
            continue
        if nkey.startswith(t.key):
            run.delete(i, kind="repeat")

    # P6 (escalation part): discourse markers in a suspicious position.
    for _, span, phrase in run.match_phrases(lexicon_en.B_MARKERS, kept_only=True):
        first, last = span[0], span[-1]
        before_comma = first > 0 and run.is_comma(first - 1) and not toks[first - 1].deleted
        after = last + 1
        after_comma = after >= len(toks) or (
            toks[after].punct and toks[after].text in COMMAS | TERMINALS
        )
        near_deleted = any(
            toks[k].deleted and not toks[k].punct
            for k in (_prev_word(toks, first), _next_word(toks, last))
            if k is not None
        )
        if (before_comma and after_comma) or near_deleted:
            run.escalate("discourse_marker:" + " ".join(phrase))


def _prev_word(toks: list[Token], i: int) -> int | None:
    for k in range(i - 1, -1, -1):
        if not toks[k].punct:
            return k
    return None


def _next_word(toks: list[Token], i: int) -> int | None:
    for k in range(i + 1, len(toks)):
        if not toks[k].punct:
            return k
    return None


_PHRASE_MAX = 6


def _phrase_repeats(run: _Run) -> None:
    toks = run.toks
    for length in range(_PHRASE_MAX, 1, -1):
        words = run.kept_lex()
        p = 0
        while p + 2 * length <= len(words):
            a = words[p : p + length]
            b = words[p + length : p + 2 * length]
            if (
                all(a[k + 1] == a[k] + 1 for k in range(length - 1))
                and all(b[k + 1] == b[k] + 1 for k in range(length - 1))
                and [toks[i].key for i in a] == [toks[i].key for i in b]
                and not any(toks[i].protected for i in a + b)
                and run.soft_next(a[-1]) == b[0]
                and run.soft_next(b[-1]) is not None
            ):
                for i in a:
                    run.delete(i, kind="repeat")
                p += 2 * length
                continue
            p += 1


# --- Vietnamese -------------------------------------------------------------------------------


def _vi(run: _Run, prev_q: bool, standalone: bool | None) -> None:
    toks = run.toks
    lex = run.lex()

    # P1 protect.
    for i in lex:
        t = toks[i]
        if any(c.isdigit() for c in t.key) or t.key in lexicon_vi.C_KEEP:
            t.protected = True
    for _, span, _ in run.match_phrases(lexicon_vi.PROTECTED_REDUPLICATIONS):
        run.protect(span)
    for pos, i in enumerate(lex):
        t = toks[i]
        if t.key != "à":
            continue
        prev = lex[pos - 1] if pos > 0 else None
        if prev is not None and prev == i - 1 and toks[prev].key in lexicon_vi.KINSHIP_TERMS:
            t.protected = True  # "Chị à" — a vocative
    for _, span, _ in run.match_phrases(lexicon_vi.SELF_REPAIR_MARKERS):
        run.protect(span)
        run.escalate("self_repair")

    # P2 fillers.
    def tier(t: Token) -> str:
        if t.protected or not _ALPHA_RE.match(t.key):
            return ""
        sq = squeeze_key(t.key)
        if sq in lexicon_vi.A1_FILLERS:
            return "A1"
        if sq in lexicon_vi.A2_FILLERS:
            return "A2"
        return ""

    if standalone is None:
        standalone = all(tier(toks[i]) or toks[i].key in lexicon_vi.C_KEEP for i in lex)
    for _, span, _ in run.match_phrases(lexicon_vi.A2_BIGRAMS):
        if standalone or any(toks[i].protected for i in span):
            continue
        if not run.has_kept_before(span[0]) and (prev_q or not run.has_kept_after(span[-1])):
            continue
        for i in span:
            run.delete(i, kind="filler")
    for i in lex:
        if toks[i].deleted:
            continue
        kind = tier(toks[i])
        if kind == "A1":
            run.delete(i, kind="filler")
        elif kind == "A2" and not standalone:
            if not run.has_kept_before(i) and (prev_q or not run.has_kept_after(i)):
                continue
            run.delete(i, kind="filler")

    # P3 repeats — whitelist only; anything else identical is kept and escalated.
    def repeat(group: list[int]) -> None:
        if any(toks[i].protected for i in group):
            return
        key = toks[group[0]].key
        if key in lexicon_vi.STUTTER_WHITELIST:
            for i in group[:-1]:
                run.delete(i, kind="repeat")
            return
        if key == lexicon_vi.LA:
            before = run.soft_prev(group[0])
            if before is not None and toks[before].key in lexicon_vi.LA_LA_SPEECH_VERBS:
                for i in group[:-1]:
                    run.delete(i, kind="repeat")
                return
        if key in lexicon_vi.NUMBER_SYLLABLES:
            return  # reading digits out: "không không bảy"
        run.escalate("vi_unlisted_repeat")

    _collapse_runs(run, repeat)

    # Two discourse markers in a row ("thì cái", "là kiểu") — probably filler, maybe not.
    matches = run.match_phrases(lexicon_vi.B_MARKERS, kept_only=True)
    for (_, s1, _), (_, s2, _) in zip(matches, matches[1:], strict=False):
        if any(toks[i].protected for i in s1 + s2):
            continue  # "bay là là" is a word, not two markers
        if s2[0] == s1[-1] + 1:
            run.escalate("discourse_marker_sequence")
            break


# --- Japanese ---------------------------------------------------------------------------------

_JA_ASCII_PUNCT_RE = re.compile(r"(?<!\d)[,.](?!\d)|[?!]")
_JA_ASCII_MAP = {",": "、", ".": "。", "?": "？", "!": "！"}


def _ja_prepare(text: str) -> tuple[str, list[int]]:
    """Width-normalise and turn ASCII punctuation into Japanese — both one char to one char."""
    work, origin = normalize_width_ja(text)
    work = _JA_ASCII_PUNCT_RE.sub(lambda m: _JA_ASCII_MAP[m.group(0)], work)
    return work, origin


def _ja(run: _Run, prev_q: bool, standalone: bool | None) -> None:
    toks = run.toks
    lex = run.lex()
    morphology = ja_morphology_available()
    if not morphology:
        run.escalate("ja_fallback_tokenizer")

    # P1 protect.
    for i in lex:
        t = toks[i]
        if (
            t.pos2 == "数詞"
            or any(c in lexicon_ja.NUMERAL_CHARS for c in t.key)
            or t.key in lexicon_ja.PROTECTED_REDUPLICATIONS
            or t.key in lexicon_ja.C_KEEP
            or t.key in lexicon_ja.B_MARKERS
        ):
            t.protected = True
        if t.key in lexicon_ja.SELF_REPAIR_MARKERS:
            t.protected = True
            run.escalate("self_repair")

    # P2 fillers.
    def tier(t: Token) -> str:
        if t.protected:
            return ""
        if t.key in lexicon_ja.A1_FILLERS:
            return "A1"
        if t.key in lexicon_ja.A2_FILLERS:
            return "A2"
        return ""

    def delimited(i: int) -> bool:
        before_ok = i == 0 or toks[i - 1].punct
        after_ok = i + 1 >= len(toks) or toks[i + 1].punct
        return before_ok and after_ok

    if standalone is None:
        standalone = all(tier(toks[i]) or toks[i].key in lexicon_ja.C_KEEP for i in lex)
    for i in lex:
        kind = tier(toks[i])
        if not kind or (not morphology and not delimited(i)):
            continue
        if kind == "A2":
            if standalone or (not run.has_kept_before(i) and prev_q):
                continue
        run.delete(i, kind="filler")

    if not morphology:
        return  # the fallback segmentation is too coarse for anything but fillers

    # P3 morpheme/phrase repeated across 、: "その、その件" → "その件".
    for length in range(4, 0, -1):
        words = run.kept_lex()
        for p in range(len(words)):
            a = words[p : p + length]
            if len(a) < length or any(toks[i].deleted for i in a):
                continue
            if any(a[k + 1] != a[k] + 1 for k in range(length - 1)):
                continue
            comma = a[-1] + 1
            if comma >= len(toks) or not run.is_comma(comma) or toks[comma].deleted:
                continue
            b = list(range(comma + 1, comma + 1 + length))
            if b[-1] >= len(toks) or any(toks[i].punct or toks[i].deleted for i in b):
                continue
            if [toks[i].key for i in a] != [toks[i].key for i in b]:
                continue
            if any(toks[i].protected and toks[i].key not in lexicon_ja.B_MARKERS for i in a):
                continue
            for i in a:
                run.delete(i, kind="repeat")

    # P4 kana fragment: "わ、私が" → "私が".
    for i in run.kept_lex():
        t = toks[i]
        if t.protected or t.deleted or not _HIRAGANA_RE.match(t.key) or len(t.key) > 2:
            continue
        if i > 0 and not toks[i - 1].punct:
            continue
        comma = i + 1
        if comma + 1 >= len(toks) or not run.is_comma(comma):
            continue
        nxt = toks[comma + 1]
        if nxt.punct:
            continue
        candidates = [nxt.key, nxt.reading]
        if any(c.startswith(t.key) and len(c) > len(t.key) for c in candidates if c):
            run.delete(i, kind="repeat")


# --- entry point ------------------------------------------------------------------------------


def prepass(
    text: str,
    language: str,
    *,
    prev_turn_is_question: bool = False,
    standalone_turn: bool | None = None,
) -> PrepassResult:
    """Clean one STT segment deterministically. Pure; safe to call on every final segment.

    `language`: "en", "en-US", "vi", "ja", "auto"... — see normalize.resolve_language.
    `prev_turn_is_question`: the previous turn in the meeting asked something; a turn-initial
      "hmm"/"ờ"/"うーん" is then an answer, not a hesitation.
    `standalone_turn`: whether this segment is a whole turn on its own. None infers it: a turn
      made only of fillers and backchannels is standalone.
    """
    lang = resolve_language(language, text)
    if lang is None or not text.strip():
        return PrepassResult(text)

    if lang == "ja":
        work, origin = _ja_prepare(text)
    else:
        work, origin = text, None
    toks = tokenize_spans(work, lang)
    if not any(not t.punct for t in toks):
        return PrepassResult(text)

    run = _Run(toks, lang)
    if lang == "en":
        _en(run, prev_turn_is_question, standalone_turn)
    elif lang == "vi":
        _vi(run, prev_turn_is_question, standalone_turn)
    else:
        _ja(run, prev_turn_is_question, standalone_turn)

    flags: set[str] = set()
    if run.fillers_removed:
        flags.add(FLAG_FILLERS_REMOVED)
    if run.stutter_removed:
        flags.add(FLAG_STUTTER_REMOVED)

    if not run.kept_lex():
        flags.add(FLAG_FILLER_ONLY)
        if run.reasons:
            flags.add(FLAG_ESCALATE)
        return PrepassResult("", frozenset(flags), _removed_spans(toks, origin), run.reasons)

    clean = _render(run, capitalize=lang in ("en", "vi"))
    clean = normalize_terminal_punctuation(clean, lang)

    violations = check_invariants(text, clean, lang)
    if violations:
        return PrepassResult(text, frozenset({FLAG_ESCALATE}), [], [*run.reasons, *violations])
    if run.reasons:
        flags.add(FLAG_ESCALATE)
    return PrepassResult(clean, frozenset(flags), _removed_spans(toks, origin), run.reasons)
