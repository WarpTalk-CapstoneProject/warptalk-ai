"""Terminal punctuation and question detection for clean transcript lines (WT-716).

POLICY — what this module may do to a line, and nothing else:

1. Never remove a "?"/"？" that STT produced. For ja, ASCII "?"/"!" become "？"/"！".
2. A sentence with NO terminal punctuation gets one: "?" ("？" for ja) when `detect_question`
   says so, otherwise "." (en/vi) or "。" (ja).
3. A sentence that ends with "." / "。" becomes a question only when `detect_question` fires. The
   rules below are the "strong" signals — an aux/modal inversion, a sentence-final particle —
   and deliberately miss many real questions: a missing "?" costs a reader far less than a
   statement dressed up as a question.
4. No terminal is added when the line is visibly unfinished: en/vi ending on "," ";" ":" or a
   dash; ja ending on a conjunctive particle (が, けど, ので ...), a bare interrogative-ish
   ending that is not strong enough (の, よね, かな, かどうか), or "、". Adding a period would
   not complete the sentence, but it would claim it was complete.

Adding or swapping terminal punctuation is punctuation-only, so it never violates invariant I1.

Question rules (conservative):
- en: first word (after leading "so/and/but/okay/well..." and a comma) is an aux/modal followed
  by a pronoun, determiner (not after do/have: "Do the dishes" is an imperative) or a
  capitalised name; or a wh-word followed by an aux ("where are", "how is"), a "wh's"
  contraction, "how many/much/long/about", or "what/which + noun + aux"; or a tag at the end:
  ", right", ", okay", ", correct", ", isn't it" (", <aux>n't <pronoun>"), "or not".
  "What I mean is..." and "I know what you mean" are statements.
- vi: sentence-final (after stripping trailing ạ/vậy/thế) à, hả, hở, nhỉ, chứ (but "à" right
  after a kinship term is vocative); final "không" not preceded by cũng/là/hay/hoặc and with no
  earlier negation; final "chưa" not after cũng/vẫn/còn; final sao/gì/đâu/nào with no earlier
  negation; a wh-phrase (bao nhiêu, bao giờ, thế nào, như thế nào, ai, gì, đâu, sao) in the last
  three syllables not followed by "cũng"; a sentence that starts with ai/sao/tại sao/vì sao/
  bao giờ/bao nhiêu/mấy giờ/có phải. Endings nhé/nha/nhá/đi/rồi are never questions.
- ja: final particle か (ですか/ますか/ませんか/でしょうか...) unless the sentence ends in
  かどうか or is an か…か enumeration; final っけ. Bare の/よね/かな stay as STT wrote them.
"""

from __future__ import annotations

import re
import unicodedata

from shared.disfluency import lexicon_ja, lexicon_vi
from shared.disfluency.normalize import normalize_key, resolve_language
from shared.disfluency.tokenize import ja_morphology_available, tokenize_spans

# --- English ------------------------------------------------------------------------------

_EN_AUX = frozenset(
    {
        "do",
        "does",
        "did",
        "is",
        "are",
        "was",
        "were",
        "am",
        "can",
        "could",
        "will",
        "would",
        "should",
        "shall",
        "may",
        "might",
        "must",
        "have",
        "has",
        "had",
    }
)
_EN_AUX_NEG = frozenset(
    {
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "can't",
        "couldn't",
        "won't",
        "wouldn't",
        "shouldn't",
        "haven't",
        "hasn't",
        "hadn't",
        "mustn't",
    }
)
_EN_PRONOUNS = frozenset(
    {
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "it",
        "there",
        "that",
        "this",
        "these",
        "those",
        "anyone",
        "anybody",
        "someone",
        "somebody",
        "everyone",
        "everybody",
        "anything",
        "something",
        "everything",
        "y'all",
    }
)
_EN_DETERMINERS = frozenset(
    {"the", "a", "an", "my", "your", "our", "their", "his", "her", "its", "any", "some", "all"}
)
_EN_WH = frozenset({"what", "why", "how", "when", "where", "who", "which", "whose", "whom"})
_EN_WH_CONTRACTIONS = frozenset(
    {"what's", "where's", "who's", "how's", "why's", "when's", "what're", "where're", "who're"}
)
_EN_HOW_NEXT = frozenset({"many", "much", "long", "often", "far", "come", "about", "soon"})
_EN_LEADING = frozenset(
    {
        "so",
        "and",
        "but",
        "okay",
        "ok",
        "well",
        "oh",
        "then",
        "also",
        "now",
        "alright",
        "hey",
        "or",
        "um",
        "uh",
    }
)
_EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*(?:-[A-Za-z0-9]+)*")
_EN_TAG_RE = re.compile(
    r",\s*(?:right|okay|ok|correct|yeah)\s*$"
    r"|,\s*(?:" + "|".join(re.escape(a) for a in sorted(_EN_AUX_NEG)) + r")\s+"
    r"(?:" + "|".join(sorted(_EN_PRONOUNS)) + r")\s*$"
    r"|\bor\s+not\s*$",
    re.IGNORECASE,
)


def _detect_question_en(sentence: str) -> bool:
    body = sentence.replace("’", "'").rstrip(" .!?…")
    if _EN_TAG_RE.search(body):
        return True
    words = _EN_WORD_RE.findall(body)
    lowered = [w.lower() for w in words]
    i = 0
    while i < len(lowered) - 1 and lowered[i] in _EN_LEADING:
        i += 1
    words, lowered = words[i:], lowered[i:]
    if len(lowered) < 2:
        return False
    w0, w1 = lowered[0], lowered[1]
    if w0 in _EN_AUX or w0 in _EN_AUX_NEG:
        if w1 in _EN_PRONOUNS:
            return True
        if w1 in _EN_DETERMINERS:
            return w0 not in ("do", "have")
        # A capitalised name: "Did John send it". "I" is a pronoun and handled above.
        return bool(words[1][:1].isupper())
    if w0 in _EN_WH_CONTRACTIONS:
        return True
    if w0 in _EN_WH:
        if w1 in _EN_AUX or w1 in _EN_AUX_NEG:
            return True
        if w0 == "how" and w1 in _EN_HOW_NEXT:
            return True
        if w0 == "what" and w1 == "about":
            return True
        if (
            w0 in ("what", "which", "whose")
            and len(lowered) >= 3
            and w1 not in _EN_PRONOUNS
            and lowered[2] in _EN_AUX
        ):
            return True
    return False


# --- Vietnamese ---------------------------------------------------------------------------


def _vk(word: str) -> str:
    return normalize_key(word, "vi")


_VI_FINAL_PARTICLES = frozenset(_vk(w) for w in ("à", "hả", "hở", "nhỉ", "chứ"))
_VI_NEVER_QUESTION_FINAL = frozenset(_vk(w) for w in ("nhé", "nha", "nhá", "đi", "rồi"))
_VI_TRAILING_SOFTENERS = frozenset(_vk(w) for w in ("ạ", "vậy", "thế", "nhể"))
_VI_FINAL_WH = frozenset(_vk(w) for w in ("sao", "gì", "đâu", "nào"))
_VI_NEGATIONS = frozenset(_vk(w) for w in ("không", "chẳng", "chả", "chưa", "đừng"))
_VI_KHONG = _vk("không")
_VI_CHUA = _vk("chưa")
_VI_CUNG = _vk("cũng")
_VI_KHONG_BLOCKERS = frozenset(_vk(w) for w in ("cũng", "là", "hay", "hoặc", "chẳng"))
_VI_CHUA_BLOCKERS = frozenset(_vk(w) for w in ("cũng", "vẫn", "còn", "là"))
_VI_WH_PHRASES = tuple(
    tuple(_vk(s) for s in p.split())
    for p in ("như thế nào", "bao nhiêu", "bao giờ", "thế nào", "ai", "gì", "đâu", "sao")
)
_VI_WH_START = tuple(
    tuple(_vk(s) for s in p.split())
    for p in ("tại sao", "vì sao", "bao giờ", "bao nhiêu", "mấy giờ", "có phải", "ai", "sao")
)
_VI_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _vi_followed_by_cung(words: list[str], end: int) -> bool:
    return _VI_CUNG in words[end : end + 2]


def _detect_question_vi(sentence: str) -> bool:
    words = [_vk(w) for w in _VI_WORD_RE.findall(unicodedata.normalize("NFC", sentence))]
    if not words:
        return False
    if words[-1] in _VI_NEVER_QUESTION_FINAL:
        # "Mấy giờ rồi" is the one common question ending on rồi.
        return words[:2] == [_vk("mấy"), _vk("giờ")]
    while len(words) > 1 and words[-1] in _VI_TRAILING_SOFTENERS:
        words = words[:-1]
    last = words[-1]
    prev = words[-2] if len(words) >= 2 else ""
    earlier_negation = any(w in _VI_NEGATIONS for w in words[:-1])

    if last in _VI_NEVER_QUESTION_FINAL:
        return False
    if last in _VI_FINAL_PARTICLES:
        # "Chị à" is a vocative, not "... à?".
        return not (last == _vk("à") and prev in lexicon_vi.KINSHIP_TERMS)
    if last == _VI_KHONG:
        return len(words) >= 2 and prev not in _VI_KHONG_BLOCKERS and not earlier_negation
    if last == _VI_CHUA:
        return len(words) >= 2 and prev not in _VI_CHUA_BLOCKERS
    if last in _VI_FINAL_WH:
        return not earlier_negation
    for phrase in _VI_WH_START:
        n = len(phrase)
        if tuple(words[:n]) == phrase and not _vi_followed_by_cung(words, n):
            return True
    if earlier_negation:
        return False
    tail_start = max(0, len(words) - 3)
    for phrase in _VI_WH_PHRASES:
        n = len(phrase)
        for i in range(tail_start, len(words) - n + 1):
            if tuple(words[i : i + n]) == phrase and not _vi_followed_by_cung(words, i + n):
                return True
    return False


# --- Japanese -----------------------------------------------------------------------------

_JA_FALLBACK_QUESTION_RE = re.compile(
    r"(?:ですか|ますか|ませんか|でしょうか|ましたか|でしたか|っけ)$"
)
_JA_INDEFINITE_BEFORE_KA = frozenset({"なん", "何", "誰", "だれ", "どこ", "いつ", "どれ", "どう"})
_JA_STRIP = " 　。．.？?！!…"


def _detect_question_ja(sentence: str) -> bool:
    body = sentence.rstrip(_JA_STRIP)
    if not body:
        return False
    if body.endswith("かどうか"):
        return False
    if not ja_morphology_available():
        return bool(_JA_FALLBACK_QUESTION_RE.search(body))
    morphemes = [t for t in tokenize_spans(body, "ja") if not t.punct]
    if not morphemes:
        return False
    last = morphemes[-1]
    if last.text == "っけ" or body.endswith("っけ"):
        return True
    if last.text != "か" or last.pos1 != "助詞":
        return False
    # "行くか行かないか" — an か…か enumeration names options; it does not ask.
    for i, tok in enumerate(morphemes[:-1]):
        if tok.text == "か" and tok.pos1 == "助詞":
            before = morphemes[i - 1].text if i > 0 else ""
            if before not in _JA_INDEFINITE_BEFORE_KA:
                return False
    return True


def _ja_is_unfinished(body: str) -> bool:
    """Whether a ja sentence without terminal punctuation should be left without one."""
    stripped = body.rstrip(" 　")
    if not stripped or stripped.endswith(("、", ",", "，")):
        return True
    if stripped.endswith(("かどうか", "よね", "かな", "かしら")):
        return True
    if not ja_morphology_available():
        return stripped.endswith(tuple(lexicon_ja.CONTINUATION_PARTICLES))
    morphemes = [t for t in tokenize_spans(stripped, "ja") if not t.punct]
    if not morphemes:
        return True
    last = morphemes[-1]
    if last.pos1 == "助詞" and last.text in lexicon_ja.CONTINUATION_PARTICLES:
        return True
    # A bare の / か that detect_question did not accept is too ambiguous to call a statement.
    return last.pos1 == "助詞" and last.text in ("の", "か")


# --- Public API ---------------------------------------------------------------------------


def detect_question(text: str, language: str) -> bool:
    """Whether `text` (one sentence; the last one if several) reads as a question.

    Deterministic and conservative — see the module docstring for the exact rules. A sentence
    that already ends in "?"/"？" is a question: STT heard the intonation, we did not.
    """
    lang = resolve_language(language, text)
    stripped = text.strip()
    if not stripped or lang is None:
        return False
    if stripped.rstrip(" \"'”’」』)").endswith(("?", "？")):
        return True
    sentences = _split_sentences(stripped, lang)
    last = sentences[-1] if sentences else stripped
    if lang == "en":
        return _detect_question_en(last)
    if lang == "vi":
        return _detect_question_vi(last)
    return _detect_question_ja(last)


_LATIN_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?…]+(?=\s|$)|$)", re.DOTALL)
_JA_SENTENCE_RE = re.compile(r"[^。！？]+[。！？]*|[。！？]+")


def _split_sentences(text: str, lang: str) -> list[str]:
    pattern = _JA_SENTENCE_RE if lang == "ja" else _LATIN_SENTENCE_RE
    return [m.group(0) for m in pattern.finditer(text) if m.group(0).strip()]


def _normalize_latin(text: str, lang: str) -> str:
    out: list[str] = []
    last_end = 0
    for m in _LATIN_SENTENCE_RE.finditer(text):
        sentence = m.group(0)
        out.append(text[last_end : m.start()])
        last_end = m.end()
        stripped = sentence.rstrip()
        trailing_ws = sentence[len(stripped) :]
        detect = _detect_question_en if lang == "en" else _detect_question_vi
        if stripped.endswith(("?", "!", "…", "...")):
            pass
        elif stripped.endswith("."):
            if detect(stripped):
                stripped = stripped[:-1] + "?"
        elif stripped[-1:] in (",", ";", ":", "-", "—", "–"):
            pass
        elif stripped:
            # Only the last sentence can get here: the pattern ends a sentence at terminal
            # punctuation or at the end of the text.
            stripped += "?" if detect(stripped) else "."
        out.append(stripped + trailing_ws)
    out.append(text[last_end:])
    return "".join(out)


def _normalize_ja(text: str) -> str:
    text = text.replace("?", "？").replace("!", "！")
    out: list[str] = []
    for m in _JA_SENTENCE_RE.finditer(text):
        chunk = m.group(0)
        body = chunk.rstrip("。！？")
        terminal = chunk[len(body) :]
        if not body.strip():
            out.append(chunk)
            continue
        if "？" in terminal or "！" in terminal:
            out.append(chunk)
        elif terminal.startswith("。"):
            out.append(body + ("？" if _detect_question_ja(body) else "。") + terminal[1:])
        else:
            trimmed = body.rstrip(" 　")
            if _detect_question_ja(trimmed):
                out.append(trimmed + "？")
            elif _ja_is_unfinished(trimmed):
                out.append(body)
            else:
                out.append(trimmed + "。")
    return "".join(out)


def normalize_terminal_punctuation(text: str, language: str) -> str:
    """Apply the terminal-punctuation policy (module docstring) to a whole line.

    Every sentence is checked for "." → "?" (strong signals only); only the LAST sentence can
    lack terminal punctuation, and that is where one is added. Whitespace is preserved.
    """
    lang = resolve_language(language, text)
    if lang is None or not text.strip():
        return text
    if lang == "ja":
        return _normalize_ja(text)
    return _normalize_latin(text, lang)
