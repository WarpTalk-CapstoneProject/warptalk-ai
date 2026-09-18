"""Lookup keys and language resolution for the disfluency prepass (WT-716, stage P0).

WHY KEYS AND NOT TEXT. Every decision the prepass makes — "is this a filler", "are these two
tokens the same word" — is a comparison, and speech-to-text spells the same sound many ways:
"ummm", "Umm", "ừmmm", "エーーと", "ｴｰﾄ". Normalising the TEXT would be an edit the prepass is not
allowed to make (deletion-only, invariant I1), so normalisation happens on a separate lookup key
and the surface text is never rewritten by this module.
"""

from __future__ import annotations

import re
import unicodedata

from shared.lang import base_language

SUPPORTED_LANGUAGES = frozenset({"en", "vi", "ja"})

# Three or more of the same letter is elongation ("ummm", "sooo"), never spelling: English and
# Vietnamese have no word with a tripled letter, so collapsing a 3+ run cannot merge two words.
_ELONGATION_RE = re.compile(r"(\w)\1{2,}", re.UNICODE)
# Any run at all — used only for filler lookup, where "umm" and "hmm" must meet "um" and "hm".
_SQUEEZE_RE = re.compile(r"(\w)\1+", re.UNICODE)

_KANA_RE = re.compile(r"[぀-ヿｦ-ﾟ]")
_KANJI_RE = re.compile(r"[一-鿿㐀-䶿]")
# Letters that exist in Vietnamese orthography and not in English. Tone marks on plain vowels
# (à, á...) also occur in French loanwords, so the set is the vowels and consonant that ONLY
# Vietnamese uses plus the dot-below / hook-above tones.
_VI_RE = re.compile(
    r"[ăâđêôơưĂÂĐÊÔƠƯạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
    r"ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ]"
)

# Old-style ("hòa") vs new-style ("hoà") tone placement in the oa/oe/uy diphthongs. Both are in
# live use and STT emits either; the key maps the old placement onto the new one so "hòa" and
# "hoà" compare equal. Safe as a blind substitution: with a final consonant ("hoàn") both styles
# already agree, so the old-style sequence never occurs where it would mean something else.
_VI_TONE_PAIRS = {
    "òa": "oà",
    "óa": "oá",
    "ỏa": "oả",
    "õa": "oã",
    "ọa": "oạ",
    "òe": "oè",
    "óe": "oé",
    "ỏe": "oẻ",
    "õe": "oẽ",
    "ọe": "oẹ",
    "ùy": "uỳ",
    "úy": "uý",
    "ủy": "uỷ",
    "ũy": "uỹ",
    "ụy": "uỵ",
}
_VI_TONE_RE = re.compile("|".join(_VI_TONE_PAIRS))

# Half-width katakana → full-width. Only kana and Japanese punctuation, deliberately NOT full
# NFKC: NFKC would also rewrite full-width digits and ① style characters, which are text.
_HALFWIDTH_KATA = "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
_FULLWIDTH_KATA = (
    "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチ"
    "ツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"
)
_HALFWIDTH_PUNCT = {"｡": "。", "､": "、", "｢": "「", "｣": "」", "･": "・"}
_DAKUTEN = "ﾞ"
_HANDAKUTEN = "ﾟ"
_LONG_MARKS_RE = re.compile("[ー～〜ｰ~]+")


def resolve_language(language: str, text: str) -> str | None:
    """The prepass language for `language`, or None when the prepass should not touch the text.

    "en-US" → "en". "auto", empty or an unsupported tag falls back to a script heuristic only
    for "auto"/empty; an explicit unsupported tag ("zh", "ko") returns None — running English
    rules over Chinese would be guessing, and the rule of this package is: when unsure, keep.
    """
    base = base_language(language or "")
    if base in SUPPORTED_LANGUAGES:
        return base
    if base not in ("", "auto", "und"):
        return None
    return detect_script_language(text)


def detect_script_language(text: str) -> str | None:
    """Kana/kanji → ja, Vietnamese-only letters → vi, Latin letters → en, otherwise None."""
    if _KANA_RE.search(text) or _KANJI_RE.search(text):
        return "ja"
    nfc = unicodedata.normalize("NFC", text)
    if _VI_RE.search(nfc):
        return "vi"
    if re.search(r"[A-Za-z]", nfc):
        return "en"
    return None


def katakana_to_hiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def normalize_width_ja(text: str) -> tuple[str, list[int]]:
    """Half-width kana/punctuation → full-width, returning the text and an offset map.

    `origin[i]` is the raw offset of normalised character i (plus one trailing entry for the
    end), so removed spans can be reported against the text the caller actually passed in —
    a voiced half-width pair ("ｶﾞ") becomes ONE character, so offsets do shift.
    """
    out: list[str] = []
    origin: list[int] = []
    for idx, ch in enumerate(text):
        if ch in (_DAKUTEN, _HANDAKUTEN) and out:
            composed = unicodedata.normalize("NFC", out[-1] + ("゙" if ch == _DAKUTEN else "゚"))
            if len(composed) == 1:
                out[-1] = composed
                continue
        pos = _HALFWIDTH_KATA.find(ch)
        if pos >= 0:
            out.append(_FULLWIDTH_KATA[pos])
        else:
            out.append(_HALFWIDTH_PUNCT.get(ch, ch))
        origin.append(idx)
    origin.append(len(text))
    return "".join(out), origin


def normalize_key(token: str, language: str) -> str:
    """The comparison key for one token. Never shown to anyone; never written back to text.

    - all: NFC, casefold, curly apostrophe → straight, a trailing fragment dash stripped.
    - en/vi: runs of 3+ identical letters collapse to one ("ummm" → "um", "ừmmm" → "ừm").
    - vi: old/new tone placement unified ("hòa" → "hoà").
    - ja: half-width → full-width, katakana → hiragana, ー/～/〜/ｰ unified and collapsed
      ("エーーと" → "えーと"); っ/ッ fall out of the katakana mapping.
    """
    base = base_language(language or "")
    key = unicodedata.normalize("NFC", token).casefold().replace("’", "'")
    key = key.rstrip("-‐—–")
    if base == "ja":
        key, _ = normalize_width_ja(key)
        key = katakana_to_hiragana(key)
        key = _LONG_MARKS_RE.sub("ー", key)
        return key
    key = _ELONGATION_RE.sub(r"\1", key)
    if base == "vi":
        key = _VI_TONE_RE.sub(lambda m: _VI_TONE_PAIRS[m.group(0)], key)
    return key


def squeeze_key(key: str) -> str:
    """Every repeated-letter run collapsed — filler lookup only ("hmm" and "hmmmm" → "hm")."""
    return _SQUEEZE_RE.sub(r"\1", key)
