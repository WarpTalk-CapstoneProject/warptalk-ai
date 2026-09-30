"""Vietnamese disfluency lexicon (WT-716). Data only — rules live in prepass.py.

Vietnamese is worked on SYLLABLES (space-separated). The hard problem is reduplication: "từ từ"
(slowly), "ba ba" (soft-shell turtle), "bay là là" (fly low) are words, and a generic "collapse
repeated syllables" rule would destroy them. So repeats collapse ONLY for a whitelist of
syllables people restart on; any other identical pair is kept and escalated.

Toneless look-alikes ("u", "o", "a") are deliberately absent: without the tone mark they are
letters, abbreviations or English, not hesitation.
"""

from __future__ import annotations

from shared.disfluency.normalize import normalize_key, squeeze_key


def _k(text: str) -> str:
    return normalize_key(text, "vi")


def _phrases(*phrases: str) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_k(s) for s in phrase.split()) for phrase in phrases)


A1_FILLERS = frozenset(squeeze_key(_k(w)) for w in ("ừm", "ờm", "hừm", "um", "uhm", "ưm", "ừhm"))
A2_FILLERS = frozenset(squeeze_key(_k(w)) for w in ("ờ", "ơ", "ừ", "hmm", "hm"))
A2_BIGRAMS: tuple[tuple[str, ...], ...] = _phrases("à ờ")

B_MARKERS = _phrases(
    "nói chung là",
    "thực ra là",
    "kiểu như",
    "tức là",
    "ý là",
    "thì",
    "là",
    "mà",
    "kiểu",
    "cái",
    "đấy",
    "thế",
    "rồi",
)

C_KEEP = frozenset({"dạ", "vâng"})
KINSHIP_TERMS = frozenset(
    _k(w)
    for w in (
        "chị",
        "anh",
        "em",
        "bác",
        "cô",
        "chú",
        "bố",
        "mẹ",
        "ông",
        "bà",
        "con",
        "cháu",
        "thầy",
        "dì",
        "cậu",
        "mợ",
        "thím",
        "sếp",
        "bạn",
    )
)

STUTTER_WHITELIST = frozenset(
    _k(w)
    for w in (
        "tôi",
        "mình",
        "chúng",
        "của",
        "và",
        "để",
        "cho",
        "với",
        "thì",
        "mà",
        "cái",
        "những",
        "các",
    )
)
# "là là" is a stutter only right after a verb of saying/thinking ("nghĩ là là").
LA = _k("là")
LA_LA_SPEECH_VERBS = frozenset(
    _k(w) for w in ("nghĩ", "nói", "thấy", "bảo", "biết", "hiểu", "tức", "nghĩa")
)

PROTECTED_REDUPLICATIONS = _phrases(
    "bay là là",
    "từ từ",
    "dần dần",
    "luôn luôn",
    "mãi mãi",
    "ngày ngày",
    "người người",
    "nhà nhà",
    "ai ai",
    "đâu đâu",
    "xanh xanh",
    "nhanh nhanh",
    "cao cao",
    "lâu lâu",
    "thường thường",
    "sơ sơ",
    "vừa vừa",
    "đi đi",
    "thôi thôi",
    "được được",
    "rồi rồi",
    "có có",
    "không không",
    "vâng vâng",
    "dạ dạ",
    "ừ ừ",
    "này này",
    "ba ba",
    "chuồn chuồn",
    "cào cào",
    "chôm chôm",
    "đa đa",
    "le le",
)

SELF_REPAIR_MARKERS = _phrases(
    "ý tôi là",
    "à không",
    "à nhầm",
    "à quên",
    "ý là",
    "không phải",
    "xin lỗi",
    "nói lại",
)

NEGATIONS = frozenset(_k(w) for w in ("không", "chưa", "chẳng", "đừng"))
NUMBER_SYLLABLES = frozenset(
    _k(w)
    for w in (
        "không",
        "một",
        "hai",
        "ba",
        "bốn",
        "tư",
        "năm",
        "lăm",
        "sáu",
        "bảy",
        "bẩy",
        "tám",
        "chín",
        "mười",
        "mươi",
        "linh",
        "lẻ",
        "trăm",
        "nghìn",
        "ngàn",
        "triệu",
        "tỷ",
        "tỉ",
        "mốt",
    )
)
