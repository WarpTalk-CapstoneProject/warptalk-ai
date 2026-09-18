"""Deterministic transcript disfluency cleanup — the rule tier of clean transcript (WT-716).

Public API:
    prepass(text, language, *, prev_turn_is_question=False, standalone_turn=None)
        -> PrepassResult(clean_text, flags, removed_spans, escalate_reasons)
    tokenize(text, language) -> list[str]
    normalize_key(token, language) -> str
    check_invariants(raw, clean, language) -> list[str]
    detect_question(text, language) -> bool
    normalize_terminal_punctuation(text, language) -> str

Flags: "filler_only", "fillers_removed", "stutter_removed", "escalate".

Terminal punctuation policy (details in punctuation.py): "." (en/vi) / "。" (ja) is added only
when a sentence has none and the text is non-empty and not visibly unfinished; "?" / "？" is
added — or replaces a "." — only on the strong question signals `detect_question` knows; an
STT "?" is never removed; ja gets full-width "？"/"！".

Imports are cheap: the Japanese tagger is created on first Japanese use only.
"""

from shared.disfluency.invariants import check_invariants
from shared.disfluency.normalize import normalize_key
from shared.disfluency.prepass import (
    FLAG_ESCALATE,
    FLAG_FILLER_ONLY,
    FLAG_FILLERS_REMOVED,
    FLAG_STUTTER_REMOVED,
    PrepassResult,
    prepass,
)
from shared.disfluency.punctuation import detect_question, normalize_terminal_punctuation
from shared.disfluency.tokenize import tokenize

__all__ = [
    "FLAG_ESCALATE",
    "FLAG_FILLERS_REMOVED",
    "FLAG_FILLER_ONLY",
    "FLAG_STUTTER_REMOVED",
    "PrepassResult",
    "check_invariants",
    "detect_question",
    "normalize_key",
    "normalize_terminal_punctuation",
    "prepass",
    "tokenize",
]
