"""Drop a bridge stand-in line that is a named WarpTalk speaker heard back through Meet.

The ingress overlap gate (livekit_ingress_worker/far_side_gate.py) zeroes most of these copies
before they are transcribed, but it works on a guessed path delay and VAD verdicts. This is the
text-level check behind it, the same shape as the dub-echo guard (`_matches_recent_dub` in
model.py): a stand-in segment whose normalized text matches a FINAL line a named speaker in the
same room published moments ago is the same sentence twice, and the named copy is the one worth
keeping — it carries the person's identity and came from a clean near-field mic.

Pure functions here; the worker fetches the reference lines from `stt:results:{room}`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

_UNKNOWN_LANGUAGES = {"", "auto", "unknown", "und"}


@dataclass(frozen=True)
class NamedSegmentRef:
    """A line a named (non-stand-in) speaker in the room just published."""

    speaker_id: str
    #: Already normalized with stt_worker.model._normalize_overheard_text.
    text: str
    language: str
    #: The source chunk's ingress timestamp (STTResultMessage.timestamp_ms), epoch ms.
    timestamp_ms: int


@dataclass(frozen=True)
class DedupeConfig:
    window_ms: int = 15_000
    min_ratio: float = 0.8
    min_chars: int = 8
    same_language: bool = True


def _primary_language(language: str | None) -> str:
    return (language or "").strip().lower().replace("_", "-").split("-", 1)[0]


def languages_compatible(a: str | None, b: str | None) -> bool:
    """Equal primary subtags, or either side unknown (which cannot rule a match out)."""
    pa, pb = _primary_language(a), _primary_language(b)
    if pa in _UNKNOWN_LANGUAGES or pb in _UNKNOWN_LANGUAGES:
        return True
    return pa == pb


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def find_far_side_duplicate(
    normalized_text: str,
    language: str | None,
    timestamp_ms: int,
    refs: Iterable[NamedSegmentRef],
    config: DedupeConfig | None = None,
) -> NamedSegmentRef | None:
    """The named line this stand-in segment duplicates, or None. Pure.

    A match is any reference within `window_ms` (either direction — the two copies race through
    different STT sessions), language-compatible when `same_language`, and EITHER:

    * the stand-in text is the named line or a contiguous piece of it (the stand-in's VAD cut the
      same sentence differently), or
    * the two lines are at least `min_ratio` similar (Meet's codec and a second recognition pass
      rarely yield identical text).

    Stand-in lines shorter than `min_chars` never match: short back-channel ("ok", "yeah") is
    said by real far-side people all the time, and dropping real speech is the failure this
    must not have. A stand-in line that merely CONTAINS a named line plus more is not dropped
    either — the rest of it is somebody else.
    """
    cfg = config or DedupeConfig()
    text = normalized_text
    if len(text) < cfg.min_chars:
        return None
    best: tuple[float, NamedSegmentRef] | None = None
    for ref in refs:
        if not ref.text:
            continue
        if abs(timestamp_ms - ref.timestamp_ms) > cfg.window_ms:
            continue
        if cfg.same_language and not languages_compatible(language, ref.language):
            continue
        if text == ref.text or text in ref.text:
            return ref
        if min(len(text), len(ref.text)) < cfg.min_chars:
            continue
        ratio = text_similarity(text, ref.text)
        if ratio >= cfg.min_ratio and (best is None or ratio > best[0]):
            best = (ratio, ref)
    return best[1] if best is not None else None
