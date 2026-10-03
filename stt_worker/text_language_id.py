"""Which of the ROOM's Latin-script languages a transcript line is written in.

WHY THIS EXISTS (bridge room 01a10069, 3 Oct)
    The Realtime transcription model returns no language, so a line's label is: unambiguous
    evidence (a non-Latin script, or a Vietnamese-UNIQUE letter), else the speaker's declared
    language, else a guess. Between two Latin-script languages that leaves the declaration as
    the last word far too often:

    * "Anh làm gì?" from a Vietnamese speaker declared `en` carries only à and ì, which French,
      Italian and Portuguese use too, so it is not evidence and the line was labelled English.
    * "AI opportunities?" / "Morning is great." from the Meet side of a bridge room whose
      stand-in was declared `vi` is plain ASCII, and ASCII is never evidence, so both were
      labelled Vietnamese and "translated" vi -> en. Nothing could ever re-pin that speaker
      TOWARD English either: `_learn_language_evidence` only learns from labels that differ
      from the declaration, and no rule ever produced `en`.

WHAT IT DOES, AND WHY IT CANNOT WANDER
    A statistical language identifier (lingua, n-gram models, deterministic, no network) is
    asked to choose ONLY among the room's own Latin-script languages. It never proposes a
    language the room did not declare, it is not consulted when the room declared fewer than two
    Latin-script languages (then there is nothing to choose between), and its answer is used
    only above a confidence floor and a minimum length, so "OK", "Hi" and half-English
    code-switching ("Anh deploy cái backend API nha", en 0.86) keep the declaration. Script
    evidence still comes first; this only replaces "the declaration wins by default" on text
    that proves nothing by itself.

FAIL-OPEN
    If the library is missing or a language has no model, the answer is None and the pipeline
    behaves exactly as before.
"""

from __future__ import annotations

import re
import threading
from functools import lru_cache
from typing import Any

import structlog

from shared.lang import base_language

logger = structlog.get_logger(__name__)

#: Below this relative confidence (among the room's candidates only) the line keeps its
#: declared language. Measured on vi/en lines: real sentences score >= 0.94, mixed code-switching
#: and two-letter interjections score 0.5-0.88.
MIN_CONFIDENCE = 0.9
#: Shortest line worth asking about, in letters and in words. One word ("Yes", "Hi") is said in
#: every language of this product and proves nothing.
MIN_LETTERS = 8
MIN_WORDS = 2
#: A label from this module re-pins a speaker's session (stt_worker.model
#: `_learn_language_evidence`) only from a line at least this long: re-pinning costs the whole
#: next chunk if it is wrong, a single mislabelled line costs one line.
MIN_WORDS_TO_LEARN = 3

_WORD_RE = re.compile(r"[^\W\d_]+")

_unavailable_logged = False


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


@lru_cache(maxsize=64)
def _detector(codes: frozenset[str]) -> tuple[Any, dict[Any, str]] | None:
    """A lingua detector restricted to `codes`, with its Language -> code map; None if unusable."""
    global _unavailable_logged
    try:
        from lingua import IsoCode639_1, Language, LanguageDetectorBuilder
    except ImportError:  # pragma: no cover - the dependency is declared; fail open regardless
        if not _unavailable_logged:
            _unavailable_logged = True
            logger.warning("text_language_id_unavailable", reason="lingua not installed")
        return None

    languages: dict[Any, str] = {}
    for code in codes:
        try:
            languages[Language.from_iso_code_639_1(IsoCode639_1.from_str(code))] = code
        except ValueError:
            # A room language lingua has no model for. Leaving it out would let the detector
            # pick a wrong candidate for that language's text, so do not decide at all.
            return None
    if len(languages) < 2:
        return None
    return LanguageDetectorBuilder.from_languages(*languages).build(), languages


_built: set[frozenset[str]] = set()
_building: set[frozenset[str]] = set()
_build_lock = threading.Lock()


def _build_off_thread(codes: frozenset[str]) -> None:
    try:
        _detector(codes)
    finally:
        with _build_lock:
            _building.discard(codes)
            _built.add(codes)


def _detector_if_ready(codes: frozenset[str]) -> tuple[Any, dict[Any, str]] | None:
    """The detector for `codes` if it is already built; otherwise start building it and say None.

    Building a detector for a new language set loads n-gram models (0.15-0.5 s). The STT label
    path runs on the event loop, so the first lines of a room with a new set keep their declared
    label while a thread loads the models, instead of stalling every meeting on the worker.
    """
    with _build_lock:
        if codes in _built:
            ready = True
        else:
            ready = False
            if codes not in _building:
                _building.add(codes)
                threading.Thread(
                    target=_build_off_thread, args=(codes,), name="text-lid-load", daemon=True
                ).start()
    return _detector(codes) if ready else None


def identify_room_language(
    text: str,
    candidates: set[str] | frozenset[str],
    *,
    min_confidence: float = MIN_CONFIDENCE,
    build_inline: bool = True,
) -> str | None:
    """The candidate language `text` is written in, or None when it cannot be told confidently.

    `candidates` must already be the room's LATIN-SCRIPT languages; fewer than two of them, a
    line shorter than MIN_LETTERS / MIN_WORDS, or a best score under `min_confidence` answer
    None. Pure apart from the cached detector. With `build_inline=False` (the event-loop caller)
    a language set whose detector is not loaded yet answers None and loads in the background.
    """
    codes = frozenset(base_language(code) for code in candidates if code)
    codes = frozenset(code for code in codes if code and code not in {"auto", "unknown", "und"})
    if len(codes) < 2:
        return None
    if _letters(text) < MIN_LETTERS or word_count(text) < MIN_WORDS:
        return None

    if build_inline:
        built = _detector(codes)
        with _build_lock:
            _built.add(codes)
    else:
        built = _detector_if_ready(codes)
    if built is None:
        return None
    detector, languages = built
    try:
        values = detector.compute_language_confidence_values(text)
    except Exception:  # pragma: no cover - defensive: a label is never worth a dropped line
        logger.warning("text_language_id_failed", exc_info=True)
        return None
    if not values:
        return None
    best = values[0]
    if best.value < min_confidence:
        return None
    return languages.get(best.language)
