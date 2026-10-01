"""Decide which copy of a sentence heard on BOTH a named mic and the bridge stand-in to keep.

The ingress overlap gate (livekit_ingress_worker/far_side_gate.py) zeroes most of these copies
before they are transcribed, but it works on a guessed path delay and VAD verdicts. This is the
text-level check behind it, the same shape as the dub-echo guard (`_matches_recent_dub` in
model.py). A bridge room produces the same sentence twice in two different ways, and they need
OPPOSITE answers:

FORWARD - a WarpTalk user is also in the Meet. Their own mic carries the sentence first; Meet
    plays it back and the stand-in (Chrome process loopback) carries it again ~300-900 ms LATER.
    The named copy is the real one: drop the stand-in copy (`find_far_side_duplicate`).

LEAK - the host listens on laptop speakers. The Meet-side speech is played by Chrome, captured
    cleanly by the loopback (stand-in) and ALSO picked up acoustically by the host's real mic,
    which Electron's AEC cannot cancel (no cross-process reference). The stand-in copy comes
    FIRST or at about the same time; the named copy is the leak and carries the wrong identity.
    Keep the stand-in, drop the named copy (`find_far_side_leak`, separate flag).

WHICH WAY IT WENT IS READ FROM AUDIO TIME, NEVER FROM PUBLISH ORDER. Two STT sessions race, so
whichever line was published first says nothing about whose audio started first. Each line's
audio start is `anchor_ms + start_ms` (see `audio_start_epoch_ms`) - both tracks pass through the
same ingress process, so they are stamped on one clock. When either side's audio start is unknown
the pair is AMBIGUOUS and nothing is dropped: a duplicate line is a cosmetic defect, a deleted or
mis-attributed line is lost meeting content.

Pure functions here; the worker fetches the reference lines from `stt:results:{room}`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

_UNKNOWN_LANGUAGES = {"", "auto", "unknown", "und"}


@dataclass(frozen=True)
class NamedSegmentRef:
    """A line already published in the room - by a named speaker, or by the stand-in.

    (The name predates the reverse/leak check, which uses the same shape for stand-in lines.)
    """

    speaker_id: str
    #: Already normalized with stt_worker.model._normalize_overheard_text.
    text: str
    language: str
    #: The source chunk's ingress timestamp (STTResultMessage.timestamp_ms), epoch ms. Marks the
    #: END of the chunk's audio (the ingress publishes once the speech is over).
    timestamp_ms: int
    #: Epoch ms the line's audio started, or None when it cannot be known (see
    #: `audio_start_epoch_ms`). None makes the line unusable for a drop decision.
    audio_start_ms: int | None = None


@dataclass(frozen=True)
class DedupeConfig:
    """Forward dedupe: drop a stand-in line that is a named speaker heard back through Meet."""

    #: Coarse prefilter on chunk timestamps; the direction rule below is what actually decides.
    window_ms: int = 15_000
    min_ratio: float = 0.8
    min_chars: int = 8
    same_language: bool = True
    #: The stand-in's audio must start at least this much AFTER the named line's. Same floor as
    #: the ingress gate's FAR_SIDE_GATE_LAG_MIN_MS: Meet's path is never faster than this, so a
    #: smaller lag is not a Meet echo (it is the leak case, or the same instant).
    min_lag_ms: int = 150
    #: ...and at most this much after it. Further apart is not an echo of that line, or the two
    #: chunks were cut so differently that their starts no longer describe the sentence.
    max_lag_ms: int = 2_000


@dataclass(frozen=True)
class LeakConfig:
    """Reverse dedupe: drop a named line that is Meet audio leaking into that person's mic.

    Stricter than the forward thresholds on purpose: what this drops is a WarpTalk user's line.
    """

    min_ratio: float = 0.85
    min_chars: int = 12
    same_language: bool = True
    #: How much LATER than the stand-in's audio the named copy may start (speaker->mic is ~ms,
    #: plus the mic VAD tripping later on a quieter copy). The other bound is the forward
    #: `min_lag_ms`: a named line up to that much EARLIER still counts as "about simultaneous".
    max_named_delay_ms: int = 1_000


def audio_start_epoch_ms(anchor_ms: int, start_ms: int) -> int | None:
    """Epoch ms a segment's audio started, from STTResultMessage.anchor_ms + start_ms.

    `start_ms` is the chunk's audio start (ingress publish time minus PCM length) plus the
    model's in-chunk offset, counted from the room's `anchor_ms`. None when it cannot be trusted:
    `anchor_ms` 0 means no origin was stated, and `start_ms` 0 is what `_elapsed_ms` clamps the
    room's first chunk(s) to - where anchor+0 is the chunk's END, not its start.
    """
    if anchor_ms <= 0 or start_ms <= 0:
        return None
    return anchor_ms + start_ms


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


def _text_match(
    text: str,
    language: str | None,
    ref: NamedSegmentRef,
    *,
    min_ratio: float,
    min_chars: int,
    same_language: bool,
) -> float | None:
    """How strongly the CANDIDATE `text` is a copy of `ref` (1.0 = equal or contained), or None.

    Containment is one-way: the candidate may be the ref or a piece of it (its VAD cut the same
    sentence differently), but a candidate that CONTAINS the ref plus more is not a copy - the
    rest of it is somebody else's words.
    """
    if not ref.text or len(text) < min_chars:
        return None
    if same_language and not languages_compatible(language, ref.language):
        return None
    if text == ref.text or text in ref.text:
        return 1.0
    if min(len(text), len(ref.text)) < min_chars:
        return None
    ratio = text_similarity(text, ref.text)
    return ratio if ratio >= min_ratio else None


def find_far_side_duplicate(
    normalized_text: str,
    language: str | None,
    timestamp_ms: int,
    refs: Iterable[NamedSegmentRef],
    config: DedupeConfig | None = None,
    *,
    audio_start_ms: int | None = None,
) -> NamedSegmentRef | None:
    """The NAMED line this stand-in segment echoes (forward case), or None. Pure.

    A match needs, against one named ref:

    * both audio starts known, and the stand-in's starting `min_lag_ms`..`max_lag_ms` LATER;
    * chunk timestamps within `window_ms`, languages compatible when `same_language`;
    * the stand-in text equal to / a contiguous piece of the named line, or `min_ratio` similar.

    Stand-in lines shorter than `min_chars` never match: short back-channel ("ok", "yeah") is
    said by real far-side people all the time. Missing timing on either side never matches.
    """
    cfg = config or DedupeConfig()
    if audio_start_ms is None or len(normalized_text) < cfg.min_chars:
        return None
    best: tuple[float, NamedSegmentRef] | None = None
    for ref in refs:
        if ref.audio_start_ms is None:
            continue
        if abs(timestamp_ms - ref.timestamp_ms) > cfg.window_ms:
            continue
        lag = audio_start_ms - ref.audio_start_ms
        if not cfg.min_lag_ms <= lag <= cfg.max_lag_ms:
            continue
        score = _text_match(
            normalized_text,
            language,
            ref,
            min_ratio=cfg.min_ratio,
            min_chars=cfg.min_chars,
            same_language=cfg.same_language,
        )
        if score is not None and (best is None or score > best[0]):
            best = (score, ref)
    return best[1] if best is not None else None


def find_far_side_leak(
    normalized_text: str,
    language: str | None,
    timestamp_ms: int,
    standin_refs: Iterable[NamedSegmentRef],
    dedupe: DedupeConfig | None = None,
    leak: LeakConfig | None = None,
    *,
    audio_start_ms: int | None = None,
) -> NamedSegmentRef | None:
    """The STAND-IN line this named segment is a leaked copy of (reverse case), or None. Pure.

    `standin_refs` are stand-in lines ALREADY published in the room; an empty list means "not a
    bridge room, or nothing from Meet yet", and nothing is dropped. A match needs:

    * both audio starts known, the named line starting no more than `dedupe.min_lag_ms` before
      the stand-in's and no more than `leak.max_named_delay_ms` after it - exactly the band the
      forward rule leaves alone, so one pair can never satisfy both rules;
    * the stricter `leak` text thresholds, with the NAMED text as the candidate: equal to or a
      piece of the stand-in line (the leak is quieter, so its VAD often catches less), or
      `leak.min_ratio` similar. A named line that contains the stand-in line plus more is kept -
      the rest is the user's own speech.
    """
    dcfg = dedupe or DedupeConfig()
    lcfg = leak or LeakConfig()
    if audio_start_ms is None or len(normalized_text) < lcfg.min_chars:
        return None
    best: tuple[float, NamedSegmentRef] | None = None
    for ref in standin_refs:
        if ref.audio_start_ms is None:
            continue
        if abs(timestamp_ms - ref.timestamp_ms) > dcfg.window_ms:
            continue
        # Same sign convention as the forward rule: stand-in start minus named start.
        lag = ref.audio_start_ms - audio_start_ms
        if not -lcfg.max_named_delay_ms <= lag < dcfg.min_lag_ms:
            continue
        score = _text_match(
            normalized_text,
            language,
            ref,
            min_ratio=lcfg.min_ratio,
            min_chars=lcfg.min_chars,
            same_language=lcfg.same_language,
        )
        if score is not None and (best is None or score > best[0]):
            best = (score, ref)
    return best[1] if best is not None else None
