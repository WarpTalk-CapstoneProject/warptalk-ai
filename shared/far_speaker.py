"""Who on the far side of a bridged call said a stand-in segment.

THE PROBLEM
    In an EXTERNAL_BRIDGE room every person on the Google Meet side is transcribed under ONE
    LiveKit identity — the stand-in seat (shared.control_markers.EXTERNAL_BRIDGE_SPEAKER_ID).
    The transcript can say "Google Meet participants" and nothing more, because the audio is a
    single mixed feed.

THE PHASE-1 ANSWER: HINTS, NOT A MODEL
    Meet already knows who is talking — its live captions name the speaker. The WarpTalk desktop
    that captures the Meet tab will publish what it reads there as a stream of hints:

        XADD meeting:{room}:far_speaker_hints *
             name "Lan Pham" t_ms 1759300000123 source meet_caption

    `t_ms` is the unix-epoch millisecond the desktop SAW the caption, which is always somewhat
    after the words were spoken. `attribute_far_speaker` shifts each hint back by a configurable
    lag and asks which name lands inside the segment's time window.

    Nothing here downloads or runs a diarization model. `FarSpeakerTracker` is the seam WT-677
    plugs one into: the STT worker only ever asks a tracker "who said this window", so swapping
    the caption-hint tracker for an embedding tracker (or chaining the two) changes no caller.

WIRE CONTRACT (read by warptalk-backend)
    The answer travels on `stt:results:{room}` as three OPTIONAL fields of STTResultMessage —
    `far_speaker_name`, `far_speaker_source`, `far_speaker_confidence` — absent when there is no
    answer, so a message without one is byte-for-byte what it was before.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

#: Redis stream the desktop writes caption hints to. `{room}` is the translation room id — the
#: same id every other `meeting:{room}:*` key in the ingress worker uses.
FAR_SPEAKER_HINTS_KEY = "meeting:{room}:far_speaker_hints"

#: The only source phase 1 produces. A diarization tracker (WT-677) will report its own.
SOURCE_MEET_CAPTION = "meet_caption"

#: Confidence ceiling for a hint that only lands NEAR the window rather than inside it. A name
#: inside the window is evidence about the window; a name next to it is a guess about a turn
#: boundary, and must never outrank the former.
_NEAREST_HINT_MAX_CONFIDENCE = 0.5


def far_speaker_hints_key(room_id: str) -> str:
    return FAR_SPEAKER_HINTS_KEY.format(room=room_id)


@dataclass(frozen=True)
class FarSpeakerHint:
    """One "this person was talking at t_ms" observation."""

    name: str
    t_ms: int
    source: str = SOURCE_MEET_CAPTION


@dataclass(frozen=True)
class SegmentWindow:
    """A segment's span in unix-epoch milliseconds, the clock hints are stamped in."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class FarSpeakerAttribution:
    name: str
    source: str
    confidence: float


class FarSpeakerTracker(Protocol):
    """Answers "who on the far side said this window". The seam WT-677 implements.

    Implementations must fail open (return None) rather than raise: an attribution is a label
    on a line that is going to be published either way.
    """

    async def attribute(
        self, meeting_id: str, window: SegmentWindow
    ) -> FarSpeakerAttribution | None: ...


def parse_hint(fields: Mapping[Any, Any]) -> FarSpeakerHint | None:
    """One stream entry -> a hint, or None when it is not a usable one."""
    data = {
        (k.decode() if isinstance(k, bytes) else str(k)): (
            v.decode() if isinstance(v, bytes) else str(v)
        )
        for k, v in fields.items()
    }
    name = " ".join(data.get("name", "").split())
    if not name:
        return None
    try:
        t_ms = int(float(data.get("t_ms", "")))
    except ValueError:
        return None
    if t_ms <= 0:
        return None
    source = data.get("source", "").strip() or SOURCE_MEET_CAPTION
    return FarSpeakerHint(name=name, t_ms=t_ms, source=source)


def attribute_far_speaker(
    segment_window: SegmentWindow,
    hints: Iterable[FarSpeakerHint],
    lag_ms: int,
    max_gap_ms: int = 1500,
) -> FarSpeakerAttribution | None:
    """Pick the far-side speaker for one segment from caption hints. Pure.

    Each hint is moved back by `lag_ms` (captions trail speech) to estimate when the words were
    spoken. Then:

    * Hints that land INSIDE the window vote; the name with the most votes wins (ties go to the
      latest one, which is the speaker the window ends on). Confidence is that name's share of
      the votes — 1.0 when every hint in the window agrees, lower when the window spans a
      hand-over.
    * Otherwise the hint NEAREST the window wins if it is within `max_gap_ms`, with a confidence
      that decays from 0.5 to 0 across that gap.
    * Otherwise None — no hint is a guess nobody should store.
    """
    start, end = segment_window.start_ms, segment_window.end_ms
    if end < start:
        start, end = end, start

    inside: list[FarSpeakerHint] = []
    nearest: tuple[int, FarSpeakerHint] | None = None
    for hint in hints:
        spoken_at = hint.t_ms - lag_ms
        if start <= spoken_at <= end:
            inside.append(hint)
            continue
        gap = start - spoken_at if spoken_at < start else spoken_at - end
        if gap <= max_gap_ms and (nearest is None or gap < nearest[0]):
            nearest = (gap, hint)

    if inside:
        votes: dict[str, list[FarSpeakerHint]] = {}
        for hint in inside:
            votes.setdefault(hint.name.casefold(), []).append(hint)
        _key, winners = max(
            votes.items(), key=lambda item: (len(item[1]), max(h.t_ms for h in item[1]))
        )
        latest = max(winners, key=lambda h: h.t_ms)
        return FarSpeakerAttribution(
            name=latest.name,
            source=latest.source,
            confidence=round(len(winners) / len(inside), 3),
        )

    if nearest is not None:
        gap, hint = nearest
        span = max(1, max_gap_ms)
        confidence = _NEAREST_HINT_MAX_CONFIDENCE * (1.0 - gap / span)
        if confidence <= 0:
            return None
        return FarSpeakerAttribution(
            name=hint.name, source=hint.source, confidence=round(confidence, 3)
        )
    return None


class CaptionHintTracker:
    """FarSpeakerTracker over `meeting:{room}:far_speaker_hints`.

    `read_hints(key, count)` returns newest-first stream entries — `redis.xrevrange` bound to a
    client, in production. Injected so the tracker has no Redis dependency of its own and tests
    can hand it a list.
    """

    def __init__(
        self,
        read_hints: Callable[[str, int], Awaitable[Sequence[tuple[Any, Mapping[Any, Any]]]]],
        *,
        lag_ms: int,
        max_gap_ms: int,
        scan_count: int = 64,
        cache_ttl_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read_hints = read_hints
        self._lag_ms = lag_ms
        self._max_gap_ms = max_gap_ms
        self._scan_count = scan_count
        self._cache_ttl_s = cache_ttl_s
        self._clock = clock
        self._cache: dict[str, tuple[list[FarSpeakerHint], float]] = {}

    async def _hints(self, meeting_id: str) -> list[FarSpeakerHint]:
        now = self._clock()
        cached = self._cache.get(meeting_id)
        if cached is not None and now - cached[1] < self._cache_ttl_s:
            return cached[0]
        hints: list[FarSpeakerHint] = []
        try:
            entries = await self._read_hints(far_speaker_hints_key(meeting_id), self._scan_count)
        except Exception:
            entries = []
        for _entry_id, fields in entries or []:
            if not fields:
                continue
            hint = parse_hint(fields)
            if hint is not None:
                hints.append(hint)
        self._cache[meeting_id] = (hints, now)
        return hints

    async def attribute(
        self, meeting_id: str, window: SegmentWindow
    ) -> FarSpeakerAttribution | None:
        hints = await self._hints(meeting_id)
        if not hints:
            return None
        return attribute_far_speaker(window, hints, self._lag_ms, self._max_gap_ms)

    def forget(self, meeting_id: str) -> None:
        self._cache.pop(meeting_id, None)
