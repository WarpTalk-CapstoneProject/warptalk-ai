"""A far-side speaker's name that arrives after their line was already published.

THE HAND-OVER LINE
    Lan stops talking on the Google Meet side and Minh starts. Minh's first line is finalized
    and published before any caption naming Minh has reached the desktop - Meet's captions trail
    speech by 0.5-1.5 s, and the desktop's hints trail the captions. At that moment the only hints
    near the line are Lan's, all BEFORE it, and attribute_far_speaker scores that as the hand-over
    guess (<= 0.5). The gateway shows a name only from 0.6 (Bridge:FarSpeakerNameMinConfidence),
    so the line says "Google Meet participants". Lowering the threshold is not the answer: the
    name it would show is LAN's, on Minh's words.

THE PO'S ANSWER (2026-10-03): PUBLISH NOW, NAME LATER
    The line goes out exactly as before - fallback label, translation and dub not held back by a
    single millisecond. This module then asks the same tracker again, about the same window, a
    little later (by default +1 s and +2.5 s after the line was finalized), when Minh's own hints
    have had time to land. The first CONFIDENT answer is published once, keyed by the line's
    segment_id, and the backend renames that one line - live in the popup and in the saved
    transcript. No confident answer by the last attempt: give up, log it, the line keeps the
    fallback label (and the post-meeting relabel from Google's transcript may still name it).

WHAT A LATE ANSWER MAY DO, AND WHAT IT MAY NOT
    * Only fallback -> name. A line is scheduled only when what it was published with would NOT
      have been shown (no attribution, or one below the display threshold), so a late answer can
      never replace a name somebody already read.
    * Only the two rules that are evidence, not a guess: hints INSIDE the window (BASIS_INSIDE),
      or every near hint naming one person with one lying after the window (BASIS_SOLE_NEAR). The
      hand-over guess (BASIS_NEAREST) is refused by name, whatever confidence it carries - a
      threshold configured below its 0.5 cap must not let LAN's name onto Minh's line late.
    * At most one late message per segment.

WIRE CONTRACT (read by warptalk-backend: the gateway and TranscriptService, nobody else)
    XADD stt:far_speaker_late:{room}  (and the global stt:far_speaker_late, like every publish)
        type                    "far_speaker_late"
        meeting_id              the translation room id, as on stt:results
        segment_id              the SAME id the line carried on stt:results
        far_speaker_name        whitespace-normalised name
        far_speaker_source      e.g. "meet_caption"
        far_speaker_confidence  str(float), always >= the display threshold
        t_ms                    unix-epoch ms the late answer was produced

    Its own stream, NOT stt:results: translation, tts, billing and the assistant all read
    stt:results, and an entry there is a sentence to them (shared/control_markers.py is what
    happened the last time something that was not speech rode that stream). A backend that
    predates this never reads the new stream, and its MAXLEN/TTL trims it like any other.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from shared.far_speaker import (
    BASIS_INSIDE,
    BASIS_SOLE_NEAR,
    FarSpeakerAttribution,
    FarSpeakerTracker,
    SegmentWindow,
)

#: The `<prefix>` handed to BaseWorker.publish, which writes `<prefix>:<room>` and `<prefix>`.
FAR_SPEAKER_LATE_STREAM = "stt:far_speaker_late"

#: The `type` field of every entry on that stream.
FAR_SPEAKER_LATE_TYPE = "far_speaker_late"

#: The rules a late answer may come from. See the module docstring.
LATE_ANSWER_BASES = frozenset({BASIS_INSIDE, BASIS_SOLE_NEAR})

#: Bounds on the configured schedule. More attempts than this is polling Redis for a line nobody
#: is looking at any more; later than this and the reader has scrolled past it - the
#: post-meeting relabel is the tool for anything that late.
MAX_LATE_ATTEMPTS = 4
MAX_LATE_DELAY_MS = 10_000


def name_hash(name: str) -> str:
    """What the logs carry instead of a Meet participant's name - they agreed to nothing."""
    return hashlib.sha256(name.encode()).hexdigest()[:12]


def was_shown(attribution: FarSpeakerAttribution | None, min_confidence: float) -> bool:
    """Whether the line was published with a name the gateway displays.

    Mirrors WarpTalk.Shared.FarSpeakerNames.ResolveLive: a non-blank name at or above the
    threshold. Such a line is never renamed late.
    """
    return (
        attribution is not None
        and bool(attribution.name.strip())
        and attribution.confidence >= min_confidence
    )


def needs_late_attribution(
    attribution: FarSpeakerAttribution | None, min_confidence: float
) -> bool:
    """Whether a just-published stand-in line should be asked about again later."""
    return not was_shown(attribution, min_confidence)


def is_late_answer(attribution: FarSpeakerAttribution | None, min_confidence: float) -> bool:
    """Whether a re-attribution is good enough to rename a line that went out unnamed."""
    return (
        attribution is not None
        and attribution.basis in LATE_ANSWER_BASES
        and was_shown(attribution, min_confidence)
    )


def late_attempt_delays_ms(configured: Iterable[int]) -> tuple[int, ...]:
    """The configured schedule made safe: positive, at most MAX_LATE_DELAY_MS, ascending, unique,
    at most MAX_LATE_ATTEMPTS. Empty means the feature is off."""
    delays = sorted({int(d) for d in configured if 0 < int(d) <= MAX_LATE_DELAY_MS})
    return tuple(delays[:MAX_LATE_ATTEMPTS])


@dataclass(frozen=True)
class LateFarSpeakerName:
    """One late rename, as it goes on the wire."""

    meeting_id: str
    segment_id: str
    name: str
    source: str
    confidence: float
    t_ms: int

    def to_redis(self) -> dict[str, str]:
        return {
            "type": FAR_SPEAKER_LATE_TYPE,
            "meeting_id": self.meeting_id,
            "segment_id": self.segment_id,
            "far_speaker_name": " ".join(self.name.split()),
            "far_speaker_source": self.source,
            "far_speaker_confidence": str(self.confidence),
            "t_ms": str(self.t_ms),
        }


class LateFarSpeakerNamer:
    """Schedules and runs the bounded re-attribution of lines published without a shown name.

    One asyncio task per scheduled line, which sleeps to each attempt, asks the tracker, and
    stops at the first acceptable answer or after the last attempt. Never awaited by the publish
    path: `schedule` only creates the task.

    Bounded memory: at most `max_pending` lines are waiting at once (a new one beyond that is not
    scheduled, and says so), and a task removes itself when it finishes. `cancel_meeting` drops a
    room's pending lines when the room ends; `cancel_all` on shutdown.

    `logger` is a structlog-style logger (`.info(event, **fields)`). `publish` puts one message on
    the wire; a failure is logged and the line keeps its fallback label.
    """

    def __init__(
        self,
        tracker: FarSpeakerTracker,
        publish: Callable[[LateFarSpeakerName], Awaitable[Any]],
        logger: Any,
        *,
        min_confidence: float,
        delays_ms: Iterable[int],
        max_pending: int = 256,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self._tracker = tracker
        self._publish = publish
        self._logger = logger
        self._min_confidence = min_confidence
        self._delays_ms = late_attempt_delays_ms(delays_ms)
        self._max_pending = max(0, max_pending)
        self._sleep = sleep
        self._clock_ms = clock_ms
        # segment_id -> (meeting_id, task)
        self._pending: dict[str, tuple[str, asyncio.Task[None]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self._delays_ms) and self._max_pending > 0

    def pending_tasks(self) -> list[asyncio.Task[None]]:
        return [task for _meeting_id, task in self._pending.values()]

    def schedule(self, meeting_id: str, segment_id: str, window: SegmentWindow) -> bool:
        """Start asking about this line later. True when a task was started."""
        if not self.enabled or segment_id in self._pending:
            return False
        if len(self._pending) >= self._max_pending:
            self._logger.info(
                "far_speaker_late_skipped",
                meeting_id=meeting_id,
                segment_id=segment_id,
                reason="capacity",
                pending=len(self._pending),
            )
            return False
        task = asyncio.create_task(self._run(meeting_id, segment_id, window))
        self._pending[segment_id] = (meeting_id, task)
        task.add_done_callback(lambda done: self._forget(segment_id, done))
        return True

    def cancel_meeting(self, meeting_id: str) -> int:
        """Drop every pending line of a room that ended. Returns how many were dropped."""
        dropped = [sid for sid, (mid, _task) in self._pending.items() if mid == meeting_id]
        for sid in dropped:
            _mid, task = self._pending.pop(sid)
            task.cancel()
        return len(dropped)

    def cancel_all(self) -> int:
        dropped = list(self._pending.values())
        self._pending.clear()
        for _mid, task in dropped:
            task.cancel()
        return len(dropped)

    def _forget(self, segment_id: str, task: asyncio.Task[None]) -> None:
        current = self._pending.get(segment_id)
        if current is not None and current[1] is task:
            self._pending.pop(segment_id, None)

    async def _run(self, meeting_id: str, segment_id: str, window: SegmentWindow) -> None:
        slept_ms = 0
        best: float | None = None
        for attempt, delay_ms in enumerate(self._delays_ms, start=1):
            await self._sleep((delay_ms - slept_ms) / 1000)
            slept_ms = delay_ms
            try:
                attribution = await self._tracker.attribute(meeting_id, window)
            except Exception:
                # Trackers fail open by contract; one that does not must not kill the task with
                # an unretrieved exception. The next attempt may still succeed.
                self._logger.warning(
                    "far_speaker_late_attribution_failed",
                    meeting_id=meeting_id,
                    segment_id=segment_id,
                    attempt=attempt,
                    exc_info=True,
                )
                continue
            if attribution is not None:
                best = attribution.confidence if best is None else max(best, attribution.confidence)
            if attribution is None or not is_late_answer(attribution, self._min_confidence):
                continue
            message = LateFarSpeakerName(
                meeting_id=meeting_id,
                segment_id=segment_id,
                name=" ".join(attribution.name.split()),
                source=attribution.source,
                confidence=attribution.confidence,
                t_ms=self._clock_ms(),
            )
            try:
                await self._publish(message)
            except Exception:
                self._logger.warning(
                    "far_speaker_late_publish_failed",
                    meeting_id=meeting_id,
                    segment_id=segment_id,
                    exc_info=True,
                )
                return
            self._logger.info(
                "far_speaker_late_attributed",
                meeting_id=meeting_id,
                segment_id=segment_id,
                # A Meet participant is not a user and agreed to nothing: a hash, not the name.
                far_speaker_name_hash=name_hash(message.name),
                far_speaker_confidence=attribution.confidence,
                basis=attribution.basis,
                attempt=attempt,
                delay_ms=delay_ms,
                window_start_ms=window.start_ms,
                window_end_ms=window.end_ms,
            )
            return
        self._logger.info(
            "far_speaker_late_gave_up",
            meeting_id=meeting_id,
            segment_id=segment_id,
            attempts=len(self._delays_ms),
            # The best score any attempt reached, whatever its rule - no name.
            best_confidence=best,
            window_start_ms=window.start_ms,
            window_end_ms=window.end_ms,
        )
