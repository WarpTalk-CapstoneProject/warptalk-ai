"""Zero the bridge stand-in's copy of a WarpTalk participant who is also in the Meet.

THE DUPLICATE
    One WarpTalk desktop per Meet (the "capturer") publishes Meet's mixed audio under the stand-in
    identity. Other WarpTalk users in the same Meet join the same WarpTalk room and publish their
    OWN mic. Meet carries their voice to the capturer too, so the same sentence reaches ingress
    twice: directly, under their name, and a few hundred ms later inside the stand-in feed.

THE TIME GATE
    Every non-stand-in track already runs Silero per 32ms frame. Those per-frame verdicts are
    remembered here as speech intervals (monotonic seconds, the arrival clock both tracks share
    because one ingress process owns the room). A stand-in frame arriving at [s, e] can contain
    human speech that arrived at [s - lag_max, e - lag_min]; if any is there, the frame is
    `suppressed_overlap` and the caller zeroes it before VAD and STT.

    Per FRAME, not per turn: a far-side person talking over a WarpTalk user loses only the
    overlapping frames, and a turn that merely starts during somebody's speech keeps the rest.

    The capturer's own voice is not normally in its loopback (Meet does not echo you to
    yourself), so gating on every non-stand-in human mostly costs far-side crosstalk while a
    WarpTalk user speaks — the accepted trade for phase 1.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

#: One Silero frame: 512 samples at 16kHz.
FRAME_S = 0.032


@dataclass(frozen=True)
class FarSideGateConfig:
    enabled: bool = True
    lag_min_ms: int = 300
    lag_max_ms: int = 600
    min_overlap_ms: float = 0.0
    history_ms: int = 5000

    @classmethod
    def from_settings(cls, settings: object) -> FarSideGateConfig:
        lag_min = int(getattr(settings, "far_side_gate_lag_min_ms", cls.lag_min_ms))
        lag_max = int(getattr(settings, "far_side_gate_lag_max_ms", cls.lag_max_ms))
        if lag_max < lag_min:
            lag_min, lag_max = lag_max, lag_min
        return cls(
            enabled=bool(getattr(settings, "far_side_gate_enabled", cls.enabled)),
            lag_min_ms=max(0, lag_min),
            lag_max_ms=max(0, lag_max),
            min_overlap_ms=float(getattr(settings, "far_side_gate_min_overlap_ms", 0.0)),
            history_ms=max(
                int(getattr(settings, "far_side_gate_history_ms", cls.history_ms)),
                lag_max + 1000,
            ),
        )


def lagged_overlap_ms(
    frame_start_s: float,
    frame_end_s: float,
    intervals: Iterable[tuple[float, float]],
    lag_min_ms: float,
    lag_max_ms: float,
) -> float:
    """How much human speech could have produced this stand-in frame, in ms. Pure.

    The frame arrived at [frame_start, frame_end]. With a path delay anywhere in
    [lag_min, lag_max], the speech it could carry arrived on the direct track during
    [frame_start - lag_max, frame_end - lag_min]. Returns the length of that range covered by
    `intervals` (assumed non-overlapping, as SpeechActivity keeps them).
    """
    lo = frame_start_s - lag_max_ms / 1000.0
    hi = frame_end_s - lag_min_ms / 1000.0
    if hi <= lo:
        return 0.0
    covered = 0.0
    for start, end in intervals:
        a = max(lo, start)
        b = min(hi, end)
        if b > a:
            covered += b - a
    return covered * 1000.0


class FarSideOverlapGate:
    """Per-room memory of who was speaking when, and the suppression decision for the stand-in."""

    def __init__(
        self,
        config: FarSideGateConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or FarSideGateConfig()
        self._clock = clock
        # room -> speaker -> merged (start_s, end_s) intervals, oldest first.
        self._speech: dict[str, dict[str, deque[tuple[float, float]]]] = {}

    def now(self) -> float:
        return self._clock()

    # -- recording human speech ---------------------------------------------------------------

    def note_speech(self, room: str, speaker: str, start_s: float, end_s: float) -> None:
        if not self.config.enabled or end_s <= start_s:
            return
        intervals = self._speech.setdefault(room, {}).setdefault(speaker, deque())
        # Frames arrive in order; merge with the last interval when contiguous (1ms slack for
        # float drift between consecutive frame boundaries).
        if intervals and start_s <= intervals[-1][1] + 0.001:
            last_start, last_end = intervals[-1]
            intervals[-1] = (last_start, max(last_end, end_s))
        else:
            intervals.append((start_s, end_s))
        horizon = end_s - self.config.history_ms / 1000.0
        while intervals and intervals[0][1] < horizon:
            intervals.popleft()

    def note_frames(
        self,
        room: str,
        speaker: str,
        window_end_s: float,
        frame_probabilities: Sequence[float],
        threshold: float,
        frame_s: float = FRAME_S,
    ) -> None:
        """Record one VAD window's verdicts for a NON-stand-in track, ending at window_end_s."""
        n = len(frame_probabilities)
        for i, prob in enumerate(frame_probabilities):
            if prob >= threshold:
                end = window_end_s - (n - 1 - i) * frame_s
                self.note_speech(room, speaker, end - frame_s, end)

    # -- the stand-in decision -----------------------------------------------------------------

    def _human_intervals(self, room: str, exclude: Iterable[str]) -> list[tuple[float, float]]:
        speakers = self._speech.get(room)
        if not speakers:
            return []
        skip = set(exclude)
        merged: list[tuple[float, float]] = []
        for speaker, intervals in speakers.items():
            if speaker in skip:
                continue
            merged.extend(intervals)
        if len(merged) <= 1:
            return merged
        # Different speakers overlap; union them so overlap is not counted twice.
        merged.sort()
        union = [merged[0]]
        for start, end in merged[1:]:
            if start <= union[-1][1]:
                union[-1] = (union[-1][0], max(union[-1][1], end))
            else:
                union.append((start, end))
        return union

    def suppression_mask(
        self,
        room: str,
        window_end_s: float,
        n_frames: int,
        exclude: Iterable[str] = (),
        frame_s: float = FRAME_S,
    ) -> list[bool]:
        """Which frames of a stand-in window ending at window_end_s are `suppressed_overlap`."""
        if not self.config.enabled or n_frames <= 0:
            return [False] * max(0, n_frames)
        intervals = self._human_intervals(room, exclude)
        if not intervals:
            return [False] * n_frames
        mask: list[bool] = []
        for i in range(n_frames):
            end = window_end_s - (n_frames - 1 - i) * frame_s
            overlap = lagged_overlap_ms(
                end - frame_s,
                end,
                intervals,
                self.config.lag_min_ms,
                self.config.lag_max_ms,
            )
            mask.append(overlap > self.config.min_overlap_ms)
        return mask

    def forget_speaker(self, room: str, speaker: str) -> None:
        speakers = self._speech.get(room)
        if speakers is not None:
            speakers.pop(speaker, None)

    def forget_room(self, room: str) -> None:
        self._speech.pop(room, None)


def zero_frames(window: bytes, mask: Sequence[bool], frame_bytes: int) -> bytes:
    """Return `window` with every masked frame replaced by digital silence."""
    if not any(mask):
        return window
    out = bytearray(window)
    for i, suppressed in enumerate(mask):
        if suppressed:
            start = i * frame_bytes
            out[start : start + frame_bytes] = bytes(min(frame_bytes, max(0, len(out) - start)))
    return bytes(out)
