"""Zero the bridge stand-in's copy of a WarpTalk participant who is also in the Meet.

THE DUPLICATE
    One WarpTalk desktop per Meet (the "capturer") publishes Meet's mixed audio under the stand-in
    identity. Other WarpTalk users in the same Meet join the same WarpTalk room and publish their
    OWN mic. Meet carries their voice to the capturer too, so the same sentence reaches ingress
    twice: directly, under their name, and a few hundred ms later inside the stand-in feed, after
    Meet's AEC/NS/AGC, two Opus legs, jitter buffering, loopback capture and network lag.

WHAT IS REMOVED, AND WHAT IS NOT
    Native WarpTalk rooms never cut a speaker because somebody else talks: every track has its
    own pipeline. The bridge must behave the same, so a stand-in frame is zeroed ONLY when its
    content is explained by a real participant's own track, delayed and gain/EQ-shaped. A
    Meet-side person talking over a WarpTalk user is a second source the reference cannot
    explain, and those frames pass through untouched. When unsure: keep (fail-open). The STT
    text dedupe (stt_worker/far_side_dedupe.py) stays the second layer for whatever is kept.

WHY SPECTRAL ENVELOPES, NOT WAVEFORMS (measured, scripts/far_side_gate_eval)
    A block frequency-domain NLMS canceller on the simulated Meet chain (NS, EQ, AGC, 2x Opus
    ~31kbps) reaches ~14dB median ERLE while the path delay is FIXED, and ~0dB once the delay
    wanders by +-15ms the way a jitter buffer's accelerate/expand and clock drift make it do:
    the filter never re-converges. GCC-PHAT still finds the coarse lag, but waveform
    subtraction cannot be trusted per frame. What survives is the time-frequency envelope, so
    every track is reduced to log band energies (16 bands, 125Hz-7kHz) and per-bin log power
    per 16ms hop, and:

    1. LAG TRACKING (per participant, slow): the last ~1s of the stand-in's band trajectories,
       each band minus its own long-term mean (cancels Meet's EQ and gain), is correlated with
       the participant's at every lag in [lag_min, lag_max]. A distinct, non-edge peak >=
       lock_min_corr confirms the lock; a different lag must win twice in a row to replace it
       (re-acquire). A lock not re-confirmed for lock_hold_ms expires.
    2. PER-HOP DUPLICATE TEST, all of:
       a. the lock is live and the participant's own track carried speech at that lag;
       b. SIMILARITY: the last ~64ms spectro-temporal patch of the stand-in correlates with the
          lag-aligned reference patch at >= frame_min_corr (positive evidence of the copy; this
          is what keeps a Meet-side person when the WarpTalk user is muted in Meet);
       c. RESIDUAL (per-bin ERLE): the stand-in's power is predicted per FFT bin as band gain x
          the loudest reference bin within +-2 hops / +-1 bin (gain = a LOW percentile of the
          observed offsets, so crosstalk cannot inflate it). The share of stand-in power more
          than residual_margin_db above the prediction must be <= residual_max_ratio.
    3. SECOND-VOICE HANGOVER: when >= 3 of the last 5 hops put >= second_voice_ratio of the
       stand-in's power more than second_voice_margin_db above the prediction, somebody else
       is talking; nothing is zeroed for second_voice_hold_ms (nor in the current window). A
       talker's quiet syllable under a loud duplicate looks "explained" on its own; the
       hangover is what keeps it.
    A 32ms Silero frame is zeroed only if both of its hops are duplicates. The residual is never
    passed on: a kept frame is the original audio.

CLOCK
    One ingress process owns a room, so every track's windows are stamped by one monotonic
    clock. Each stream maps its sample counter to that clock through a slowly-leaking minimum
    of (arrival - samples/sr), which strips delivery jitter but follows drift and re-anchors
    after a gap.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import numpy.typing as npt

_Arr = npt.NDArray[Any]

#: One Silero frame: 512 samples at 16kHz.
FRAME_S = 0.032
SAMPLE_RATE = 16000
HOP = 256  # 16ms
WIN = 512
HOP_S = HOP / SAMPLE_RATE
N_BANDS = 16
_EPS = 1e-10


@dataclass(frozen=True)
class FarSideGateConfig:
    """Defaults mirror WorkerSettings.far_side_gate_* in shared/config.py (explained there)."""

    enabled: bool = False
    lag_min_ms: int = 150
    lag_max_ms: int = 1200
    history_ms: int = 4000
    lock_min_corr: float = 0.5
    lock_hold_ms: int = 8000
    frame_min_corr: float = 0.7
    residual_margin_db: float = 6.0
    residual_max_ratio: float = 0.15
    ref_active_db: float = 15.0
    second_voice_margin_db: float = 12.0
    second_voice_ratio: float = 0.15
    second_voice_hold_ms: int = 1500

    @classmethod
    def from_settings(cls, settings: object) -> FarSideGateConfig:
        values: dict[str, object] = {}
        for f in fields(cls):
            default = getattr(cls, f.name)
            raw = getattr(settings, f"far_side_gate_{f.name}", default)
            values[f.name] = type(default)(raw)
        lag_min, lag_max = int(values["lag_min_ms"]), int(values["lag_max_ms"])  # type: ignore[call-overload]
        if lag_max < lag_min:
            lag_min, lag_max = lag_max, lag_min
        lag_min = max(0, lag_min)
        lag_max = max(lag_min + 2 * int(HOP_S * 1000), lag_max)
        values["lag_min_ms"], values["lag_max_ms"] = lag_min, lag_max
        values["history_ms"] = max(int(values["history_ms"]), lag_max + 1500)  # type: ignore[call-overload]
        return cls(**values)  # type: ignore[arg-type]


def _band_matrix() -> _Arr:
    """(WIN//2+1, N_BANDS) 0/1 matrix: log-spaced bands from 125Hz to 7kHz."""
    freqs = np.fft.rfftfreq(WIN, 1.0 / SAMPLE_RATE)
    edges = np.geomspace(125.0, 7000.0, N_BANDS + 1)
    m = np.zeros((freqs.size, N_BANDS), dtype=np.float64)
    for b in range(N_BANDS):
        m[(freqs >= edges[b]) & (freqs < edges[b + 1]), b] = 1.0
    return m


_BANDS = _band_matrix()
#: The residual test works per FFT bin (31.25Hz) inside the same 125Hz-7kHz span: a second
#: voice's harmonics land between the duplicate's, which a 16-band envelope would average away.
_FINE = np.flatnonzero(_BANDS.sum(axis=1) > 0)
_FINE_BAND = np.argmax(_BANDS[_FINE], axis=1)
_WINDOW = np.hanning(WIN + 1)[:-1]


class _Stream:
    """One track: its band features on the shared clock, kept for history_ms."""

    def __init__(self, capacity: int) -> None:
        self.cap = capacity
        self.times = np.zeros(capacity)
        self.db = np.zeros((capacity, N_BANDS))  # log band energy, dB
        self.fine = np.zeros((capacity, _FINE.size), dtype=np.float32)  # per-bin dB
        self.dev = np.zeros((capacity, N_BANDS))  # db minus long-term band mean, 0 when silent
        self.cdb = np.zeros((capacity, N_BANDS))  # db minus long-term band mean, always
        self.level = np.full(capacity, -120.0)  # total hop energy, dB
        self.active = np.zeros(capacity, dtype=bool)
        self.n = 0  # hops stored (<= cap)
        self.tail = np.zeros(WIN - HOP)  # first hop's look-back
        self.samples = 0
        self.t0: float | None = None
        self.last_arrival = 0.0
        self.mean = np.zeros(N_BANDS)
        self.mean_init = False
        self.active_hops = 0  # how many active hops the long-term mean has seen
        self.floor = 0.0
        self.floor_init = False

    def _clock(self, n_new: int, arrival_end_s: float) -> None:
        end = self.samples + n_new
        offset = arrival_end_s - end / SAMPLE_RATE
        if self.t0 is None or offset - self.t0 > 0.25:
            self.t0 = offset  # first window, or a gap: re-anchor
        else:
            dt = max(0.0, arrival_end_s - self.last_arrival)
            self.t0 = min(self.t0 + 0.002 * dt, offset)
        self.last_arrival = arrival_end_s

    def push(self, pcm: _Arr, arrival_end_s: float, active_db: float) -> int:
        """Append a window of float samples; returns how many hops were added."""
        self._clock(pcm.size, arrival_end_s)
        assert self.t0 is not None
        buf = np.concatenate([self.tail, pcm])
        start_abs = self.samples - self.tail.size  # absolute index of buf[0]
        n_hops = max(0, (buf.size - WIN) // HOP + 1)
        consumed = n_hops * HOP
        self.tail = buf[consumed:]
        if n_hops == 0:
            self.samples += pcm.size
            return 0
        idx = np.arange(n_hops)[:, None] * HOP + np.arange(WIN)[None, :]
        frames = buf[idx] * _WINDOW
        power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        bands = power @ _BANDS
        db = 10.0 * np.log10(bands + _EPS)
        level = 10.0 * np.log10(bands.sum(axis=1) + _EPS)
        fine = (10.0 * np.log10(power[:, _FINE] + _EPS)).astype(np.float32)
        hop_end = start_abs + HOP * np.arange(n_hops) + WIN
        times = self.t0 + hop_end / SAMPLE_RATE
        self.samples += pcm.size

        active = np.zeros(n_hops, dtype=bool)
        dev = np.zeros_like(db)
        cdb = np.zeros_like(db)
        for i in range(n_hops):
            lv = level[i]
            if not self.floor_init:
                self.floor, self.floor_init = lv, True
            elif lv < self.floor:
                self.floor = 0.7 * self.floor + 0.3 * lv
            else:
                self.floor += min(lv - self.floor, 0.05)  # creeps up ~3dB/s
            if lv > self.floor + active_db and lv > -80.0:
                active[i] = True
                self.active_hops += 1
                if not self.mean_init:
                    self.mean, self.mean_init = db[i].copy(), True
                else:
                    # Fast at first (a plain running average), then ~1.6s time constant.
                    self.mean += max(0.01, 1.0 / self.active_hops) * (db[i] - self.mean)
                dev[i] = db[i] - self.mean
            cdb[i] = db[i] - self.mean
        self._append(times, db, fine, dev, cdb, level, active)
        return int(n_hops)

    def _append(self, *rows: _Arr) -> None:
        stores: tuple[_Arr, ...] = (
            self.times,
            self.db,
            self.fine,
            self.dev,
            self.cdb,
            self.level,
            self.active,
        )
        k = rows[0].shape[0]
        if k >= self.cap:
            rows = tuple(a[-self.cap :] for a in rows)
            k = self.cap
        overflow = self.n + k - self.cap
        if overflow > 0:
            for a in stores:
                a[: self.n - overflow] = a[overflow : self.n]
            self.n -= overflow
        for store, row in zip(stores, rows, strict=True):
            store[self.n : self.n + k] = row
        self.n += k


@dataclass
class _Lock:
    lag: float | None = None  # hops
    confirmed_at: float = -1e9
    candidate: int | None = None
    gain_db: _Arr = field(default_factory=lambda: np.zeros(N_BANDS))
    acquired_at: float | None = None


@dataclass
class _Room:
    standin: _Stream
    refs: dict[str, _Stream]
    locks: dict[str, _Lock]
    second_voice_until: float = -1e9
    recent_flags: deque[bool] = field(default_factory=lambda: deque(maxlen=5))


class FarSideOverlapGate:
    """Per-room same-source test between the stand-in and every real participant's own track."""

    CONTEXT_HOPS = 64  # ~1s of stand-in for the lag search
    PATCH_HOPS = 4  # ~64ms for the per-hop similarity
    SECOND_VOICE_MIN_HOPS = 3  # of the last 5 (80ms)
    GAIN_PERCENTILE = 20
    SIM_SHIFTS = (-2, -1, 0, 1, 2)  # hops of lag jitter the similarity tolerates

    def __init__(
        self,
        config: FarSideGateConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or FarSideGateConfig()
        self._clock = clock
        self._rooms: dict[str, _Room] = {}
        self._cap = int(self.config.history_ms / 1000.0 / HOP_S) + 8
        self.lag_min_hops = int(round(self.config.lag_min_ms / 1000.0 / HOP_S))
        self.lag_max_hops = int(round(self.config.lag_max_ms / 1000.0 / HOP_S))
        #: Set to a list to record (time, similarity, unexplained, second-voice share) per hop.
        self.debug: list[tuple[float, float, float, float]] | None = None

    def now(self) -> float:
        return self._clock()

    def _room(self, room: str) -> _Room:
        r = self._rooms.get(room)
        if r is None:
            r = _Room(standin=_Stream(self._cap), refs={}, locks={})
            self._rooms[room] = r
        return r

    @staticmethod
    def _to_float(window: bytes | _Arr) -> _Arr:
        if isinstance(window, (bytes, bytearray, memoryview)):
            return np.frombuffer(window, dtype=np.int16).astype(np.float64) / 32768.0
        arr = np.asarray(window)
        if arr.dtype == np.int16:
            return arr.astype(np.float64) / 32768.0
        return arr.astype(np.float64)

    # -- real participants -------------------------------------------------------------------

    def push_reference(
        self, room: str, speaker: str, window: bytes | _Arr, window_end_s: float
    ) -> None:
        """Feed one 16kHz mono window of a NON-stand-in participant's own track."""
        if not self.config.enabled:
            return
        r = self._room(room)
        stream = r.refs.get(speaker)
        if stream is None:
            stream = r.refs[speaker] = _Stream(self._cap)
            r.locks[speaker] = _Lock()
        stream.push(self._to_float(window), window_end_s, self.config.ref_active_db)

    # -- the stand-in decision ---------------------------------------------------------------

    def process_standin(
        self,
        room: str,
        window: bytes | _Arr,
        window_end_s: float,
        frame_samples: int = 512,
        exclude: Iterable[str] = (),
    ) -> list[bool]:
        """Feed one stand-in window; return which of its Silero frames are a duplicate."""
        pcm = self._to_float(window)
        n_frames = pcm.size // frame_samples
        if not self.config.enabled or n_frames <= 0:
            return [False] * max(0, n_frames)
        r = self._room(room)
        n_new = r.standin.push(pcm, window_end_s, 6.0)
        skip = set(exclude)
        refs = {k: v for k, v in r.refs.items() if k not in skip and v.n > 0}
        if n_new == 0 or not refs:
            return [False] * n_frames
        now = r.standin.times[r.standin.n - 1]
        hold_until = r.second_voice_until
        for speaker, ref in refs.items():
            self._track_lag(r.standin, ref, r.locks[speaker], now)
        verdicts = [
            self._judge_hop(r.standin, r.standin.n - n_new + i, refs, r.locks, now)
            for i in range(n_new)
        ]
        # SECOND-VOICE HANGOVER. A hop that shows a voice the references cannot explain marks
        # the stand-in as carrying somebody else; nothing is zeroed for second_voice_hold_ms
        # after it, and nothing in this window either (the window is the only look-ahead there
        # is). A talker's quiet syllable under a loud duplicate looks "explained" on its own —
        # the hangover is what keeps it.
        times = r.standin.times[r.standin.n - n_new : r.standin.n]
        # One odd hop is a codec/NS artifact of the duplicate; a talker persists. Count a hop
        # only when it is the SECOND_VOICE_MIN_HOPS-th flagged hop of the last SECOND_VOICE_SPAN.
        flagged = []
        for t, (_, second) in zip(times, verdicts, strict=True):
            r.recent_flags.append(second)
            if second and sum(r.recent_flags) >= self.SECOND_VOICE_MIN_HOPS:
                flagged.append(float(t))
        if flagged:
            r.second_voice_until = max(
                r.second_voice_until, flagged[-1] + self.config.second_voice_hold_ms / 1000.0
            )
        hop_mask = [
            dup and not flagged and float(t) > hold_until
            for t, (dup, _) in zip(times, verdicts, strict=True)
        ]
        hops_per_frame = max(1, frame_samples // HOP)
        # Align the hop decisions to the END of the window (hops == frames*2 when aligned).
        hop_mask = ([False] * max(0, n_frames * hops_per_frame - len(hop_mask)) + hop_mask)[
            -n_frames * hops_per_frame :
        ]
        return [
            all(hop_mask[f * hops_per_frame : (f + 1) * hops_per_frame]) for f in range(n_frames)
        ]

    def _ref_index(self, ref: _Stream, t: _Arr | float) -> _Arr:
        """Index of the reference hop at (or just before) clock time t."""
        return np.searchsorted(ref.times[: ref.n], t, side="right") - 1

    def _track_lag(self, s: _Stream, ref: _Stream, lock: _Lock, now: float) -> None:
        c = min(self.CONTEXT_HOPS, s.n)
        if c < 24 or s.active_hops < 30 or ref.active_hops < 30:
            return  # the long-term means are not settled yet
        sl = slice(s.n - c, s.n)
        x = s.dev[sl]
        s_act = s.active[sl]
        if s_act.sum() < 0.3 * c:
            return
        base = self._ref_index(ref, s.times[sl])
        lags = np.arange(self.lag_min_hops, self.lag_max_hops + 1)
        idx = base[None, :] - lags[:, None]  # (nL, c)
        valid = idx >= 0
        y = ref.dev[np.clip(idx, 0, None)] * valid[..., None]
        y_act = ref.active[np.clip(idx, 0, None)] & valid
        both = (y_act & s_act[None, :]).sum(axis=1)
        num = np.einsum("lcb,cb->l", y, x)
        den = np.sqrt((y * y).sum(axis=(1, 2)) * (x * x).sum()) + _EPS
        corr = np.where(both >= 0.3 * c, num / den, -1.0)
        j = int(np.argmax(corr))
        peak = float(corr[j])
        if peak < self.config.lock_min_corr or j in (0, lags.size - 1):
            return  # weak, or at the edge of the search range (the true lag may lie outside)
        if peak - float(np.median(corr[corr > -1.0])) < 0.25:
            return  # not a distinct peak
        lag = int(lags[j])
        if lock.lag is None or abs(lag - lock.lag) <= 2:
            if lock.lag is None:
                if lock.candidate is None or abs(lag - lock.candidate) > 2:
                    lock.candidate = lag
                    return
                lock.lag = float(lag)
                lock.acquired_at = now
            else:
                lock.lag = 0.7 * lock.lag + 0.3 * lag
            lock.candidate = None
        else:
            if lock.candidate is None or abs(lag - lock.candidate) > 2:
                lock.candidate = lag
                return
            lock.lag, lock.candidate, lock.acquired_at = float(lag), None, now
        lock.confirmed_at = now
        # Band gain: a LOW percentile of stand-in minus reference over hops where both speak.
        # Crosstalk only ever adds stand-in energy, so it cannot drag a low percentile up.
        ri = base - int(round(lock.lag))
        ok = (ri >= 0) & s_act & ref.active[np.clip(ri, 0, None)]
        if ok.sum() >= 8:
            diff = s.db[sl][ok] - ref.db[ri[ok]]
            lock.gain_db = np.percentile(diff, self.GAIN_PERCENTILE, axis=0)

    def _judge_hop(
        self,
        s: _Stream,
        k: int,
        refs: dict[str, _Stream],
        locks: dict[str, _Lock],
        now: float,
    ) -> tuple[bool, bool]:
        """(the hop is a duplicate, the hop shows a second voice). Pure over the stored state."""
        cfg = self.config
        p = self.PATCH_HOPS
        if k < p or s.level[k] <= -100.0:
            return False, False  # warm-up, or digital silence: nothing to zero
        loud = bool(s.level[k] > s.floor + cfg.ref_active_db)
        best_sim = -1.0
        predicted = np.zeros(_FINE.size)
        any_locked = any_live = False
        for speaker, ref in refs.items():
            lock = locks[speaker]
            if lock.lag is None or now - lock.confirmed_at > cfg.lock_hold_ms / 1000.0:
                continue
            ri = int(self._ref_index(ref, s.times[k])) - int(round(lock.lag))
            if ri - p < 0 or ri + 1 >= ref.n:
                continue
            any_locked = True
            near = slice(max(0, ri - 2), ri + 3)  # +-2 hops of lag jitter
            live = bool(ref.active[near].any())
            any_live = any_live or live
            # Prediction of the stand-in's per-bin powers from this reference: the loudest of
            # +-2 hops and +-1 bin (lag jitter, pitch smear), shaped by the band gain.
            ref_db = ref.fine[near].max(axis=0)
            ref_db = np.maximum(ref_db, np.maximum(np.roll(ref_db, 1), np.roll(ref_db, -1)))
            pred_db = ref_db + lock.gain_db[_FINE_BAND]
            predicted += 10.0 ** (pred_db / 10.0)
            if not live:
                continue  # its silence still predicts the stand-in's floor, nothing more
            # Similarity of the last ~64ms patch, best over +-2 hops of jitter.
            xs = s.cdb[k - p + 1 : k + 1].ravel()
            xs = xs - xs.mean()
            for d in self.SIM_SHIFTS:
                a = ri + d
                if a - p + 1 < 0 or a >= ref.n:
                    continue
                ys = ref.cdb[a - p + 1 : a + 1].ravel()
                ys = ys - ys.mean()
                den = np.sqrt(float(xs @ xs) * float(ys @ ys))
                if den > 0:
                    best_sim = max(best_sim, float(xs @ ys) / den)
        if not any_locked:
            return False, False
        s_db = s.fine[k].astype(np.float64)
        s_pow = 10.0 ** (s_db / 10.0)
        total = float(s_pow.sum()) + _EPS
        excess = s_db - 10.0 * np.log10(predicted + _EPS)
        unexplained = float(s_pow[excess > cfg.residual_margin_db].sum()) / total
        strong = float(s_pow[excess > cfg.second_voice_margin_db].sum()) / total
        if self.debug is not None:
            self.debug.append((float(s.times[k]), best_sim, unexplained, strong))
        second = loud and strong >= cfg.second_voice_ratio
        dup = any_live and best_sim >= cfg.frame_min_corr and unexplained <= cfg.residual_max_ratio
        return dup, second

    # -- introspection / lifecycle -----------------------------------------------------------

    def lag_ms(self, room: str, speaker: str) -> float | None:
        """The tracked lag of `speaker`'s copy in the stand-in, or None when not locked."""
        r = self._rooms.get(room)
        lock = r.locks.get(speaker) if r else None
        if lock is None or lock.lag is None:
            return None
        return lock.lag * HOP_S * 1000.0

    def forget_speaker(self, room: str, speaker: str) -> None:
        r = self._rooms.get(room)
        if r is not None:
            r.refs.pop(speaker, None)
            r.locks.pop(speaker, None)

    def forget_room(self, room: str) -> None:
        self._rooms.pop(room, None)


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
