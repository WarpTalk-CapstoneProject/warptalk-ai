"""The ingress same-source gate: zero the stand-in's copy of a WarpTalk participant, nothing else.

A WarpTalk user who is also in the Meet reaches ingress twice — on their own mic, and a few
hundred ms later inside the stand-in's mixed Meet feed. A stand-in frame is zeroed only when the
participant's own track explains it (lagged, gain/EQ-shaped). A Meet-side person talking at the
same time is a second voice and is kept, as a native WarpTalk room would keep them.

The signals here are synthetic "speech": harmonic syllables with a random pitch and formant
shape, so two talkers differ the way two voices do. The realistic measurement (TTS speech, Meet
NS/EQ/AGC, real Opus) lives in scripts/far_side_gate_eval/evaluate.py.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from livekit_ingress_worker.far_side_gate import (
    HOP_S,
    FarSideGateConfig,
    FarSideOverlapGate,
    _Stream,
    zero_frames,
)
from livekit_ingress_worker.worker import LiveKitIngressWorker, _frames_to_ms
from shared.config import WorkerSettings

SR = 16000
WINDOW = 1536  # what ingress hands the gate: three Silero frames
FRAME = 512
ROOM = "room-1"
STAND_IN = "00000000-0000-0000-0000-00000000b21d"


# -- synthetic speech ------------------------------------------------------------------------


def talker(seconds: float, seed: int, start: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Syllables (120-320ms) of harmonic sound with a random pitch and formant envelope.

    Returns (signal, per-frame "speaking" truth)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.zeros(n)
    t = int(start * SR)
    while t < n:
        dur = int(rng.uniform(0.12, 0.32) * SR)
        f0 = rng.uniform(95, 230)
        formants = rng.uniform(300, 3500, size=3)
        tt = np.arange(dur) / SR
        f0_t = f0 * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(1, 4) * tt))
        phase = 2 * np.pi * np.cumsum(f0_t) / SR
        syl = np.zeros(dur)
        for h in range(1, int(7000 / f0)):
            amp = sum(np.exp(-(((h * f0 - f) / 250.0) ** 2)) for f in formants) + 0.02
            syl += amp * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
        syl *= np.hanning(dur)
        e = min(n, t + dur)
        x[t:e] += syl[: e - t]
        # syllables come in words; words have pauses
        t = e + int((rng.uniform(0.02, 0.08) if rng.random() < 0.7 else rng.uniform(0.3, 1.0)) * SR)
    x *= 0.3 / (np.max(np.abs(x)) + 1e-9)
    lv = frame_levels(x)
    return x, lv > np.max(lv) - 35


def frame_levels(x: np.ndarray) -> np.ndarray:
    k = x.size // FRAME
    return 10 * np.log10((x[: k * FRAME].reshape(k, FRAME) ** 2).mean(axis=1) + 1e-12)


def meet_path(x: np.ndarray, lag_s: float, gain: float = 0.6, seed: int = 0) -> np.ndarray:
    """Delay + gain + a gentle low-pass + a little noise: the copy that comes back via Meet."""
    rng = np.random.default_rng(seed)
    d = int(lag_s * SR)
    y = np.concatenate([np.zeros(d), x[: x.size - d]]) * gain
    y = np.convolve(y, np.ones(3) / 3, mode="same")
    return y + rng.normal(0, 1e-4, y.size)


def delay(x: np.ndarray, lag_s: float) -> np.ndarray:
    d = int(lag_s * SR)
    return np.concatenate([np.zeros(d), x[: x.size - d]])


def windows(x: np.ndarray) -> Iterator[tuple[int, bytes]]:
    for w in range(x.size // WINDOW):
        seg = x[w * WINDOW : (w + 1) * WINDOW]
        yield w, (np.clip(seg, -1, 1) * 32767).astype(np.int16).tobytes()


def run(
    gate: FarSideOverlapGate,
    reference: np.ndarray | None,
    standin: np.ndarray,
    jitter_s: float = 0.02,
    seed: int = 1,
) -> np.ndarray:
    """Feed both tracks window by window in arrival order; return the stand-in frame mask."""
    rng = np.random.default_rng(seed)
    events = []
    for w, pcm in windows(standin):
        events.append((((w + 1) * WINDOW) / SR + rng.uniform(0, jitter_s), 1, w, pcm))
    if reference is not None:
        for w, pcm in windows(reference):
            events.append((((w + 1) * WINDOW) / SR + rng.uniform(0, jitter_s), 0, w, pcm))
    events.sort(key=lambda e: (e[0], e[1]))
    mask = np.zeros((standin.size // WINDOW) * 3, dtype=bool)
    for arrival, track, w, pcm in events:
        if track == 0:
            gate.push_reference(ROOM, "alice", pcm, arrival)
        else:
            mask[w * 3 : w * 3 + 3] = gate.process_standin(ROOM, pcm, arrival, exclude=(STAND_IN,))
    return mask


def on_gate(**overrides: object) -> FarSideOverlapGate:
    return FarSideOverlapGate(FarSideGateConfig(enabled=True, **overrides))  # type: ignore[arg-type]


# -- behaviour --------------------------------------------------------------------------------


class TestSameSourceDecision:
    def test_the_copy_alone_is_suppressed_once_the_lag_is_locked(self) -> None:
        user, _ = talker(30, seed=1)
        gate = on_gate()
        mask = run(gate, user, meet_path(user, 0.5))
        _, dupe = talker(30, seed=1)
        dupe = np.concatenate([np.zeros(int(0.5 * SR / FRAME)), dupe])[: mask.size].astype(bool)
        late = np.arange(mask.size) > int(10 * SR / FRAME)
        assert mask[dupe & late].mean() > 0.3
        assert gate.lag_ms(ROOM, "alice") == pytest.approx(500, abs=40)

    def test_meet_side_speaker_alone_is_never_touched(self) -> None:
        # The WarpTalk participant is connected but silent; somebody in the Meet talks.
        rng = np.random.default_rng(3)
        silence = rng.normal(0, 3e-4, 30 * SR)
        other, _ = talker(30, seed=7)
        mask = run(on_gate(), silence, other)
        assert not mask.any()

    def test_participant_muted_in_meet_does_not_cost_the_meet_side_speaker(self) -> None:
        # Alice talks on WarpTalk but is muted in Meet: the stand-in is only Bob. Their talk
        # overlaps in time, but nothing in the stand-in is Alice — nothing may be zeroed.
        alice, _ = talker(30, seed=1)
        bob, _ = talker(30, seed=9)
        mask = run(on_gate(), alice, delay(bob, 0.0) * 0.6)
        assert mask.mean() < 0.01

    def test_crosstalk_keeps_the_meet_side_speaker(self) -> None:
        alice, _ = talker(40, seed=1)
        bob, bob_speaking = talker(40, seed=11, start=1.0)
        standin = meet_path(alice, 0.45) + bob * 0.6
        mask = run(on_gate(), alice, standin)
        n = min(mask.size, bob_speaking.size)
        assert mask[:n][bob_speaking[:n]].mean() <= 0.03

    def test_lag_change_is_reacquired(self) -> None:
        user, _ = talker(40, seed=2)
        standin = np.concatenate(
            [meet_path(user, 0.4)[: 20 * SR], meet_path(user, 0.65)[20 * SR :]]
        )
        gate = on_gate()
        run(gate, user, standin)
        assert gate.lag_ms(ROOM, "alice") == pytest.approx(650, abs=40)

    def test_no_participant_track_means_nothing_is_suppressed(self) -> None:
        other, _ = talker(10, seed=4)
        assert not run(on_gate(), None, other).any()

    def test_the_stand_in_is_never_its_own_reference(self) -> None:
        user, _ = talker(20, seed=5)
        gate = on_gate()
        rng = np.random.default_rng(0)
        for w, pcm in windows(meet_path(user, 0.5)):
            arrival = (w + 1) * WINDOW / SR + rng.uniform(0, 0.02)
            gate.push_reference(ROOM, STAND_IN, pcm, arrival)
            assert not any(gate.process_standin(ROOM, pcm, arrival, exclude=(STAND_IN,)))

    def test_disabled_gate_suppresses_nothing_and_keeps_no_state(self) -> None:
        user, _ = talker(20, seed=1)
        gate = FarSideOverlapGate(FarSideGateConfig(enabled=False))
        assert not run(gate, user, meet_path(user, 0.5)).any()
        assert gate._rooms == {}

    def test_other_rooms_do_not_leak(self) -> None:
        user, _ = talker(20, seed=1)
        gate = on_gate()
        rng = np.random.default_rng(0)
        hits = 0
        dupe = meet_path(user, 0.5)
        for (w, ref), (_, st) in zip(windows(user), windows(dupe), strict=True):
            arrival = (w + 1) * WINDOW / SR + rng.uniform(0, 0.02)
            gate.push_reference("other-room", "alice", ref, arrival)
            hits += sum(gate.process_standin(ROOM, st, arrival))
        assert hits == 0

    def test_forget_speaker_and_room(self) -> None:
        user, _ = talker(20, seed=1)
        gate = on_gate()
        run(gate, user, meet_path(user, 0.5))
        assert gate.lag_ms(ROOM, "alice") is not None
        gate.forget_speaker(ROOM, "alice")
        assert gate.lag_ms(ROOM, "alice") is None
        gate.forget_room(ROOM)
        assert ROOM not in gate._rooms


class TestStreamClock:
    def test_hops_are_evenly_spaced_despite_delivery_jitter(self) -> None:
        stream = _Stream(capacity=400)
        rng = np.random.default_rng(0)
        for w in range(60):
            arrival = 100.0 + (w + 1) * WINDOW / SR + rng.uniform(0, 0.03)
            stream.push(np.zeros(WINDOW), arrival, active_db=15.0)
        gaps = np.diff(stream.times[: stream.n])
        # Within a window hops are exactly 16ms apart; across windows the clock may step back
        # by at most the jitter when a faster delivery lowers the anchor. Never backwards.
        assert np.all(gaps > 0)
        assert np.median(gaps) == pytest.approx(HOP_S)
        assert np.all(np.abs(gaps - HOP_S) <= 0.03)

    def test_a_gap_in_delivery_re_anchors(self) -> None:
        stream = _Stream(capacity=400)
        for w in range(10):
            stream.push(np.zeros(WINDOW), 1.0 + (w + 1) * WINDOW / SR, active_db=15.0)
        before = stream.times[stream.n - 1]
        stream.push(np.zeros(WINDOW), 1.0 + 11 * WINDOW / SR + 2.0, active_db=15.0)  # 2s mute
        assert stream.times[stream.n - 1] - before > 1.9

    def test_odd_sized_windows_keep_their_leftover(self) -> None:
        stream = _Stream(capacity=400)
        for w in range(10):
            stream.push(np.zeros(1000), (w + 1) * 1000 / SR, active_db=15.0)
        # The first hop looks back over 256 zeros, so hops end at samples 256, 512, ... <= 10000.
        assert stream.n == 10000 // 256

    def test_history_is_bounded(self) -> None:
        stream = _Stream(capacity=50)
        for w in range(100):
            stream.push(np.zeros(WINDOW), (w + 1) * WINDOW / SR, active_db=15.0)
        assert stream.n == 50
        assert np.all(np.diff(stream.times[: stream.n]) > 0)


class TestConfigAndZeroing:
    def test_settings_default_the_gate_off(self) -> None:
        config = FarSideGateConfig.from_settings(WorkerSettings())
        assert config.enabled is False
        assert config == FarSideGateConfig(**{**config.__dict__, "enabled": False})

    def test_settings_defaults_match_the_dataclass(self) -> None:
        config = FarSideGateConfig.from_settings(WorkerSettings())
        assert config == FarSideGateConfig()

    def test_thresholds_come_from_settings(self) -> None:
        settings = WorkerSettings(
            far_side_gate_enabled=True,
            far_side_gate_frame_min_corr=0.9,
            far_side_gate_second_voice_hold_ms=200,
        )
        config = FarSideGateConfig.from_settings(settings)
        assert config.enabled is True
        assert config.frame_min_corr == 0.9
        assert config.second_voice_hold_ms == 200

    def test_config_swaps_an_inverted_window_and_covers_it_with_history(self) -> None:
        settings = WorkerSettings(
            far_side_gate_lag_min_ms=900,
            far_side_gate_lag_max_ms=200,
            far_side_gate_history_ms=100,
        )
        config = FarSideGateConfig.from_settings(settings)
        assert (config.lag_min_ms, config.lag_max_ms) == (200, 900)
        assert config.history_ms >= config.lag_max_ms + 1500

    def test_zero_frames_only_touches_masked_frames(self) -> None:
        frame_bytes = 512 * 2
        pcm = (np.ones(512 * 3, dtype=np.int16) * 1000).tobytes()
        out = np.frombuffer(zero_frames(pcm, [False, True, False], frame_bytes), dtype=np.int16)
        assert out[:512].tolist() == [1000] * 512
        assert out[512:1024].tolist() == [0] * 512
        assert out[1024:].tolist() == [1000] * 512

    def test_zero_frames_without_mask_is_identity(self) -> None:
        pcm = b"\x01\x02" * 1536
        assert zero_frames(pcm, [False, False, False], 1024) is pcm

    def test_frames_to_ms(self) -> None:
        assert _frames_to_ms(3) == 96


class TestWorkerWiring:
    def test_default_settings_build_no_gate(self) -> None:
        worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
        worker.settings = WorkerSettings()
        assert worker._far_side_overlap_gate() is None

    def test_enabled_gate_is_built_once(self) -> None:
        worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
        worker.settings = WorkerSettings(far_side_gate_enabled=True)
        gate = worker._far_side_overlap_gate()
        assert gate is not None
        assert worker._far_side_overlap_gate() is gate
