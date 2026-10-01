"""Synthetic evaluation of the far-side same-source gate (livekit_ingress_worker/far_side_gate.py).

Builds a WarpTalk user's direct track and the bridge stand-in that carries their copy back
through a simulated Meet path, optionally with a second (Meet-side) speaker, feeds both to the
gate window by window in arrival order, and scores the stand-in frames it zeroes.

Simulated Meet path for the duplicate (all applied to the user's speech + room noise):
    Meet-sender NS (spectral gating) -> EQ (HP 120Hz, LP 6.5kHz, presence tilt) -> slow AGC ->
    Opus (sender leg) -> mix with the Meet-side speaker (own EQ/NS/Opus) -> Opus (capturer ->
    LiveKit leg) -> time-varying delay (base 300-900ms + drift + jitter + optional step).
The direct track: light NS -> Opus (WarpTalk leg).
Opus is REAL libopus through libsndfile (soundfile OGG/OPUS); the script refuses to run without it.

Speech: any folder of 16kHz mono wavs per speaker. make_speech.ps1 synthesizes one with the
Windows built-in voices (David, Zira).

Usage:
    python scripts/far_side_gate_eval/evaluate.py --speech <dir with David/ Zira/> [--seeds 3]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import zlib
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from livekit_ingress_worker.far_side_gate import (  # noqa: E402
    FarSideGateConfig,
    FarSideOverlapGate,
)

SR = 16000
WINDOW = 1536  # what ingress hands the gate: 3 Silero frames
FRAME = 512
#: libsndfile compression_level; 0.9 ~ 31 kbps (WebRTC voice default), 1.0 ~ 7 kbps (SILK floor).
OPUS_LEVEL = 0.9


# -- signal processing helpers ----------------------------------------------------------------


def load_dir(d: Path) -> list[np.ndarray]:
    out = []
    for p in sorted(d.glob("*.wav")):
        x, sr = sf.read(p, dtype="float64")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == SR, f"{p} is {sr}Hz, need 16kHz"
        # trim leading/trailing near-silence
        nz = np.flatnonzero(np.abs(x) > 1e-3)
        if nz.size:
            x = x[nz[0] : nz[-1] + 1]
        out.append(x / (np.max(np.abs(x)) + 1e-9) * 0.5)
    return out


def place(
    sentences: list[np.ndarray], total: int, rng: np.random.Generator, gap=(0.4, 2.5), start=0.5
) -> np.ndarray:
    y = np.zeros(total)
    t = int(start * SR)
    while t < total:
        s = sentences[rng.integers(len(sentences))] * rng.uniform(0.6, 1.0)
        e = min(total, t + s.size)
        y[t:e] += s[: e - t]
        t = e + int(rng.uniform(*gap) * SR)
    return y


def fft_filter(x: np.ndarray, resp) -> np.ndarray:  # type: ignore[no-untyped-def]
    n = 1 << int(np.ceil(np.log2(x.size)))
    spec = np.fft.rfft(x, n)
    f = np.fft.rfftfreq(n, 1 / SR)
    return np.fft.irfft(spec * resp(f), n)[: x.size]


def eq(x: np.ndarray, hp=120.0, lp=6500.0, tilt_db=3.0) -> np.ndarray:
    def resp(f):  # type: ignore[no-untyped-def]
        h = 1 / np.sqrt(1 + (hp / np.maximum(f, 1)) ** 4) / np.sqrt(1 + (f / lp) ** 8)
        presence = 10 ** (tilt_db / 20 * np.exp(-(((f - 3000) / 1200) ** 2)))
        return h * presence

    return fft_filter(x, resp)


def spectral_gate(x: np.ndarray, strength=1.0, floor_db=-25.0) -> np.ndarray:
    """Noise-suppression-like: Wiener-style gain from a percentile noise estimate."""
    n, h = 512, 256
    w = np.hanning(n + 1)[:-1]
    pad = np.concatenate([np.zeros(n), x, np.zeros(n)])
    idx = np.arange(0, pad.size - n, h)
    frames = np.stack([pad[i : i + n] * w for i in idx])
    spec = np.fft.rfft(frames, axis=1)
    power = np.abs(spec) ** 2
    noise = np.percentile(power, 10, axis=0) + 1e-12
    g = np.maximum(10 ** (floor_db / 20), 1 - strength * noise / (power + 1e-12))
    # smooth gains over time like real NS
    for i in range(1, g.shape[0]):
        g[i] = np.maximum(g[i], 0.6 * g[i - 1])
    y = np.fft.irfft(spec * g, n, axis=1) * w
    out = np.zeros(pad.size)
    for k, i in enumerate(idx):
        out[i : i + n] += y[k]
    out /= 1.5  # hann^2 overlap-add at 50% sums to 1.5
    return out[n : n + x.size]


def opus(x: np.ndarray, level: float | None = None) -> np.ndarray:
    """Real libopus round trip via libsndfile. level 0..1 = libsndfile compression_level."""
    level = OPUS_LEVEL if level is None else level
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1), SR, format="OGG", subtype="OPUS", compression_level=level)
    buf.seek(0)
    y, _ = sf.read(buf, dtype="float64")
    if y.size < x.size:
        y = np.concatenate([y, np.zeros(x.size - y.size)])
    return y[: x.size]


def variable_delay(x: np.ndarray, delay_s: np.ndarray) -> np.ndarray:
    t = np.arange(x.size) / SR
    return np.interp(t - delay_s, t, x, left=0.0, right=0.0)


def frame_levels(x: np.ndarray) -> np.ndarray:
    n = x.size // FRAME
    e = (x[: n * FRAME].reshape(n, FRAME) ** 2).mean(axis=1)
    return 10 * np.log10(e + 1e-12)


def active_frames(x: np.ndarray) -> np.ndarray:
    lv = frame_levels(x)
    return lv > np.percentile(lv, 99) - 30


# -- scenarios ---------------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    dupe: bool = True  # user's voice reaches the stand-in through Meet
    user_talks: bool = True
    other: str | None = None  # voice dir of the Meet-side speaker, None = nobody
    other_gain_db: float = 0.0  # Meet-side speaker level relative to the duplicate
    lag_ms: float = 600.0
    step_ms: float = 0.0  # lag jump at half time
    wander: bool = True  # drift + jitter-buffer-like delay wander; False = a fixed delay
    duration_s: float = 60.0


def build(
    sc: Scenario, voices: dict[str, list[np.ndarray]], user_voice: str, rng: np.random.Generator
):
    total = int(sc.duration_s * SR)
    t = np.arange(total) / SR
    user = place(voices[user_voice], total, rng) if sc.user_talks else np.zeros(total)
    room_noise = rng.normal(0, 10 ** (-55 / 20), total)
    direct = opus(spectral_gate(user + room_noise, 0.6), OPUS_LEVEL)

    # delay: base + drift (+-40ms over the clip) + slow jitter (+-15ms) + optional step
    drift = rng.uniform(-0.04, 0.04) * t / sc.duration_s
    jit = np.cumsum(rng.normal(0, 1, total // 1600 + 1))
    jit = np.repeat(jit - jit.mean(), 1600)[:total]
    jit = 0.015 * jit / (np.max(np.abs(jit)) + 1e-9)
    if not sc.wander:
        drift = jit = np.zeros(total)
    delay = sc.lag_ms / 1000 + drift + jit + np.where(t > sc.duration_s / 2, sc.step_ms / 1000, 0)

    if sc.dupe:
        agc = 10 ** ((3 * np.sin(2 * np.pi * t / 7.0)) / 20)
        dupe_meet = opus(eq(spectral_gate(user + room_noise, 1.2)) * agc * 0.8, OPUS_LEVEL)
    else:
        dupe_meet = np.zeros(total)
    if sc.other:
        other_src = place(voices[sc.other], total, rng, gap=(0.3, 2.0), start=rng.uniform(0.5, 3))
        other_meet = opus(
            eq(
                spectral_gate(other_src + rng.normal(0, 10 ** (-58 / 20), total)),
                hp=150,
                lp=7000,
                tilt_db=-2,
            ),
            OPUS_LEVEL,
        )
        # scale the Meet-side speaker relative to the duplicate's speech level
        ref_lvl = np.percentile(frame_levels(dupe_meet if sc.dupe else other_meet), 99)
        oth_lvl = np.percentile(frame_levels(other_meet), 99)
        other_meet *= 10 ** ((ref_lvl - oth_lvl + sc.other_gain_db) / 20)
    else:
        other_meet = np.zeros(total)
    # The stand-in: duplicate (delayed) + Meet-side speaker (not delayed relative to itself),
    # then the capturer's leg.
    dupe_delayed = variable_delay(dupe_meet, delay)
    standin = opus(dupe_delayed + other_meet, OPUS_LEVEL)
    truth = {
        "dupe_active": active_frames(variable_delay(user, delay))
        if sc.dupe and sc.user_talks
        else np.zeros(total // FRAME, bool),
        "other_active": active_frames(other_meet) if sc.other else np.zeros(total // FRAME, bool),
        "delay": delay,
        # per-frame levels of each component as it sits in the stand-in (before the last leg)
        "dupe_lv": frame_levels(dupe_delayed),
        "other_lv": frame_levels(other_meet),
    }
    return direct, standin, truth


def run_gate(
    direct: np.ndarray, standin: np.ndarray, config: FarSideGateConfig, rng: np.random.Generator
):
    gate = FarSideOverlapGate(config, clock=lambda: 0.0)
    events = []
    n_win = direct.size // WINDOW
    for w in range(n_win):
        end = (w + 1) * WINDOW / SR
        # ingress delivery jitter: 20-60ms network + up to 30ms event-loop wobble per track
        events.append((end + 0.04 + rng.uniform(0, 0.03), 0, w))
        events.append((end + 0.04 + rng.uniform(0, 0.03), 1, w))
    events.sort()
    mask = np.zeros(n_win * 3, bool)
    lag_trace = []
    t0 = time.perf_counter()
    for arrival, track, w in events:
        seg = slice(w * WINDOW, (w + 1) * WINDOW)
        if track == 0:
            pcm = (np.clip(direct[seg], -1, 1) * 32767).astype(np.int16).tobytes()
            gate.push_reference("room", "user", pcm, arrival)
        else:
            pcm = (np.clip(standin[seg], -1, 1) * 32767).astype(np.int16).tobytes()
            m = gate.process_standin("room", pcm, arrival)
            mask[w * 3 : w * 3 + 3] = m
            lag_trace.append(((w + 1) * WINDOW / SR, gate.lag_ms("room", "user")))
    cpu = time.perf_counter() - t0
    return mask, lag_trace, cpu


def convergence(
    lag_trace, delay: np.ndarray, first_speech_s: float, from_s: float, tol_ms=48.0
) -> float | None:
    for t, lag in lag_trace:
        if t < from_s:
            continue
        true = delay[min(delay.size - 1, int(t * SR))] * 1000
        if lag is not None and abs(lag - true) <= tol_ms:
            return t - first_speech_s
    return None


def score(sc: Scenario, mask: np.ndarray, truth, lag_trace) -> dict:
    n = min(mask.size, truth["dupe_active"].size, truth["other_active"].size)
    m, d, o = mask[:n], truth["dupe_active"][:n], truth["other_active"][:n]
    res = {}
    if d.any():
        res["dupe_recall"] = m[d & ~o].mean() if (d & ~o).any() else float("nan")
    if o.any():
        res["other_zeroed"] = m[o].mean()
        if (o & d).any():
            res["other_zeroed_in_overlap"] = m[o & d].mean()
            res["overlap_s"] = (o & d).sum() * FRAME / SR
        if "other_lv" in truth:
            # The frames where the Meet-side person is NOT buried under the duplicate (within
            # 6dB of it or louder): what a listener would actually lose.
            audible = o & d & (truth["other_lv"][:n] >= truth["dupe_lv"][:n] - 6.0)
            if audible.any():
                res["other_zeroed_audible"] = m[audible].mean()
    res["zeroed_idle"] = m[~d & ~o].mean() if (~d & ~o).any() else float("nan")
    if d.any():
        first = np.flatnonzero(d)[0] * FRAME / SR
        res["converge_s"] = convergence(lag_trace, truth["delay"], first, 0.0)
        if sc.step_ms:
            half = sc.duration_s / 2
            after = d.copy()
            after[: int(half * SR / FRAME)] = False
            first2 = np.flatnonzero(after)[0] * FRAME / SR if after.any() else half
            res["reconverge_s"] = convergence(lag_trace, truth["delay"], first2, first2)
    return res


def gcc_phat_hit_rate(direct: np.ndarray, standin: np.ndarray, delay: np.ndarray) -> float:
    """Share of 1s windows where waveform GCC-PHAT finds the true lag within 5ms."""
    hits = tot = 0
    span = SR
    for s in range(2 * SR, standin.size - span, span):
        y = standin[s : s + span]
        x = direct[s - int(1.2 * SR) : s + span]
        if np.std(y) < 1e-3 or np.std(x) < 1e-3:
            continue
        n = 1 << int(np.ceil(np.log2(x.size + y.size)))
        cross = np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n))
        r = np.fft.irfft(cross / (np.abs(cross) + 1e-12), n)
        # y[k] ~ x[k + 1.2s - lag]  -> peak at shift (lag - 1.2s) mod n ... search lags 0..1.2s
        lags = np.arange(0, int(1.2 * SR))
        shifts = (int(1.2 * SR) - lags) % n
        vals = r[(-shifts) % n]
        est = lags[int(np.argmax(vals))] / SR
        tot += 1
        hits += abs(est - delay[s + span // 2]) < 0.005
    return hits / tot if tot else float("nan")


_VOICES: dict[str, list[np.ndarray]] = {}


def _init_worker(opus_level: float) -> None:
    global OPUS_LEVEL
    OPUS_LEVEL = opus_level


def _run_case(job) -> tuple[dict, float]:  # type: ignore[no-untyped-def]
    sc, user_voice, seed, cache, config, gcc, speech = job
    rng = np.random.default_rng(seed * 7919 + zlib.crc32(sc.name.encode()) % 1000)
    key = f"{sc.name}|{user_voice}|{seed}|{OPUS_LEVEL}".replace(" ", "_")
    key = "".join(c if c.isalnum() or c in "_-." else "_" for c in key)
    cached = cache / f"{key}.npz" if cache else None
    if cached is not None and cached.exists():
        z = np.load(cached)
        direct, standin = z["direct"], z["standin"]
        truth = {k: z[k] for k in z.files if k not in ("direct", "standin")}
    else:
        if not _VOICES:
            _VOICES.update({d.name: load_dir(d) for d in sorted(speech.iterdir()) if d.is_dir()})
        direct, standin, truth = build(sc, _VOICES, user_voice, rng)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cached, direct=direct, standin=standin, **truth)
    mask, trace, cpu = run_gate(direct, standin, config, rng)
    metrics = score(sc, mask, truth, trace)
    if gcc and sc.name.startswith("dupe_only"):
        metrics["gcc_phat_hit"] = gcc_phat_hit_rate(direct, standin, truth["delay"])
        metrics["fdaf_erle_db"] = fdaf_erle_db(direct, standin, truth["delay"])
    return metrics, cpu


def fdaf_erle_db(direct: np.ndarray, standin: np.ndarray, delay: np.ndarray) -> float:
    """Median ERLE of a block frequency-domain NLMS canceller (the waveform alternative).

    Overlap-save, 4 partitions x 256 (64ms filter), coarse-aligned by the TRUE lag rounded to
    16ms (an ideal lock) with the echo centred in the filter. Measured over blocks where the
    stand-in carries speech. This is the "subtract the reference and look at the residual"
    design; it needs the waveform to survive and the delay to hold still.
    """
    blk, parts = 256, 4
    w = np.zeros((parts, blk + 1), complex)
    hist = np.zeros((parts, blk + 1), complex)
    pw = np.full(blk + 1, 1e-4)
    prev = np.zeros(blk)
    out = []
    floor = np.percentile(frame_levels(standin), 99) - 30
    for k in range(4, min(direct.size, standin.size) // blk - 1):
        t0 = k * blk
        lag = int(round(delay[t0] * SR / blk)) * blk - 2 * blk
        xb = direct[t0 - lag : t0 - lag + blk] if t0 - lag >= 0 else np.zeros(blk)
        hist = np.roll(hist, 1, axis=0)
        hist[0] = np.fft.rfft(np.concatenate([prev, xb]))
        prev = xb
        y = np.fft.irfft((w * hist).sum(axis=0))[blk:]
        d = standin[t0 : t0 + blk]
        e = d - y
        err = np.fft.rfft(np.concatenate([np.zeros(blk), e]))
        pw = 0.9 * pw + 0.1 * (np.abs(hist) ** 2).sum(axis=0)
        grad = 0.5 * np.conj(hist) * err / (pw + 1e-6)
        for p in range(parts):
            g = np.fft.irfft(grad[p])
            g[blk:] = 0  # gradient constraint
            w[p] += np.fft.rfft(g)
        if 10 * np.log10((d**2).mean() + 1e-12) > floor:
            out.append(10 * np.log10((d**2).sum() / ((e**2).sum() + 1e-12)))
    return float(np.median(out)) if out else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", required=True, type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--opus-level", type=float, default=0.9)
    ap.add_argument("--cache", type=Path, help="reuse built signals across runs (npz per case)")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="override a FarSideGateConfig field, e.g. --set frame_min_corr=0.7",
    )
    ap.add_argument("--only", default="", help="run only scenarios whose name contains this")
    ap.add_argument(
        "--jobs", type=int, default=1, help="parallel cases (CPU figures are only clean with 1)"
    )
    ap.add_argument(
        "--gcc",
        action="store_true",
        help="also measure the waveform alternatives (GCC-PHAT lag, NLMS ERLE)",
    )
    args = ap.parse_args()
    global OPUS_LEVEL
    OPUS_LEVEL = args.opus_level
    voices = {d.name: load_dir(d) for d in sorted(args.speech.iterdir()) if d.is_dir()}
    names = sorted(voices)
    assert len(names) >= 2, "need two speaker folders"
    a, b = names[0], names[1]
    overrides: dict[str, float] = {}
    for item in args.set:
        k, v = item.split("=", 1)
        overrides[k] = float(v)
    config = FarSideGateConfig(enabled=True, **overrides)  # type: ignore[arg-type]
    print(f"config: {config}  opus_level={OPUS_LEVEL}")

    scenarios: list[tuple[Scenario, str]] = []
    for lag in (300, 600, 900):
        scenarios.append((Scenario(f"dupe_only lag={lag}", lag_ms=lag), a))
    scenarios.append((Scenario("dupe_only fixed delay 600", wander=False), a))
    for g in (6.0, 0.0, -6.0, -12.0):
        scenarios.append((Scenario(f"crosstalk other={b} {g:+.0f}dB", other=b, other_gain_db=g), a))
        scenarios.append((Scenario(f"crosstalk other={a} {g:+.0f}dB", other=a, other_gain_db=g), b))
    scenarios.append((Scenario(f"crosstalk SAME voice {a} 0dB", other=a), a))
    scenarios.append((Scenario("meet_side_only", user_talks=False, other=b), a))
    scenarios.append((Scenario("user muted in Meet + crosstalk", dupe=False, other=b), a))
    scenarios.append((Scenario("lag step +250ms", lag_ms=450, step_ms=250), a))

    jobs = [
        (sc, user_voice, seed, args.cache, config, args.gcc, args.speech)
        for sc, user_voice in scenarios
        if not args.only or args.only in sc.name
        for seed in range(args.seeds)
    ]
    if args.jobs > 1:
        with Pool(args.jobs, initializer=_init_worker, initargs=(OPUS_LEVEL,)) as pool:
            results = pool.map(_run_case, jobs)
    else:
        _VOICES.update(voices)
        results = [_run_case(j) for j in jobs]

    rows = []
    cpu_total = audio_total = 0.0
    by_name: dict[str, dict[str, list[float]]] = {}
    for (sc, *_), (metrics, cpu) in zip(jobs, results, strict=True):
        cpu_total += cpu
        audio_total += sc.duration_s
        agg = by_name.setdefault(sc.name, {})
        for k, v in metrics.items():
            agg.setdefault(k, []).append(np.nan if v is None else float(v))
    for name, agg in by_name.items():
        rows.append(
            (
                name,
                {
                    k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else None
                    for k, v in agg.items()
                },
                {k: int(np.isnan(v).sum()) for k, v in agg.items()},
            )
        )

    for name, r, misses in rows:
        parts = []
        for k, v in r.items():
            if v is None:
                parts.append(f"{k}=never")
            elif k.endswith("_db"):
                parts.append(f"{k}={v:.1f}")
            elif k.endswith("_s"):
                parts.append(f"{k}={v:.2f}" + (f"(miss {misses[k]})" if misses[k] else ""))
            else:
                parts.append(f"{k}={100 * v:.1f}%")
        print(f"{name:40s} " + "  ".join(parts))
    # Both tracks were processed; cost per realtime second of ONE room with one reference.
    print(
        f"\nCPU: {1000 * cpu_total / audio_total:.1f} ms per realtime second "
        f"(stand-in + 1 reference track, single thread)"
    )


if __name__ == "__main__":
    main()
