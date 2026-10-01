"""DER (pyannote.metrics) and cluster-stability metrics for every system in out/."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

ROOT = Path(__file__).parent
FRAME = 0.01


def read_rttm(path: Path) -> list[tuple[float, float, str]]:
    segs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            s, d = float(p[3]), float(p[4])
            segs.append((s, s + d, p[7]))
    return segs


def to_annotation(segs, uri) -> Annotation:
    ann = Annotation(uri=uri)
    for i, (s, e, k) in enumerate(segs):
        ann[Segment(s, e), i] = k
    return ann


def ref_frames(segs, n: int) -> tuple[np.ndarray, list[str]]:
    names = sorted({k for _, _, k in segs})
    m = np.zeros((n, len(names)), dtype=bool)
    for s, e, k in segs:
        m[int(round(s / FRAME)): int(round(e / FRAME)), names.index(k)] = True
    return m, names


def ref_turns(segs, merge_gap: float = 0.5, min_dur: float = 0.0):
    """Same-speaker segments separated by < merge_gap become one turn (what a Meet caption block covers)."""
    by = defaultdict(list)
    for s, e, k in segs:
        by[k].append((s, e))
    turns = []
    for k, lst in by.items():
        lst.sort()
        cs, ce = lst[0]
        for s, e in lst[1:]:
            if s - ce < merge_gap:
                ce = max(ce, e)
            else:
                turns.append((cs, ce, k))
                cs, ce = s, e
        turns.append((cs, ce, k))
    return sorted(t for t in turns if t[1] - t[0] >= min_dur)


def dominant_cluster(probs: np.ndarray) -> np.ndarray:
    """Per 10 ms frame: index of the most probable active cluster, -1 if none above 0.5."""
    best = probs.argmax(axis=1)
    best[probs.max(axis=1) <= 0.5] = -1
    return best


def stability(segs, probs: np.ndarray, min_turn: float = 1.0) -> dict:
    """How often a reference speaker's dominant cluster changes between consecutive turns.

    flips: count of turn-to-turn changes of the dominant cluster (turns >= min_turn s, single-speaker
    frames only). primary_share: share of the speaker's speech time spent on their most common cluster.
    """
    n = probs.shape[0]
    rf, names = ref_frames(segs, n)
    single = rf.sum(axis=1) == 1
    dom = dominant_cluster(probs)
    out_flips, out_turns, shares = 0, 0, []
    per_spk = {}
    for k in names:
        seq = []
        for s, e, kk in ref_turns(segs):
            if kk != k or e - s < min_turn:
                continue
            a, b = int(s / FRAME), min(int(e / FRAME), n)
            sel = dom[a:b][single[a:b] & (dom[a:b] >= 0)]
            if len(sel) < 0.3 / FRAME:
                continue
            seq.append(Counter(sel.tolist()).most_common(1)[0][0])
        flips = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        j = names.index(k)
        sel = dom[rf[:, j] & single & (dom >= 0)]
        share = Counter(sel.tolist()).most_common(1)[0][1] / len(sel) if len(sel) else float("nan")
        per_spk[k] = {"turns": len(seq), "flips": flips, "primary_share": round(share, 3)}
        out_flips += flips
        out_turns += max(len(seq) - 1, 0)
        shares.append(share)
    return {"flips": out_flips, "turn_transitions": out_turns,
            "flip_rate": out_flips / out_turns if out_turns else 0.0,
            "primary_share_mean": float(np.nanmean(shares)), "per_speaker": per_spk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="", help="comma list of out/<system>; default all")
    ap.add_argument("--variant", default="meet")
    ap.add_argument("--collar", type=float, default=0.25)
    args = ap.parse_args()
    systems = args.systems.split(",") if args.systems else sorted(p.name for p in (ROOT / "out").iterdir() if p.is_dir())
    results = {}
    for system in systems:
        der = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
        der0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        rows, tim = [], []
        for npz in sorted((ROOT / "out" / system).glob(f"*.{args.variant}.npz")):
            uri = npz.name.split(".")[0]
            info = json.loads(npz.with_suffix(".json").read_text())
            end = info["audio_s"]  # streaming runs may be cropped; score the same span
            ref = [(s, min(e, end), k) for s, e, k in read_rttm(ROOT / f"data/clips/{uri}.rttm") if s < end]
            hyp = read_rttm(npz.with_suffix(".rttm"))
            probs = np.load(npz)["probs"].astype(np.float32)
            r, h = to_annotation(ref, uri), to_annotation(hyp, uri)
            comp = der(r, h, detailed=True)
            der0(r, h)
            st = stability(ref, probs)
            tim.append(info)
            rows.append({"uri": uri, "der": comp["diarization error rate"],
                         "miss": comp["missed detection"] / comp["total"],
                         "fa": comp["false alarm"] / comp["total"],
                         "conf": comp["confusion"] / comp["total"],
                         "n_ref": len({k for *_, k in ref}), "n_hyp": len({k for *_, k in hyp}),
                         "flips": st["flips"], "transitions": st["turn_transitions"],
                         "primary_share": st["primary_share_mean"], "rtf": info["rtf"]})
        if not rows:
            continue
        audio = sum(t["audio_s"] for t in tim)
        results[system] = {
            "files": len(rows), "audio_min": audio / 60,
            "DER_collar": abs(der), "DER_nocollar": abs(der0),
            "flip_rate": sum(r["flips"] for r in rows) / max(1, sum(r["transitions"] for r in rows)),
            "primary_share": float(np.mean([r["primary_share"] for r in rows])),
            "rtf": sum(t["wall_s"] for t in tim) / audio,
            "cpu_s_per_audio_s": sum(t["cpu_s_per_audio_s"] * t["audio_s"] for t in tim) / audio,
            "buffer_latency_ms": tim[0].get("buffer_latency_ms"),
            "chunk_compute_ms_mean": float(np.mean([t["chunk_compute_ms_mean"] for t in tim])),
            "rows": rows,
        }
        print(f"\n== {system} ({len(rows)} files, {audio/60:.1f} min)  DER(c={args.collar})={abs(der):.3f}  DER(c=0)={abs(der0):.3f}")
        for r in rows:
            print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))
    (ROOT / "out" / f"metrics.{args.variant}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
