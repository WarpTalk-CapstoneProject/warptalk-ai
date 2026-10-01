"""Anchor-mapping simulation: can sparse Meet captions name the diarizer's clusters?

Anchors are generated from the REFERENCE turns (a Meet caption block ~= one speaker turn) and
deliberately thinned to 10 / 25 / 50 % of turns. Caption timing follows the live UIA log
(meet-captions-watch-vi-multi.log): the tree is polled ~every 0.6 s, a block appears 0.3-1.2 s
after speech onset (ASR delay) and stops growing 0.3-1.2 s after the speaker stops. An anchor is
only usable once its block stops growing (+ one poll). The "gap55" scenario additionally blanks
every anchor in a 55 s window (user switched tab / PiP -> the caption tree disappears), the same
length as the hole in the real log (09:13:29 -> 09:14:24).

Mapper (causal, mirrors the approved design):
  * an anchor (name, t0, t1) votes for the cluster with the most activity inside [t0-1.5, t1+1.5]
    s, if that cluster holds >= 50 % of the window's activity (else the anchor is discarded);
  * a cluster gets a name after >= 2 consistent votes for that name;
  * the name is kept through any gap; it is replaced only after >= 2 consecutive conflicting
    votes for one other name;
  * names are bounded by the roster (the reference speaker set).

Speech time is scored on single-speaker reference frames, labelled with the dominant emitted
cluster and the mapping state at emission time (frame time + buffer latency).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from metrics import FRAME, dominant_cluster, read_rttm, ref_frames, ref_turns

ROOT = Path(__file__).parent
CAPTION_LOG = (
    ROOT.parent.parent.parent
    / "cap1001-desktop/src/main/__tests__/fixtures/meet-captions-watch-vi-multi.log"
)


def caption_cadence(path: Path = CAPTION_LOG) -> dict:
    """Poll interval and text growth per poll, from the real Meet caption log."""
    if not path.exists():
        return {"available": False}
    ts, lens = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d\.\d+) ", line)
        if m:
            h, mi, s = m.groups()
            ts.append(int(h) * 3600 + int(mi) * 60 + float(s))
            lens.append(len(line))
    d = np.diff(ts)
    growth = np.diff(lens)
    short = d[d < 5]
    return {
        "available": True,
        "polls": len(ts),
        "poll_s_median": float(np.median(short)),
        "poll_s_p90": float(np.percentile(short, 90)),
        "longest_gap_s": float(d.max()),
        "chars_per_poll_median": float(np.median(growth[d < 5])),
        "note": "full text is re-emitted every poll; earlier blocks get rewritten",
    }


def make_anchors(
    turns,
    density: float,
    rng: np.random.Generator,
    poll: float,
    gap=None,
    wrong_rate=0.0,
    roster=None,
):
    anchors = []
    for s, e, k in turns:
        if e - s < 0.6 or rng.random() > density:  # sub-0.6 s turns rarely get their own block
            continue
        t0 = s + rng.uniform(0.3, 1.2)
        t1 = e + rng.uniform(0.3, 1.2)
        avail = np.ceil(t1 / poll) * poll + poll
        if gap and (gap[0] <= t0 <= gap[1] or gap[0] <= avail <= gap[1]):
            continue
        name = k
        if wrong_rate and rng.random() < wrong_rate:
            name = rng.choice([r for r in roster if r != k])
        # the consumer only sees poll times; it back-dates the block by a fixed 0.75 s
        # ASR-delay guess
        anchors.append((avail, t0 - 0.75, t1 - 0.75, name))
    anchors.sort()
    return anchors


class Mapper:
    def __init__(
        self, window_pad=1.5, min_votes=2, min_conflicts=2, min_share=0.5, min_act=0.3, tuned=False
    ):
        """tuned=False is the rule set exactly as briefed. tuned=True adds four cheap guards that
        only use what a live consumer has: ignore caption blocks shorter than 1 s, weight activity
        inside the caption span 2:1 over the +-1.5 s tolerance, need a 60 % share, and require
        the winning name to lead the runner-up by >= 1 vote before the first assignment."""
        self.pad, self.min_votes, self.min_conf = window_pad, min_votes, min_conflicts
        self.min_share, self.min_act = (0.6 if tuned else min_share), min_act
        self.tuned = tuned
        self.votes = defaultdict(lambda: defaultdict(int))
        self.name = {}
        self.conflict = {}  # cluster -> (name, count)
        self.used = self.discarded = 0

    def feed(self, dom: np.ndarray, t0: float, t1: float, name: str):
        if self.tuned and t1 - t0 < 1.0:
            self.discarded += 1
            return
        a, b = max(0, int((t0 - self.pad) / FRAME)), min(len(dom), int((t1 + self.pad) / FRAME))
        seg = dom[a:b]
        w = np.ones(len(seg))
        if self.tuned:  # frames inside the caption span count double
            ia, ib = int(t0 / FRAME) - a, int(t1 / FRAME) - a
            w[max(0, ia) : max(0, ib)] = 2.0
        keep = seg >= 0
        seg, w = seg[keep], w[keep]
        if len(seg) * FRAME < self.min_act:
            self.discarded += 1
            return
        vals = np.unique(seg)
        cnt = np.array([w[seg == v].sum() for v in vals])
        c = int(vals[cnt.argmax()])
        if cnt.max() / cnt.sum() < self.min_share:
            self.discarded += 1
            return
        self.used += 1
        self.votes[c][name] += 1
        cur = self.name.get(c)
        if cur is None:
            ranked = sorted(self.votes[c].values(), reverse=True)
            lead = ranked[0] - (ranked[1] if len(ranked) > 1 else 0)
            if (
                self.votes[c][name] >= self.min_votes
                and self.votes[c][name] == ranked[0]
                and (not self.tuned or lead >= 1)
            ):
                self.name[c] = name
        elif name == cur:
            self.conflict.pop(c, None)
        else:
            prev, n = self.conflict.get(c, (name, 0))
            n = n + 1 if prev == name else 1
            self.conflict[c] = (name, n)
            if n >= self.min_conf:
                self.name[c] = name
                self.conflict.pop(c, None)


def simulate(ref, probs, anchors, latency_s: float, gap=None, tuned=False):
    n = probs.shape[0]
    rf, names = ref_frames(ref, n)
    dom = dominant_cluster(probs)
    single = rf.sum(axis=1) == 1
    who = np.where(single, rf.argmax(axis=1), -1)
    mapper = Mapper(tuned=tuned)
    labels = np.full(n, "", dtype=object)
    ai = 0
    # walk in 1 s steps; the frames emitted in that step are labelled with the state at emission
    step = int(1.0 / FRAME)
    lat = int(latency_s / FRAME)
    for a in range(0, n, step):
        b = min(n, a + step)
        now = b * FRAME + latency_s
        while ai < len(anchors) and anchors[ai][0] <= now:
            _, t0, t1, nm = anchors[ai]
            # only the part of the cluster stream already emitted is visible to the mapper
            visible = dom[: max(0, int(now / FRAME) - lat)] if lat else dom[: int(now / FRAME)]
            mapper.feed(visible, t0, t1, nm)
            ai += 1
        for f in range(a, b):
            c = dom[f]
            labels[f] = "" if c < 0 else mapper.name.get(int(c), f"Speaker {c + 1}")
    scored = who >= 0
    tot = scored.sum()
    correct = unnamed = wrong = missed = 0
    first = {}
    first_speech = {}
    gap_tot = gap_ok = 0
    for f in np.where(scored)[0]:
        r = names[who[f]]
        first_speech.setdefault(r, f * FRAME)
        lab = labels[f]
        ok = lab == r
        if lab == "":
            missed += 1
        elif ok:
            correct += 1
            first.setdefault(r, f * FRAME)
        elif lab.startswith("Speaker "):
            unnamed += 1
        else:
            wrong += 1
        if gap and gap[0] <= f * FRAME <= gap[1]:
            gap_tot += 1
            gap_ok += ok
    ttfn = {r: (first[r] - first_speech[r]) if r in first else None for r in names}
    return {
        "correct": correct / tot,
        "speaker_n": unnamed / tot,
        "wrong": wrong / tot,
        "no_cluster": missed / tot,
        "ttfn": ttfn,
        "named_speakers": sum(v is not None for v in ttfn.values()),
        "speakers": len(names),
        "anchors": len(anchors),
        "anchors_used": mapper.used,
        "anchors_discarded": mapper.discarded,
        "gap_correct": (gap_ok / gap_tot) if gap_tot else None,
    }


SCENARIOS = {
    "p10": dict(density=0.10),
    "p100": dict(density=1.00),
    "p25": dict(density=0.25),
    "p50": dict(density=0.50),
    "p25_gap55": dict(density=0.25, gap=True),
    "p25_wrong5": dict(density=0.25, wrong_rate=0.05),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="")
    ap.add_argument("--variant", default="meet")
    ap.add_argument("--uris", default="", help="only these clips (paired comparison)")
    ap.add_argument(
        "--crop", type=float, default=0, help="score only the first N seconds (paired comparison)"
    )
    ap.add_argument("--tag", default="", help="suffix for the output json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gap-start", type=float, default=150.0)
    ap.add_argument(
        "--compute-s", type=float, default=0.3, help="assumed per-chunk compute on the target host"
    )
    ap.add_argument(
        "--tuned", action="store_true", help="mapper with the extra guards (see Mapper)"
    )
    args = ap.parse_args()
    cad = caption_cadence()
    print("caption cadence:", cad)
    # 0.66 s = median poll of the real log, so results reproduce without the (local-only) log
    poll = cad.get("poll_s_median", 0.66) if cad.get("available") else 0.66
    systems = (
        args.systems.split(",")
        if args.systems
        else sorted(p.name for p in (ROOT / "out").iterdir() if p.is_dir())
    )
    out = {"caption_cadence": cad}
    for system in systems:
        npzs = [
            p
            for p in sorted((ROOT / "out" / system).glob(f"*.{args.variant}.npz"))
            if not args.uris or p.name.split(".")[0] in args.uris.split(",")
        ]
        if not npzs:
            continue
        out[system] = {}
        for sc, cfg in SCENARIOS.items():
            agg = defaultdict(list)
            ttfns = []
            for npz in npzs:
                uri = npz.name.split(".")[0]
                info = json.loads(npz.with_suffix(".json").read_text())
                end = min(info["audio_s"], args.crop or 1e9)
                ref = [
                    (s, min(e, end), k)
                    for s, e, k in read_rttm(ROOT / f"data/clips/{uri}.rttm")
                    if s < end
                ]
                probs = np.load(npz)["probs"].astype(np.float32)[: int(round(end / FRAME))]
                # emission latency = input buffer + an assumed realtime-capable compute budget. The
                # compute measured on this (overloaded, CPU-only) laptop is far beyond realtime and
                # would only measure the laptop; see the timing table instead.
                lat = (info.get("buffer_latency_ms") or 0) / 1000 + (
                    args.compute_s if info.get("buffer_latency_ms") else 0
                )
                turns = ref_turns(ref)
                roster = sorted({k for *_, k in ref})
                gap = (args.gap_start, args.gap_start + 55.0) if cfg.get("gap") else None
                for seed in range(args.seeds):
                    rng = np.random.default_rng(seed)
                    anchors = make_anchors(
                        turns,
                        cfg["density"],
                        rng,
                        poll,
                        gap=gap,
                        wrong_rate=cfg.get("wrong_rate", 0.0),
                        roster=roster,
                    )
                    r = simulate(ref, probs, anchors, lat, gap, tuned=args.tuned)
                    for k in ("correct", "speaker_n", "wrong", "no_cluster", "anchors"):
                        agg[k].append(r[k])
                    agg["named_frac"].append(r["named_speakers"] / r["speakers"])
                    if r["gap_correct"] is not None:
                        agg["gap_correct"].append(r["gap_correct"])
                    ttfns += [v for v in r["ttfn"].values() if v is not None]
            res = {k: float(np.mean(v)) for k, v in agg.items()}
            res["ttfn_median_s"] = float(np.median(ttfns)) if ttfns else None
            res["ttfn_p90_s"] = float(np.percentile(ttfns, 90)) if ttfns else None
            out[system][sc] = res
            print(
                f"{system:28s} {sc:11s} "
                + " ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()
                )
            )
    (
        ROOT / "out" / f"anchor_sim.{args.variant}{'.tuned' if args.tuned else ''}{args.tag}.json"
    ).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
