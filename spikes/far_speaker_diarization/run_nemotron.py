"""Run nvidia/Nemotron-3-Diarization (Streaming Sortformer + AOSC, 8 speakers) via Transformers.

Streaming mode feeds the audio chunk by chunk exactly as a live loopback would arrive, so the
per-frame speaker channel is what a realtime consumer would see at emission time (Sortformer
never relabels a frame after it is emitted). Offline mode is the model's own 30.4 s chunked
forward over the whole file and serves as the same-model upper bound.

Writes out/<system>/<uri>.<variant>.rttm, .npz (10 ms speaker probabilities) and .json (timing).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

ROOT = Path(__file__).parent
MODEL_ID = "nvidia/Nemotron-3-Diarization"
MODES = ["ultra_low_latency", "very_low_latency", "low_latency", "offline"]


def load(threads: int):
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor

    torch.set_num_threads(threads)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID).eval()
    return processor, model


def stream(processor, model, audio: np.ndarray, sr: int):
    """Returns (logits [T,8] at 10 ms, per-chunk compute seconds, per-chunk audio seconds)."""
    chunks, compute = [], []

    def gen():
        yield processor(
            audio[: processor.num_samples_first_audio_chunk],
            sampling_rate=sr,
            is_streaming=True,
            is_first_audio_chunk=True,
        )
        mel_idx = processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel_idx)
        while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
            yield processor(
                audio[start:end], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False
            )
            mel_idx += processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(mel_idx)
        yield processor(
            audio[start:],
            sampling_rate=sr,
            is_streaming=True,
            is_first_audio_chunk=False,
            is_last_audio_chunk=True,
        )

    cache = None
    with torch.inference_mode():
        for inputs in gen():
            t0 = time.perf_counter()
            out = model(**inputs, speaker_cache=cache)
            compute.append(time.perf_counter() - t0)
            chunks.append(out.logits[0])
            cache = out.speaker_cache
    return torch.cat(chunks, dim=0), compute


def offline(processor, model, audio, sr):
    with torch.inference_mode():
        inputs = processor(audio, sampling_rate=sr)
        t0 = time.perf_counter()
        logits = model(**inputs).logits[0]
        return logits, [time.perf_counter() - t0]


def probs_to_rttm(
    probs: np.ndarray, uri: str, frame_s: float, thr: float = 0.5, min_dur: float = 0.0
):
    lines = []
    active = probs > thr
    for k in range(active.shape[1]):
        a = np.concatenate([[0], active[:, k].astype(np.int8), [0]])
        d = np.diff(a)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if (e - s) * frame_s >= min_dur:
                lines.append((s * frame_s, (e - s) * frame_s, k))
    lines.sort()
    return "".join(
        f"SPEAKER {uri} 1 {s:.3f} {d:.3f} <NA> <NA> spk{k} <NA> <NA>\n" for s, d, k in lines
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--variants", default="meet")
    ap.add_argument("--uris", default="", help="comma list; default = all in manifest")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--max-seconds", type=float, default=0, help="crop audio (streaming on CPU is slow)"
    )
    args = ap.parse_args()

    manifest = json.loads((ROOT / "data/clips/manifest.json").read_text())
    uris = args.uris.split(",") if args.uris else [m["uri"] for m in manifest]
    processor, model = load(args.threads)
    proc = psutil.Process(os.getpid())
    for mode in args.modes.split(","):
        system = f"nemotron_{mode}"
        out_dir = ROOT / "out" / system
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode != "offline":
            processor.set_streaming_mode(mode)
        for uri in uris:
            for variant in args.variants.split(","):
                audio, sr = sf.read(ROOT / f"data/clips/{uri}.{variant}.wav", dtype="float32")
                if args.max_seconds:
                    audio = audio[: int(args.max_seconds * sr)]
                dur = len(audio) / sr
                cpu0 = proc.cpu_times()
                w0 = time.perf_counter()
                if mode == "offline":
                    logits, compute = offline(processor, model, audio, sr)
                else:
                    logits, compute = stream(processor, model, audio, sr)
                wall = time.perf_counter() - w0
                cpu1 = proc.cpu_times()
                cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
                probs = torch.sigmoid(logits).float().numpy()
                frame_s = 0.01
                stem = f"{uri}.{variant}"
                (out_dir / f"{stem}.rttm").write_text(probs_to_rttm(probs, uri, frame_s))
                np.savez_compressed(
                    out_dir / f"{stem}.npz", probs=probs.astype(np.float16), frame_s=frame_s
                )
                c = np.array(compute)
                info = {
                    "system": system,
                    "uri": uri,
                    "variant": variant,
                    "audio_s": dur,
                    "threads": args.threads,
                    "wall_s": wall,
                    "rtf": wall / dur,
                    "cpu_s_per_audio_s": cpu / dur,
                    "chunks": len(compute),
                    "buffer_latency_ms": None
                    if mode == "offline"
                    else processor.streaming_latency_ms,
                    "chunk_compute_ms_mean": float(c.mean() * 1000),
                    "chunk_compute_ms_p95": float(np.percentile(c, 95) * 1000),
                    "chunk_compute_ms_max": float(c.max() * 1000),
                    "chunk_step_ms": None
                    if mode == "offline"
                    else processor.num_mel_frames_per_step * 10,
                }
                (out_dir / f"{stem}.json").write_text(json.dumps(info, indent=1))
                print(
                    json.dumps(
                        {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
