"""Micro-benchmark: per-chunk compute of Nemotron-3-Diarization streaming on this CPU.

Usage: bench_step.py <mode> <seconds> <threads> [int8]
"""

import sys
import time

import numpy as np
import soundfile as sf
import torch
from run_nemotron import load


def main() -> None:
    mode, secs, threads = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    quant = len(sys.argv) > 4 and sys.argv[4] == "int8"
    p, m = load(threads)
    if quant:
        m = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    p.set_streaming_mode(mode)
    a, sr = sf.read("data/clips/vox02.meet.wav", dtype="float32")
    a = a[: int(secs * sr)]
    first = a[: p.num_samples_first_audio_chunk]
    ins = [p(first, sampling_rate=sr, is_streaming=True, is_first_audio_chunk=True)]
    mi = p.num_mel_frames_per_step
    st = p.audio_chunk_start(mi)
    while st + p.num_samples_per_audio_chunk <= len(a):
        chunk = a[st : st + p.num_samples_per_audio_chunk]
        ins.append(p(chunk, sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False))
        mi += p.num_mel_frames_per_step
        st = p.audio_chunk_start(mi)
    cache = None
    ts = []
    with torch.inference_mode():
        for x in ins:
            t = time.perf_counter()
            o = m(**x, speaker_cache=cache)
            ts.append(time.perf_counter() - t)
            cache = o.speaker_cache
    ms = np.array(ts) * 1000
    step = p.num_mel_frames_per_step * 10
    print(
        f"mode={mode} threads={threads} int8={quant} steps={len(ms)} step_ms={step} "
        f"first10={ms[:10].mean():.0f} last10={ms[-10:].mean():.0f} "
        f"rtf_steady={ms[-10:].mean() / step:.2f}"
    )


if __name__ == "__main__":
    main()
