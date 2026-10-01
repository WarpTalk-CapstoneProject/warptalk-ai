"""Build the evaluation set from public data (no WarpTalk audio, no recording).

Sources (Hugging Face, not gated):
  diarizers-community/ami          ihm/test-00000-of-00003.parquet  -> AMI IHM-mix (sum of close-talk mics)
  diarizers-community/voxconverse  data/test-00000-of-00011.parquet -> VoxConverse test (2-4 speaker files)

AMI IHM-mix is the closest public analogue of a Meet loopback track: every participant has
their own mic, and the far end hears the sum. Each clip is also passed through a real Opus
round trip (libsndfile OGG/OPUS, ~24 kbit/s VBR) plus 80 Hz-7 kHz band-limiting to mimic what
Chrome actually plays out of a Meet call ("meet" variant).

Output: data/clips/<clip_id>.<variant>.wav (16 kHz mono) and data/clips/<clip_id>.rttm
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import butter, sosfilt

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CLIPS = DATA / "clips"
SR = 16000

SHARDS = {
    "ami": ("ami_ihm_test0.parquet",
            "https://huggingface.co/datasets/diarizers-community/ami/resolve/main/ihm/test-00000-of-00003.parquet"),
    "vox": ("vox_test0.parquet",
            "https://huggingface.co/datasets/diarizers-community/voxconverse/resolve/main/data/test-00000-of-00011.parquet"),
}


def ensure_shard(name: str) -> Path:
    fname, url = SHARDS[name]
    p = DATA / fname
    if not p.exists():
        import urllib.request
        DATA.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, p)
    return p


def decode(audio_bytes: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return x


def meetify(x: np.ndarray) -> np.ndarray:
    """Band-limit + real Opus round trip (libsndfile >= 1.1 ships an Opus encoder)."""
    sos = butter(4, [80, 7000], btype="bandpass", fs=SR, output="sos")
    y = sosfilt(sos, x).astype(np.float32)
    peak = float(np.max(np.abs(y)) or 1.0)
    y = 0.7 * y / peak
    buf = io.BytesIO()
    # compression_level: 0 = highest quality, 1 = lowest; with libsndfile 1.2.2, 0.88 ~= 24-28 kbit/s for 16 kHz speech.
    sf.write(buf, y, SR, format="OGG", subtype="OPUS", compression_level=0.88)
    size = buf.tell()
    buf.seek(0)
    z, sr = sf.read(buf, dtype="float32")
    assert sr == SR
    kbps = size * 8 / (len(y) / SR) / 1000
    return z[: len(x)], kbps


def write_rttm(path: Path, uri: str, starts, ends, spks, max_s: float):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s, e, k in sorted(zip(starts, ends, spks)):
            if s >= max_s:
                continue
            e = min(e, max_s)
            if e - s <= 0.01:
                continue
            f.write(f"SPEAKER {uri} 1 {s:.3f} {e - s:.3f} <NA> <NA> {k} <NA> <NA>\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=600.0, help="crop each recording to this length")
    ap.add_argument("--ami-rows", default="0,1,2,4,5", help="row indices in the AMI IHM shard (one per meeting)")
    ap.add_argument("--vox-rows", default="3,2,16,18", help="row indices in the VoxConverse shard (2-4 speakers)")
    args = ap.parse_args()
    CLIPS.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, rows in (("ami", args.ami_rows), ("vox", args.vox_rows)):
        rows = [int(r) for r in rows.split(",") if r != ""]
        table = pq.read_table(ensure_shard(name))
        for r in rows:
            rec = table.slice(r, 1).to_pylist()[0]
            x = decode(rec["audio"]["bytes"])
            x = x[: int(args.max_seconds * SR)]
            dur = len(x) / SR
            uri = f"{name}{r:02d}"
            nseg = write_rttm(CLIPS / f"{uri}.rttm", uri, rec["timestamps_start"], rec["timestamps_end"],
                              rec["speakers"], dur)
            sf.write(CLIPS / f"{uri}.clean.wav", x, SR)
            y, kbps = meetify(x)
            sf.write(CLIPS / f"{uri}.meet.wav", y, SR)
            spk = sorted({k for s, k in zip(rec["timestamps_start"], rec["speakers"]) if s < dur})
            item = {"uri": uri, "source": rec["audio"].get("path"), "duration": round(dur, 1),
                    "num_speakers": len(spk), "segments": nseg, "opus_kbps": round(kbps, 1)}
            print(item)
            manifest.append(item)
    (CLIPS / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
