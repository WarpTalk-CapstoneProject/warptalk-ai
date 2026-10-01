# Spike: far-speaker diarization on the Meet loopback track

Offline prototype only. Nothing here is imported by a worker; nothing touches production.

Question: on ONE mixed track of the Meet side (Chrome per-process loopback), can
(a) streaming diarization give stable `cluster_id`s and (b) sparse Meet captions (UIA) name those
clusters well enough for live labels? Ground truth after the meeting stays Google
`transcripts.entries` (d); the roster (c) bounds the name set.

## Data (public, no recording of anyone's machine)

`prepare_data.py` pulls two non-gated Hugging Face parquet shards and writes 16 kHz mono clips +
reference RTTM to `data/clips/` (git-ignored):

* AMI IHM-mix test (EN2002b, TS3003a/b, ES2004c, IS1009a): 4 speakers, sum of close-talk mics, the
  closest public analogue of a Meet mix. First 600 s of each.
* VoxConverse test (2, 3, 4-speaker files), up to 600 s.

Each clip exists as `.clean.wav` and `.meet.wav` (80 Hz-7 kHz band-pass + real Opus round trip at
~28-37 kbit/s via libsndfile). Everything is evaluated on `.meet.wav`.

No local loopback / stand-in recordings exist in warptalk-desktop, warptalk-ai, warptalk-web or
`_worktrees/` (bench C1a used synthetic sines). The only live artefact is the UIA caption log in
`_worktrees/cap1001-desktop/.../meet-captions-watch-vi-multi.log`, used to calibrate caption cadence.

## Models

* `nvidia/Nemotron-3-Diarization` (Streaming Sortformer + AOSC, 8 speakers, not gated) through
  🤗 Transformers 5.18 on CPU (`run_nemotron.py`). Streaming modes: `ultra_low_latency` 0.32 s,
  `very_low_latency` 0.64 s, `low_latency` 1.04 s input buffer; `offline` = the model's own 30.4 s
  chunked pass (upper bound for the same model).
* NeMo-Speech.cpp 0.1.0 (`nemo-speech.exe`, GGUF Q8): rejects Nemotron-3 ("pre_ln transformer variant
  is not supported"); runs `diar_streaming_sortformer_4spk-v2` only.
* diart / pyannote: blocked — `pyannote/segmentation(-3.0)`, `pyannote/embedding`,
  `pyannote/speaker-diarization-3.1` and `-community-1` are gated (HF token + accepted terms). diart
  0.9.2 also fails to import with torchaudio >= 2.9 (`torchaudio.AudioMetaData` removed).

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe "transformers>=5.18" soundfile librosa scipy pandas pyarrow psutil pyannote.metrics pyannote.core
.venv/Scripts/python.exe prepare_data.py                     # ~1.1 GB download, writes data/clips
.venv/Scripts/python.exe run_nemotron.py --modes offline     # whole set
.venv/Scripts/python.exe run_nemotron.py --modes low_latency --uris ami00,vox03 --max-seconds 300
.venv/Scripts/python.exe metrics.py                          # DER + cluster stability -> out/metrics.meet.json
.venv/Scripts/python.exe anchor_sim.py                       # caption anchor mapping -> out/anchor_sim.meet.json
.venv/Scripts/python.exe bench_step.py low_latency 50 4      # per-chunk compute micro-benchmark
```

## Files

* `prepare_data.py` — data + Meet-like degradation.
* `run_nemotron.py` — streaming/offline inference; writes RTTM, 10 ms speaker probabilities (`.npz`) and timing (`.json`) to `out/<system>/`.
* `metrics.py` — DER (pyannote.metrics, collar 0.25 and 0) and cluster stability (turn-to-turn flips of a reference speaker's dominant cluster; primary-cluster share).
* `anchor_sim.py` — caption-anchor generator (10/25/50 %, 55 s blackout, 5 % misattributed) and the causal cluster->name mapper.
* `bench_step.py` — per-chunk latency on this CPU.
