# Spike: far-speaker diarization on the Meet loopback track

Offline prototype only. Nothing here is imported by a worker, the Dockerfile does not copy
`spikes/`, mypy does not check it and pytest does not collect it; nothing touches production.
The scripts only have to pass `ruff format --check` / `ruff check` (CI lints the whole repo).

## Why

Meet captions read through UIA lose the caption tree whenever the user switches tab or Meet goes
PiP (a 55 s hole in the real log). The proposed hybrid is: diarize the single Meet-side loopback
track into stable `Speaker N` clusters, name those clusters from whatever caption blocks we do see
(anchors), and bound names by the roster. After the meeting, Google `transcripts.entries` stays the
ground truth.

Questions this spike answers, offline:

1. On ONE mixed, Meet-degraded track, does a streaming diarizer give clusters that are accurate
   (DER) and stable (a speaker keeps the same cluster across turns)?
2. Does streaming (live) cost much accuracy compared with the same model's offline pass?
3. Given sparse captions (10/25/50/100 % of turns), a 55 s caption blackout or 5 % misattributed
   captions, how much speech gets the right name, how much a wrong name, and how fast?
4. Can it run in real time on a CPU?

## Data (public, no recording of anyone's machine)

`prepare_data.py` pulls three non-gated Hugging Face parquet shards (~1.1 GB) and writes 16 kHz
mono clips + reference RTTM to `data/clips/` (git-ignored):

* AMI IHM-mix test (EN2002b, TS3003a, ES2004c, TS3003b, IS1009a -> `ami00-02, ami04, ami05`): 4 speakers, sum of
  close-talk mics, the closest public analogue of a Meet mix. First 600 s of each.
* VoxConverse test (`vox02`, `vox03`, `vox16`, `vox18`; 2-4 speakers), up to 600 s.

Each clip exists as `.clean.wav` and `.meet.wav` (80 Hz-7 kHz band-pass + a real Opus round trip at
~27-37 kbit/s via libsndfile). Everything below is scored on `.meet.wav`.

No real loopback / stand-in recording exists in any WarpTalk repo (bench C1a used synthetic
sines). The only live artefact used is the UIA caption log
`_worktrees/cap1001-desktop/src/main/__tests__/fixtures/meet-captions-watch-vi-multi.log`, read by
`anchor_sim.py` for caption cadence only (median poll 0.66 s, p90 1.26 s, longest gap 54.7 s); its
text is never written out. Without that worktree the script falls back to the same 0.66 s poll.

## Models

* `nvidia/Nemotron-3-Diarization` (Streaming Sortformer + AOSC, up to 8 speakers, not gated)
  through Transformers >= 5.18 on CPU (`run_nemotron.py`). Streaming modes: `ultra_low_latency`
  0.32 s, `very_low_latency` 0.64 s, `low_latency` 1.04 s input buffer; `offline` = the model's own
  chunked pass over the whole file (non-causal upper bound for the same model).
* NeMo-Speech.cpp 0.1.0 (GGUF Q8): rejects Nemotron-3 ("pre_ln transformer variant is not
  supported"); only runs `diar_streaming_sortformer_4spk-v2`. Not pursued.
* diart / pyannote: blocked. All pyannote segmentation/embedding/pipeline checkpoints are gated (HF
  token + accepted terms), and diart 0.9.2 fails to import with torchaudio >= 2.9.

## Run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe "transformers>=5.18" soundfile librosa scipy pandas pyarrow psutil pyannote.metrics pyannote.core

# 1. data (~1.1 GB download) -> data/clips
.venv/Scripts/python.exe prepare_data.py
# 2. inference -> out/<system>/{uri}.meet.{rttm,npz,json}   (git-ignored)
.venv/Scripts/python.exe run_nemotron.py --modes offline                     # ~1.5 h on CPU
.venv/Scripts/python.exe run_nemotron.py --modes low_latency --uris ami00,vox16 --max-seconds 300   # ~2.5 h
# 3. scoring (minutes) -> out/*.json   (committed)
.venv/Scripts/python.exe metrics.py --systems nemotron_offline
.venv/Scripts/python.exe metrics.py --uris ami00,vox16 --crop 300 --tag .paired300
.venv/Scripts/python.exe anchor_sim.py --systems nemotron_offline
.venv/Scripts/python.exe anchor_sim.py --systems nemotron_offline --tuned
.venv/Scripts/python.exe anchor_sim.py --uris ami00,vox16 --crop 300 --tag .paired300
.venv/Scripts/python.exe anchor_sim.py --uris ami00,vox16 --crop 300 --tag .paired300 --tuned
# optional: per-chunk compute micro-benchmark
.venv/Scripts/python.exe bench_step.py low_latency 50 4
```

Steps 1-2 need the downloads and hours of CPU; their outputs (audio, probabilities, RTTM, logs) are
git-ignored. The committed `out/*.json` were regenerated from those outputs with the code in this
commit (step 3).

| Output | What |
| --- | --- |
| `out/metrics.meet.json` | DER + stability, offline, all 9 clips (85.3 min) |
| `out/metrics.meet.paired300.json` | offline vs `low_latency`, same 2 clips, first 300 s each |
| `out/anchor_sim.meet.json` / `.tuned.json` | anchor naming, offline clusters, 9 clips, baseline / tuned mapper |
| `out/anchor_sim.meet.paired300.json` / `.tuned.paired300.json` | anchor naming, paired subset, both systems |

## Method

**Diarization.** DER via pyannote.metrics, overlap scored, collar 0.25 s (and 0). Stability: for
each reference speaker, the dominant cluster of each turn >= 1 s; `flip_rate` = turn-to-turn
changes / transitions; `primary_share` = share of the speaker's single-speaker time on their most
common cluster.

**Anchors** (`anchor_sim.py`). One caption block ~= one reference turn (same-speaker segments
< 0.5 s apart merged), kept with probability 10/25/50/100 %. Block timing follows the UIA log: it
starts 0.3-1.2 s after speech, stops growing 0.3-1.2 s after it, is usable one poll after it stops,
and the consumer back-dates it by a fixed 0.75 s. `gap55` blanks every anchor in 150-205 s;
`wrong5` gives 5 % of blocks a wrong roster name. 5 seeds per scenario.

Mapper (causal): an anchor votes for the cluster with the most activity in its span +-1.5 s if that
cluster holds >= 50 % of it; a cluster is named after >= 2 votes for the leading name; the name
sticks through gaps and is replaced only after 2 consecutive conflicting votes. **Tuned** adds four
guards that only use what a live consumer has: drop blocks < 1 s, weight frames inside the span 2:1
over the tolerance, need a 60 % share, and need a strict lead over the runner-up name before the
first assignment.

Each 10 ms single-speaker reference frame is labelled with the cluster and mapping state at
emission time (frame time + latency; for streaming 1.04 s buffer + an assumed 0.3 s compute, since
the measured laptop compute is not real time, see below) and scored as **correct** name, **wrong**
name, **Speaker N** (cluster known, no name yet) or **no cluster** (model saw no speech).
**TTFN** = time from a speaker's first speech to their first correctly named frame.

## Results

### Diarization (offline, 9 clips, 85.3 min) — `out/metrics.meet.json`

| | DER c=0.25 | DER c=0 | flip rate | primary share |
| --- | --- | --- | --- | --- |
| all | **16.8 %** | 18.9 % | 2.0 % | 98.4 % |
| VoxConverse (4 clips) | 1.0-4.1 % | | | |
| AMI IHM-mix (5 clips) | 21-36 % | | | |

AMI DER is almost entirely **missed speech** (17-35 %): quiet back-channels and overlap in the mic
sum. Speaker **confusion is < 0.5 %** on 8 of 9 clips (4.4 % on ami05, which got a 5th cluster).
The cluster count was right on 7 of 9 clips.

### Streaming vs offline (paired: ami00 + vox16, first 300 s) — `out/metrics.meet.paired300.json`

| | DER c=0.25 | DER c=0 | confusion ami00 / vox16 | primary share | CPU RTF (4 threads) |
| --- | --- | --- | --- | --- | --- |
| offline | 19.0 % | 21.7 % | 0.6 % / 0.0 % | 99.1 % | 0.92 |
| `low_latency` (1.04 s) | **20.0 %** | 22.7 % | 1.3 % / 1.5 % | 91.9 % | **15.2** (11.7-18.7) |

Streaming costs ~1 DER point and some cluster purity (vox16 primary share 86 %), no turn flips.
Compute is the blocker: on this CPU-only laptop a 0.72 s streaming step took 8-13 s on average
(RTF 12-19, also inflated by the machine being loaded). The offline pass is ~real time on CPU.

### Naming clusters from captions — speech share, offline clusters, 9 clips

| scenario | correct | wrong name | Speaker N | no cluster | TTFN median |
| --- | --- | --- | --- | --- | --- |
| 10 % of turns | 16.0 % / 16.0 % | 2.6 % / **0.8 %** | 68.2 % / 70.0 % | 13.2 % | 279 s / 290 s |
| 25 % | 39.3 % / 39.9 % | 5.7 % / **2.2 %** | 41.8 % / 44.7 % | 13.2 % | 193 s / 198 s |
| 50 % | 57.3 % / 58.3 % | 4.0 % / **1.3 %** | 25.5 % / 27.1 % | 13.2 % | 145 s / 145 s |
| 100 % | 71.2 % / **72.8 %** | 4.9 % / **1.5 %** | 10.7 % / 12.5 % | 13.2 % | 85 s / 85 s |
| 25 % + 55 s blackout | 31.5 % / 31.9 % | 6.0 % / **1.9 %** | 49.3 % / 52.9 % | 13.2 % | 232 s / 268 s |
| 25 % + 5 % wrong captions | 39.7 % / 39.9 % | 6.4 % / **1.7 %** | 40.7 % / 45.1 % | 13.2 % | 203 s / 215 s |

(baseline / tuned; `out/anchor_sim.meet.json`, `out/anchor_sim.meet.tuned.json`)

Paired subset (300 s, `*.paired300.json`), baseline / tuned:

| scenario | `low_latency` correct | `low_latency` wrong | offline correct | offline wrong |
| --- | --- | --- | --- | --- |
| 25 % | 25.2 % / 25.9 % | 6.7 % / 3.8 % | 25.3 % / 25.8 % | 7.0 % / 0.1 % |
| 50 % | 44.9 % / 43.3 % | 0.9 % / 0.9 % | 42.8 % / 43.3 % | 1.6 % / 0.1 % |
| 100 % | 63.0 % / 61.2 % | 5.0 % / 2.3 % | 59.4 % / 61.4 % | 3.4 % / 0.2 % |
| 25 % + 55 s blackout | 17.2 % / 17.3 % | 6.4 % / 3.4 % | 17.7 % / 17.1 % | 7.0 % / 0.0 % |

## Conclusions

1. **Clusters are good enough to carry names.** Confusion is near zero and a speaker's turns stay
   on one cluster (flip rate 2 %, primary share 98 %). What DER loses on AMI is missed quiet speech,
   which also shows up as the constant ~13 % "no cluster" share; it does not mislabel people.
2. **Streaming is almost as accurate as offline** (DER 20.0 % vs 19.0 % on the same 10 min), but
   **not real time on a client CPU** (RTF 12-19 here). Live diarization needs a GPU worker (or a
   much lighter model); it does not belong on the desktop app's CPU.
3. **Captions are the bottleneck, not the diarizer.** Correctly named speech scales with caption
   density: ~40 % at 25 % of turns, ~72 % when every turn yields a block. A name needs >= 2 votes,
   so the first correct name arrives after a median of 1.5-5 minutes; until then speech shows as
   `Speaker N`.
4. **A blackout costs only the anchors it hides.** Names already assigned survive the 55 s gap
   (sticky mapping); the 25 % scenario drops from 39 % to 32 % correct because fewer anchors arrive,
   not because names are lost.
5. **The tuned mapper cuts wrong names ~3x** (e.g. 4.9 % -> 1.5 % at 100 %, 5.7 % -> 2.2 % at 25 %,
   6.4 % -> 1.7 % with 5 % bad captions) at equal or better correct share; the price is a later
   first name in some scenarios (median TTFN +0-36 s) and slightly more `Speaker N`.

## Recommendation

* Keep the hybrid, but frame it as **`Speaker N` first, name when proven**: live labels start
  anonymous and switch to roster names once captions agree; never show a guess.
* Use the **tuned mapper** rules (min block 1 s, in-span weighting, 60 % share, strict lead).
* Feed it **every** caption block we can read (density is the main lever), and keep the mapping
  sticky across PiP / tab-switch gaps.
* Run diarization **server-side** (AI worker with GPU, Nemotron-3 `low_latency`), not on the
  desktop CPU. Post-meeting, keep Google `transcripts.entries` as truth; the offline pass (~real
  time on CPU) can relabel the stored transcript.
* Before building: (a) measure GPU per-chunk latency for `low_latency`; (b) record one real Meet
  loopback session with known speakers (consented) to replace the AMI/VoxConverse proxy; (c) test
  whether caption speaker names from UIA are as reliable as the 5 % error assumed here.

## Files

* `prepare_data.py` — data + Meet-like degradation.
* `run_nemotron.py` — streaming/offline inference; writes RTTM, 10 ms speaker probabilities
  (`.npz`) and timing (`.json`) to `out/<system>/`.
* `metrics.py` — DER and cluster stability; `--uris/--crop/--tag` for paired comparisons.
* `anchor_sim.py` — caption-anchor generator and the causal cluster->name mapper (`--tuned`).
* `bench_step.py` — per-chunk latency micro-benchmark on this CPU.
