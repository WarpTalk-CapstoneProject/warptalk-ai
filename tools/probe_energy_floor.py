"""What does the ingress energy floor actually reject?

_publish_speech_chunk drops any chunk whose RMS is below 0.02 — an ABSOLUTE threshold, the only
audio gate in the ingress that is always on (near_field_gate_enabled defaults to False). It is
described in one line as "skip chunks that are too quiet (noise, not speech)" and has never been
measured against what the pipeline actually assembles.

TWO THINGS MAKE IT STRICTER THAN IT LOOKS
    1. The RMS is taken over the WHOLE chunk, and a chunk is not just speech: VAD prepends
       vad_pre_speech_ms of pre-onset audio and appends vad_silence_hangover_ms of trailing near
       silence. At the defaults that is 192ms + 576ms = 768ms of padding around the speech.
    2. So a SHORT utterance is diluted much harder than a long one. vad_min_speech_ms is 288ms,
       tuned deliberately to keep "short English keywords and acknowledgements" — the very
       utterances this arithmetic pushes closest to the floor.

No API calls and no network: every number here comes from the constants in shared/config.py and
the arithmetic in _publish_speech_chunk.
"""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, ".")

from shared.config import WorkerSettings  # noqa: E402

# The floor itself, copied from stt/ingress rather than imported because it is a literal there.
ENERGY_FLOOR = 0.02

SPEECH_WAV = "artifacts/world_probe/identity.wav"

# Padding is never digital silence in a real room. Two bounds are reported: a quiet office noise
# floor, and pure zeros as the worst case, because the truth sits between them and quoting only
# one of them would be picking the answer.
PAD_NOISE_RMS = (0.005, 0.0)

UTTERANCE_MS = (288, 500, 1000, 2000, 3000)


def db(ratio: float) -> float:
    return 20.0 * np.log10(ratio) if ratio > 0 else float("-inf")


def main() -> int:
    settings = WorkerSettings()
    pre_ms = settings.vad_pre_speech_ms
    tail_ms = settings.vad_silence_hangover_ms

    speech, sr = sf.read(SPEECH_WAV)
    speech = np.asarray(speech, dtype=np.float64)
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    speech_rms = float(np.sqrt(np.mean(speech**2)))

    print(f"source              {SPEECH_WAV}")
    print(f"  sample rate       {sr} Hz")
    print(f"  speech-only RMS   {speech_rms:.4f}")
    print(f"energy floor        {ENERGY_FLOOR}  ({db(ENERGY_FLOOR):+.1f} dBFS)")
    print(
        f"padding             {pre_ms}ms pre-speech + {tail_ms}ms hangover = {pre_ms + tail_ms}ms"
    )
    print(f"min speech          {settings.vad_min_speech_ms}ms (vad_min_speech_ms)")
    print()
    print("How much a speaker may drop below this source level before the chunk is discarded.")
    print("Negative dB = quieter than the source. Bigger magnitude = more headroom = safer.")
    print()

    for pad_rms in PAD_NOISE_RMS:
        label = f"padding noise RMS {pad_rms:.3f}" if pad_rms else "padding = digital silence"
        print(f"  {label}")
        print(f"    {'utterance':>10}  {'speech share':>13}  {'headroom':>9}  {'drops below':>12}")
        for utt_ms in UTTERANCE_MS:
            pad_samples = int((pre_ms + tail_ms) * sr / 1000)
            utt_samples = int(utt_ms * sr / 1000)
            total = pad_samples + utt_samples

            # Chunk RMS at attenuation a: sqrt((utt*(a*speech_rms)^2 + pad*pad_rms^2) / total).
            # Solve for the a where that equals the floor.
            pad_energy = pad_samples * pad_rms**2
            budget = ENERGY_FLOOR**2 * total - pad_energy
            if budget <= 0:
                # Padding noise alone already clears the floor: nothing can be dropped.
                print(
                    f"    {utt_ms:>8}ms  {utt_samples / total:>12.0%}  {'n/a':>9}  "
                    f"{'never (pad alone passes)':>12}"
                )
                continue
            attenuation = float(np.sqrt(budget / (utt_samples * speech_rms**2)))
            print(
                f"    {utt_ms:>8}ms  {utt_samples / total:>12.0%}  {db(attenuation):>8.1f}dB  "
                f"{attenuation * speech_rms:>11.4f}"
            )
        print()

    print("READ IT LIKE THIS")
    print("  A 3s utterance survives ~20dB of attenuation; a 288ms one survives several dB less,")
    print("  purely because 768ms of padding is averaged in with it. The floor is not wrong, but")
    print("  it is not distance-neutral either: it is hardest on exactly the short")
    print("  acknowledgements vad_min_speech_ms was lowered to keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
