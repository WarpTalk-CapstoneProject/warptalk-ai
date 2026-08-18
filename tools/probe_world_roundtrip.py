"""Does reshaping Cartesia's pitch cost more than it buys?

WHY THIS EXISTS
---------------
Cartesia accepts no pitch input — `GenerationConfigParam` is exactly
{emotion, speed, volume}, and the words "pitch", "F0" and "contour" appear nowhere in its
SDK. So the only way to move a dub's intonation without changing TTS provider is to reshape
the audio AFTER Cartesia returns it, with a vocoder.

That is attractive: Cartesia keeps doing what it is good at (Vietnamese, voice cloning,
~90ms time-to-first-audio, no GPU) and the supplement only touches the waveform.

It also has one risk big enough to kill the whole idea, and it is not the modification —
it is the ROUND TRIP. Analysing speech into (F0, spectral envelope, aperiodicity) and
resynthesising it is not lossless, even when nothing is changed in between. If that alone
audibly degrades Cartesia's output, then no amount of clever contour work is worth it, and
this is a one-afternoon finding rather than a discovery made after the pipeline is wired.

So this measures the null case first, deliberately, before anything else.

WHAT IT MEASURES
----------------
  identity   analyse -> synthesise, nothing changed. The cost of merely passing through.
  expand N   the same, but the F0 contour's deviation around its own mean is scaled by N.

`expand` is the operation the real feature would use, and it is the reason this approach is
safe for Vietnamese: it keeps the dub's OWN contour and scales its deviation, so the six
lexical tones keep their shapes and only the expressiveness changes. Nothing from the source
utterance is imposed, because imposing it would need a source-to-target alignment that
nothing in this pipeline produces.

Reported per case:
  mcd_db        mel-cepstral distortion vs the original. The standard objective measure of
                synthesis degradation; below ~1 dB is generally transparent, and above ~4 dB
                is usually audible.
  f0_rmse_hz    did the pitch track survive the trip
  vuv_err       fraction of frames whose voiced/unvoiced decision flipped. WORLD's own docs
                warn this is where its estimators are least reliable, and a flipped frame is
                heard as a click or a breathy patch, not as a wrong note.
  rtf           real-time factor: processing seconds per second of audio. Anything near or
                above 1.0 is unusable on a live dubbing path.

WHAT IT DOES NOT MEASURE
------------------------
Whether it SOUNDS worse. MCD is a proxy and a well-known imperfect one. The tool writes
every rendered case to disk for exactly that reason — the verdict needs ears, and the
numbers are only there to say whether ears are worth spending.

Nor does it measure the feature. It measures the FLOOR the feature would be built on.

USAGE
    uv run --with pyworld python -m tools.probe_world_roundtrip --wav dub.wav
    uv run --with pyworld python -m tools.probe_world_roundtrip --synthetic   # smoke test

`--wav` should be real Cartesia output — that is the input distribution the feature would
face. `--synthetic` generates a voiced tone-complex so the tool is runnable with no assets;
treat its numbers as a smoke test of the plumbing, never as evidence about speech.

pyworld is a `dev` dependency on purpose. Nothing here is imported by a worker, and the
production images must not grow a native vocoder for a probe.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_PERIOD = 5.0  # ms, WORLD's default

# Mel-cepstral distortion is conventionally computed over coefficients 1..24, skipping c0 —
# c0 is overall gain, which a listener hears as volume rather than as distortion.
_MCD_COEFFS = 24
_MCD_CONSTANT = 10.0 / np.log(10.0) * np.sqrt(2.0)


@dataclass(frozen=True)
class CaseResult:
    name: str
    mcd_db: float
    f0_rmse_hz: float
    vuv_err: float
    rtf: float
    path: Path | None


def _synthetic_speech(seconds: float = 3.0) -> npt.NDArray[np.float64]:
    """A voiced signal with harmonics, a moving F0 and a syllable envelope.

    Enough structure that the vocoder has something real to analyse, and honest enough about
    its own limits: it has no consonants, no noise floor and no room, so it cannot stand in
    for speech when judging quality.
    """
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 140.0 + 35.0 * np.sin(2 * np.pi * 0.4 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    signal = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    signal *= envelope
    return (signal / np.max(np.abs(signal)) * 0.5).astype(np.float64)


def _load(path: Path) -> npt.NDArray[np.float64]:
    audio, rate = sf.read(str(path), dtype="float64", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise SystemExit(
            f"{path} is {rate} Hz; this probe expects {SAMPLE_RATE} Hz — the rate the TTS "
            "worker actually requests from Cartesia. Resample before measuring, or the "
            "numbers describe the resampler."
        )
    return np.asarray(audio, dtype=np.float64)


def _analyse(audio: npt.NDArray[np.float64]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pyworld

    # harvest, not dio: slower but markedly better on voiced/unvoiced decisions, which is the
    # error mode this probe exists to catch.
    f0, timeaxis = pyworld.harvest(audio, SAMPLE_RATE, frame_period=FRAME_PERIOD)
    spectrum = pyworld.cheaptrick(audio, f0, timeaxis, SAMPLE_RATE)
    aperiodicity = pyworld.d4c(audio, f0, timeaxis, SAMPLE_RATE)
    return f0, spectrum, aperiodicity


def _mcd(reference: np.ndarray, other: np.ndarray) -> float:
    """Mel-cepstral distortion in dB between two WORLD spectral envelopes."""
    import pyworld

    ref_mc = pyworld.code_spectral_envelope(reference, SAMPLE_RATE, _MCD_COEFFS)
    oth_mc = pyworld.code_spectral_envelope(other, SAMPLE_RATE, _MCD_COEFFS)
    frames = min(len(ref_mc), len(oth_mc))
    if frames == 0:
        return float("nan")
    # c0 skipped: it is gain, heard as loudness rather than as distortion.
    diff = ref_mc[:frames, 1:] - oth_mc[:frames, 1:]
    return float(_MCD_CONSTANT * np.mean(np.sqrt(np.sum(diff * diff, axis=1))))


def _expand_contour(f0: np.ndarray, factor: float) -> np.ndarray:
    """Scale the contour's deviation around its own voiced mean.

    THIS is the operation the real feature would perform, and the reason it is safe for a
    tonal language: the dub's own contour is kept and only its deviation is scaled, so the
    six Vietnamese tones keep their shapes while the delivery becomes more or less animated.
    Nothing from the source utterance is imposed — that would require an alignment this
    pipeline does not produce.

    Unvoiced frames stay exactly 0.0. WORLD uses 0 as the unvoiced marker, and scaling it
    would invent pitch in silence.
    """
    voiced = f0 > 0
    if not np.any(voiced):
        return f0.copy()

    out = f0.copy()
    mean = float(np.mean(f0[voiced]))
    out[voiced] = np.maximum(1.0, mean + (f0[voiced] - mean) * factor)
    return out


def _run_case(
    name: str,
    audio: npt.NDArray[np.float64],
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    factor: float,
    out_dir: Path | None,
) -> CaseResult:
    import pyworld

    ref_f0, ref_spectrum, _ = reference
    f0, spectrum, aperiodicity = _analyse(audio)
    shaped = _expand_contour(f0, factor) if factor != 1.0 else f0

    started = time.perf_counter()
    rendered = pyworld.synthesize(shaped, spectrum, aperiodicity, SAMPLE_RATE, FRAME_PERIOD)
    # Analysis is the expensive half and belongs in the budget: the real feature would pay
    # both, every utterance.
    elapsed = (time.perf_counter() - started) + _analysis_seconds(audio)

    rendered = np.asarray(rendered, dtype=np.float64)
    new_f0, new_spectrum, _ = _analyse(rendered)

    frames = min(len(ref_f0), len(new_f0))
    both_voiced = (ref_f0[:frames] > 0) & (new_f0[:frames] > 0)
    f0_rmse = (
        float(np.sqrt(np.mean((ref_f0[:frames][both_voiced] - new_f0[:frames][both_voiced]) ** 2)))
        if np.any(both_voiced)
        else float("nan")
    )
    vuv_err = float(np.mean((ref_f0[:frames] > 0) != (new_f0[:frames] > 0))) if frames else 1.0

    path = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{name.replace(' ', '_')}.wav"
        sf.write(str(path), rendered.astype(np.float32), SAMPLE_RATE)

    return CaseResult(
        name=name,
        mcd_db=_mcd(ref_spectrum, new_spectrum),
        f0_rmse_hz=f0_rmse,
        vuv_err=vuv_err,
        rtf=elapsed / (len(audio) / SAMPLE_RATE),
        path=path,
    )


def _analysis_seconds(audio: npt.NDArray[np.float64]) -> float:
    started = time.perf_counter()
    _analyse(audio)
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, help="16 kHz mono WAV — ideally real Cartesia output")
    parser.add_argument("--synthetic", action="store_true", help="smoke test with no assets")
    parser.add_argument("--out", type=Path, default=Path("artifacts/world_probe"))
    args = parser.parse_args()

    if not args.wav and not args.synthetic:
        parser.error("pass --wav <file> or --synthetic")

    if args.wav:
        audio = _load(args.wav)
        source = str(args.wav)
    else:
        audio = _synthetic_speech()
        source = "SYNTHETIC — plumbing smoke test, not evidence about speech"

    reference = _analyse(audio)

    cases = [
        _run_case("identity", audio, reference, 1.0, args.out),
        _run_case("expand 1.3", audio, reference, 1.3, args.out),
        _run_case("expand 0.7", audio, reference, 0.7, args.out),
    ]

    print(f"\n  input: {source}")
    print(f"  {len(audio) / SAMPLE_RATE:.2f}s @ {SAMPLE_RATE} Hz\n")
    print(f"  {'case':<14}{'MCD dB':>9}{'F0 RMSE':>10}{'V/UV err':>10}{'RTF':>8}")
    print("  " + "-" * 51)
    for case in cases:
        print(
            f"  {case.name:<14}{case.mcd_db:>9.2f}{case.f0_rmse_hz:>10.1f}"
            f"{case.vuv_err:>10.1%}{case.rtf:>8.2f}"
        )

    identity = cases[0]
    print("\n  Latency — answered by either input:")
    if identity.rtf >= 0.5:
        print(
            f"    RTF {identity.rtf:.2f} — too slow for a live dubbing path regardless of quality."
        )
    else:
        print(f"    RTF {identity.rtf:.2f}. Comfortably inside a live path.")

    print("\n  THE ONE THAT DECIDES IT — identity MCD:")
    if args.synthetic:
        # Refusing to print a verdict here is the point. The synthetic signal is a pure
        # harmonic complex with no noise floor, so its spectral envelope between harmonics is
        # essentially -inf dB and ANY change fills those valleys — measured directly, adding
        # noise 60 dB below the signal moves MCD by ~39 dB. Real speech always has a noise
        # floor and does not behave this way. Reporting the synthetic number as a quality
        # finding would be inventing evidence.
        print("    NOT MEASURABLE on --synthetic, and the number above is not a small one to")
        print("    ignore — it is meaningless. A pure harmonic complex has no noise floor, so")
        print("    its inter-harmonic valleys sit near -inf dB and any change at all reads as")
        print("    enormous distortion. This run proved the plumbing and the latency; it")
        print("    proved nothing about quality.")
        print("\n    Re-run against a real Cartesia render to get the answer:")
        print("      uv run --with pyworld python -m tools.probe_world_roundtrip --wav dub.wav")
    else:
        print(f"    MCD {identity.mcd_db:.2f} dB. Below ~1 dB is usually transparent; above ~4 dB")
        print("    is usually audible. This is the cost of merely passing through the vocoder,")
        print("    paid on every dub before any pitch work buys anything back.")
    print("\n  Numbers are proxies. Listen to the files before believing them:")
    for case in cases:
        if case.path:
            print(f"    {case.path}")
    print()


if __name__ == "__main__":
    main()
