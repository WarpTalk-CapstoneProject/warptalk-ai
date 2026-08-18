"""How long does a listener wait to hear the FIRST sample of a dubbed sentence?

WHY THIS PROBE EXISTS
    Two numbers in this codebase are asserted and have never been measured:
      * prosody_context.py: "Cartesia's time-to-first-audio is ~90ms"
      * synthesizer.py:     "TTFA: 40ms (Sonic Turbo)"
    Production runs `sonic-3.5`, and the pipeline's own `tts_first_audio` metric was last read
    at p50 1.00s. If the vendor really answers in ~90ms then a second is being spent somewhere
    other than generation, and swapping models would not recover it.

WHAT IS MEASURED, AND WHY BOTH
    COLD  — a fresh context: websocket connect + session setup + generation. This is what the
            FIRST sentence of every spoken turn actually pays.
    WARM  — the next sentence on that same context. This is generation alone.
    The gap between them is the handshake, and it is the difference between "change the model"
    and "open the socket earlier" as the fix. Measuring only one cannot tell them apart.

METHOD — copied from tools/probe_latency_ab.py, for the same reasons it gives:
    * Paired      every model speaks every sentence, so sentence length drops out.
    * Interleaved model order is shuffled per sentence, so API load drifting across the run
                  cannot settle on one model.
    * Warmed      one throwaway call per model before measuring, so connection setup in the
                  client library is not charged to the first cell.

TTFA is taken from the `on_pcm` tee that WT-397 added — the same callback that feeds the
LiveKit track — so this is the instant the listener could first have heard something, not the
instant the sentence finished generating.
"""

from __future__ import annotations

import asyncio
import random
import statistics
import sys
import time

sys.path.insert(0, ".")

from tts_worker.synthesizer import CartesiaSynthesizer  # noqa: E402

MODELS = ["sonic-3.5", "sonic-3", "sonic-2", "sonic-turbo"]

# Real meeting sentences, not tongue-twisters: the question is what this product's traffic
# costs. Both product languages, because Cartesia is multilingual and the dub is usually the
# non-English direction.
UTTERANCES = [
    ("en", "Let's go over the deployment plan for next week before we run out of time."),
    ("en", "I think the backend API is fine, the problem is on the client side."),
    ("en", "Can you share the document with the whole team after this meeting?"),
    ("vi", "Mình sẽ xem lại kế hoạch triển khai tuần sau trước khi hết giờ."),
    ("vi", "Phần backend thì ổn rồi, vấn đề nằm ở phía client."),
    ("vi", "Bạn gửi tài liệu cho cả nhóm sau cuộc họp này nhé."),
]

REPEATS = 2
SAMPLE_RATE = 44100


def api_key() -> str:
    for line in open(".env"):
        if line.startswith("TTS_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("TTS_API_KEY not found in .env")


async def measure(
    synth: CartesiaSynthesizer,
    language: str,
    text: str,
) -> tuple[float, float, float, int]:
    """(cold_ttfa_s, warm_ttfa_s, cold_total_s, audio_ms) for one sentence on one model."""
    first_at: list[float] = []

    async def on_pcm(_chunk: bytes) -> None:
        if not first_at:
            first_at.append(time.monotonic())

    t0 = time.monotonic()
    context, connection = await synth.open_prosody_context(
        context_id=f"probe-{random.randint(0, 1 << 30)}",
        language=language,
    )
    try:
        _audio, duration_ms = await context.speak(text, on_pcm=on_pcm)
        cold_total = time.monotonic() - t0
        cold_ttfa = (first_at[0] - t0) if first_at else float("nan")

        # Second sentence on the SAME context — generation without the handshake.
        first_at.clear()
        t1 = time.monotonic()
        await context.speak(text, on_pcm=on_pcm)
        warm_ttfa = (first_at[0] - t1) if first_at else float("nan")
        return cold_ttfa, warm_ttfa, cold_total, duration_ms
    finally:
        await context.aclose()
        try:
            await connection.close()
        except Exception:
            pass


async def main() -> None:
    key = api_key()
    results: dict[str, dict[str, list[float]]] = {
        m: {"cold": [], "warm": [], "total": [], "audio": []} for m in MODELS
    }
    synths: dict[str, CartesiaSynthesizer] = {}

    for model in MODELS:
        s = CartesiaSynthesizer(api_key=key, model=model, sample_rate=SAMPLE_RATE)
        await s.load()
        synths[model] = s
        # Warm the client: the first call of a process pays TLS + pool setup.
        try:
            await measure(s, "en", "Warming up.")
        except Exception as exc:
            print(f"  warmup failed for {model}: {str(exc)[:100]}")

    print(f"\nmeasuring {len(UTTERANCES)} sentences x {REPEATS} reps x {len(MODELS)} models\n")
    for rep in range(REPEATS):
        for language, text in UTTERANCES:
            order = MODELS[:]
            random.shuffle(order)
            for model in order:
                try:
                    cold, warm, total, audio_ms = await measure(synths[model], language, text)
                except Exception as exc:
                    print(f"  {model} failed: {str(exc)[:110]}")
                    continue
                results[model]["cold"].append(cold)
                results[model]["warm"].append(warm)
                results[model]["total"].append(total)
                results[model]["audio"].append(audio_ms / 1000.0)
        print(f"  rep {rep + 1}/{REPEATS} done")

    def q(values: list[float], p: float) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))
        return ordered[idx]

    print("\n" + "=" * 78)
    header = f"{'model':<12} {'n':>3} {'cold TTFA':>18} {'warm TTFA':>18}"
    print(header + f" {'cold total':>12} {'audio':>7}")
    print(f"{'':12} {'':>3} {'p50 / p95':>18} {'p50 / p95':>18} {'p50':>12} {'p50':>7}")
    print("-" * 78)
    for model in MODELS:
        r = results[model]
        n = len(r["cold"])
        if not n:
            print(f"{model:<12} {0:>3}  (no successful measurements)")
            continue
        print(
            f"{model:<12} {n:>3} "
            f"{q(r['cold'], 0.5):>8.3f} / {q(r['cold'], 0.95):<7.3f} "
            f"{q(r['warm'], 0.5):>8.3f} / {q(r['warm'], 0.95):<7.3f} "
            f"{q(r['total'], 0.5):>11.3f} "
            f"{statistics.median(r['audio']):>6.2f}s"
        )
    print("=" * 78)

    base = results["sonic-3.5"]
    if base["cold"]:
        print("\nhandshake cost (cold TTFA - warm TTFA), p50, per model:")
        for model in MODELS:
            r = results[model]
            if r["cold"] and r["warm"]:
                print(f"  {model:<12} {q(r['cold'], 0.5) - q(r['warm'], 0.5):+.3f}s")

    for synth in synths.values():
        close = getattr(synth, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(main())
