"""Self-contained E2E latency test: Translation → TTS pipeline.

Directly calls OpenAI Translator + Cartesia Synthesizer to measure
real-world latency without Redis worker infrastructure.

Usage:
    cd warptalk-ai
    .venv/bin/python tests/test_e2e_latency.py
"""

import asyncio
import os
import sys
import time

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.text_utils import split_into_sentences
from translation_worker.translator import OpenAITranslator
from tts_worker.synthesizer import CartesiaSynthesizer

# ── Test sentences ───────────────────────────────────────────
TEST_CASES = [
    {
        "text": "Hello everyone, welcome to the meeting today.",
        "language": "en",
        "target": "vi",
        "description": "Short (EN→VI)",
    },
    {
        "text": "We need to discuss the quarterly results and plan for next quarter.",
        "language": "en",
        "target": "vi",
        "description": "Medium (EN→VI)",
    },
    {
        "text": (
            "The new feature implementation is going well. We expect to finish by "
            "the end of the week and start user testing next Monday."
        ),
        "language": "en",
        "target": "vi",
        "description": "Long (EN→VI)",
    },
    {
        "text": "Can you share the latest version of the design document?",
        "language": "en",
        "target": "vi",
        "description": "Question (EN→VI)",
    },
    {
        "text": "I agree with your proposal. Let's move forward with it.",
        "language": "en",
        "target": "vi",
        "description": "Agreement (EN→VI)",
    },
]


async def main():
    print("=" * 72)
    print("WarpTalk E2E Latency Test — Translation + TTS Pipeline")
    print("  Translator: OpenAI gpt-4.1-mini")
    print("  TTS Engine: Cartesia Sonic Turbo")
    print("=" * 72)
    print()

    # ── 1. Load models ────────────────────────────────────────
    openai_api_key = os.getenv("TRANSLATION_API_KEY") or os.getenv("OPENAI_API_KEY")
    cartesia_api_key = os.getenv("TTS_API_KEY") or os.getenv("CARTESIA_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY or TRANSLATION_API_KEY is required")
    if not cartesia_api_key:
        raise RuntimeError("TTS_API_KEY or CARTESIA_API_KEY is required")

    print("📦 Initializing OpenAI translator...")
    t0 = time.perf_counter()

    translator = OpenAITranslator(
        api_key=openai_api_key,
        model="gpt-4.1-mini",
        max_tokens=512,
        temperature=0.1,
    )
    await translator.load()
    load_time = time.perf_counter() - t0
    print(f"   ✅ Translator ready in {load_time:.1f}s")
    print()

    print("🔊 Initializing Cartesia...")
    synthesizer = CartesiaSynthesizer(
        api_key=cartesia_api_key,
        model="sonic-turbo",
        sample_rate=44100,
    )
    await synthesizer.load()
    print("   ✅ Cartesia ready")
    print()

    # ── 2. Run test cases ─────────────────────────────────────
    results = []
    output_dir = os.path.join(os.path.dirname(__file__), "e2e_output")
    os.makedirs(output_dir, exist_ok=True)

    for i, case in enumerate(TEST_CASES):
        print(f"─── Test {i+1}/{len(TEST_CASES)}: {case['description']} ───")
        print(f"  Input: \"{case['text']}\"")

        # ── 1. Text Chunking (Simulate STT -> Translation Pipelining) ──
        sentences = split_into_sentences(case["text"])

        t_start = time.perf_counter()

        # Only measure latency to the FIRST chunk (Time To First Byte)
        # In a real pipelined system, subsequent chunks process in background
        first_sentence = sentences[0] if sentences else ""
        print(f"  First Chunk: \"{first_sentence}\"")

        # ── Translation ──
        translated = await translator.translate(
            first_sentence, case["language"], case["target"]
        )
        t_translate = time.perf_counter()
        translate_ms = (t_translate - t_start) * 1000

        print(f"  Translated: \"{translated}\"")
        print(f"  ⏱ Translation (Chunk 1): {translate_ms:.0f}ms")

        # ── TTS ──
        audio_bytes, duration_ms = await synthesizer.synthesize(
            text=translated,
            language=case["target"],
        )
        t_tts = time.perf_counter()
        tts_ms = (t_tts - t_translate) * 1000
        total_ms = (t_tts - t_start) * 1000

        # Note: in real pipeline, other chunks are being generated simultaneously.
        # But E2E latency is defined by TTFB for conversational responsiveness.

        audio_kb = len(audio_bytes) / 1024

        print(f"  ⏱ TTS (Chunk 1):       {tts_ms:.0f}ms")
        print(f"  📊 Total TTFB:          {total_ms:.0f}ms")
        print(f"  🔈 Audio:               {audio_kb:.1f} KB, ~{duration_ms}ms")

        # Save audio
        audio_path = os.path.join(output_dir, f"test_{i+1}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        print(f"  💾 Saved: {audio_path}")
        print()

        results.append({
            "desc": case["description"],
            "translate_ms": translate_ms,
            "tts_ms": tts_ms,
            "total_ms": total_ms,
            "audio_kb": audio_kb,
            "translated": translated,
        })

        # Small gap between tests
        await asyncio.sleep(0.2)

    # ── 3. Summary ────────────────────────────────────────────
    print("=" * 72)
    print("LATENCY SUMMARY")
    print("=" * 72)
    header = f"{'Test':<22} {'Translate':>10} {'TTS':>10} {'TOTAL':>10} {'Audio':>8}"
    print(header)
    print("-" * 72)

    for r in results:
        print(
            f"{r['desc']:<22} "
            f"{r['translate_ms']:>8.0f}ms "
            f"{r['tts_ms']:>8.0f}ms "
            f"{r['total_ms']:>8.0f}ms "
            f"{r['audio_kb']:>6.1f}KB"
        )

    avg_translate = sum(r["translate_ms"] for r in results) / len(results)
    avg_tts = sum(r["tts_ms"] for r in results) / len(results)
    avg_total = sum(r["total_ms"] for r in results) / len(results)
    min_total = min(r["total_ms"] for r in results)
    max_total = max(r["total_ms"] for r in results)

    print("-" * 72)
    print(f"{'Average':<22} {avg_translate:>8.0f}ms {avg_tts:>8.0f}ms {avg_total:>8.0f}ms")
    print(f"{'Min':<22} {'':>10} {'':>10} {min_total:>8.0f}ms")
    print(f"{'Max':<22} {'':>10} {'':>10} {max_total:>8.0f}ms")
    print()

    target = 1500
    if avg_total < target:
        print(f"✅ Average total {avg_total:.0f}ms is UNDER the {target}ms target!")
    else:
        print(f"⚠️  Average total {avg_total:.0f}ms EXCEEDS the {target}ms target!")
        print(f"   Breakdown: Translation={avg_translate:.0f}ms, TTS={avg_tts:.0f}ms")
        if avg_translate > avg_tts:
            print("   💡 Bottleneck: Translation (check prompt/model latency)")
        else:
            print("   💡 Bottleneck: TTS (check Cartesia network latency)")

    print()
    print(f"Audio files saved in: {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
