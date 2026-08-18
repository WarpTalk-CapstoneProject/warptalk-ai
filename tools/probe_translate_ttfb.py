"""How long does the translation stage take, warmed, on the path production uses?

WHY
    The <2.5s question turns on whether translation can be hidden. `_prefetch_from_event`
    already warms a speculative translation from STT deltas, but it is bounded by
    _SPECULATIVE_TIMEOUT_SECONDS = 1.5 and serialised behind a 1-slot semaphore — so if a
    single translation costs more than that, speculation is decoration and the stage is paid
    in full on the critical path every time.

    This measures the stage itself: pool warmed, one sentence at a time, both directions.
"""

from __future__ import annotations

import asyncio
import sys
import time

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker.translator import OpenAITranslator  # noqa: E402

PAIRS = [
    ("vi", "en", "Mình sẽ xem lại kế hoạch triển khai tuần sau trước khi hết giờ."),
    ("vi", "en", "Phần backend thì ổn rồi, vấn đề nằm ở phía client."),
    ("vi", "en", "Bạn gửi tài liệu cho cả nhóm sau cuộc họp này nhé."),
    ("en", "vi", "Let's go over the deployment plan for next week before we run out of time."),
    ("en", "vi", "I think the backend API is fine, the problem is on the client side."),
    ("en", "vi", "Can you share the document with the whole team after this meeting?"),
]
REPEATS = 3


def q(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


async def main() -> None:
    settings = TranslationSettings()
    translator = OpenAITranslator(
        api_key=resolve_openai_api_key(settings.api_key),
        model=settings.model,
        realtime_model=settings.realtime_model,
        realtime_reasoning_effort=settings.realtime_reasoning_effort,
        realtime_pool_size=settings.realtime_pool_size,
        realtime_timeout_seconds=settings.realtime_timeout_seconds,
        realtime_max_output_tokens=settings.realtime_max_output_tokens,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    await translator.load()
    await translator.warm_up()
    print(
        f"model={settings.model} realtime={settings.realtime_model} "
        f"pool={settings.realtime_pool_size}"
    )

    # One throwaway so the very first call does not carry pool setup.
    await translator.translate_with_valence(
        "Hello there.", source_lang="en", target_lang="vi", glossary_terms=[], meeting_context=[]
    )

    per_direction: dict[str, list[float]] = {"vi->en": [], "en->vi": []}
    for _rep in range(REPEATS):
        for source, target, text in PAIRS:
            t0 = time.monotonic()
            try:
                out, _valence = await translator.translate_with_valence(
                    text,
                    source_lang=source,
                    target_lang=target,
                    glossary_terms=[],
                    meeting_context=[],
                )
            except Exception as exc:
                print(f"  failed {source}->{target}: {str(exc)[:100]}")
                continue
            elapsed = time.monotonic() - t0
            per_direction[f"{source}->{target}"].append(elapsed)
            if _rep == 0:
                print(f"  {source}->{target} {elapsed:.3f}s  {out[:60]}")

    print("\n" + "=" * 52)
    print(f"{'direction':<10} {'n':>3} {'p50':>8} {'p95':>8} {'max':>8}")
    print("-" * 52)
    allv: list[float] = []
    for direction, values in per_direction.items():
        allv.extend(values)
        print(
            f"{direction:<10} {len(values):>3} "
            f"{q(values, 0.5):>7.3f}s {q(values, 0.95):>7.3f}s {max(values):>7.3f}s"
        )
    print("-" * 52)
    print(
        f"{'combined':<10} {len(allv):>3} {q(allv, 0.5):>7.3f}s "
        f"{q(allv, 0.95):>7.3f}s {max(allv):>7.3f}s"
    )
    print("=" * 52)
    over = sum(1 for v in allv if v > 1.5)
    print(f"\ncalls exceeding the 1.5s speculative budget: {over}/{len(allv)}")


if __name__ == "__main__":
    asyncio.run(main())
