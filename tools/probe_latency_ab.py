"""Latency A/B: does the guardian prompt cost time, and does the full model cost time?

Two separate questions, and only the second is expected to cost anything. The prompt adds
458 characters to every call. The model decision it enables swaps -mini for the full model
on both paths — and the code comment that pinned mini in the first place feared exactly
that. A 2x2 (prompt x model) separates the two so neither hides behind the other.

Method, because the earlier n=12 pass could not support a claim in either direction:
  * Paired. Every cell sees the same utterance, so utterance length drops out of the
    comparison and the delta is computed pair-by-pair instead of median-vs-median.
  * Interleaved. All four cells stay open at once and are called in shuffled order within
    each pair, so API load drifting across the run cannot settle on one cell.
  * Warmed. The realtime pool pays connection setup on its first call; production holds
    that pool open for the process lifetime, so steady-state is what users actually feel.

Chat cells use production's pinned models (gpt-4.1-mini / gpt-4.1), not the local default.
"""

from __future__ import annotations

import asyncio
import random
import re
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker import translator as translator_mod  # noqa: E402

UTTERANCES = [
    "Ừ đúng rồi",
    "Chào mọi người, mình bắt đầu nhé",
    "Con cu bơ nét tự restart cái pod đó rồi",
    "Bên mình deploy lên staging từ chiều hôm qua",
    "Mọi người có câu hỏi gì trước khi kết thúc không",
    "Tôi vừa xem log thì thấy lỗi xảy ra lúc gọi sang service kia",
    "Cái này để tôi kiểm tra lại rồi trả lời trong hôm nay nhé mọi người",
    "Sprint này còn ba ticket chưa xong, tôi nghĩ nên dời một cái sang tuần sau "
    "và tập trung vào phần thanh toán trước",
]
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
    {"source": "sprint", "target": "sprint"},
]
REPEATS = 3


def old_prompt() -> str:
    return translator_mod._SYSTEM_PROMPT.replace(translator_mod._ASR_REPAIR_INSTRUCTION, "")


def literal_only(text: str, terms: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reproduce production's literal-only glossary matcher."""
    hay = text.lower()
    keep = []
    for t in terms:
        src = (t.get("source") or "").strip()
        if src and re.search(rf"(?<!\w){re.escape(src.lower())}(?!\w)", hay):
            keep.append({**t, "match": "exact"})
    return keep


def message(text: str, terms: list[dict[str, str]]) -> str:
    return (
        f"Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{text}\n\n"
        f"Respond entirely in English — never leave any word in Vietnamese or switch to a "
        f"third language, {translator_mod._exception_clause(terms)}."
        f"{translator_mod._build_glossary_block(terms)}"
    )


def report(name: str, samples: dict[str, list[float]], baseline: str) -> None:
    print(f"\n{'=' * 78}\n{name}   n={len(next(iter(samples.values())))} per cell\n{'=' * 78}")
    print(f"  {'cell':<22}{'median':>9}{'p90':>9}{'max':>9}   vs {baseline} (paired median)")
    base = samples[baseline]
    for label, xs in samples.items():
        p90 = statistics.quantiles(xs, n=10)[8] if len(xs) >= 10 else max(xs)
        delta = ""
        if label != baseline and len(xs) == len(base):
            diffs = [a - b for a, b in zip(xs, base, strict=True)]
            med = statistics.median(diffs)
            worse = sum(1 for d in diffs if d > 0)
            delta = f"{med:+7.0f} ms   slower in {worse}/{len(diffs)}"
        elif label != baseline:
            delta = "unpaired (dropped calls)"
        print(
            f"  {label:<22}{statistics.median(xs):>7.0f}ms{p90:>7.0f}ms{max(xs):>7.0f}ms   {delta}"
        )


async def build(cfg: TranslationSettings, key: str, model: str, realtime: bool) -> Any:
    tx = translator_mod.OpenAITranslator(
        api_key=key,
        model=cfg.model if realtime else model,
        realtime_model=model if realtime else "",
        realtime_pool_size=1,
        realtime_timeout_seconds=30.0,
        realtime_max_output_tokens=cfg.realtime_max_output_tokens,
    )
    await tx.load()
    return tx


async def call(tx: Any, cfg: TranslationSettings, system: str, user: str, realtime: bool) -> float:
    """One timed call. Returns elapsed ms, or -1.0 if it broke."""
    started = time.perf_counter()
    try:
        if realtime:
            await tx._translate_realtime(user, system)
        else:
            await tx._create_with_retry(
                model=tx.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **tx._completion_options(cfg.max_tokens),
            )
    except Exception:
        return -1.0
    return (time.perf_counter() - started) * 1000


async def sweep(name: str, mini: str, full: str, realtime: bool) -> None:
    cfg = TranslationSettings()
    key = resolve_openai_api_key(cfg.api_key)
    new, old = translator_mod._SYSTEM_PROMPT, old_prompt()

    cells = {
        "prod (old+mini)": (old, mini, literal_only),
        "prompt (new+mini)": (new, mini, translator_mod._select_relevant_glossary_terms),
        "old+full": (old, full, literal_only),
        "branch (new+full)": (new, full, translator_mod._select_relevant_glossary_terms),
    }
    tx = {label: await build(cfg, key, model, realtime) for label, (_, model, _) in cells.items()}

    # Warm every pool so connection setup is not charged to whichever cell ran first.
    for label, (system, _, match) in cells.items():
        warm = message(UTTERANCES[0], match(UTTERANCES[0], GLOSSARY))
        await call(tx[label], cfg, system, warm, realtime)

    samples: dict[str, list[float]] = {label: [] for label in cells}
    broke = 0
    order = list(cells)
    for _ in range(REPEATS):
        for text in UTTERANCES:
            random.shuffle(order)
            for label in order:
                system, _, match = cells[label]
                user = message(text, match(text, GLOSSARY))
                ms = await call(tx[label], cfg, system, user, realtime)
                if ms < 0:
                    broke += 1
                    ms = float("nan")
                samples[label].append(ms)

    for handle in tx.values():
        await handle.close()

    clean = {label: [x for x in xs if x == x] for label, xs in samples.items()}
    if broke:
        print(f"\n  ! {broke} call(s) broke and were dropped; pairing is per-cell from here")
        report(name, clean, "prod (old+mini)")
    else:
        report(name, samples, "prod (old+mini)")

    for label, (system, _, match) in cells.items():
        size = len(system) + len(message(UTTERANCES[2], match(UTTERANCES[2], GLOSSARY)))
        print(f"  {label:<22}{size:>5} chars sent (system + user, hard-case utterance)")


async def main() -> None:
    random.seed(11)
    await sweep("REALTIME — first sentence", "gpt-realtime-2.1-mini", "gpt-realtime-2.1", True)
    await sweep("CHAT — sentences 2..N", "gpt-4.1-mini", "gpt-4.1", False)


if __name__ == "__main__":
    asyncio.run(main())
