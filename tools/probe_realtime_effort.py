"""Sweep the Realtime reasoning effort against ASR-mishearing repair.

The realtime path carries the FIRST sentence of every utterance, so it is the one
translation call a listener waits on directly. `_ASR_REPAIR_INSTRUCTION` reaches it,
but `gpt-realtime-2.1-mini` at `effort: minimal` does not act on the hint: it renders
"cu bo net" back as a quoted Vietnamese fragment instead of "Kubernetes".

This asks whether more thinking fixes that, and what it costs. Both ceilings can bite:
`realtime_timeout_seconds` (2.0 in production) and `realtime_max_output_tokens` (128,
which must also absorb hidden reasoning tokens). A response that ends `incomplete` or
times out raises, and production then falls back to the slower HTTP path — so a
"repair" bought at the price of always falling back is not a win.

`_translate_realtime` is called directly rather than `translate()`, because
`translate()` swallows a realtime failure and silently returns the chat answer, which
would make a broken realtime path look healthy.

Run: uv run python tools/probe_realtime_effort.py
"""

from __future__ import annotations

import asyncio
import statistics
import time

from shared.config import TranslationSettings, resolve_openai_api_key
from translation_worker import translator as translator_mod

GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]

# (label, utterance, expected token in a correct repair, must-not-appear)
CASES: list[tuple[str, str, str | None]] = [
    ("REPAIR-hard", "Con cu bơ nét tự restart cái pod đó rồi", "kubernetes"),
    ("REPAIR-easy", "Deploy lên xì ta ging trước khi release", "staging"),
    ("NO-INVENT", "Hôm nay trời đẹp quá mọi người ạ", None),
]

EFFORTS = ["minimal", "low", "medium"]
RUNS = 3


def _build_messages(text: str) -> tuple[str, str]:
    """Rebuild exactly what translate() would send, so the probe measures the real prompt."""
    relevant = translator_mod._select_relevant_glossary_terms(text, GLOSSARY)
    user_message = (
        "Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{text}\n\n"
        "Respond entirely in English — never leave any word in Vietnamese or "
        f"switch to a third language, {translator_mod._exception_clause(relevant)}."
        f"{translator_mod._build_glossary_block(relevant)}"
    )
    return translator_mod._SYSTEM_PROMPT, user_message


async def _sweep(max_tokens: int) -> None:
    settings = TranslationSettings()
    api_key = resolve_openai_api_key(settings.api_key)
    print(f"\n{'=' * 78}")
    print(f"max_output_tokens = {max_tokens}   timeout = {settings.realtime_timeout_seconds}s")
    print(f"{'=' * 78}")

    for effort in EFFORTS:
        tx = translator_mod.OpenAITranslator(
            api_key=api_key,
            model=settings.model,
            realtime_model=settings.realtime_model,
            realtime_reasoning_effort=effort,
            realtime_pool_size=1,
            # Deliberately generous so a slow answer is MEASURED rather than cut off.
            # Production's own 2.0s ceiling is applied afterwards, in the verdict column.
            realtime_timeout_seconds=15.0,
            realtime_max_output_tokens=max_tokens,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
        )
        await tx.load()
        await tx.warm_up()
        print(f"\n--- effort={effort}")
        for label, text, expect in CASES:
            system_prompt, user_message = _build_messages(text)
            lats: list[float] = []
            outs: list[str] = []
            errs: list[str] = []
            for _ in range(RUNS):
                t0 = time.perf_counter()
                try:
                    out = await tx._translate_realtime(user_message, system_prompt)
                    lats.append((time.perf_counter() - t0) * 1000)
                    outs.append(out)
                except Exception as exc:  # noqa: BLE001 - the failure mode IS the datum
                    lats.append((time.perf_counter() - t0) * 1000)
                    errs.append(f"{type(exc).__name__}: {str(exc)[:60]}")
            med = statistics.median(lats)
            worst = max(lats)
            if errs:
                verdict = f"BROKE ({errs[0]})"
            elif expect is None:
                # Control: the only wrong answer here is an invented technical term.
                invented = any(g["source"].lower() in o.lower() for o in outs for g in GLOSSARY)
                verdict = "INVENTED" if invented else "literal ok"
            else:
                hits = sum(1 for o in outs if expect in o.lower())
                verdict = f"repaired {hits}/{len(outs)}"
            over = "  OVER-2s" if med > settings.realtime_timeout_seconds * 1000 else ""
            print(f"  [{label:12}] med {med:7.0f}ms  max {worst:7.0f}ms  {verdict}{over}")
            if outs:
                print(f"                 -> {outs[0][:88]}")
        await tx.close()


async def main() -> None:
    # 128 is production. 512 separates "thinking does not help" from "thinking had no
    # room to happen" — if a repair appears only at the larger ceiling, the blocker was
    # the token budget, not the model.
    for max_tokens in (128, 512):
        await _sweep(max_tokens)


if __name__ == "__main__":
    asyncio.run(main())
