"""Which half of the guardian actually repairs the sentence?

The repair permission is currently stated twice. _ASR_REPAIR_INSTRUCTION sits in the system
prompt and ships on every call; the "misheard" section of _build_glossary_block ships only
when the fuzzy matcher suspects something. Both say roughly "read it as the term if that
makes sense, otherwise change nothing".

If the glossary section alone repairs, the system-prompt copy is 458 characters of dead
weight on every utterance in every meeting, and deleting it is free. If the system-prompt
copy is what does the work, it stays and the cost is real. Measured, not guessed.

Cells, all on the FULL models (mini repairs nothing, so it cannot tell these apart):
  both    - system instruction + misheard glossary section   (current branch)
  block   - misheard glossary section only
  instr   - system instruction only, glossary matched literally like production
  neither - production
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker import translator as translator_mod  # noqa: E402

HARD = "Con cu bơ nét tự restart cái pod đó rồi"
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]
RUNS = 6


def without_instruction() -> str:
    return translator_mod._SYSTEM_PROMPT.replace(translator_mod._ASR_REPAIR_INSTRUCTION, "")


def literal_only(text: str, terms: list[dict[str, str]]) -> list[dict[str, str]]:
    """Production's literal-only matcher: no term matches HARD, so no misheard section."""
    hay = text.lower()
    return [
        {**t, "match": "exact"}
        for t in terms
        if (t.get("source") or "").strip()
        and re.search(rf"(?<!\w){re.escape((t['source']).lower())}(?!\w)", hay)
    ]


def message(terms: list[dict[str, str]]) -> str:
    return (
        f"Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{HARD}\n\n"
        f"Respond entirely in English — never leave any word in Vietnamese or switch to a "
        f"third language, {translator_mod._exception_clause(terms)}."
        f"{translator_mod._build_glossary_block(terms)}"
    )


async def sweep(model: str, realtime: bool) -> None:
    cfg = TranslationSettings()
    key = resolve_openai_api_key(cfg.api_key)
    fuzzy = translator_mod._select_relevant_glossary_terms(HARD, GLOSSARY)
    literal = literal_only(HARD, GLOSSARY)

    cells = {
        "both   ": (translator_mod._SYSTEM_PROMPT, fuzzy),
        "block  ": (without_instruction(), fuzzy),
        "instr  ": (translator_mod._SYSTEM_PROMPT, literal),
        "neither": (without_instruction(), literal),
    }
    print(f"\n{'=' * 78}\n{model}   {'REALTIME' if realtime else 'CHAT'}   n={RUNS}\n{'=' * 78}")

    for label, (system, terms) in cells.items():
        tx: Any = translator_mod.OpenAITranslator(
            api_key=key,
            model=cfg.model if realtime else model,
            realtime_model=model if realtime else "",
            realtime_pool_size=1,
            realtime_timeout_seconds=30.0,
            realtime_max_output_tokens=cfg.realtime_max_output_tokens,
        )
        await tx.load()
        user = message(terms)
        hits = 0
        outs: list[str] = []
        for _ in range(RUNS):
            try:
                if realtime:
                    out = await tx._translate_realtime(user, system)
                else:
                    r = await tx._create_with_retry(
                        model=model,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        **tx._completion_options(cfg.max_tokens),
                    )
                    out = (r.choices[0].message.content or "").strip()
            except Exception as exc:
                out = f"BROKE {type(exc).__name__}"
            ok = "kubernetes" in out.lower()
            hits += ok
            outs.append(f"    {'OK ' if ok else '   '} {out[:84]}")
        await tx.close()
        print(f"\n  [{label}] repaired {hits}/{RUNS}   sent {len(system) + len(user)} chars")
        for line in outs:
            print(line)


async def main() -> None:
    await sweep("gpt-realtime-2.1", realtime=True)
    await sweep("gpt-4.1", realtime=False)


if __name__ == "__main__":
    asyncio.run(main())
