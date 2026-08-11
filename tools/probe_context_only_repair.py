"""Does _ASR_REPAIR_INSTRUCTION earn its 458 characters on a term the glossary cannot cover?

probe_which_part_repairs.py showed the instruction and the misheard glossary section repair
equally well (6/6 each) on a term that IS in the glossary — so for that case the instruction
is redundant and deleting it is free.

This is the case that decides whether it is redundant everywhere. The misheard term here is
NOT in the glossary, so _build_glossary_block contributes nothing and the fuzzy matcher has
nothing to match. The only evidence available is the meeting context, which is exactly what
the instruction points the model at ("When the glossary or the MEETING CONTEXT makes it
clear what was really said"). If the instruction repairs here and production does not, it
pays for itself on every meeting that has run long enough to accumulate context.

The context block is copied from OpenAITranslator.translate so the model sees what it would
really see in production.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker import translator as translator_mod  # noqa: E402

# "Grafana" misheard as Vietnamese syllables. Not in the glossary below.
HARD = "Cái gờ ra pha na đang không hiện số liệu của hôm nay"
CONTEXT = [
    "We moved all the dashboards over to Grafana last sprint.",
    "The Prometheus scraper is running every thirty seconds now.",
    "Can you check whether the metrics endpoint is still exposed?",
]
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]
RUNS = 6


def without_instruction() -> str:
    return translator_mod._SYSTEM_PROMPT.replace(translator_mod._ASR_REPAIR_INSTRUCTION, "")


def message() -> str:
    """Mirror translate()'s user message, including the context block."""
    terms = translator_mod._select_relevant_glossary_terms(HARD, GLOSSARY)
    context_block = (
        "\n\nEarlier accepted utterances from this same meeting are provided only "
        "to resolve pronouns, terminology, and topic. Never translate, repeat, or "
        "invent content from them:\n"
        + "\n".join(f"- {line}" for line in CONTEXT)
        + "\n\nOnly when the current utterance is clearly background speech with no "
        "plausible connection to this meeting context, respond exactly "
        f"{translator_mod.OUT_OF_MEETING_SCOPE}. Never suppress short acknowledgements, "
        "questions, corrections, names, technical terms, reasonable tangents, or ambiguous "
        "utterances. If uncertain, translate normally."
    )
    return (
        f"Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{HARD}\n\n"
        f"Respond entirely in English — never leave any word in Vietnamese or switch to a "
        f"third language, {translator_mod._exception_clause(terms)}."
        f"{translator_mod._build_glossary_block(terms)}"
        f"{context_block}"
    )


async def sweep(model: str, realtime: bool) -> None:
    cfg = TranslationSettings()
    key = resolve_openai_api_key(cfg.api_key)
    user = message()

    # Production also appends _CONTEXT_RELEVANCE_INSTRUCTION whenever context is present.
    relevance = translator_mod._CONTEXT_RELEVANCE_INSTRUCTION
    cells = {
        "instr  ": translator_mod._SYSTEM_PROMPT + relevance,
        "neither": without_instruction() + relevance,
    }
    print(f"\n{'=' * 78}\n{model}   {'REALTIME' if realtime else 'CHAT'}   n={RUNS}\n{'=' * 78}")

    for label, system in cells.items():
        tx: Any = translator_mod.OpenAITranslator(
            api_key=key,
            model=cfg.model if realtime else model,
            realtime_model=model if realtime else "",
            realtime_pool_size=1,
            realtime_timeout_seconds=30.0,
            realtime_max_output_tokens=cfg.realtime_max_output_tokens,
        )
        await tx.load()
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
            ok = "grafana" in out.lower()
            hits += ok
            outs.append(f"    {'OK ' if ok else '   '} {out[:84]}")
        await tx.close()
        print(f"\n  [{label}] repaired {hits}/{RUNS}   sent {len(system) + len(user)} chars")
        for line in outs:
            print(line)


async def main() -> None:
    await sweep(TranslationSettings().realtime_model, realtime=True)
    await sweep("gpt-realtime-2.1", realtime=True)
    await sweep("gpt-4.1-mini", realtime=False)
    await sweep("gpt-4.1", realtime=False)


if __name__ == "__main__":
    asyncio.run(main())
