"""Does the guardian prompt turn a VISIBLE mistranscription into a FLUENT wrong sentence?

The one risk the ASR-repair prompt was designed around: a reader spots "cô đích" as
nonsense instantly, but cannot tell a confidently repaired sentence from a real one. So a
change that makes wrong output *more fluent* without making it *more correct* is a
regression, even though every latency and leak number stays flat.

First pass (tools/probe_guardian_vs_prod.py, n=1 per cell) hinted at exactly that on the
realtime model: production left "Bơ Nét" sitting in the English, while the branch produced
"The smooth net" — fluent, plausible, wrong. This script repeats the same hard sentence
enough times to tell a pattern from a coincidence.

Classification per output, on the utterance "Con cu bơ nét tự restart cái pod đó rồi":
  REPAIRED  - says Kubernetes. The goal.
  VISIBLE   - keeps Vietnamese syllables or an ellipsis/quote/paren hedge, so a reader
              can see the recogniser failed. Wrong, but honestly wrong.
  FLUENT    - clean English that names something the speaker never said. The bad cell.
"""

from __future__ import annotations

import asyncio
import re
import sys

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker import translator as translator_mod  # noqa: E402

HARD = "Con cu bơ nét tự restart cái pod đó rồi"
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]
RUNS = 6

_VI = re.compile(r"[àáâãèéêìíòóôõùúýăđĩũơưạ-ỹ]", re.IGNORECASE)
_HEDGE = re.compile(r"[(\"“”']|\.\.\.|…|\?")


def classify(out: str) -> str:
    if "kubernetes" in out.lower():
        return "REPAIRED"
    if _VI.search(out) or _HEDGE.search(out):
        return "VISIBLE"
    return "FLUENT"


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


def message(terms: list[dict[str, str]]) -> str:
    return (
        f"Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{HARD}\n\n"
        f"Respond entirely in English — never leave any word in Vietnamese or switch to a "
        f"third language, {translator_mod._exception_clause(terms)}."
        f"{translator_mod._build_glossary_block(terms)}"
    )


async def run(model: str, realtime: bool) -> None:
    cfg = TranslationSettings()
    key = resolve_openai_api_key(cfg.api_key)
    cells = {
        "prod  ": (old_prompt(), literal_only(HARD, GLOSSARY)),
        "branch": (
            translator_mod._SYSTEM_PROMPT,
            translator_mod._select_relevant_glossary_terms(HARD, GLOSSARY),
        ),
    }
    print(f"\n{'=' * 78}\n{model}   {'REALTIME' if realtime else 'CHAT'}   n={RUNS}\n{'=' * 78}")
    for label, (system, terms) in cells.items():
        tx = translator_mod.OpenAITranslator(
            api_key=key,
            model=model if not realtime else cfg.model,
            realtime_model=model if realtime else "",
            realtime_pool_size=1,
            realtime_timeout_seconds=30.0,
            realtime_max_output_tokens=cfg.realtime_max_output_tokens,
        )
        await tx.load()
        user = message(terms)
        tally: dict[str, int] = {}
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
            kind = classify(out) if not out.startswith("BROKE") else "BROKE"
            tally[kind] = tally.get(kind, 0) + 1
            outs.append(f"    {kind:9} {out[:88]}")
        await tx.close()
        shape = "  ".join(f"{k} {v}/{RUNS}" for k, v in sorted(tally.items()))
        print(f"\n  [{label}] {shape}")
        for line in outs:
            print(line)


async def main() -> None:
    cfg = TranslationSettings()
    await run(cfg.realtime_model, realtime=True)
    await run("gpt-4.1-mini", realtime=False)


if __name__ == "__main__":
    asyncio.run(main())
