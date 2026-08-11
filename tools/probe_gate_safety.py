"""Is it safe to DROP _ASR_REPAIR_INSTRUCTION when there is no evidence to repair from?

The planned optimisation gates the instruction on evidence: send it only when the fuzzy
matcher flagged a possible mishearing, or when meeting context exists. By the instruction's
own wording that gate should be inert — "When you have no such evidence, translate exactly
what is written and invent nothing" is the only clause that applies in the gated-off cell.

But that clause is also the brake. Removing it leaves the base prompt's "translate
accurately and naturally" facing a garbled sentence with nothing to anchor on, and the
failure mode this whole change was designed around is a FLUENT invention, not a visible
mistranscription. So the gated-off cell has to be measured, not reasoned about.

The utterance below is the real production case the matcher cannot help with: "Codex" folds
to a 5-character skeleton, under _MIN_MISHEARD_SKELETON, so it is deliberately excluded
(see the table above _MISHEARD_SIMILARITY). No glossary match, no context — exactly the
cell the gate creates.

  LITERAL - keeps the misheard syllables, or hedges with quotes/ellipsis. Honest.
  INVENTED - names a specific tool/system the speaker never said. The regression.
"""

from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

sys.path.insert(0, ".")

from shared.config import TranslationSettings, resolve_openai_api_key  # noqa: E402
from translation_worker import translator as translator_mod  # noqa: E402

HARD = "Cái cô đích nó báo lỗi khi tôi chạy lại lần nữa"
GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]
RUNS = 6

# Vietnamese syllables surviving into the English output, or an explicit hedge.
_VISIBLE = re.compile(r"[àáâãèéêìíòóôõùúýăđĩũơưạ-ỹ]|co dich|codich|\.\.\.|…|[\"“”']", re.IGNORECASE)
# A confident specific noun the speaker never uttered.
_INVENTION = re.compile(
    r"\b(codex|copilot|cursor|jenkins|docker|git|github|gitlab|xcode|vscode|"
    r"compiler|linter|debugger|terminal|script|server|database|api|cache)\b",
    re.IGNORECASE,
)


def classify(out: str) -> str:
    if _VISIBLE.search(out):
        return "LITERAL "
    if _INVENTION.search(out):
        return "INVENTED"
    return "vague   "


def without_instruction() -> str:
    return translator_mod._SYSTEM_PROMPT.replace(translator_mod._ASR_REPAIR_INSTRUCTION, "")


def message() -> str:
    terms = translator_mod._select_relevant_glossary_terms(HARD, GLOSSARY)
    assert not terms, f"expected no glossary match for the gated-off cell, got {terms}"
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
    user = message()
    cells = {
        "keep (today)": translator_mod._SYSTEM_PROMPT,
        "gated off   ": without_instruction(),
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
            kind = classify(out)
            tally[kind] = tally.get(kind, 0) + 1
            outs.append(f"    {kind} {out[:84]}")
        await tx.close()
        shape = "  ".join(f"{k.strip()} {v}/{RUNS}" for k, v in sorted(tally.items()))
        print(f"\n  [{label}] {shape}   sent {len(system) + len(user)} chars")
        for line in outs:
            print(line)


async def main() -> None:
    cfg = TranslationSettings()
    await sweep(cfg.realtime_model, realtime=True)
    await sweep("gpt-realtime-2.1", realtime=True)
    await sweep("gpt-4.1-mini", realtime=False)
    await sweep("gpt-4.1", realtime=False)


if __name__ == "__main__":
    asyncio.run(main())
