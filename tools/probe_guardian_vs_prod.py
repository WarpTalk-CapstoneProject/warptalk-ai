"""Does the guardian change make PRODUCTION's own models better or worse?

Production runs gpt-realtime-2.1-mini for the first sentence of every utterance and
gpt-4.1-mini for the rest. Both were measured to IGNORE the repair hint on a hard
mishearing, so the guardian buys nothing on those models — but the longer prompt is
still sent on every call, and nothing had measured what that costs.

Three cells, so the instruction and the matcher can be told apart:

    prod    : base prompt, literal-only glossary   (what production does today)
    prompt  : new prompt,  literal-only glossary   (isolates the instruction)
    branch  : new prompt,  fuzzy glossary          (what this branch does)

The sentences that matter most are the ORDINARY ones — no mishearing, no glossary hit.
They are the overwhelming majority of real meeting traffic, so a regression there
outweighs any repair win on a rare mangled term.

Two objective failure signals, both things the base prompt already forbids:

    leak  : Vietnamese characters surviving into an English translation
            ("never leave any word in {src}")
    hedge : parentheses, question marks or quoted source words
            ("Output ONLY the translation — no explanations, no notes, no alternatives")

Usage:  uv run python tools/probe_guardian_vs_prod.py
"""

from __future__ import annotations

import asyncio
import re
import statistics
import time

from shared.config import TranslationSettings, resolve_openai_api_key
from translation_worker import translator as translator_mod

GLOSSARY = [
    {"source": "Kubernetes", "target": "Kubernetes"},
    {"source": "staging", "target": "staging"},
    {"source": "deadline", "target": "hạn chót"},
]

# Ordinary meeting speech: nothing misheard, no glossary term present. This is the
# population a regression would actually hit.
ORDINARY = [
    "Chúng ta cần chốt ngân sách cho quý sau trước thứ Sáu",
    "Anh gửi lại tài liệu cho em sau buổi họp nhé",
    "Tôi nghĩ nên hoãn phần đó sang sprint tới",
    "Mọi người có câu hỏi gì trước khi mình kết thúc không",
]

HARD = "Con cu bơ nét tự restart cái pod đó rồi"

_VIET = re.compile(
    r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)
_HEDGE = re.compile(r"[()?]|\"[^\"]*\"|“[^”]*”")


def _base_prompt() -> str:
    """Reconstruct the pre-guardian system prompt by removing the added instruction."""
    base = translator_mod._SYSTEM_PROMPT.replace(translator_mod._ASR_REPAIR_INSTRUCTION, "")
    assert "automatic speech recognition" not in base, "base prompt still carries the hint"
    return base


def _literal_only(text: str) -> list[dict[str, str]]:
    """Glossary selection as production does it: word-boundary literal matches only."""
    kept: list[dict[str, str]] = []
    haystack = text.lower()
    for term in GLOSSARY:
        source = term["source"]
        pieces = re.split(r"[\s_-]+", source.lower().strip())
        pattern = r"(?<!\w)" + r"[\s_-]+".join(map(re.escape, pieces)) + r"(?!\w)"
        if re.search(pattern, haystack):
            kept.append({**term, "match": "exact"})
    return kept


def _message(text: str, terms: list[dict[str, str]]) -> str:
    return (
        f"Translate from Vietnamese to English.\n"
        f"Current utterance (translate only this):\n{text}\n\n"
        f"Respond entirely in English — never leave any word in Vietnamese or "
        f"switch to a third language, {translator_mod._exception_clause(terms)}."
        f"{translator_mod._build_glossary_block(terms)}"
    )


CELLS = {
    "prod": (_base_prompt, _literal_only),
    "prompt": (lambda: translator_mod._SYSTEM_PROMPT, _literal_only),
    "branch": (
        lambda: translator_mod._SYSTEM_PROMPT,
        lambda t: translator_mod._select_relevant_glossary_terms(t, GLOSSARY),
    ),
}

RUNS = 3


async def main() -> None:
    settings = TranslationSettings()
    key = resolve_openai_api_key(settings.api_key)

    for model in (settings.realtime_model, "gpt-4.1-mini"):
        realtime = model.startswith("gpt-realtime")
        label = "REALTIME (sentence 1)" if realtime else "CHAT (sentences 2..N / fallback)"
        print(f"\n{'=' * 78}\n{model}   {label}\n{'=' * 78}")

        for cell, (prompt_fn, select_fn) in CELLS.items():
            system_prompt = prompt_fn()
            latencies: list[float] = []
            leaks: list[str] = []
            hedges: list[str] = []
            outputs: set[str] = set()

            tx = translator_mod.OpenAITranslator(
                api_key=key,
                model=model if not realtime else "gpt-4.1-mini",
                realtime_model=model if realtime else "",
                realtime_pool_size=1,
                realtime_timeout_seconds=settings.realtime_timeout_seconds,
                realtime_max_output_tokens=settings.realtime_max_output_tokens,
            )
            await tx.load()
            try:
                for text in ORDINARY:
                    message = _message(text, select_fn(text))
                    for _ in range(RUNS):
                        started = time.perf_counter()
                        try:
                            if realtime:
                                out = await tx._translate_realtime(message, system_prompt)
                            else:
                                response = await tx._create_with_retry(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": message},
                                    ],
                                    **tx._completion_options(tx.max_tokens),
                                )
                                out = (response.choices[0].message.content or "").strip()
                        except Exception as exc:  # noqa: BLE001 - probe records, never raises
                            out = f"<BROKE {type(exc).__name__}>"
                        latencies.append((time.perf_counter() - started) * 1000)
                        outputs.add(out)
                        if _VIET.search(out):
                            leaks.append(out)
                        if _HEDGE.search(out):
                            hedges.append(out)

                # One hard mishearing, to state the repair side in the same table.
                hard_terms = select_fn(HARD)
                hard_out = ""
                try:
                    if realtime:
                        hard_out = await tx._translate_realtime(
                            _message(HARD, hard_terms), system_prompt
                        )
                    else:
                        response = await tx._create_with_retry(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": _message(HARD, hard_terms)},
                            ],
                            **tx._completion_options(tx.max_tokens),
                        )
                        hard_out = (response.choices[0].message.content or "").strip()
                except Exception as exc:  # noqa: BLE001
                    hard_out = f"<BROKE {type(exc).__name__}>"
            finally:
                await tx.close()

            n = len(latencies)
            ordered = sorted(latencies)
            print(
                f"\n  [{cell:6}] n={n}  med {statistics.median(latencies):6.0f}ms"
                f"  p90 {ordered[int(n * 0.9) - 1]:6.0f}ms  max {max(latencies):6.0f}ms"
                f"   prompt_chars={len(system_prompt)}"
            )
            print(f"           leak {len(leaks)}/{n}   hedge {len(hedges)}/{n}")
            if hedges:
                print(f"           hedge e.g. {hedges[0][:88]}")
            if leaks:
                print(f"           leak  e.g. {leaks[0][:88]}")
            repaired = "kubernetes" in hard_out.lower()
            print(
                f"           hard: {'REPAIRED' if repaired else 'not repaired'} -> {hard_out[:74]}"
            )


if __name__ == "__main__":
    asyncio.run(main())
