"""What does translate() actually send, now that the repair clause is gated?

Every other probe here builds the prompt by hand, so none of them exercise the wiring in
translate()/translate_batch() — they would report identical numbers whether the gate were
connected or left dangling. This one calls the real method and intercepts the request, so
the prompt it prints is the prompt production would send. No API calls, no cost.
"""

from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, ".")

from translation_worker import translator as translator_mod  # noqa: E402

GLOSSARY = [
    {"source": "staging", "target": "staging"},
    {"source": "Kubernetes", "target": "Kubernetes"},
]
CONTEXT = ["We moved all the dashboards over to Grafana last sprint."]

CASES = [
    ("clean, no context", "Ừ đúng rồi, mình chốt vậy nhé", None),
    ("exact glossary hit only", "Bên mình deploy lên staging chiều nay", None),
    ("suspected mishearing", "Con cu bơ nét tự restart cái pod đó rồi", None),
    ("clean, with context", "Ừ đúng rồi, mình chốt vậy nhé", CONTEXT),
]


async def main() -> None:
    tx = translator_mod.OpenAITranslator(api_key="offline", model="gpt-4.1-mini")
    await tx.load()

    seen: dict[str, str] = {}

    async def fake_create(**kwargs: object) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        seen["system"] = str(messages[0]["content"])
        seen["user"] = str(messages[1]["content"])

        class Msg:
            content = "ok"

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    tx._create_with_retry = fake_create  # type: ignore[method-assign]

    print(f"\n  {'case':<26}{'gate':>7}{'system':>9}{'total':>8}   sent to the model")
    print(f"  {'-' * 74}")
    for label, text, context in CASES:
        await tx.translate(text, "vi", "en", GLOSSARY, context)
        system, user = seen["system"], seen["user"]
        open_ = translator_mod._ASR_REPAIR_INSTRUCTION.strip()[:40] in system
        print(
            f"  {label:<26}{'OPEN' if open_ else 'shut':>7}"
            f"{len(system):>8}c{len(system) + len(user):>7}c"
        )

    print("\n  Gate must be OPEN only where there is something to repair from:")
    print("    - a suspected mishearing (fuzzy glossary match), or")
    print("    - meeting context to reason from.")
    print("  An exact glossary hit is terminology, not evidence the recogniser slipped.")


if __name__ == "__main__":
    asyncio.run(main())
