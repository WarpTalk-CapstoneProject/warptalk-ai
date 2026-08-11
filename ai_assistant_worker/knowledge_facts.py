"""Turn a piece of workspace content into a short list of durable facts.

WHAT A "FACT" IS HERE
    One standalone statement a person could act on months later without the surrounding
    text — a decision that was made, a requirement that was agreed, a risk that was named.
    Not a paraphrase of the content, and not a summary of it: the summary already exists.

WHY THE CATEGORY SET IS CLOSED
    Six categories, fixed. An open set produces a near-unique label per fact — a tag cloud,
    not a filter — and the Knowledge page's category tabs would each match one row. The same
    six are declared on the reading side (WorkspaceKnowledgeService.FactCategories and
    warptalk-web's FACT_CATEGORIES); all three lists must move together.

WHY AN UNRECOGNISED CATEGORY DROPS THE FACT
    The model occasionally invents a seventh label. Storing it would create a chunk that no
    tab can ever surface — present in the index, invisible in the UI, and impossible to
    explain. Dropping it loses one fact; keeping it quietly corrupts the filter.
"""

from __future__ import annotations

import json
from typing import Any, cast

from openai import AsyncOpenAI

from shared.config import AssistantSettings
from shared.logger import get_logger
from shared.openai_options import completion_options

logger = get_logger(__name__)

_DEFAULTS = AssistantSettings()

FACT_CATEGORIES = (
    "decision",
    "requirement",
    "definition",
    "commitment",
    "risk",
    "reference",
)

# A bound on cost and on nonsense. Content long enough to hold more than this many durable
# facts is content that should have been split before it got here, and an unbounded list is
# how one pathological document fills a workspace's index with near-duplicates.
MAX_FACTS = 12

# The extractor reads content, not conversations. Anything past this is very unlikely to
# still be introducing new durable facts, and the tail of a long transcript is mostly
# closing pleasantries.
MAX_INPUT_CHARS = 24000

SYSTEM_PROMPT = f"""You extract durable facts from workspace content.

A fact is ONE standalone statement that stays useful months later, understandable without
the text around it. Resolve pronouns and references into names. Write each fact in the same
language as the source content.

Do NOT extract:
- Summaries or restatements of the content as a whole
- Small talk, greetings, scheduling chatter, or transitions
- Anything you are inferring rather than reading

Classify each fact into EXACTLY ONE of these categories:
- decision: a choice that was settled
- requirement: something that must be true or must be done
- definition: what a term, system, or role means in this workspace
- commitment: a named party agreeing to do something
- risk: a stated danger, blocker, or concern
- reference: a pointer to an external resource, document, or system

Return JSON: {{"facts": [{{"fact": "...", "category": "...", "quote": "..."}}]}}
`quote` is the shortest span of the source text the fact came from, copied verbatim.
Return at most {MAX_FACTS} facts. Return {{"facts": []}} if the content holds none —
an empty list is a valid and useful answer, so never pad it.
"""


class KnowledgeFactExtractor:
    """A thin, single-purpose LLM client. Owns no state beyond its connection."""

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULTS.model,
        max_tokens: int = _DEFAULTS.max_tokens,
        temperature: float = _DEFAULTS.temperature,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client: AsyncOpenAI | None = None

    async def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for knowledge fact extraction")
        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("knowledge_fact_extractor_initialized", model=self.model)

    async def extract(self, title: str, text: str) -> list[dict[str, str]]:
        """Facts found in `text`, or an empty list.

        Never raises. A workspace's indexing must not fail because fact extraction did —
        the content is already indexed by whoever published it, and a missing fact is a
        visibly empty column rather than lost knowledge.
        """
        if not text.strip():
            return []

        client = self._client
        if client is None:
            raise RuntimeError("KnowledgeFactExtractor is not loaded")

        heading = f"Content title: {title}\n\n" if title.strip() else ""
        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{heading}{text[:MAX_INPUT_CHARS]}",
                    },
                ],
                **completion_options(self.model, self.max_tokens, self.temperature),
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            parsed = cast(dict[str, Any], json.loads(raw))
        except Exception:
            logger.exception("knowledge_fact_extraction_failed", title=title)
            return []

        return _clean(parsed.get("facts"))


def _clean(raw: Any) -> list[dict[str, str]]:
    """Keep only well-formed facts in a known category, de-duplicated, capped."""
    if not isinstance(raw, list):
        return []

    facts: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        fact = str(item.get("fact") or "").strip()
        category = str(item.get("category") or "").strip().lower()
        if not fact or category not in FACT_CATEGORIES:
            continue
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        # The quote is what gets embedded and shown as "Indexed text". Falling back to the
        # fact itself keeps the row readable when the model omitted or hallucinated a span.
        quote = str(item.get("quote") or "").strip() or fact
        facts.append({"fact": fact, "category": category, "quote": quote})
        if len(facts) >= MAX_FACTS:
            break

    return facts
