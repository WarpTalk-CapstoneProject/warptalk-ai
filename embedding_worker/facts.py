"""Turning an indexed chunk into a fact a person can read.

A vector is not reviewable. A workspace owner asked to see "what the system knows about my
documents" cannot be shown 1536 floats, and the chunk text alone is the raw material rather
than the answer — a 400-word slab of a contract is not a fact.

So each chunk gets one short statement extracted from it, plus a category, and those travel
in the chunk's own Qdrant payload. Storing them beside the chunk rather than in a new table
is deliberate: a fact is derived from exactly one chunk and has no life of its own. When the
chunk is deleted or its retention state changes, the fact goes with it automatically —
`retention_state` and `deletion_state` are already on that payload. A separate table would
have to be kept in step, and that is a synchronisation problem nobody would remember to
solve until it was already wrong.

The parsing here is pure so the failure modes can be tested without a model call, and every
one of them is a real one: a model that answers in prose, in fenced JSON, with an unknown
category, or with nothing at all.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Deliberately small and closed. An open category set produces a different label for every
# chunk, which is a tag cloud, not a filter — and the table this feeds has a category column
# people are meant to click.
FACT_CATEGORIES = (
    "decision",
    "requirement",
    "definition",
    "commitment",
    "risk",
    "reference",
)

DEFAULT_CATEGORY = "reference"

# Long enough to be a sentence, short enough to read in a table row without truncation.
MAX_FACT_CHARS = 240

EXTRACTION_PROMPT = """You read one chunk of a document or meeting transcript and state the
single most useful fact it contains.

Rules:
- One sentence. Under 200 characters. No preamble.
- State it as a fact about the subject, not about the text. Write "Payment terms are net 30",
  never "The document says payment terms are net 30".
- If the chunk carries no fact worth remembering — boilerplate, a page header, filler — return
  an empty string for "fact". Saying nothing is a valid answer and better than inventing one.
- Choose exactly one category from: {categories}

Answer with JSON only: {{"fact": "...", "category": "..."}}"""


def build_extraction_prompt() -> str:
    return EXTRACTION_PROMPT.format(categories=", ".join(FACT_CATEGORIES))


def _strip_code_fence(raw: str) -> str:
    """Models wrap JSON in ```json fences often enough that not handling it is a bug."""
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", raw, re.DOTALL)
    return fenced.group(1) if fenced else raw


def parse_fact_response(raw: str | None) -> dict[str, str] | None:
    """The model's answer as {fact, category}, or None when there is no fact.

    None is a real outcome, not an error: most chunks of most documents are filler, and a
    table of invented facts about page headers is worse than a shorter table.
    """
    if not raw or not raw.strip():
        return None

    try:
        parsed: Any = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    fact = str(parsed.get("fact") or "").strip()
    if not fact:
        return None

    fact = fact[:MAX_FACT_CHARS].strip()

    category = str(parsed.get("category") or "").strip().lower()
    if category not in FACT_CATEGORIES:
        # An unrecognised category is not worth discarding a good fact over.
        category = DEFAULT_CATEGORY

    return {"fact": fact, "category": category}


def fact_payload_fields(parsed: dict[str, str] | None) -> dict[str, str]:
    """What to merge into the chunk's Qdrant payload.

    Empty when there is no fact, so a chunk without one carries no keys rather than a null
    the reader would have to interpret.
    """
    if not parsed:
        return {}
    return {"fact": parsed["fact"], "fact_category": parsed["category"]}
