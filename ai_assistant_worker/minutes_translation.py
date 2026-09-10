"""The minutes' other languages, kept independent of the summary's own. WT-665.

WHY THIS IS A SECOND PASS AND NOT A BIGGER PROMPT
    A summary can be asked for in one language, and a biên bản needs the meeting's languages
    side by side. Those are two different documents for two different readers, and until now
    asking for the first silently cancelled the second: `generate_structured_summary` only
    requested a `translations` map when nobody had chosen a summary language, because

        Asking for the whole summary in Japanese and then for a "translations" map beside it
        are contradictory instructions, and a model given both answers one of them at random.

    That reasoning is right, and this module is what it implies. Translating is a separate
    question, so it gets a separate call — with the finished summary as its input rather than
    the transcript. Nothing is asked of one prompt that pulls against itself.

WHY THE MODEL IS ONLY EVER SHOWN STRINGS
    Every item of a summary carries `atMs`, and the meeting page turns it into a jump. A
    number the model produced from nothing lands on a plausible, wrong moment and never
    surfaces as an error — the whole reason `summary_grounding` exists.

    Translations are exempt from that check on the stated grounds that they are "restatements
    of items already checked in the source language, not separate claims". That holds only if
    the restatement really does keep the original's moments, so this module makes it
    structurally impossible to do otherwise: the model is handed a flat list of strings and
    hands back a flat list of strings. Moments, owners and section keys never leave this
    process, and are re-attached here from the source. A fabricated `atMs` has nowhere to
    enter.

WHY AN OWNER IS NOT TRANSLATED
    `owner` is a person's name. "Tú" is not a word with a Japanese equivalent, and a model
    asked to translate a field will translate it.

WHAT HAPPENS WHEN IT GOES WRONG
    Nothing is published rather than something that is subtly mispaired. The minutes then read
    as they do today — one language — which is a smaller loss than a document whose two halves
    do not say the same thing, and which nobody would be able to see was wrong.
"""

from __future__ import annotations

from typing import Any

#: The fields of an item that hold words a person reads. `owner` is deliberately absent; so are
#: `atMs` and `alsoAtMs`, which are moments and not language.
TRANSLATABLE_FIELDS = ("text", "task")

#: Top-level keys of a summary that are not translatable content. `summary` is handled on its
#: own because it is a bare string rather than a list; the rest carry no prose at all.
#: Mirrors MeetingMinutesDrafter.NonSectionKeys, which is what reads the result.
NON_SECTION_KEYS = frozenset(
    {
        "summary",
        "citations",
        "translations",
        "templateKey",
        "insufficientData",
        "sections",
        "summaryLanguage",
        "generationFailed",
        "carriedOver",
    }
)


def collect_translatable(summary: dict[str, Any]) -> dict[str, Any]:
    """The words in a summary, flattened to strings, in a stable order.

    Returns `{"summary": str, "<sectionKey>": [str, ...]}`. A section whose items carry no
    readable field contributes an empty string in that position rather than being dropped —
    position IS the identity of an item here, so the list that comes back can be zipped against
    the source without a second thought about alignment.
    """
    payload: dict[str, Any] = {}

    overview = summary.get("summary")
    if isinstance(overview, str) and overview.strip():
        payload["summary"] = overview

    for key, value in summary.items():
        if key in NON_SECTION_KEYS or not isinstance(value, list):
            continue

        strings = [_readable(item) for item in value]
        if any(text for text in strings):
            payload[key] = strings

    return payload


def merge_translation(
    summary: dict[str, Any],
    # `Any`, not `dict`: the caller passes whatever the model put under this language key, and
    # deciding that a non-object is unusable is this function's job rather than the caller's.
    translated: Any,
) -> dict[str, Any] | None:
    """One language's half of the minutes, rebuilt from the source's structure.

    The model's strings are placed back into copies of the SOURCE items, so every moment,
    owner and key is the one that was already verified. Returns None when the shape does not
    line up — see the module docstring on why that is silence rather than a best effort.
    """
    if not isinstance(translated, dict):
        return None

    result: dict[str, Any] = {}

    overview = translated.get("summary")
    if isinstance(overview, str) and overview.strip():
        result["summary"] = overview

    for key, value in summary.items():
        if key in NON_SECTION_KEYS or not isinstance(value, list):
            continue

        replacements = translated.get(key)
        if not isinstance(replacements, list):
            continue

        # Length is the whole alignment guarantee. A model that returned four sentences for
        # three would otherwise shift every item after the extra one against its own moment,
        # producing a document that is wrong in a way only a bilingual reader could catch.
        if len(replacements) != len(value):
            return None

        items: list[Any] = []
        for source_item, replacement in zip(value, replacements):
            items.append(_with_text(source_item, replacement))

        if items:
            result[key] = items

    return result or None


def _readable(item: Any) -> str:
    """The one string a reader sees in this item, or "" when it has none."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for field in TRANSLATABLE_FIELDS:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _with_text(source_item: Any, replacement: Any) -> Any:
    """`source_item` with its readable string swapped, and everything else untouched."""
    if not isinstance(replacement, str) or not replacement.strip():
        # Nothing usable came back for this one. The source string is more useful to a reader
        # than a blank line, and leaves the two halves the same length.
        return source_item

    if isinstance(source_item, str):
        return replacement

    if isinstance(source_item, dict):
        translated_item = dict(source_item)
        for field in TRANSLATABLE_FIELDS:
            if isinstance(source_item.get(field), str):
                translated_item[field] = replacement
                return translated_item
        # An item with no readable field: keep it as it is rather than inventing one.
        return translated_item

    return source_item
