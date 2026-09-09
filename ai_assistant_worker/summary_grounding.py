"""Whether the moments a summary cites are moments the meeting actually had.

WHY AN INVENTED `atMs` NEVER LOOKS INVENTED
    Every item of a summary carries `atMs`, and the meeting page turns that number into a
    jump: click the item and the transcript scrolls to the moment and highlights it. The
    lookup rule is "the last segment that started at or before this moment, otherwise the
    first one after", so it ALWAYS lands on a segment. A number the model produced from
    nothing therefore never surfaces as an error — it quietly scrolls the reader to a
    plausible, wrong place. A citation that misbehaves exactly like a citation that works is
    worse than no citation at all, because the reader went and checked.

    Nothing between the model and that click was checking the number. The web only asks that
    it be finite and non-negative, which every fabricated number is. This module is the
    check, and it belongs here because here is the only place that still knows which moments
    were handed to the model.

WHY THE MEMBERSHIP TEST IS EXACT, AND WHY NOTHING IS SNAPPED TO THE NEAREST MOMENT
    `transcript_offsets` returns precisely the integers `format_transcript_line` printed into
    the prompt — the model was shown them and asked to repeat one, so there is nothing for a
    tolerance to forgive. Rounding a near-miss to the closest real moment is not leniency: it
    is the mechanism that converts an invented number into a jump that looks legitimate,
    which is the single failure this check exists to expose. A moment is in the set or it is
    not.

WHY A FAILED CITATION COSTS THE ITEM ITS LINK AND NEVER ITS PLACE
    Dropping the item too would silently shorten the summary, and a summary quietly missing a
    line is a worse lie than a line that admits it cannot point anywhere. `meeting-summary.ts`
    already renders a null `atMs` as plain text rather than a control, and footnotes why, so
    an uncited item degrades into exactly what it is: something the model said that the
    transcript will not vouch for.

    The same discipline `SourceRegistry` applies to `[S1]` markers in citations.py — resolve
    only what we ourselves handed out — written for the other half of the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_assistant_worker.summary_templates import transcript_offsets

#: Keys of a summary that are not lists of cited items. Named rather than inferred so that a
#: template adding a section (standup's `blockers`, technical's `options`) is checked without
#: anybody remembering to come back here — sections are open, this list is closed.
#: `translations` is a map of whole translated summaries and passes through whole: its items
#: are restatements of items already checked in the source language, not separate claims.
UNCITED_KEYS = frozenset({"summary", "templateKey", "insufficientData", "translations"})


@dataclass(frozen=True)
class GroundedSummary:
    """The summary with its unverifiable moments removed, and what that cost."""

    summary: dict[str, Any]
    #: Every moment the model produced — one `atMs` plus each `alsoAtMs` element.
    moments_checked: int
    #: Those the transcript could not vouch for. Duplicates removed by normalisation are NOT
    #: counted here; they were real moments, just written twice.
    moments_dropped: int
    #: Items left pointing at nothing. The number worth alarming on: a summary of them is a
    #: summary of assertions nobody can check.
    items_uncited: int


@dataclass
class _Tally:
    """Running counts while the walk is in progress; frozen into GroundedSummary at the end."""

    checked: int = 0
    dropped: int = 0
    uncited: int = 0


def _verify(raw: Any, offsets: set[int], tally: _Tally) -> int | None:
    """One claimed moment, or None when the transcript does not carry it.

    `bool` is excluded before the `int` check because `True == 1`: a JSON `true` where a
    moment belongs would otherwise validate against an offset of 1 ms. A float is accepted
    only when it is a whole number of milliseconds — that is the same integer written
    differently, not a near miss being rounded. Everything else (a string, None, a fraction)
    is a value that was never a moment.
    """
    tally.checked += 1

    moment: int | None = None
    if isinstance(raw, bool):
        moment = None
    elif isinstance(raw, int):
        moment = raw
    elif isinstance(raw, float) and raw.is_integer():
        moment = int(raw)

    if moment is None or moment not in offsets:
        tally.dropped += 1
        return None
    return moment


def _ground_item(item: Any, offsets: set[int], tally: _Tally) -> Any:
    """One item of one section, with only the moments the transcript vouches for."""
    if not isinstance(item, dict):
        # A model that wrote a bare string where an object belongs has produced a shape
        # problem, not a citation problem. Handing it back unchanged keeps that failure
        # legible to whoever debugs it instead of half-rewriting it here.
        return item

    has_at = "atMs" in item
    has_also = "alsoAtMs" in item
    if not has_at and not has_also:
        # Nothing claimed, so nothing to check. An older template's item, or a section the
        # model answered without citing — both untouched, neither counted as uncited.
        return item

    primary = _verify(item.get("atMs"), offsets, tally) if has_at else None

    raw_also = item.get("alsoAtMs")
    survivors = [primary] if primary is not None else []
    if isinstance(raw_also, list):
        survivors.extend(
            moment for raw in raw_also if (moment := _verify(raw, offsets, tally)) is not None
        )

    grounded = dict(item)
    if not survivors:
        # Kept, deliberately. See the module docstring: an item with no link reads as text.
        grounded["atMs"] = None
        grounded["alsoAtMs"] = []
        tally.uncited += 1
        return grounded

    # A surviving `atMs` keeps its own moment: it is the one the model chose to anchor the
    # item to, and reordering a citation that is already true would move the reader's jump
    # for no reason. Only a dropped one is replaced, and then by the EARLIEST survivor —
    # matching the traceable template's rule that a merged sentence is anchored to the first
    # moment it rests on, so the summary still reads in meeting order.
    at_ms = primary if primary is not None else min(survivors)
    others = sorted({moment for moment in survivors if moment != at_ms})

    grounded["atMs"] = at_ms
    if others or has_also:
        grounded["alsoAtMs"] = others
    return grounded


def _ground_citations(entries: list[Any], offsets: set[int], tally: _Tally) -> list[Any]:
    """The overview paragraph's evidence, minus the entries that are not evidence.

    A citation is removed rather than blanked, which is the opposite of what happens to an
    item — because the two are not the same thing. An item is a sentence a reader reads, and
    it survives losing its link. A citation entry is nothing BUT a link: keeping one with a
    null `atMs` leaves a chip on screen that points at nowhere, and no text is lost by taking
    it away.
    """
    grounded: list[Any] = []
    for entry in entries:
        if not isinstance(entry, dict) or "atMs" not in entry:
            grounded.append(entry)
            continue
        moment = _verify(entry.get("atMs"), offsets, tally)
        if moment is None:
            continue
        grounded.append({**entry, "atMs": moment})
    return grounded


def ground_summary(summary: dict[str, Any], transcript: str) -> GroundedSummary:
    """Strip every cited moment the transcript cannot vouch for, and report what went.

    Pure: `summary` is returned rebuilt, never modified.
    """
    offsets = transcript_offsets(transcript)
    tally = _Tally()

    grounded: dict[str, Any] = {}
    for key, value in summary.items():
        if key == "citations" and isinstance(value, list):
            grounded[key] = _ground_citations(value, offsets, tally)
        elif key in UNCITED_KEYS or not isinstance(value, list):
            grounded[key] = value
        else:
            grounded[key] = [_ground_item(item, offsets, tally) for item in value]

    return GroundedSummary(
        summary=grounded,
        moments_checked=tally.checked,
        moments_dropped=tally.dropped,
        items_uncited=tally.uncited,
    )
