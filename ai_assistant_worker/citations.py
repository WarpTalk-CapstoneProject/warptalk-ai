"""Which sources an answer actually rests on, in a form the model cannot fake.

THE PROBLEM WITH ASKING A MODEL WHERE ITS ANSWER CAME FROM
    It will tell you. It will also tell you when it did not use a source, when it half-remembers
    one from training, and when it is simply completing the shape of a citation because answers of
    this kind have citations in them. A chip reading "Q3-plan.pdf" under an answer that never
    opened Q3-plan.pdf is worse than no chip at all: it is a claim of provenance, and the reader
    has no way to check it.

    Listing the tools that ran instead is honest and useless — `semantic_search` returning five
    chunks says nothing about which of them, if any, the answer used.

HOW THIS MAKES THE CLAIM UNFORGEABLE
    The same discipline as `atMs` on a summary item: the model may only point at something that
    demonstrably exists. Every source handed to it in a tool result carries a marker this registry
    issued — S1, S2, S3 — and the model is asked to cite by marker. Afterwards, only markers THIS
    REGISTRY ISSUED THIS TURN are resolved. A model that invents [S9] cites nothing, because S9 was
    never handed out; a model that cites [S2] can only have got it from the tool result that
    contained it.

    So the answer is neither "what the tools returned" nor "what the model says it used". It is the
    intersection: sources that were genuinely retrieved AND that the model pointed at.

WHY MARKERS ARE STRIPPED FROM THE VISIBLE ANSWER
    They are machinery, not prose. The reader gets chips; leaving "[S1]" in the middle of a
    sentence would be showing them the wiring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: What a source IS, for the chip that shows it. A closed set, mirrored by the web client — an
#: open one would produce a different icon per source and no way to group them.
SOURCE_KINDS = ("document", "glossary", "knowledge", "meeting", "transcript", "web")

#: `[S1]`, `[S12]`. Bracketed so ordinary prose cannot accidentally match: a model writing about a
#: variable called S1 does not cite anything, and one writing "[S1]" is unambiguously citing.
_MARKER = re.compile(r"\[(S\d+)\]")

#: Nobody reads twelve chips, and a model that cites everything it was shown has cited nothing.
MAX_SOURCES_PER_ANSWER = 8


@dataclass(frozen=True)
class Source:
    """One thing an answer can rest on."""

    marker: str
    kind: str
    title: str
    #: What the client needs to OPEN it — a document id, a url, a room id. Absent for a source
    #: that has no destination, such as a glossary term.
    ref: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"marker": self.marker, "kind": self.kind, "title": self.title}
        if self.ref:
            payload["ref"] = self.ref
        return payload


@dataclass
class SourceRegistry:
    """Every source shown to the model during one turn, and the markers they were shown under."""

    _sources: list[Source] = field(default_factory=list)
    _by_identity: dict[tuple[str, str, str | None], Source] = field(default_factory=dict)

    def register(self, kind: str, title: str | None, ref: str | None = None) -> str | None:
        """Issue a marker for one source, or None when there is nothing to point at.

        Deduplicated by identity, so the same document quoted five times is one chip carrying one
        marker — and a model citing that marker five times still produces one chip.
        """
        cleaned = (title or "").strip()
        if not cleaned:
            # A source with no name cannot be shown to anybody, and a chip reading "Untitled" is a
            # worse answer than no chip.
            return None

        kind = kind if kind in SOURCE_KINDS else "knowledge"
        identity = (kind, cleaned.casefold(), (ref or "").strip() or None)

        existing = self._by_identity.get(identity)
        if existing:
            return existing.marker

        source = Source(
            marker=f"S{len(self._sources) + 1}",
            kind=kind,
            title=cleaned,
            ref=(ref or "").strip() or None,
        )
        self._sources.append(source)
        self._by_identity[identity] = source
        return source.marker

    def cited(self, answer: str) -> list[Source]:
        """The sources this answer actually points at, in the order it first points at them.

        First-appearance order rather than registry order: the chips then read in the same
        sequence as the argument they support.
        """
        seen: set[str] = set()
        issued = {source.marker: source for source in self._sources}
        found: list[Source] = []

        for marker in _MARKER.findall(answer or ""):
            if marker in seen:
                continue
            source = issued.get(marker)
            # A marker this registry never issued is a model completing the SHAPE of a citation.
            # Dropped in silence: it is not an error the reader can do anything about, and the
            # answer without it is exactly as good.
            if source is None:
                continue
            seen.add(marker)
            found.append(source)
            if len(found) >= MAX_SOURCES_PER_ANSWER:
                break

        return found

    def registered(self) -> list[Source]:
        """Everything shown to the model this turn — for logging what it declined to cite."""
        return list(self._sources)


def strip_markers(answer: str) -> str:
    """The answer as a person should read it.

    Markers are machinery. Removing them leaves the spacing tidy: a marker at the end of a sentence
    takes its leading space with it, so "decided on Friday [S1]." does not become "…Friday ."
    """
    without = _MARKER.sub("", answer or "")
    without = re.sub(r"[ \t]+([.,;:!?])", r"\1", without)
    without = re.sub(r"[ \t]{2,}", " ", without)
    return "\n".join(line.rstrip() for line in without.split("\n")).strip()


def instruction() -> str:
    """What the model is told about citing. Added to the system prompt when tools are available."""
    return (
        "Some tool results carry a `marker` such as S1. When a statement in your answer comes "
        "from one of those results, put its marker in square brackets at the end of that "
        "statement — like [S1]. Cite the specific source the statement came from, not everything "
        "you were shown.\n"
        "Never write a marker that did not appear in a tool result. A marker you invent is "
        "discarded, and the sentence then appears to rest on nothing.\n"
        "If an answer comes from the conversation itself rather than from a tool result, cite "
        "nothing — an uncited sentence is perfectly normal."
    )
