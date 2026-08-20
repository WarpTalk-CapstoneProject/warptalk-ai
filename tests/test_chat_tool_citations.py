"""Wiring the citation registry into the tools.

The registry itself is tested in test_citations.py. What is tested here is the join: that a tool
hands the model a marker it can cite, that an unciteable item arrives without one rather than with
a broken one, and that a context with no registry degrades to no citations rather than failing.
"""

from __future__ import annotations

from ai_assistant_worker.chat_tools import _cite, _source_kind, _with_marker
from ai_assistant_worker.citations import SourceRegistry


class _Ctx:
    """Only the field _cite reads. The real ToolContext needs live HTTP and Redis clients."""

    def __init__(self, citations: SourceRegistry | None) -> None:
        self.citations = citations


def test_a_tool_gets_a_marker_it_can_hand_the_model():
    registry = SourceRegistry()

    marker = _cite(_Ctx(registry), "document", "Q3-plan.pdf", "doc-1")

    assert marker == "S1"
    assert registry.registered()[0].title == "Q3-plan.pdf"


def test_a_context_with_no_registry_simply_produces_no_citations():
    # Every existing construction site passes no registry, and a tool must keep working there
    # rather than failing because provenance is unavailable.
    assert _cite(_Ctx(None), "document", "Q3-plan.pdf", "doc-1") is None


def test_an_item_with_nothing_to_point_at_carries_no_marker():
    # Better than a marker that resolves to nothing: the model would cite it and the chip row
    # would come out empty, which reads as a bug rather than as an uncited answer.
    registry = SourceRegistry()

    assert _cite(_Ctx(registry), "knowledge", None) is None
    assert _with_marker({"fact": "x"}, None) == {"fact": "x"}


def test_a_marker_is_attached_where_the_model_will_see_it():
    assert _with_marker({"fact": "x"}, "S2") == {"fact": "x", "marker": "S2"}


def test_the_indexers_vocabulary_maps_onto_the_chips():
    # Two different closed sets: what the indexer records as sourceType, and what a chip can be.
    assert _source_kind("document") == "document"
    assert _source_kind("FILE") == "document"
    assert _source_kind("glossary") == "glossary"
    assert _source_kind("meeting_summary") == "meeting"
    assert _source_kind("transcript_segment") == "transcript"


def test_an_unrecognised_source_type_falls_back_to_knowledge():
    # True of every chunk in the index, and the honest answer when a new producer starts
    # publishing a type this mapping has not met.
    assert _source_kind("something_new") == "knowledge"
    assert _source_kind(None) == "knowledge"
    assert _source_kind("") == "knowledge"
