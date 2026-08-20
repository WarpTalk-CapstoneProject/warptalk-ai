"""Which sources an answer is allowed to claim.

A chip is a claim of provenance the reader cannot check, so the failure that matters is a chip
under an answer that never used that source. Every case here is either the intersection being
computed correctly, or an invented citation being refused.
"""

from __future__ import annotations

from ai_assistant_worker.citations import (
    MAX_SOURCES_PER_ANSWER,
    SourceRegistry,
    strip_markers,
)


def test_only_a_cited_source_becomes_a_chip():
    # Retrieved but not pointed at is the common case: semantic_search returns five chunks and the
    # answer rests on one. Listing all five would be listing what the TOOL did.
    registry = SourceRegistry()
    used = registry.register("document", "Q3-plan.pdf", "doc-1")
    registry.register("document", "Unrelated.pdf", "doc-2")

    cited = registry.cited(f"We ship on Friday [{used}].")

    assert [source.title for source in cited] == ["Q3-plan.pdf"]


def test_a_marker_the_model_invented_cites_nothing():
    # The property the whole module exists for: S9 was never handed out, so it cannot be pointed
    # at. A model completing the SHAPE of a citation produces no chip.
    registry = SourceRegistry()
    registry.register("document", "Q3-plan.pdf", "doc-1")

    assert registry.cited("We ship on Friday [S9].") == []


def test_an_answer_with_no_tools_and_no_citations_is_fine():
    assert SourceRegistry().cited("Xin chào!") == []


def test_chips_read_in_the_order_the_argument_makes_them():
    registry = SourceRegistry()
    first = registry.register("glossary", "SLA")
    second = registry.register("document", "Q3-plan.pdf", "doc-1")

    # Cited in reverse of registration order.
    cited = registry.cited(f"The plan says Friday [{second}]. The SLA allows it [{first}].")

    assert [source.title for source in cited] == ["Q3-plan.pdf", "SLA"]


def test_the_same_source_quoted_repeatedly_is_one_chip():
    registry = SourceRegistry()
    marker = registry.register("document", "Q3-plan.pdf", "doc-1")

    cited = registry.cited(f"A [{marker}]. B [{marker}]. C [{marker}].")

    assert len(cited) == 1


def test_one_source_registered_twice_gets_one_marker():
    # The same document returned by two tools is one thing, and a reader seeing it twice would
    # reasonably think there were two.
    registry = SourceRegistry()

    first = registry.register("document", "Q3-plan.pdf", "doc-1")
    again = registry.register("document", "  q3-plan.PDF  ", "doc-1")

    assert first == again
    assert len(registry.registered()) == 1


def test_the_same_title_from_different_sources_stays_separate():
    # Two workspaces can both have "Notes.pdf", and merging them would attribute one to the other.
    registry = SourceRegistry()

    first = registry.register("document", "Notes.pdf", "doc-1")
    second = registry.register("document", "Notes.pdf", "doc-2")

    assert first != second


def test_a_source_with_no_name_is_not_registered():
    # A chip reading "Untitled" is a worse answer than no chip.
    registry = SourceRegistry()

    assert registry.register("document", None, "doc-1") is None
    assert registry.register("document", "   ", "doc-1") is None
    assert registry.registered() == []


def test_an_unknown_kind_falls_back_rather_than_inventing_a_category():
    registry = SourceRegistry()
    registry.register("wikipedia", "Something", None)

    assert registry.registered()[0].kind == "knowledge"


def test_the_chip_row_is_bounded():
    # A model that cites everything it was shown has cited nothing, and nobody reads twelve chips.
    registry = SourceRegistry()
    markers = [registry.register("knowledge", f"Fact {index}") for index in range(20)]

    cited = registry.cited(" ".join(f"x [{marker}]." for marker in markers))

    assert len(cited) == MAX_SOURCES_PER_ANSWER


def test_markers_are_stripped_from_what_the_reader_sees():
    assert strip_markers("We ship on Friday [S1].") == "We ship on Friday."


def test_stripping_does_not_leave_gaps_or_orphaned_spaces():
    assert strip_markers("A [S1] and B [S2] , then C [S3]!") == "A and B, then C!"
    assert strip_markers("Line one [S1]\n\nLine two [S2]") == "Line one\n\nLine two"


def test_an_answer_without_markers_survives_stripping_untouched():
    answer = "Không có nguồn nào ở đây."

    assert strip_markers(answer) == answer


def test_prose_that_merely_mentions_s1_is_not_a_citation():
    # Brackets are what makes a citation unambiguous; a variable named S1 is not one.
    registry = SourceRegistry()
    registry.register("document", "Q3-plan.pdf", "doc-1")

    assert registry.cited("The S1 bucket holds the plan.") == []
    assert strip_markers("The S1 bucket holds the plan.") == "The S1 bucket holds the plan."


def test_a_source_carries_what_the_client_needs_to_open_it():
    registry = SourceRegistry()
    registry.register("web", "example.com/report", "https://example.com/report")

    payload = registry.registered()[0].as_dict()
    assert payload["kind"] == "web"
    assert payload["ref"] == "https://example.com/report"


def test_a_source_with_nowhere_to_go_omits_its_ref():
    # A glossary term is a thing, not a destination, and a chip that looks clickable and is not
    # is a broken promise.
    registry = SourceRegistry()
    registry.register("glossary", "SLA")

    assert "ref" not in registry.registered()[0].as_dict()


# ---------------------------------------------------------------------------------------------
# Sources the answer anchors to directly — OpenAI's hosted web search, which never passes through
# a handler here and so can never be handed a marker.
# ---------------------------------------------------------------------------------------------


def test_a_hosted_search_result_is_cited_without_any_marker():
    registry = SourceRegistry()
    registry.note_cited("web", "VnExpress", "https://vnexpress.net/article", at=0)

    cited = registry.cited("Bộ luật mới có hiệu lực từ tháng 7.")

    assert [(source.kind, source.ref) for source in cited] == [
        ("web", "https://vnexpress.net/article")
    ]


def test_web_and_marker_sources_interleave_in_reading_order():
    # An answer that searched the knowledge base AND the web reads in one order; the chips under
    # it must too, or the reader matches the wrong chip to the wrong claim.
    registry = SourceRegistry()
    marker = registry.register("document", "Q3-plan.pdf", "doc-1")
    answer = f"The web says X. The plan says Y [{marker}]."
    registry.note_cited("web", "VnExpress", "https://vnexpress.net/x", at=answer.index("web"))

    assert [source.kind for source in registry.cited(answer)] == ["web", "document"]


def test_a_web_source_with_no_position_lands_after_the_positioned_ones():
    registry = SourceRegistry()
    marker = registry.register("glossary", "SLA")
    registry.note_cited("web", "example.com", "https://example.com/a")

    assert [source.kind for source in registry.cited(f"Định nghĩa [{marker}].")] == [
        "glossary",
        "web",
    ]


def test_the_same_source_cited_twice_is_still_one_chip():
    # Once through a marker and once through an annotation: the same url, one chip.
    registry = SourceRegistry()
    marker = registry.register("web", "example.com", "https://example.com/a")
    registry.note_cited("web", "example.com", "https://example.com/a", at=0)

    assert len(registry.cited(f"Answer [{marker}].")) == 1


def test_a_nameless_hosted_result_registers_nothing():
    registry = SourceRegistry()

    assert registry.note_cited("web", "", "https://example.com/a") is None
    assert registry.cited("Answer.") == []
