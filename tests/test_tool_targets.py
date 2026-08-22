"""A step says what it is doing AND what it is doing it to.

"Searching documents…" answers a question nobody asked. The target is the half that makes a
wrong turn visible while it is still running — "Searching documents · Q3 budget" when you asked
about hiring is a mistake you can interrupt, where four identical spinner rows are not.

These pin the two rules that keep the target useful rather than noisy: an id is never printed,
and a long value is cut rather than allowed to wrap the step onto three lines.
"""

from __future__ import annotations

from ai_assistant_worker.tool_targets import (
    MAX_TARGET_CHARS,
    describe_tool_target,
    describe_web_search_target,
    split_reasoning_summary,
)
from shared.openai_options import reasoning_summary_options


class TestDescribeToolTarget:
    def test_the_query_is_the_target_of_a_search(self) -> None:
        assert (
            describe_tool_target("search_documents", {"query": "onboarding checklist"})
            == "onboarding checklist"
        )

    def test_the_best_key_wins_over_a_later_one(self) -> None:
        # A title names a document; a query is what was typed. For search_documents the query
        # is what the user recognises as theirs.
        target = describe_tool_target(
            "search_documents", {"title": "Handbook.docx", "query": "leave policy"}
        )
        assert target == "leave policy"

    def test_a_document_title_is_shown_when_there_is_no_query(self) -> None:
        assert describe_tool_target("get_document", {"title": "Q3 Budget.xlsx"}) == "Q3 Budget.xlsx"

    def test_a_bare_id_is_never_printed(self) -> None:
        # A UUID is noise in a one-line step: "Reading the document · 0f2c…" tells a reader
        # strictly less than "Reading the document".
        assert (
            describe_tool_target(
                "get_document", {"document_id": "019f0d00-0de0-7000-9000-000000000001"}
            )
            == ""
        )

    def test_an_id_in_a_target_key_is_also_dropped(self) -> None:
        assert (
            describe_tool_target(
                "get_room_detail", {"title": "019f0d00-0de0-7000-9000-000000000001"}
            )
            == ""
        )

    def test_a_long_value_is_cut_rather_than_wrapped(self) -> None:
        target = describe_tool_target("semantic_search", {"query": "a " * 200})
        assert len(target) <= MAX_TARGET_CHARS
        assert target.endswith("…")

    def test_whitespace_is_collapsed_to_one_line(self) -> None:
        assert describe_tool_target("semantic_search", {"query": " two\n  lines "}) == "two lines"

    def test_a_tool_with_no_subject_reports_none(self) -> None:
        # Not a failure: list_recent_meetings with no query is about nothing in particular, and
        # the label alone already says what is happening.
        assert describe_tool_target("list_recent_meetings", {}) == ""
        assert describe_tool_target("ask_user", {"questions": [{"question": "Which room?"}]}) == ""

    def test_an_unmapped_tool_still_names_an_ordinary_argument(self) -> None:
        # A tool added in warptalk-ai without touching the map must degrade to no detail at
        # worst — never to a wrong one — and it names its target when it uses ordinary keys.
        assert describe_tool_target("some_new_tool", {"query": "budget"}) == "budget"
        assert describe_tool_target("some_new_tool", {"unknown_shape": "budget"}) == ""

    def test_non_string_arguments_are_not_targets(self) -> None:
        # "get_platform_analytics · 30" reads as a subject and is a setting.
        assert describe_tool_target("get_platform_analytics", {"range": 30}) == ""
        assert describe_tool_target("search_documents", None) == ""


class TestDescribeWebSearchTarget:
    def test_a_search_action_reports_its_query(self) -> None:
        assert (
            describe_web_search_target({"type": "search", "query": "livekit krisp pricing"})
            == "livekit krisp pricing"
        )

    def test_opening_a_page_reports_the_site_not_the_url(self) -> None:
        # The ask was "search web thì searching abc.com" — a full URL pushes the step off a
        # 320px widget, and the site is the part a reader recognises.
        target = describe_web_search_target(
            {"type": "open_page", "url": "https://www.livekit.io/pricing?ref=abc#plans"}
        )
        assert target == "livekit.io"

    def test_an_sdk_object_reads_the_same_as_a_dict(self) -> None:
        # The stream hands back model objects and the tests hand back dicts; a reader written
        # for one shape silently returns nothing for the other.
        class _Action:
            type = "search"
            query = "warptalk release notes"

        assert describe_web_search_target(_Action()) == "warptalk release notes"

    def test_an_unrecognised_action_is_no_target_rather_than_a_crash(self) -> None:
        assert describe_web_search_target(None) == ""
        assert describe_web_search_target({"type": "something_new"}) == ""


class TestSplitReasoningSummary:
    """The model's own account of a step, as a title and the sentence under it.

    A trail built from tool calls alone can say "Searching the web" and never say why, and
    between two calls it says nothing at all. This is the only source for that sentence — and
    it has to be split, because a title line with an indented body is how every agent UI worth
    copying draws a step.
    """

    def test_a_bold_heading_becomes_the_title(self) -> None:
        title, body = split_reasoning_summary(
            "**Clarifying specifications and sources**\n\nI will re-check the official Qwen "
            "announcements before answering."
        )
        assert title == "Clarifying specifications and sources"
        assert body.startswith("I will re-check the official Qwen")

    def test_a_markdown_heading_works_too(self) -> None:
        title, body = split_reasoning_summary("## Weighing the options\nBoth paths are viable.")
        assert title == "Weighing the options"
        assert body == "Both paths are viable."

    def test_a_paragraph_with_no_heading_keeps_all_of_itself(self) -> None:
        # No invented title: cutting the first sentence out to sit in bold would misrepresent
        # the model as having structured something it did not.
        title, body = split_reasoning_summary("Just a plain thought about the question.")
        assert title == ""
        assert body == "Just a plain thought about the question."

    def test_an_overlong_heading_is_prose_not_a_title(self) -> None:
        long_heading = "**" + ("word " * 40).strip() + "**\n\nbody"
        title, body = split_reasoning_summary(long_heading)
        assert title == ""
        assert "word" in body

    def test_whitespace_only_is_nothing_at_all(self) -> None:
        assert split_reasoning_summary("   \n  ") == ("", "")
        assert split_reasoning_summary("") == ("", "")

    def test_a_heading_with_no_body_still_titles_the_step(self) -> None:
        title, body = split_reasoning_summary("**Checking the transcript**")
        assert title == "Checking the transcript"
        assert body == ""


class TestReasoningSummaryOptions:
    def test_a_gpt5_model_is_asked_to_narrate(self) -> None:
        assert reasoning_summary_options("gpt-5.6-luna") == {"reasoning": {"summary": "auto"}}

    def test_anything_else_is_asked_for_nothing(self) -> None:
        # Sending `reasoning` to a model that does not take it is a 400 on the FIRST request,
        # which is the trap shared/openai_options.py exists to describe.
        assert reasoning_summary_options("gpt-4o-mini") == {}
