"""What WarpBot can reach, and what it is told it can reach.

The bug behind these tests rendered perfectly: asked "meeting b có decision gì", WarpBot answered
"chưa có summary tự động" while the Knowledge page behind the chat window displayed the answer —
a fact tagged `Decision`, from meeting `b`. Nothing had failed. The facts were indexed, in the
collection semantic_search queries, and the tool simply never described itself as covering them,
so the model had no reason to route there.

A tool description is not documentation; it is the only routing signal the model gets. These
tests treat it as behaviour, because that is what it is.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker.chat_tools import TOOLS, ToolContext, _list_recent_meetings, _search_facts
from shared.config import ChatAssistantSettings

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _ctx(response: MagicMock | None = None) -> ToolContext:
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return ToolContext(
        workspace_id="w1",
        user_id="u1",
        bearer_token="Bearer t",
        workspace_client=client,
        transcript_client=client,
        translation_room_client=client,
        openai_client=MagicMock(),
        model="gpt-4o-mini",
        redis=MagicMock(),
    )


def _ok(payload: object) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(return_value=payload)
    return response


# ── The description IS the routing signal ────────────────────────────────────────────────────


def test_semantic_search_says_it_covers_facts_and_decisions() -> None:
    """It always did cover them — knowledge_fact_worker publishes facts into the same
    `workspace_{id}` collection this queries. The description named only documents, transcripts
    and glossaries, so the model picked get_meeting_summary for "what was decided" and reported
    that no summary existed."""
    description = TOOLS_BY_NAME["semantic_search"].description.lower()

    assert "fact" in description
    assert "decision" in description


def test_there_is_a_tool_for_facts_by_category() -> None:
    tool = TOOLS_BY_NAME["search_facts"]

    assert tool.parameters["properties"]["fact_category"]["enum"] == [
        "decision",
        "requirement",
        "definition",
        "commitment",
        "risk",
        "reference",
    ]
    # Both optional: "what do we know" is as valid a question as "what was decided".
    assert tool.parameters["required"] == []


# ── search_facts ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_facts_returns_the_facts_and_where_they_came_from() -> None:
    ctx = _ctx(
        _ok(
            {
                "items": [
                    {
                        "fact": "Tiếng Nhật chưa được chọn.",
                        "factCategory": "decision",
                        "sourceTitle": "b",
                        "sourceType": "meeting_summary",
                    }
                ]
            }
        )
    )

    result = json.loads(await _search_facts(ctx, {"fact_category": "decision"}))

    assert result["facts"] == [
        {
            "fact": "Tiếng Nhật chưa được chọn.",
            "category": "decision",
            "source": "b",
            "source_type": "meeting_summary",
        }
    ]


@pytest.mark.asyncio
async def test_search_facts_drops_indexed_chunks_that_carry_no_fact() -> None:
    # The endpoint returns every indexed chunk. Rows with no extracted fact are not knowledge,
    # and listing them buries six real decisions under a hundred transcript fragments.
    ctx = _ctx(
        _ok(
            {
                "items": [
                    {"fact": None, "text": "half a sentence", "sourceType": "document"},
                    {"fact": "Ship on Friday.", "factCategory": "commitment", "sourceTitle": "a"},
                ]
            }
        )
    )

    result = json.loads(await _search_facts(ctx, {}))

    assert [f["fact"] for f in result["facts"]] == ["Ship on Friday."]


@pytest.mark.asyncio
async def test_search_facts_refuses_a_category_the_extractor_never_writes() -> None:
    # The set is closed. Passing "blocker" through would query for a label no row can have and
    # return an empty list, which the model reports as "there are none" — a confident wrong answer.
    ctx = _ctx()

    result = json.loads(await _search_facts(ctx, {"fact_category": "blocker"}))

    assert "error" in result
    assert "decision" in result["allowed"]
    ctx.workspace_client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_facts_explains_an_empty_result_instead_of_implying_none_exist() -> None:
    ctx = _ctx(_ok({"items": []}))

    result = json.loads(await _search_facts(ctx, {"fact_category": "risk"}))

    assert result["facts"] == []
    assert "note" in result, "an empty list with no explanation reads as 'this workspace has none'"


# ── Diagnosability ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_failed_meeting_lookup_carries_its_status_to_the_transcript() -> None:
    """The status was logged and dropped. What a user can screenshot is the tool's return value,
    and it said only "could not look up recent meetings" — so the WT-373 report could not say
    whether that was auth, routing or an outage, and the logs that knew were rotated away by a
    deploy before anyone looked."""
    response = MagicMock()
    response.status_code = 403
    ctx = _ctx(response)

    result = json.loads(await _list_recent_meetings(ctx, {}))

    assert result["status"] == 403


@pytest.mark.asyncio
async def test_a_thrown_meeting_lookup_names_the_exception_type() -> None:
    ctx = _ctx()
    ctx.translation_room_client.get = AsyncMock(side_effect=TimeoutError("upstream"))

    result = json.loads(await _list_recent_meetings(ctx, {}))

    assert result["reason"] == "TimeoutError"


# ── Hosted web search ────────────────────────────────────────────────────────────────────────


def test_web_search_is_on_by_default_and_can_be_switched_off_without_a_rebuild() -> None:
    assert ChatAssistantSettings().web_search_enabled is True
    assert ChatAssistantSettings(web_search_enabled=False).web_search_enabled is False


def test_web_search_is_not_a_local_tool() -> None:
    # It is OpenAI's hosted tool: the model calls it, OpenAI runs it. Registering it as a ChatTool
    # would put a name in TOOLS_BY_NAME with no handler behind it, and the dispatch loop would
    # answer every call with "Unknown tool".
    assert "web_search" not in TOOLS_BY_NAME


# ---------------------------------------------------------------------------------------------
# What a hosted search leaves behind: annotations on the answer, and the step the reader sees.
# ---------------------------------------------------------------------------------------------


class _Annotation:
    def __init__(self, type_: str, url: str = "", title: str = "", start_index: int = 0) -> None:
        self.type = type_
        self.url = url
        self.title = title
        self.start_index = start_index


class _Part:
    def __init__(self, annotations: list[_Annotation]) -> None:
        self.annotations = annotations


class _Item:
    def __init__(self, type_: str, content: list[_Part] | None = None) -> None:
        self.type = type_
        self.content = content or []


def test_a_url_citation_becomes_a_web_source() -> None:
    from ai_assistant_worker.chat_worker import _web_citations

    items = [
        _Item(
            "message",
            [_Part([_Annotation("url_citation", "https://vnexpress.net/x", "VnExpress", 42)])],
        )
    ]

    assert _web_citations(items) == [("VnExpress", "https://vnexpress.net/x", 42)]


def test_an_untitled_result_is_named_by_its_host() -> None:
    # "vnexpress.net" is a source a reader recognises. An untitled chip is not.
    from ai_assistant_worker.chat_worker import _web_citations

    items = [_Item("message", [_Part([_Annotation("url_citation", "https://vnexpress.net/x")])])]

    assert _web_citations(items) == [("vnexpress.net", "https://vnexpress.net/x", 0)]


def test_annotations_that_are_not_url_citations_are_ignored() -> None:
    from ai_assistant_worker.chat_worker import _web_citations

    items = [
        _Item("message", [_Part([_Annotation("file_citation", "https://example.com/a", "A")])]),
        # A reasoning or web_search_call item carries no answer text to anchor anything to.
        _Item("web_search_call"),
    ]

    assert _web_citations(items) == []


def test_a_response_shape_without_annotations_does_not_raise() -> None:
    # Provider response shapes change. Losing a chip is acceptable; losing the answer is not.
    from ai_assistant_worker.chat_worker import _web_citations

    class _Bare:
        type = "message"

    assert _web_citations([_Bare()]) == []
