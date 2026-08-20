"""Tests for OpenAISuggester — response parsing, clamping and failure containment.

No network: the OpenAI client is replaced with a stub that records the request it was
given and returns a canned completion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from suggestion_worker.suggester import (
    NullSuggester,
    OpenAISuggester,
    SuggestionDecision,
    TranscriptTurn,
)


class StubCompletions:
    def __init__(self, payload: Any, total_tokens: int = 50) -> None:
        self.payload = payload
        self.total_tokens = total_tokens
        self.requests: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(total_tokens=self.total_tokens),
        )


def build_suggester(
    payload: Any, total_tokens: int = 50
) -> tuple[OpenAISuggester, StubCompletions]:
    suggester = OpenAISuggester(
        api_key="test-key",
        decide_model="decide-model",
        generate_model="generate-model",
        decide_max_tokens=64,
        generate_max_tokens=200,
        temperature=0.2,
        max_suggestion_chars=140,
        request_timeout_seconds=8.0,
    )
    completions = StubCompletions(payload, total_tokens)
    suggester._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]
    return suggester, completions


WINDOW = [
    TranscriptTurn(speaker_id="alice", text="Phần tích hợp thanh toán tới đâu rồi?", language="vi"),
]
SEGMENT = TranscriptTurn(
    speaker_id="bob",
    text="Chắc tuần sau xong, để mình xem lại đã.",
    language="vi",
)


class TestLoad:
    @pytest.mark.asyncio
    async def test_missing_api_key_fails_loudly(self) -> None:
        """Enabling the feature without a key must not degrade to silent inactivity."""
        suggester = OpenAISuggester(
            api_key="",
            decide_model="m",
            generate_model="m",
            decide_max_tokens=64,
            generate_max_tokens=200,
            temperature=0.2,
            max_suggestion_chars=140,
            request_timeout_seconds=8.0,
        )

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await suggester.load()


class TestDecide:
    @pytest.mark.asyncio
    async def test_parses_an_approving_verdict(self) -> None:
        suggester, _ = build_suggester(
            {
                "should_suggest": True,
                "category": "action",
                "confidence": 0.86,
                "reason": "commitment without an owner",
            },
            total_tokens=42,
        )

        decision = await suggester.decide(WINDOW, SEGMENT)

        assert decision.should_suggest is True
        assert decision.category == "action"
        assert decision.confidence == pytest.approx(0.86)
        assert decision.token_count == 42

    @pytest.mark.asyncio
    async def test_declining_verdict_still_reports_tokens(self) -> None:
        """The decide call is billable whether or not it approves — most never approve."""
        suggester, _ = build_suggester({"should_suggest": False, "reason": "small talk"}, 31)

        decision = await suggester.decide(WINDOW, SEGMENT)

        assert decision.should_suggest is False
        assert decision.token_count == 31

    @pytest.mark.asyncio
    async def test_category_is_normalized(self) -> None:
        suggester, _ = build_suggester(
            {"should_suggest": True, "category": "  ACTION ", "confidence": 0.9}
        )

        assert (await suggester.decide(WINDOW, SEGMENT)).category == "action"

    @pytest.mark.asyncio
    async def test_unknown_category_is_passed_through_for_the_worker_to_reject(self) -> None:
        """Coercing it here would hide a misbehaving model behind a valid-looking result."""
        suggester, _ = build_suggester(
            {"should_suggest": True, "category": "banter", "confidence": 0.9}
        )

        assert (await suggester.decide(WINDOW, SEGMENT)).category == "banter"

    @pytest.mark.asyncio
    async def test_confidence_is_clamped(self) -> None:
        suggester, _ = build_suggester(
            {"should_suggest": True, "category": "term", "confidence": 4.2}
        )

        assert (await suggester.decide(WINDOW, SEGMENT)).confidence == 1.0

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_becomes_zero(self) -> None:
        """Zero fails the worker's min_confidence gate, so a malformed field stays silent."""
        suggester, _ = build_suggester(
            {"should_suggest": True, "category": "term", "confidence": "very sure"}
        )

        assert (await suggester.decide(WINDOW, SEGMENT)).confidence == 0.0

    @pytest.mark.asyncio
    async def test_malformed_json_declines_instead_of_raising(self) -> None:
        suggester, _ = build_suggester("not json at all")

        assert (await suggester.decide(WINDOW, SEGMENT)).should_suggest is False

    @pytest.mark.asyncio
    async def test_api_error_declines_instead_of_raising(self) -> None:
        """Raising would leave the message pending and pay for both calls again on redelivery."""
        suggester, completions = build_suggester({})
        completions.error = RuntimeError("upstream 503")

        assert (await suggester.decide(WINDOW, SEGMENT)).should_suggest is False

    @pytest.mark.asyncio
    async def test_unloaded_client_declines_instead_of_raising(self) -> None:
        suggester, _ = build_suggester({})
        suggester._client = None

        assert (await suggester.decide(WINDOW, SEGMENT)).should_suggest is False

    @pytest.mark.asyncio
    async def test_transcript_is_fenced_inside_the_user_message(self) -> None:
        """Participant speech must never arrive as system-level instruction."""
        suggester, completions = build_suggester({"should_suggest": False})

        await suggester.decide(WINDOW, SEGMENT)

        messages = completions.requests[0]["messages"]
        system, user = messages[0], messages[1]
        assert system["role"] == "system"
        assert SEGMENT.text not in system["content"]
        assert "<transcript>" in user["content"]
        assert user["content"].rstrip().endswith("</transcript>")
        assert "LATEST" in user["content"]

    @pytest.mark.asyncio
    async def test_uses_the_cheap_model_and_its_token_ceiling(self) -> None:
        suggester, completions = build_suggester({"should_suggest": False})

        await suggester.decide(WINDOW, SEGMENT)

        request = completions.requests[0]
        assert request["model"] == "decide-model"
        assert request["max_tokens"] == 64
        assert request["response_format"] == {"type": "json_object"}


class TestGenerate:
    APPROVED = SuggestionDecision(
        should_suggest=True, category="action", confidence=0.9, reason="no owner"
    )

    @pytest.mark.asyncio
    async def test_parses_a_suggestion(self) -> None:
        suggester, _ = build_suggester(
            {
                "content": "Chưa có ai nhận phần này.",
                "detail": "Deadline nêu ra nhưng thiếu owner.",
            },
            total_tokens=120,
        )

        suggestion = await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        assert suggestion is not None
        assert suggestion.content == "Chưa có ai nhận phần này."
        assert suggestion.detail == "Deadline nêu ra nhưng thiếu owner."
        assert suggestion.category == "action", "category comes from the decide stage"
        assert suggestion.token_count == 120

    @pytest.mark.asyncio
    async def test_empty_content_yields_nothing(self) -> None:
        """The prompt tells the model to return "" when it cannot be specific."""
        suggester, _ = build_suggester({"content": "   ", "detail": "x"})

        assert await suggester.generate(WINDOW, SEGMENT, self.APPROVED) is None

    @pytest.mark.asyncio
    async def test_api_error_yields_nothing(self) -> None:
        suggester, completions = build_suggester({})
        completions.error = RuntimeError("upstream 503")

        assert await suggester.generate(WINDOW, SEGMENT, self.APPROVED) is None

    @pytest.mark.asyncio
    async def test_reference_documents_go_in_the_user_message(self) -> None:
        """Workspace documents are no more trusted than the transcript itself."""
        suggester, completions = build_suggester({"content": "x"})

        await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, context_snapshot="Doanh thu 1.2 tỷ"
        )

        system, user = completions.requests[0]["messages"]
        assert "Doanh thu 1.2 tỷ" not in system["content"]
        assert "<reference_documents>" in user["content"]
        assert "Doanh thu 1.2 tỷ" in user["content"]

    @pytest.mark.asyncio
    async def test_no_reference_block_when_there_are_no_documents(self) -> None:
        suggester, completions = build_suggester({"content": "x"})

        await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        assert "<reference_documents>" not in completions.requests[0]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_segment_language_is_passed_through(self) -> None:
        suggester, completions = build_suggester({"content": "x"})

        await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        assert "vi" in completions.requests[0]["messages"][1]["content"]

    @pytest.mark.asyncio
    async def test_uses_the_full_model_and_char_budget(self) -> None:
        suggester, completions = build_suggester({"content": "x"})

        await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        request = completions.requests[0]
        assert request["model"] == "generate-model"
        assert request["max_tokens"] == 200
        assert "140 characters" in request["messages"][0]["content"]


class TestNullSuggester:
    @pytest.mark.asyncio
    async def test_declines_everything(self) -> None:
        suggester = NullSuggester()
        await suggester.load()

        decision = await suggester.decide(WINDOW, SEGMENT)

        assert decision.should_suggest is False
        assert await suggester.generate(WINDOW, SEGMENT, decision) is None


class TestEntrypointWiring:
    """build_suggester decides whether a deployment can produce anything at all."""

    def test_disabled_deployment_needs_no_api_key(self) -> None:
        from shared.config import SuggestionSettings
        from suggestion_worker.__main__ import build_suggester

        suggester = build_suggester(SuggestionSettings(enabled=False, api_key=""))

        assert isinstance(suggester, NullSuggester)

    def test_enabled_deployment_gets_the_model_backed_suggester(self) -> None:
        from shared.config import SuggestionSettings
        from suggestion_worker.__main__ import build_suggester

        suggester = build_suggester(
            SuggestionSettings(
                enabled=True,
                api_key="test-key",
                decide_model="d",
                generate_model="g",
            )
        )

        assert isinstance(suggester, OpenAISuggester)
        assert suggester.decide_model == "d"
        assert suggester.generate_model == "g"


# -------------------------------------------------------------------------------------------
# Where a hint says it came from.
#
# The same rule the chat assistant's markers enforce, arrived at from the other side: a model
# asked for its source will always produce one, and a plausible filename under an invented
# figure is worse than the bare hint — it turns a guess into a citation.
# -------------------------------------------------------------------------------------------

SNAPSHOT = (
    "RAG CONTEXT (STATIC SNAPSHOT FOR MEETING):\n"
    "--- Document: Q3-budget.xlsx ---\n"
    "Marketing spend: 1.2 tỷ\n"
    "-----------------------------------\n"
    "--- Document: Kế hoạch 2026.docx ---\n"
    "Mục tiêu doanh thu\n"
    "-----------------------------------\n"
)


class TestGeneratedSources:
    APPROVED = SuggestionDecision(
        should_suggest=True, category="fact", confidence=0.8, reason="figure discussed"
    )

    @pytest.mark.asyncio
    async def test_a_document_the_snapshot_contained_is_kept(self) -> None:
        suggester, _ = build_suggester(
            {"content": "Ngân sách marketing là 1.2 tỷ.", "source": "Q3-budget.xlsx"}
        )

        suggestion = await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, context_snapshot=SNAPSHOT
        )

        assert suggestion is not None
        assert suggestion.sources == ("Q3-budget.xlsx",)

    @pytest.mark.asyncio
    async def test_an_invented_document_is_dropped_and_the_hint_survives(self) -> None:
        # The hint is still a correct hint about the transcript. Losing it over its footnote
        # would be the worse trade.
        suggester, _ = build_suggester(
            {"content": "Ngân sách marketing là 1.2 tỷ.", "source": "Q4-forecast.pdf"}
        )

        suggestion = await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, context_snapshot=SNAPSHOT
        )

        assert suggestion is not None
        assert suggestion.content == "Ngân sách marketing là 1.2 tỷ."
        assert suggestion.sources == ()

    @pytest.mark.asyncio
    async def test_a_name_is_returned_as_the_snapshot_spelled_it(self) -> None:
        # Matching is case-insensitive; what reaches the chip is the document's own casing, not
        # whatever the model typed.
        suggester, _ = build_suggester(
            {"content": "Mục tiêu doanh thu.", "source": "kế hoạch 2026.DOCX"}
        )

        suggestion = await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, context_snapshot=SNAPSHOT
        )

        assert suggestion is not None
        assert suggestion.sources == ("Kế hoạch 2026.docx",)

    @pytest.mark.asyncio
    async def test_a_hint_from_the_transcript_names_nothing(self) -> None:
        suggester, _ = build_suggester({"content": "Ai nhận phần này?", "source": ""})

        suggestion = await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        assert suggestion is not None
        assert suggestion.sources == ()

    @pytest.mark.asyncio
    async def test_a_source_named_with_no_snapshot_at_all_is_dropped(self) -> None:
        suggester, _ = build_suggester({"content": "Ngân sách 1.2 tỷ.", "source": "Q3-budget.xlsx"})

        suggestion = await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        assert suggestion is not None
        assert suggestion.sources == ()
