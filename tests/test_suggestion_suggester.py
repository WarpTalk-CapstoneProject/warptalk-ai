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
    _DECIDE_SYSTEM_PROMPT,
    NullSuggester,
    OpenAISuggester,
    SuggestionDecision,
    TranscriptTurn,
    _generate_system_prompt,
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
    payload: Any,
    total_tokens: int = 50,
    *,
    decide_model: str = "decide-model",
    generate_model: str = "generate-model",
    reasoning_effort: str | None = None,
) -> tuple[OpenAISuggester, StubCompletions]:
    suggester = OpenAISuggester(
        api_key="test-key",
        decide_model=decide_model,
        generate_model=generate_model,
        decide_max_tokens=64,
        generate_max_tokens=200,
        temperature=0.2,
        max_suggestion_chars=140,
        request_timeout_seconds=8.0,
        reasoning_effort=reasoning_effort,
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


class TestReasoningModelBudget:
    """Production generates on gpt-5.6-luna, a reasoning model.

    Its `max_completion_tokens` is shared between hidden reasoning and the visible answer. At
    the default effort the 200-token cap was sometimes spent on reasoning alone, which OpenAI
    reports as 400 "max_tokens or model output limit was reached" (prod, 1 Oct 2026).
    """

    APPROVED = TestGenerate.APPROVED

    @pytest.mark.asyncio
    async def test_generate_on_a_reasoning_model_sends_the_configured_effort(self) -> None:
        suggester, completions = build_suggester(
            {"content": "x"}, generate_model="gpt-5.6-luna", reasoning_effort="none"
        )

        await suggester.generate(WINDOW, SEGMENT, self.APPROVED)

        request = completions.requests[0]
        assert request["reasoning_effort"] == "none"
        assert request["max_completion_tokens"] == 200
        assert "max_tokens" not in request
        assert "temperature" not in request

    @pytest.mark.asyncio
    async def test_decide_on_a_reasoning_model_sends_the_configured_effort(self) -> None:
        suggester, completions = build_suggester(
            {"should_suggest": False}, decide_model="gpt-5.6-luna", reasoning_effort="low"
        )

        await suggester.decide(WINDOW, SEGMENT)

        assert completions.requests[0]["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_a_non_reasoning_model_never_receives_an_effort(self) -> None:
        """gpt-4o-mini (prod's decide model) answers reasoning_effort with a 400."""
        suggester, completions = build_suggester(
            {"should_suggest": False}, decide_model="gpt-4o-mini", reasoning_effort="none"
        )

        await suggester.decide(WINDOW, SEGMENT)

        assert "reasoning_effort" not in completions.requests[0]

    def test_production_settings_turn_reasoning_off(self) -> None:
        from shared.config import SuggestionSettings
        from suggestion_worker.__main__ import build_suggester as build_from_settings

        settings = SuggestionSettings(
            enabled=True, api_key="test-key", generate_model="gpt-5.6-luna"
        )
        suggester = build_from_settings(settings)

        assert settings.reasoning_effort == "none"
        assert isinstance(suggester, OpenAISuggester)
        assert suggester.reasoning_effort == "none"

    @pytest.mark.asyncio
    async def test_the_client_does_not_retry(self) -> None:
        """The SDK default re-sent a timed-out hint twice: ~27s of a stalled consumer each."""
        suggester = OpenAISuggester(
            api_key="test-key",
            decide_model="m",
            generate_model="m",
            decide_max_tokens=64,
            generate_max_tokens=200,
            temperature=0.2,
            max_suggestion_chars=140,
            request_timeout_seconds=8.0,
        )

        await suggester.load()

        assert suggester._require_client().max_retries == 0


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


# ── WT-582: the term category could fire but never answer ────────────────────────────────


def test_term_may_answer_from_general_knowledge() -> None:
    """`decide` fires `term` on a term the meeting has NOT defined, and `generate` used to be
    told to ground every claim in the transcript or the documents and otherwise return "".

    Those two rules cannot both hold. The trigger condition for the category was the same
    condition that forced an empty answer, so "we should look into Next.js" produced a badge
    that opened onto nothing. The reader saw an assistant that had noticed the term and had
    nothing to say about it.
    """
    prompt = _generate_system_prompt(160, "term")

    assert "general knowledge" in prompt
    # The instruction that used to silence it must now be explicitly disclaimed.
    assert "do NOT reach for it merely because the meeting never explained the term" in prompt


def test_meeting_specific_facts_still_may_not_come_from_memory() -> None:
    """Opening the door to public knowledge must not open it to invented meeting facts.

    An invented owner or date is far worse than a missing hint: it is indistinguishable from a
    sourced one, and it is about people in the room.
    """
    prompt = _generate_system_prompt(160, "term")

    assert "SPECIFIC TO THIS MEETING OR THIS TEAM" in prompt
    assert "Never supply one of those from memory" in prompt
    for meeting_fact in ("who owns something", "dates", "figures", "decisions"):
        assert meeting_fact in prompt


def test_only_term_and_correction_may_use_public_knowledge() -> None:
    """`fact` in particular must not: it means "the meeting's own documents cover this", and a
    recalled figure reaches the reader looking exactly like a sourced one.
    """
    prompt = _generate_system_prompt(160, "fact")
    assert "only the `term` and `correction` hints may use it" in prompt


def test_correction_covers_a_claim_that_is_wrong_about_the_world() -> None:
    """It used to mean intra-transcript contradictions only, so a statement that was simply
    wrong — ".NET 10 for the frontend" — matched no category at all and passed in silence.
    """
    assert "misstates a widely established fact" in _DECIDE_SYSTEM_PROMPT
    assert ".NET 10 for the frontend" in _DECIDE_SYSTEM_PROMPT


def test_correction_keeps_a_high_bar_and_corrects_the_fact_not_the_person() -> None:
    """The failure mode of a wider `correction` is an assistant that contradicts people over
    preferences, or over an unusual-but-deliberate choice. Both are named in the prompts.
    """
    assert "not a matter of opinion" in _DECIDE_SYSTEM_PROMPT
    assert "may simply be doing differently on purpose" in _DECIDE_SYSTEM_PROMPT

    contract = _generate_system_prompt(160, "correction")
    assert "Correct the FACT, never the person" in contract


def test_term_fires_on_a_term_being_used_not_only_asked_about() -> None:
    """ "we should look into Next.js" is a statement, not a question. Reading `term` as
    question-shaped is why a plain mention produced nothing.
    """
    assert "fires on a term being USED as much as on one being asked about" in _DECIDE_SYSTEM_PROMPT


class TestReaderLanguages:
    """WT-922 — one generate call also writes the hint in every reader's language."""

    APPROVED = SuggestionDecision(
        should_suggest=True, category="term", confidence=0.9, reason="undefined acronym"
    )

    def test_the_prompt_asks_for_translations_only_when_there_are_readers(self) -> None:
        assert "translations" not in _generate_system_prompt(140, "term")
        prompt = _generate_system_prompt(140, "term", ["en", "ja"])
        assert '"en" (English)' in prompt
        assert '"ja" (Japanese)' in prompt
        assert '"translations": object' in prompt

    @pytest.mark.asyncio
    async def test_translations_for_requested_languages_are_kept(self) -> None:
        suggester, completions = build_suggester(
            {
                "content": "SLA là cam kết mức dịch vụ.",
                "detail": "Bob nhắc SLA mà chưa giải thích.",
                "translations": {
                    "en": {
                        "content": "SLA is a service-level agreement.",
                        "detail": "Bob said SLA.",
                    },
                    "ja": {"content": "", "detail": "empty content is dropped"},
                    "fr": {"content": "never requested", "detail": ""},
                },
            }
        )

        suggestion = await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, reader_languages=["en", "ja"]
        )

        assert suggestion is not None
        assert suggestion.content == "SLA là cam kết mức dịch vụ."
        assert suggestion.translations == {
            "en": ("SLA is a service-level agreement.", "Bob said SLA.")
        }
        assert '"en" (English)' in completions.requests[0]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_malformed_translations_cost_only_the_translations(self) -> None:
        suggester, _ = build_suggester({"content": "Hint.", "translations": ["not", "a", "dict"]})

        suggestion = await suggester.generate(
            WINDOW, SEGMENT, self.APPROVED, reader_languages=["en"]
        )

        assert suggestion is not None
        assert suggestion.content == "Hint."
        assert suggestion.translations == {}

    @pytest.mark.asyncio
    async def test_the_token_budget_grows_with_each_reader_language(self) -> None:
        """At a flat budget two translations truncated the JSON and lost the hint for everyone."""
        suggester, completions = build_suggester({"content": "Hint."})

        await suggester.generate(WINDOW, SEGMENT, self.APPROVED)
        await suggester.generate(WINDOW, SEGMENT, self.APPROVED, reader_languages=["en", "ja"])

        def budget(request: dict[str, Any]) -> int:
            return request.get("max_completion_tokens") or request["max_tokens"]

        assert budget(completions.requests[0]) == 200
        assert budget(completions.requests[1]) == 600


def test_a_reader_copy_translates_the_quote_too() -> None:
    """A hint for an English reader quoted the Vietnamese line verbatim, in the middle of an
    English sentence — the one part of the card that reader could not read (prod, 2 Oct)."""
    from suggestion_worker.suggester import _translations_rule

    rule = _translations_rule(["en"])

    assert "INCLUDING anything quoted from the transcript" in rule
    assert "keep the quoted words" not in rule
