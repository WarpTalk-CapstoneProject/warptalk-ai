"""Tests for ai_assistant_worker.chat_tools._search_terminology.

Focus: the token-cost question flagged in docs/global-glossary-plan.md — GlossaryTerm.Context
and GlobalGlossaryTerm.Definition are unbounded TEXT columns (no VARCHAR cap the way
Term/PreferredTranslation have), so without _truncate_terminology_context, one long-winded
term definition would make every _search_terminology call that surfaces it cost proportionally
many (unbounded) tokens. These tests measure the actual worst-case JSON size and pin it to a
concrete ceiling so that bound can't silently regress.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_tools import (
    TERMINOLOGY_CONTEXT_CHAR_LIMIT,
    ToolContext,
    _search_terminology,
    _truncate_terminology_context,
)


def _response(status_code: int, payload) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _make_client(routes: dict[str, MagicMock]) -> AsyncMock:
    """A fake httpx.AsyncClient.get() that dispatches on the request path prefix."""
    client = AsyncMock()

    async def _get(url: str, **_kwargs):
        for path, response in routes.items():
            if url.startswith(path):
                return response
        raise AssertionError(f"Unexpected request: {url}")

    client.get.side_effect = _get
    return client


def _make_ctx(transcript_client: AsyncMock) -> ToolContext:
    return ToolContext(
        workspace_id="ws-1",
        user_id="user-1",
        bearer_token="Bearer test-token",
        workspace_client=AsyncMock(),
        transcript_client=transcript_client,
        translation_room_client=AsyncMock(),
        openai_client=None,
        model="gpt-4.1",
        redis=MagicMock(),
    )


class TestTruncateTerminologyContext:
    def test_passes_short_text_through_unchanged(self) -> None:
        assert _truncate_terminology_context("A design role.") == "A design role."

    def test_passes_none_through_unchanged(self) -> None:
        assert _truncate_terminology_context(None) is None

    def test_truncates_text_longer_than_limit(self) -> None:
        long_text = "x" * (TERMINOLOGY_CONTEXT_CHAR_LIMIT + 500)
        result = _truncate_terminology_context(long_text)
        assert result is not None
        assert len(result) == TERMINOLOGY_CONTEXT_CHAR_LIMIT


class TestSearchTerminologyGlobalFallback:
    async def test_global_fallback_triggers_only_when_workspace_matches_are_sparse(self) -> None:
        """The fallback (and its token cost) is only paid when the workspace glossary alone
        didn't already find >= 5 matches — see docs/global-glossary-plan.md.
        """
        five_terms = [
            {
                "sourceTerm": f"term{i}",
                "targetTerm": f"t{i}",
                "context": "arch note",
                "domain": None,
                "isActive": True,
            }
            for i in range(5)
        ]
        client = _make_client(
            {
                "/api/v1/glossaries/workspace/ws-1": _response(
                    200, [{"id": "g1", "name": "Eng", "isActive": True}]
                ),
                "/api/v1/glossaries/g1/terms": _response(200, five_terms),
            }
        )
        ctx = _make_ctx(client)

        result = json.loads(await _search_terminology(ctx, {"query": "arch"}))

        assert len(result) == 5
        assert all(m["source"] == "workspace" for m in result)
        # /api/v1/glossaries/global must never even be requested once the workspace glossary
        # alone already satisfied the match budget.
        requested_paths = [call.args[0] for call in client.get.await_args_list]
        assert not any(path.startswith("/api/v1/glossaries/global") for path in requested_paths)

    async def test_global_term_truncates_long_definition(self) -> None:
        client = _make_client(
            {
                "/api/v1/glossaries/workspace/ws-1": _response(200, []),
                "/api/v1/glossaries/global": _response(
                    200,
                    [
                        {
                            "term": "architect",
                            "preferredTranslation": "architect",
                            "definition": "A "
                            * 1000,  # deliberately far longer than the DB VARCHAR caps
                            "businessDomain": None,
                        }
                    ],
                ),
            }
        )
        ctx = _make_ctx(client)

        result = json.loads(await _search_terminology(ctx, {"query": "architect"}))

        assert len(result) == 1
        assert result[0]["source"] == "global"
        assert len(result[0]["context"]) == TERMINOLOGY_CONTEXT_CHAR_LIMIT

    async def test_global_terms_already_covered_by_workspace_are_skipped(self) -> None:
        """A workspace term always wins over a global term with the same name — same rule
        GlossaryStartedEventConsumer uses for the STT/MT prompt merge — so the global fallback
        must not re-add (and re-cost tokens for) a term the workspace already defined.
        """
        client = _make_client(
            {
                "/api/v1/glossaries/workspace/ws-1": _response(
                    200, [{"id": "g1", "name": "Eng", "isActive": True}]
                ),
                "/api/v1/glossaries/g1/terms": _response(
                    200,
                    [
                        {
                            "sourceTerm": "architect",
                            "targetTerm": "kien truc su",
                            "context": None,
                            "domain": None,
                            "isActive": True,
                        }
                    ],
                ),
                "/api/v1/glossaries/global": _response(
                    200,
                    [
                        {
                            "term": "architect",
                            "preferredTranslation": "architect",
                            "definition": "A design role.",
                            "businessDomain": None,
                        },
                        {
                            "term": "sprint",
                            "preferredTranslation": "sprint",
                            "definition": "A work cycle.",
                            "businessDomain": None,
                        },
                    ],
                ),
            }
        )
        ctx = _make_ctx(client)

        result = json.loads(await _search_terminology(ctx, {"query": "a"}))

        sources_by_term = {m["term"].lower(): m["source"] for m in result}
        assert sources_by_term["architect"] == "workspace"
        assert "sprint" in sources_by_term  # not shadowed, still surfaced from global

    async def test_worst_case_response_size_is_bounded(self) -> None:
        """The actual "token cost" measurement: 8 matches (the hard cap in _search_terminology)
        each carrying a maximally long context/definition must still produce a JSON response
        whose size is a small, predictable multiple of TERMINOLOGY_CONTEXT_CHAR_LIMIT — not
        unbounded. ~4 chars/token (OpenAI's own rule-of-thumb estimate) puts this well under
        1,000 tokens even in the worst case, versus unbounded before this fix.
        """
        global_terms = [
            {
                "term": f"term-{i}",
                "preferredTranslation": f"translation-{i}",
                "definition": "A " * 1000,
                "businessDomain": "Engineering",
            }
            for i in range(10)
        ]
        client = _make_client(
            {
                "/api/v1/glossaries/workspace/ws-1": _response(200, []),
                "/api/v1/glossaries/global": _response(200, global_terms),
            }
        )
        ctx = _make_ctx(client)

        raw = await _search_terminology(ctx, {"query": "term"})
        result = json.loads(raw)

        assert len(result) == 8  # the hard cap in _search_terminology
        # Rough per-match ceiling: context (300) + term/translation/domain/glossary label
        # overhead (~100 chars) + JSON punctuation/keys (~100 chars) = ~500 chars/match.
        assert len(raw) < 8 * 500
