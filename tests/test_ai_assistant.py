"""Tests for WarpBot assistant configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_assistant_worker.assistant import MeetingAssistant


async def test_assistant_requires_openai_api_key() -> None:
    assistant = MeetingAssistant(api_key="")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await assistant.load()


def _make_assistant_with_fake_client(response_content: str) -> MeetingAssistant:
    assistant = MeetingAssistant(api_key="test-key")
    fake_message = SimpleNamespace(content=response_content)
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=fake_response))
        )
    )
    assistant._client = fake_client
    return assistant


async def test_generate_structured_summary_returns_insufficient_data_for_empty_transcript() -> None:
    assistant = MeetingAssistant(api_key="test-key")

    result = await assistant.generate_structured_summary("   ")

    assert result["insufficientData"] is True
    assert result["decisions"] == []
    assert result["actionItems"] == []


async def test_a_transcript_of_empty_segments_is_insufficient_not_summarised() -> None:
    """WT-478 — the bug, at the layer that decides.

    Timestamps and speaker labels with nothing said between them are truthy to `.strip()`,
    so this transcript used to reach the model. The model then reported the transcript was
    empty, the call SUCCEEDED, insufficientData stayed False, and the UI rendered that
    report as the meeting's summary. The fake client below would answer anything, so if the
    gate regresses this test fails on the assertion rather than on a missing client.
    """
    assistant = _make_assistant_with_fake_client('{"summary": "the model was asked anyway"}')

    result = await assistant.generate_structured_summary("[t=0] [Nhi] \n[t=1200] [Ky]    ")

    assert result["insufficientData"] is True
    assert result["summary"] == "No transcript content to summarize."


async def test_a_short_but_real_transcript_is_still_summarised() -> None:
    # The other half of the ticket: "kể cả khi nội dung ngắn". Two sentences is a meeting.
    payload = {"summary": "Nhi confirmed the Q3 receivables.", "decisions": [], "actionItems": []}
    assistant = _make_assistant_with_fake_client(json.dumps(payload))

    result = await assistant.generate_structured_summary("[t=0] [Nhi] chốt công nợ quý ba")

    assert result["insufficientData"] is False
    assert result["summary"] == payload["summary"]


async def test_generate_structured_summary_parses_model_json() -> None:
    payload = {
        "summary": "The team reviewed the Q3 roadmap.",
        "decisions": ["Ship the beta by August"],
        "actionItems": [{"owner": "Alice", "task": "Draft the release notes"}],
    }
    assistant = _make_assistant_with_fake_client(json.dumps(payload))

    result = await assistant.generate_structured_summary("Alice: let's ship the beta by August.")

    assert result["summary"] == payload["summary"]
    assert result["decisions"] == payload["decisions"]
    assert result["actionItems"] == payload["actionItems"]
    assert result["insufficientData"] is False


async def test_structured_summary_requests_bilingual_output() -> None:
    assistant = _make_assistant_with_fake_client(json.dumps({"summary": "ok"}))

    await assistant.generate_structured_summary(
        "Some transcript text.",
        target_languages=["en", "vi"],
    )

    call_kwargs = assistant._client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "en, vi" in system_message
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_generate_structured_summary_falls_back_gracefully_on_malformed_json() -> None:
    assistant = _make_assistant_with_fake_client("not valid json")

    result = await assistant.generate_structured_summary("Some transcript text.")

    assert result["insufficientData"] is True
    assert result["decisions"] == []
    assert result["actionItems"] == []
