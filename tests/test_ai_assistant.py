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


async def test_generate_structured_summary_requests_bilingual_output_for_multiple_target_languages() -> None:
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
