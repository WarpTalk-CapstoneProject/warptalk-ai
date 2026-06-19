"""Tests for WarpBot assistant configuration."""

from __future__ import annotations

import pytest

from ai_assistant_worker.assistant import MeetingAssistant


async def test_assistant_requires_openai_api_key() -> None:
    assistant = MeetingAssistant(api_key="")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await assistant.load()
