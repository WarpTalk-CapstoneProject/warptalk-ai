"""Tests for OpenAI SDK configuration shared across AI workers."""

from __future__ import annotations

import pytest

from shared.config import AssistantSettings, resolve_openai_api_key


def test_resolve_openai_api_key_uses_shared_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shared-test-key")

    assert resolve_openai_api_key("") == "shared-test-key"


def test_resolve_openai_api_key_prefers_stage_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "shared-test-key")

    assert resolve_openai_api_key("stage-test-key") == "stage-test-key"


def test_assistant_model_uses_assistant_env_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "legacy-shared-model")
    monkeypatch.setenv("ASSISTANT_MODEL", "assistant-test-model")

    assert AssistantSettings().model == "assistant-test-model"
