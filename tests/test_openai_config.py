"""Tests for OpenAI SDK configuration shared across AI workers."""

from __future__ import annotations

import pytest

from shared.config import AssistantSettings, SuggestionSettings, resolve_openai_api_key


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


def test_suggestion_settings_ship_disabled() -> None:
    """Production rolls the worker out dark — enabling it is a deliberate second step."""
    assert SuggestionSettings().enabled is False


def test_suggestion_settings_use_own_env_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SUGGESTION_ must not collide with the two other assistant surfaces' prefixes."""
    monkeypatch.setenv("ASSISTANT_MODEL", "assistant-test-model")
    monkeypatch.setenv("SUGGESTION_ENABLED", "true")
    monkeypatch.setenv("SUGGESTION_DECIDE_MODEL", "decide-test-model")
    monkeypatch.setenv("SUGGESTION_COOLDOWN_SECONDS", "90")

    settings = SuggestionSettings()

    assert settings.enabled is True
    assert settings.decide_model == "decide-test-model"
    assert settings.cooldown_seconds == 90
    # Untouched by ASSISTANT_MODEL above — the two surfaces are configured independently.
    assert settings.generate_model == "gpt-4.1"
