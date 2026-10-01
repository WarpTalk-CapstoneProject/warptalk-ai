"""completion_options: the one place that knows which generation controls a model accepts."""

from __future__ import annotations

from shared.openai_options import completion_options


def test_reasoning_effort_reaches_a_gpt5_model() -> None:
    assert completion_options("gpt-5.6-luna", 200, 0.2, reasoning_effort="none") == {
        "max_completion_tokens": 200,
        "reasoning_effort": "none",
    }


def test_reasoning_effort_is_never_sent_to_a_non_reasoning_model() -> None:
    assert completion_options("gpt-4o-mini", 64, 0.2, reasoning_effort="none") == {
        "max_tokens": 64,
        "temperature": 0.2,
    }


def test_no_effort_is_invented_when_the_caller_did_not_ask() -> None:
    assert completion_options("gpt-5.6-luna", 200, 0.2) == {"max_completion_tokens": 200}
