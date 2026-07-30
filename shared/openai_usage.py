"""Helpers for extracting token usage from OpenAI SDK responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _read(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    @property
    def uncached_prompt_tokens(self) -> int:
        return max(self.prompt_tokens - self.cached_tokens, 0)

    @property
    def has_tokens(self) -> bool:
        return self.prompt_tokens > 0 or self.cached_tokens > 0 or self.completion_tokens > 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    @classmethod
    def from_openai_usage(cls, usage: Any) -> TokenUsage:
        if usage is None:
            return cls()

        prompt_tokens = int(
            _read(usage, "prompt_tokens", _read(usage, "input_tokens", 0)) or 0
        )
        completion_tokens = int(
            _read(usage, "completion_tokens", _read(usage, "output_tokens", 0)) or 0
        )

        details = _read(
            usage,
            "prompt_tokens_details",
            _read(usage, "input_token_details"),
        )
        cached_tokens = int(_read(details, "cached_tokens", 0) or 0)

        return cls(
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
        )
