"""Model-compatible generation controls for OpenAI chat completions.

Lifted out of translation_worker/translator.py, which was the ONLY place that knew
GPT-5 models reject the legacy `max_tokens` parameter and only accept their default
temperature. Every other worker — assistant, chat tools, suggestion, security —
passed both unconditionally, so pointing any of them at a gpt-5 model turned what
looks like a one-line config change into an API error on the very first request.
That is exactly the trap ASSISTANT_MODEL=gpt-5.6-luna would have walked into.

Keep this the single place that encodes the rule. A worker that builds its own
options dict is a worker that will break the next time a model family changes its
parameter contract.
"""

from __future__ import annotations

from typing import Any


def completion_options(
    model: str,
    token_limit: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Return the generation controls `model` actually accepts.

    Both arguments are optional and omitted entirely when None, so this never
    invents a cap or a temperature a caller did not already have. That matters:
    chat_tools._translate_text deliberately runs uncapped, and quietly acquiring a
    ceiling here would truncate long translations that used to succeed.

    Note for callers that depend on deterministic output: a gpt-5 model silently
    does NOT honour `temperature`, because there is no way to send it. Treat such a
    model as non-deterministic rather than assuming 0.0 took effect — translation
    caching learned this the expensive way (see translator.py's TTS-cache comment).
    """
    options: dict[str, Any] = {}

    if model.startswith("gpt-5"):
        # GPT-5 renamed the cap and accepts only its default temperature, so the
        # legacy pair is dropped rather than translated.
        if token_limit is not None:
            options["max_completion_tokens"] = token_limit
        return options

    if token_limit is not None:
        options["max_tokens"] = token_limit
    if temperature is not None:
        options["temperature"] = temperature
    return options


def responses_options(
    model: str,
    token_limit: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """The same question for /v1/responses, which names the cap differently.

    Responses calls it `max_output_tokens`, and the temperature rule is unchanged:
    verified against the live API, gpt-5.6-luna answers `temperature` with

        400 Unsupported parameter: 'temperature' is not supported with this model

    on this endpoint exactly as it does on chat completions, while gpt-4o-mini accepts
    it. Keeping the two helpers side by side means a caller switching endpoints cannot
    accidentally carry a parameter name the new one rejects — which is the mistake that
    took the chat assistant down in v47.
    """
    options: dict[str, Any] = {}
    if token_limit is not None:
        options["max_output_tokens"] = token_limit
    if temperature is not None and not model.startswith("gpt-5"):
        options["temperature"] = temperature
    return options
