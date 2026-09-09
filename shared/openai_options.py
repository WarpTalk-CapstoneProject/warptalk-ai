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

The same argument now covers one non-generation rule: how long a Realtime socket may be
held before OpenAI closes it. Two workers pool those sockets, both learned the 60-minute
cap the same way on the same evening, and a number that lives in one of them is a number
the other will get wrong.
"""

from __future__ import annotations

import time
from typing import Any

# OpenAI closes a Realtime session at 60 minutes, whoever is holding it and whatever it is
# doing, with `1001 (going away) Your session hit the maximum duration of 60 minutes`.
#
# THIS IS A CLOCK ON THE SOCKET, NOT ON ITS USE, which is what both pools got wrong. They
# evicted on idleness and on failure — neither of which a still-connected, recently-used,
# 61-minute-old socket triggers. Production, 2026-09-08 17:33Z: the first audio chunk of the
# only meeting held that evening claimed a socket the STT warm pool had opened at worker
# startup, 4h40m earlier. It failed. So did translation's, at the same moment, on its own
# pooled connection. The user experience of that is ~6 extra seconds before the first caption
# of the meeting, and it lands on whoever speaks first after a quiet hour.
#
# Ten minutes of headroom under the cap: enough that a session claimed just under the line
# cannot cross it mid-utterance, and cheap, because retiring one costs a background reconnect
# while failing on one costs a real sentence.
REALTIME_SESSION_MAX_AGE_S = 50 * 60.0


def realtime_session_expired(opened_at: float | None, now: float | None = None) -> bool:
    """Whether a Realtime socket opened at `opened_at` is too old to hand to a caller.

    `opened_at` is a `time.monotonic()` reading. None means "nobody stamped it", which is
    treated as EXPIRED: an unstamped socket is one this rule cannot vouch for, and the cost of
    being wrong is a reconnect, not a failed transcription.
    """
    if opened_at is None:
        return True
    return (time.monotonic() if now is None else now) - opened_at >= REALTIME_SESSION_MAX_AGE_S


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


def reasoning_summary_options(model: str) -> dict[str, Any]:
    """Ask a reasoning model to narrate what it is doing, when it can.

    WHY THIS IS WORTH ASKING FOR
        A trail built only from tool calls can say "Searching the web" and never say WHY — and
        between two calls, where the model is deciding what to do next, it says nothing at all.
        The summary is the model's own account of the step it is taking, which is the only
        source for that sentence: everything else the client could show is a label somebody
        wrote in advance about a tool, not a description of this turn.

    Off for anything outside the gpt-5 family, and separate from `responses_options` on
    purpose. Sending `reasoning` to a model that does not take it is a 400 on the FIRST
    request — the exact trap the header of this module exists to describe — and the assistant
    is the only caller that wants it, so widening the shared helper would hand it to
    translation and suggestion as well.
    """
    if not model.startswith("gpt-5"):
        return {}
    return {"reasoning": {"summary": "auto"}}
