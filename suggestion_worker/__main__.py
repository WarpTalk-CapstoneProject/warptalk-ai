"""Suggestion Worker entry point.

Runs one consumer: SuggestionWorker over stt:results under its own consumer group.

While SUGGESTION_ENABLED is unset the worker is wired with NullSuggester — it starts,
reports healthy and consumes its stream while producing nothing and needing no API key.
That is what lets the container be rolled out and verified in production before the
feature is switched on. Enabling it without an API key is a hard startup failure rather
than silent inactivity, so a misconfigured rollout is visible immediately.
"""

import asyncio

from shared.config import SuggestionSettings, WorkerSettings, resolve_openai_api_key
from shared.logger import setup_logging
from suggestion_worker.suggester import NullSuggester, OpenAISuggester, Suggester
from suggestion_worker.worker import SuggestionWorker


def build_suggester(settings: SuggestionSettings) -> Suggester:
    if not settings.enabled:
        return NullSuggester()

    return OpenAISuggester(
        api_key=resolve_openai_api_key(settings.api_key),
        decide_model=settings.decide_model,
        generate_model=settings.generate_model,
        decide_max_tokens=settings.decide_max_tokens,
        generate_max_tokens=settings.generate_max_tokens,
        temperature=settings.temperature,
        max_suggestion_chars=settings.max_suggestion_chars,
        request_timeout_seconds=settings.request_timeout_seconds,
    )


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    suggestion_settings = SuggestionSettings()
    worker = SuggestionWorker(
        suggestion_settings=suggestion_settings,
        suggester=build_suggester(suggestion_settings),
        settings=worker_settings,
    )

    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
