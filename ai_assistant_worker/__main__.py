"""AI Assistant Worker entry point.

Runs three independent Redis-stream consumers concurrently in one process:
    - AIAssistantWorker — per-meeting summarization (stt:results)
    - ChatAssistantWorker — global "Ask WarpTalk" tool-calling chat (assistant:chat_requests)
    - SummaryTemplateWorker — re-summarise a finished meeting (assistant:summary_requests)
All three are lightweight consumers with their own consumer group; no need for separate
containers.

ONE WORKER'S DEATH IS NOT THE OTHERS' DEATH
    They used to be started with a bare asyncio.gather, which has no return_exceptions — the
    first worker to raise brought main() down and took its siblings with it. Three features
    in one process, any one of which could silently remove the other two.

    They are supervised individually now: a worker that dies is logged and restarted, and its
    neighbours keep serving. The process only exits when every worker has stopped, which is a
    real fault the container health check should act on.
"""

import asyncio
import contextlib

from ai_assistant_worker.chat_worker import ChatAssistantWorker
from ai_assistant_worker.summary_template_worker import SummaryTemplateWorker
from ai_assistant_worker.worker import AIAssistantWorker
from shared.config import AssistantSettings, ChatAssistantSettings, WorkerSettings
from shared.logger import get_logger, setup_logging


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    summarization_worker = AIAssistantWorker(
        assistant_settings=AssistantSettings(),
        settings=worker_settings,
    )
    chat_worker = ChatAssistantWorker(
        chat_settings=ChatAssistantSettings(),
        settings=worker_settings,
    )
    # Re-summarising a finished meeting under a different template. Separate from the
    # summarisation worker above because it reads the SAVED transcript rather than the live
    # accumulator, which is gone by the time anyone asks.
    summary_template_worker = SummaryTemplateWorker(
        assistant_settings=AssistantSettings(),
        settings=worker_settings,
    )

    await asyncio.gather(
        _supervise("assistant", summarization_worker),
        _supervise("assistant-chat", chat_worker),
        _supervise("summary-template", summary_template_worker),
    )


RESTART_DELAY_SECONDS = 5


async def _supervise(name: str, worker: object) -> None:
    """Keep one worker running without letting its failures reach its siblings.

    A worker that cannot start at all — a missing key, an unreachable dependency — would
    otherwise take the whole container with it on the first attempt, so the chat assistant
    could disappear because an unrelated summariser was misconfigured. Retrying with a fixed
    delay keeps that fault local and visible in the logs rather than fatal and silent.
    """
    logger = get_logger(__name__)

    while True:
        try:
            await worker.start()  # type: ignore[attr-defined]
            logger.warning("worker_stopped", worker=name)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("worker_crashed", worker=name)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
