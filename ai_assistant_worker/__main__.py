"""AI Assistant Worker entry point.

Runs two independent Redis-stream consumers concurrently in one process:
    - AIAssistantWorker — per-meeting summarization (stt:results)
    - ChatAssistantWorker — global "Ask WarpTalk" tool-calling chat (assistant:chat_requests)
Both are lightweight consumers with their own consumer group; no need for separate containers.
"""

import asyncio

from ai_assistant_worker.chat_worker import ChatAssistantWorker
from ai_assistant_worker.worker import AIAssistantWorker
from shared.config import AssistantSettings, ChatAssistantSettings, WorkerSettings
from shared.logger import setup_logging


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

    await asyncio.gather(summarization_worker.start(), chat_worker.start())


if __name__ == "__main__":
    asyncio.run(main())
