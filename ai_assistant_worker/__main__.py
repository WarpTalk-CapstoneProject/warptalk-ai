"""AI Assistant Worker entry point."""

import asyncio

from shared.config import AssistantSettings, WorkerSettings
from shared.logger import setup_logging

from ai_assistant_worker.worker import AIAssistantWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = AIAssistantWorker(
        assistant_settings=AssistantSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
