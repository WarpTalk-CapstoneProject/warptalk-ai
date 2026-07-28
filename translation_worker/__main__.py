"""Translation Worker entry point."""

import asyncio

from shared.config import TranslationSettings, WorkerSettings
from shared.logger import setup_logging
from translation_worker.worker import TranslationWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = TranslationWorker(
        translation_settings=TranslationSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
