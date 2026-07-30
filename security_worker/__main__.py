"""Security Worker entry point."""

import asyncio

from security_worker.worker import SecurityWorker
from shared.config import SecuritySettings, WorkerSettings
from shared.logger import setup_logging


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = SecurityWorker(
        security_settings=SecuritySettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
