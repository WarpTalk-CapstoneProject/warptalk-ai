"""STT Worker entry point."""

import asyncio

from shared.config import STTSettings, WorkerSettings
from shared.logger import setup_logging

from stt_worker.worker import STTWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = STTWorker(
        stt_settings=STTSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
