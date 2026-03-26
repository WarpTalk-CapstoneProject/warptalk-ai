"""TTS Worker entry point."""

import asyncio

from shared.config import TTSSettings, WorkerSettings
from shared.logger import setup_logging

from tts_worker.worker import TTSWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = TTSWorker(
        tts_settings=TTSSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
