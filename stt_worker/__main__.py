"""STT Worker entry point."""

import asyncio

from shared.logger import get_logger, setup_logging
from shared.config import settings

from stt_worker.worker import STTWorker

logger = get_logger(__name__)


async def main() -> None:
    setup_logging(settings.log_level)
    logger.info("Starting STT Worker", model=settings.log_level)

    worker = STTWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
