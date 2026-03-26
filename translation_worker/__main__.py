"""Translation Worker entry point."""

import asyncio

from shared.logger import get_logger, setup_logging
from shared.config import settings

logger = get_logger(__name__)


async def main() -> None:
    setup_logging(settings.log_level)
    logger.info("Starting Translation Worker")
    # TODO: Implement translation worker loop
    # Consume from stt:results:{meetingId}
    # Translate text
    # Publish to translate:results:{meetingId}


if __name__ == "__main__":
    asyncio.run(main())
