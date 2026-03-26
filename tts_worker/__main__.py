"""TTS Worker entry point."""

import asyncio

from shared.logger import get_logger, setup_logging
from shared.config import settings

logger = get_logger(__name__)


async def main() -> None:
    setup_logging(settings.log_level)
    logger.info("Starting TTS Worker")
    # TODO: Implement TTS worker loop
    # Consume from translate:results:{meetingId}
    # Synthesize speech (with voice cloning)
    # Publish to tts:results:{meetingId}


if __name__ == "__main__":
    asyncio.run(main())
