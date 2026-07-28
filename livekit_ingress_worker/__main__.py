"""Entry point for LiveKit Ingress Worker."""

import asyncio

from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import WorkerSettings
from shared.logger import setup_logging


async def main() -> None:
    settings = WorkerSettings()
    setup_logging(log_level=settings.log_level)

    worker = LiveKitIngressWorker(settings=settings)
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
