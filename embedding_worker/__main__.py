"""Embedding Worker entry point."""

import asyncio

from embedding_worker.worker import EmbeddingWorker
from shared.config import EmbeddingSettings, VectorDbSettings, WorkerSettings
from shared.logger import setup_logging


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = EmbeddingWorker(
        embedding_settings=EmbeddingSettings(),
        vector_settings=VectorDbSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
