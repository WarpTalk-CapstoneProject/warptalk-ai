"""Embedding Worker entry point.

Runs two independent Redis-stream consumers concurrently in one process:
    - EmbeddingWorker — document/transcript/glossary indexing (embedding:index_requests)
    - EmbeddingSearchWorker — semantic-search RPC for the chat assistant (embedding:search_requests)
"""

import asyncio

from embedding_worker.search_worker import EmbeddingSearchWorker
from embedding_worker.worker import EmbeddingWorker
from shared.config import EmbeddingSettings, VectorDbSettings, WorkerSettings
from shared.logger import setup_logging


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    embedding_settings = EmbeddingSettings()
    vector_settings = VectorDbSettings()

    indexing_worker = EmbeddingWorker(
        embedding_settings=embedding_settings,
        vector_settings=vector_settings,
        settings=worker_settings,
    )
    search_worker = EmbeddingSearchWorker(
        embedding_settings=embedding_settings,
        vector_settings=vector_settings,
        settings=worker_settings,
    )

    await asyncio.gather(indexing_worker.start(), search_worker.start())


if __name__ == "__main__":
    asyncio.run(main())
