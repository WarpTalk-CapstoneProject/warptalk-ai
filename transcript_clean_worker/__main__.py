"""Transcript Clean Worker entry point.

Runs one consumer: TranscriptCleanWorker over stt:results under its own consumer group.

Starting without an OpenAI key is deliberately NOT a failure, unlike suggestion_worker: the
deterministic prepass tier needs no model and still produces every line, so a missing key
degrades the stage to rule-only cleaning instead of stopping the transcript. The startup log
says which tier is live.
"""

import asyncio

from shared.config import WorkerSettings
from shared.logger import setup_logging
from transcript_clean_worker.config import TranscriptCleanSettings
from transcript_clean_worker.worker import TranscriptCleanWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = TranscriptCleanWorker(
        clean_settings=TranscriptCleanSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
