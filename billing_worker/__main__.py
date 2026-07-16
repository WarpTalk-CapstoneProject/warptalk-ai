"""Billing settlement worker entry point.

Env vars beyond the shared REDIS_* ones:
    BILLING_DB_DSN=postgresql://user:pass@host:5432/warptalk
"""

import asyncio

from shared.config import BillingSettings, WorkerSettings
from shared.logger import setup_logging

from billing_worker.worker import BillingSettlementWorker


async def main() -> None:
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = BillingSettlementWorker(
        billing_settings=BillingSettings(),
        worker_settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
