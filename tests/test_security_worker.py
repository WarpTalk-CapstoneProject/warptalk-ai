from unittest.mock import AsyncMock, MagicMock

import pytest

from security_worker.worker import SecurityWorker


@pytest.mark.asyncio
async def test_pii_scan_fails_closed_when_provider_is_unavailable() -> None:
    worker = SecurityWorker.__new__(SecurityWorker)
    worker.logger = MagicMock()
    worker.openai_client = None
    worker.openai_scanner = None
    worker._save_result = AsyncMock()

    await worker.process(
        b"message-1",
        {
            b"scan_id": b"scan-1",
            b"content": b"secret customer data",
            b"pii_enabled": b"true",
            b"dlp_enabled": b"false",
            b"keywords": b"[]",
        },
    )

    worker._save_result.assert_awaited_once_with(
        "scan-1",
        pii_detected=False,
        dlp_detected=False,
        violation_found=True,
        masked_content="",
        scan_failed=True,
    )
