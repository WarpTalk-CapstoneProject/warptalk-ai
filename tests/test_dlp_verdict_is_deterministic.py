"""The DLP verdict comes from the document, not from the model's opinion of it.

THE BUG THESE PIN
    `security_worker/worker.py` decided DLP with `llm_dlp_detected or dlp_detected_local`. The
    local half answers "does this text contain one of these words" exactly, case-insensitively,
    over the whole document. The model half answers it approximately, over at most
    `max_analyze_length` characters of it. OR-ing the approximate answer onto the exact one can
    only ADD false positives; it has no way to correct one.

    And a DLP hit is terminal. `ProcessDocumentUploadAsync` builds `canIndex` with
    `&& !scanResult.DlpDetected` and gives DLP no masking route, so a hallucinated match drops the
    document out of the index permanently, with `ingestion_failure_reason=dlp_detected` and no way
    to appeal it.

PRODUCTION, 2026-09-06
    Counting the blacklist (`Fuck`, `Shit`, `Damn`) in the stored bodies of all thirteen
    DLP-enabled scans in `security:scan_requests`: zero hits, every single time. Three documents
    were nonetheless recorded `DlpDetected: true` — 18 Aug, 19 Aug, and 5 Sep. Three out of three
    DLP blocks this system has ever produced were invented. The 5 Sep one was a 293,950-character
    test report; its owner deleted the file three minutes later.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from security_worker.scanners import SecurityScanReport
from security_worker.worker import SecurityWorker, split_claims_by_evidence

BLACKLIST = ["Fuck", "Shit", "Damn"]


def _worker(report: SecurityScanReport) -> SecurityWorker:
    worker = SecurityWorker.__new__(SecurityWorker)
    worker.logger = MagicMock()
    worker.openai_client = SimpleNamespace(api_key="sk-test")
    worker.openai_scanner = SimpleNamespace(scan_and_mask=AsyncMock(return_value=report))
    worker._save_result = AsyncMock()
    return worker


async def _scan(worker: SecurityWorker, content: str) -> dict[str, object]:
    await worker.process(
        b"message-1",
        {
            b"scan_id": b"scan-1",
            b"content": content.encode(),
            b"pii_enabled": b"true",
            b"dlp_enabled": b"true",
            b"keywords": json.dumps(BLACKLIST).encode(),
        },
    )
    call = worker._save_result.await_args
    return {**dict(zip(("scan_id", "pii", "dlp", "violation", "masked"), call.args)), **call.kwargs}


@pytest.mark.asyncio
async def test_a_match_the_document_does_not_contain_does_not_block_it() -> None:
    """The 5 Sep production case, reduced: a clean document, a model that says otherwise."""
    content = "Test Report. Round 1: 4 passed, 0 failed. Tester: Nhi."
    worker = _worker(
        SecurityScanReport(
            pii_detected=False,
            masked_content=content,
            dlp_terms_claimed=("Damn",),  # nowhere in the text above
        )
    )

    result = await _scan(worker, content)

    assert result["dlp"] is False, "an invented match must not block a document"
    assert result["violation"] is False
    assert result["dlp_flagged_terms"] == ()
    worker.logger.warning.assert_called_once()
    assert worker.logger.warning.call_args.args[0] == "dlp_claim_absent_from_text"


@pytest.mark.asyncio
async def test_a_keyword_that_is_really_there_still_blocks() -> None:
    """The guardrail still guards: the local check alone is enough to block, and the model
    staying silent about a real hit cannot unblock it."""
    content = "This damn spreadsheet again."  # 'Damn', case-insensitively
    worker = _worker(
        SecurityScanReport(pii_detected=False, masked_content=content, dlp_terms_claimed=())
    )

    result = await _scan(worker, content)

    assert result["dlp"] is True
    assert result["violation"] is True
    assert result["dlp_flagged_terms"] == ("Damn",), "the verdict must say which word caused it"


@pytest.mark.asyncio
async def test_a_variant_the_text_supports_is_reported_but_does_not_block() -> None:
    """An obfuscated hit is real signal the substring search cannot see — and still not grounds
    to drop the document, because fuzzy blocking is what cost three of them."""
    content = "this F*ck of a build"
    worker = _worker(
        SecurityScanReport(pii_detected=False, masked_content=content, dlp_terms_claimed=("F*ck",))
    )

    result = await _scan(worker, content)

    assert result["dlp"] is False
    assert result["dlp_flagged_terms"] == ("F*ck",)
    worker.logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_pii_is_still_the_model_s_call_and_still_masks() -> None:
    """Nothing here demotes PII. It has no exact answer and its hit is recoverable, which is
    exactly why the two are treated differently."""
    content = "call Nhi on 0912345678"
    worker = _worker(
        SecurityScanReport(
            pii_detected=True,
            masked_content="call [PII_REDACTED] on [PHONE_REDACTED]",
            dlp_terms_claimed=(),
        )
    )

    result = await _scan(worker, content)

    assert result["pii"] is True
    assert result["violation"] is True
    assert result["dlp"] is False
    assert result["masked"] == "call [PII_REDACTED] on [PHONE_REDACTED]"


def test_evidence_is_checked_against_the_whole_document_not_the_analysed_slice() -> None:
    """The model sees at most `max_analyze_length` characters; the check reads all of them. A
    claim about the tail of a long document is still verifiable."""
    content = "a" * 50_000 + " Damn"

    supported, unsupported = split_claims_by_evidence(content, ("Damn", "Shit"))

    assert supported == ("Damn",)
    assert unsupported == ("Shit",)
