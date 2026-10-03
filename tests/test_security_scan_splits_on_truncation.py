"""A chunk whose reply was cut off is halved and scanned again instead of failing the document.

Production, 3 Oct 2026: two documents failed with ~19,800 characters analysed against a ~10,000
token reply budget. The two-characters-per-token guess is too generous for dense scripts, and no
single ratio is right for every language — so the reply decides.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from security_worker.scanners import MIN_SPLIT_CHARS, OpenAISecurityScanner, ScanReplyTruncated


def _scanner(max_chars_per_reply: int) -> tuple[OpenAISecurityScanner, list[int]]:
    seen: list[int] = []

    async def create(**kwargs: Any) -> SimpleNamespace:
        text = kwargs["messages"][-1]["content"]
        body = text.split("Text to analyze:\n", 1)[-1]
        seen.append(len(body))
        if len(body) > max_chars_per_reply:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"piiDe'), finish_reason="length"
                    )
                ]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"piiDetected": False, "dlpMatches": [], "maskedContent": "<ok>"}
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = SimpleNamespace(
        model=None,
        max_tokens=None,
        temperature=None,
        max_analyze_length=None,
        max_total_analyze_length=None,
        scan_concurrency=None,
    )
    return OpenAISecurityScanner(client, settings), seen  # type: ignore[arg-type]


def _doc(chars: int) -> str:
    line = "Nguyễn Văn An gửi báo cáo quý ba cho phòng tài chính.\n"
    return (line * (chars // len(line) + 1))[:chars]


@pytest.mark.asyncio
async def test_a_chunk_cut_off_is_split_and_the_document_still_scans() -> None:
    scanner, seen = _scanner(max_chars_per_reply=6_000)

    report = await scanner.scan_and_mask(
        _doc(20_000), pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
    )

    assert report.masked_content.count("<ok>") >= 4, "every piece was scanned and reassembled"
    assert max(seen) >= 19_000 and min(seen) <= 6_000


@pytest.mark.asyncio
async def test_below_the_floor_a_cut_reply_is_still_an_error() -> None:
    scanner, _ = _scanner(max_chars_per_reply=10)

    with pytest.raises(ScanReplyTruncated):
        await scanner.scan_and_mask(
            _doc(MIN_SPLIT_CHARS * 3), pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
        )


@pytest.mark.asyncio
async def test_a_reply_that_fits_is_not_split() -> None:
    scanner, seen = _scanner(max_chars_per_reply=50_000)

    await scanner.scan_and_mask(
        _doc(8_000), pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
    )

    assert len(seen) == 1
