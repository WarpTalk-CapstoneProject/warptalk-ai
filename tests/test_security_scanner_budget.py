"""WT-460: the security scan must be able to return the document it was given.

THE BUG THESE PIN
    The prompt asks the model for `maskedContent` — the whole analysed text, echoed back with
    PII replaced — while the input was allowed 20,000 CHARACTERS and the reply capped at 2,000
    TOKENS. Past roughly six thousand characters the model cannot finish the JSON object, the
    reply is cut mid-string, and `json.loads` raises three frames down.

    That is the entirety of "approved documents never embed in production": the scan throws, the
    document never becomes AiEligible, and nothing ever reaches Qdrant. The ticket blamed a
    missing OpenAI key, a refused Qdrant connection and an unset VectorDb:Url. The scan was in
    fact reaching OpenAI perfectly well and being truncated on the way back.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from security_worker.scanners import MAX_TOKENS, OpenAISecurityScanner


def _completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def _scanner(completion: SimpleNamespace) -> tuple[OpenAISecurityScanner, AsyncMock]:
    create = AsyncMock(return_value=completion)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = SimpleNamespace(
        model=None, max_tokens=None, temperature=None, max_analyze_length=None
    )
    return OpenAISecurityScanner(client, settings), create  # type: ignore[arg-type]


def _ok_body(text: str) -> str:
    return json.dumps({"piiDetected": False, "dlpMatches": [], "maskedContent": text})


@pytest.mark.asyncio
async def test_a_long_document_is_given_room_to_come_back() -> None:
    # 20,000 characters is exactly what max_analyze_length allows through, and it could never
    # have fitted in the old flat 2,000-token reply budget.
    text = "a" * 20_000
    scanner, create = _scanner(_completion(_ok_body(text)))

    await scanner.scan_and_mask(text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[])

    budget = create.await_args.kwargs["max_tokens"]
    assert budget > MAX_TOKENS, "a document larger than the flat cap must widen the reply budget"
    assert budget >= len(text) // 2, "the reply has to be able to carry the text back"


@pytest.mark.asyncio
async def test_a_short_document_keeps_the_configured_budget() -> None:
    # The configured cap stays a floor, so nothing about small documents changes.
    text = "hello world"
    scanner, create = _scanner(_completion(_ok_body(text)))

    await scanner.scan_and_mask(text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[])

    assert create.await_args.kwargs["max_tokens"] == MAX_TOKENS


@pytest.mark.asyncio
async def test_a_truncated_reply_says_it_was_truncated() -> None:
    """The old failure was a bare JSONDecodeError from json.loads, which reads as a malformed
    model reply and sent the investigation at the API key and at Qdrant. finish_reason is the
    API stating plainly that it ran out of room."""
    scanner, _ = _scanner(_completion('{"piiDetected": false, "maskedCon', finish_reason="length"))

    with pytest.raises(ValueError, match="truncated"):
        await scanner.scan_and_mask(
            "some text", pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
        )


@pytest.mark.asyncio
async def test_a_normal_reply_is_still_parsed() -> None:
    text = "contact me at a@b.com"
    body = json.dumps(
        {
            "piiDetected": True,
            "dlpMatches": [],
            "maskedContent": "contact me at [EMAIL_REDACTED]",
        }
    )
    scanner, _ = _scanner(_completion(body))

    report = await scanner.scan_and_mask(
        text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
    )

    assert report.pii_detected is True
    assert report.dlp_terms_claimed == ()
    assert report.masked_content == "contact me at [EMAIL_REDACTED]"
