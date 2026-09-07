"""The scan reads the whole document, and what it gives back is the whole document.

THE BUG THESE PIN
    `scan_and_mask` handed the model `text[:SECURITY_MAX_ANALYZE_LENGTH] + "... [truncated]"` and
    returned the answer as though it described the file. Production sets no `SECURITY_*` variable,
    so that cap stood at 20,000 characters — 6.8% of the 293,950-character upload of 5 Sep 2026.

    Both ways it went wrong were silent, and which one you got depended on the flag:

    * PII found     → the backend sets `textToIngest = scanResult.MaskedContent`, which was the
                      model's echo of the 20,000 characters it was shown. The remaining 93% never
                      reached the index, and nobody was told.
    * PII not found → the backend indexes `content.FullText`. Any PII in the unread 93% went into
                      Qdrant unmasked, under a scan that had just declared the document clean.

    The second is the worse half: the guardrail reports on text it never read.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from security_worker.scanners import (
    MAX_TOTAL_ANALYZE_LENGTH,
    OpenAISecurityScanner,
    split_for_analysis,
)


def _scanner(
    reply_for: object = None, **overrides: object
) -> tuple[OpenAISecurityScanner, AsyncMock]:
    """A scanner whose model echoes each chunk back untouched, unless told otherwise."""

    async def echo(**kwargs: object) -> SimpleNamespace:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        submitted = messages[1]["content"].split("Text to analyze:\n", 1)[1]
        body = (
            reply_for(submitted)
            if callable(reply_for)
            else json.dumps({"piiDetected": False, "dlpMatches": [], "maskedContent": submitted})
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=body),
                    finish_reason="stop",
                )
            ]
        )

    create = AsyncMock(side_effect=echo)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = SimpleNamespace(
        model=None,
        max_tokens=None,
        temperature=None,
        max_analyze_length=None,
        max_total_analyze_length=None,
        scan_concurrency=None,
        **overrides,
    )
    return OpenAISecurityScanner(client, settings), create  # type: ignore[arg-type]


def _submitted(create: AsyncMock) -> str:
    """Everything the model was actually shown, across every call, in order."""
    return "".join(
        call.kwargs["messages"][1]["content"].split("Text to analyze:\n", 1)[1]
        for call in create.await_args_list
    )


@pytest.mark.asyncio
async def test_every_character_of_a_long_document_is_shown_to_the_model() -> None:
    """The 5 Sep document's size. Under the old cap the model saw 20,000 of these characters and
    the scan reported on all 293,950 of them."""
    text = "Round 1: 4 passed, 0 failed.\n" * 10_500  # ~294,000 characters

    scanner, create = _scanner()
    await scanner.scan_and_mask(text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[])

    assert _submitted(create) == text, "part of the document was never shown to the scanner"
    assert create.await_count > 1, "a 294k document must not fit in one request"


@pytest.mark.asyncio
async def test_the_masked_text_that_comes_back_is_the_whole_document() -> None:
    """`masked_content` is what the backend indexes when PII was found. Anything missing from it
    is missing from search, silently."""
    text = "".join(f"line {i} with content\n" for i in range(4_000))

    scanner, _ = _scanner()
    report = await scanner.scan_and_mask(
        text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
    )

    assert report.masked_content == text
    assert "[truncated]" not in report.masked_content


@pytest.mark.asyncio
async def test_pii_in_the_tail_is_found() -> None:
    """The half that leaked rather than the half that lost data: PII past the old cap was indexed
    unmasked, under a scan that called the document clean."""

    def reply(submitted: str) -> str:
        found = "0912345678" in submitted
        return json.dumps(
            {
                "piiDetected": found,
                "dlpMatches": [],
                "maskedContent": submitted.replace("0912345678", "[PHONE_REDACTED]"),
            }
        )

    text = "harmless filler.\n" * 3_000 + "call me on 0912345678\n"

    scanner, _ = _scanner(reply_for=reply)
    report = await scanner.scan_and_mask(
        text, pii_enabled=True, dlp_enabled=False, keywords_blacklist=[]
    )

    assert report.pii_detected is True, "PII past the first chunk went unnoticed"
    assert "0912345678" not in report.masked_content
    assert report.masked_content.endswith("call me on [PHONE_REDACTED]\n")


@pytest.mark.asyncio
async def test_a_document_beyond_the_total_cap_fails_instead_of_being_half_scanned() -> None:
    """Fail closed. A guardrail that quietly inspects a prefix produces a document that LOOKS
    scanned, and only an outright failure can be acted on."""
    scanner, create = _scanner()

    with pytest.raises(ValueError, match="beyond the"):
        await scanner.scan_and_mask(
            "x" * (MAX_TOTAL_ANALYZE_LENGTH + 1),
            pii_enabled=True,
            dlp_enabled=False,
            keywords_blacklist=[],
        )

    # An oversize document must not cost a single API call before it is refused.
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_dlp_claims_from_every_chunk_are_collected_once() -> None:
    def reply(submitted: str) -> str:
        return json.dumps(
            {"piiDetected": False, "dlpMatches": ["Damn"], "maskedContent": submitted}
        )

    scanner, create = _scanner(reply_for=reply)
    report = await scanner.scan_and_mask(
        "Damn.\n" * 8_000, pii_enabled=True, dlp_enabled=True, keywords_blacklist=["Damn"]
    )

    assert create.await_count > 1
    assert report.dlp_terms_claimed == ("Damn",), "the same term claimed per chunk must collapse"


def test_chunks_reassemble_into_the_original_exactly() -> None:
    """The property the whole design rests on: no overlap, nothing dropped, nothing inserted."""
    for text in (
        "short",
        "no boundaries anywhere" + "x" * 5_000,
        "paragraph one\n\nparagraph two\n" * 400,
        "word " * 4_000,
        "",
    ):
        assert "".join(split_for_analysis(text, 500)) == text


def test_a_chunk_never_exceeds_the_requested_size() -> None:
    chunks = split_for_analysis("word " * 4_000, 500)

    assert chunks, "a non-empty document must produce at least one chunk"
    assert all(len(chunk) <= 500 for chunk in chunks)


def test_the_cut_prefers_a_line_break_so_a_match_is_not_split_in_half() -> None:
    """A phone number cut down the middle is a match neither half can see."""
    text = "a" * 90 + "\n" + "b" * 90 + "\n" + "c" * 90 + "\n"

    chunks = split_for_analysis(text, 100)

    assert all(chunk.endswith("\n") for chunk in chunks[:-1])
    assert "".join(chunks) == text


def test_text_with_no_boundary_at_all_still_makes_progress() -> None:
    """A base64 blob offers neither a line break nor a space; the hard offset has to hold."""
    chunks = split_for_analysis("Z" * 2_000, 300)

    assert len(chunks) == 7
    assert "".join(chunks) == "Z" * 2_000
