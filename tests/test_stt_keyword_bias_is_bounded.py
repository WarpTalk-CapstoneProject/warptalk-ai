"""WT-426 — the keyword bias list is a thumb on the scale, so its size is a safety property.

A noisy production meeting on 15 Aug transcribed "voice clone" as "Cũng là ChatGPT", and emitted
"WarpTalk, WarpBot, Codex." as a whole utterance nobody spoke. All four are glossary terms.

On marginal audio the model resolves ambiguity INTO whatever list it was handed, so the size of
that list is the size of the hallucination surface. This module pins the reader's ceiling — the
last bound before the list reaches the provider.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import STTSettings, WorkerSettings
from stt_worker.worker import _MAX_STT_KEYWORDS, STTWorker


class _KeywordRedis:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    async def get(self, _key: str) -> bytes:
        return json.dumps(self._payload).encode()


def _worker(payload: Any) -> STTWorker:
    worker = STTWorker.__new__(STTWorker)
    worker.settings = WorkerSettings()
    worker.stt_settings = STTSettings()
    worker.logger = MagicMock()
    worker._stt_keywords = {}
    worker.redis = _KeywordRedis(payload)  # type: ignore[assignment]
    return worker


def test_the_reader_is_not_more_permissive_than_the_writer() -> None:
    """GlossaryStartedEventConsumer sends 10. A reader that accepts 100 is not a bound.

    It is the last check before the provider sees the list, so anything that ever wrote more — by
    hand, by a migration, by a future change to the writer — would sail straight through.
    """
    assert _MAX_STT_KEYWORDS <= 16, (
        "the safety ceiling drifted back above what the writer intends to send"
    )


@pytest.mark.asyncio
async def test_an_oversized_list_is_truncated_rather_than_trusted() -> None:
    oversized = [f"Term{index}" for index in range(200)]

    keywords = await _worker(oversized)._get_stt_keywords("m1")

    assert len(keywords) == _MAX_STT_KEYWORDS


@pytest.mark.asyncio
async def test_a_list_within_the_ceiling_is_passed_through_whole() -> None:
    # The negative control: a cap that silently trimmed a legitimate list would remove the terms
    # this feature exists for — "Codex" came back as "cô đích" when it was missing.
    terms = ["Codex", "Kubernetes", "WarpTalk"]

    keywords = await _worker(terms)._get_stt_keywords("m1")

    assert keywords == terms


@pytest.mark.asyncio
async def test_duplicates_do_not_consume_the_ceiling() -> None:
    # Source and target are both offered for every term, and a term whose translation is itself
    # ("Codex" -> "Codex") arrives twice. Spending two slots on one word would halve the budget.
    keywords = await _worker(["Codex", "codex", "CODEX", "Kubernetes"])._get_stt_keywords("m1")

    assert keywords == ["Codex", "Kubernetes"]
