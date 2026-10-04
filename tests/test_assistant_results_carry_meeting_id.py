"""Every summary and action-items entry on `ai_assistant:results` names its meeting.

BaseWorker.publish() uses its `meeting_id` argument only to build the per-room stream key, then
writes the SAME payload to the flat `ai_assistant:results` stream as well. The flat stream is the
one the .NET gateway's `gateway-consumers` group reads, and AiResultConsumerService routes each
entry to its SignalR room by the payload's `meeting_id` field. The assistant built its payload by
hand as {type, content, timestamp_ms}, so on the flat stream nothing said which meeting a summary
belonged to: the gateway skipped every one without acking it, and they piled up as pending
entries until WarpTalkAiPendingStuck fired (62 in prod on 30 Sep 2026).

`publish` is deliberately NOT mocked here. The defect lives in the gap between the stream key and
the payload, and a mocked publish hides exactly that gap.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker.worker import AIAssistantWorker

MEETING_ID = "019fbb91-a381-74d7-abe9-330fea589d81"
GLOBAL_STREAM = "ai_assistant:results"


def _worker() -> tuple[AIAssistantWorker, list[tuple[str, dict[str, Any]]]]:
    worker = AIAssistantWorker.__new__(AIAssistantWorker)
    # Bypassing __init__ means bypassing every field it sets.
    worker._transcripts = {MEETING_ID: [("speaker-a", "we ship the beta on Friday", 1000)]}
    worker._pause_gaps = {}
    worker._gap_open = set()
    worker._filler_only_ms = {}
    worker.logger = MagicMock()

    published: list[tuple[str, dict[str, Any]]] = []

    async def _publish(stream: str, data: dict[str, Any]) -> bytes:
        published.append((stream, dict(data)))
        return b"1-0"

    worker.redis = SimpleNamespace(  # type: ignore[assignment]
        publish=AsyncMock(side_effect=_publish),
        get=AsyncMock(return_value=None),
        hgetall=AsyncMock(return_value={}),
        hset=AsyncMock(),
        lrange=AsyncMock(return_value=[]),
        delete=AsyncMock(),
    )

    async def _summarize(transcript_text: str, **kwargs: Any) -> str:
        return "The team agreed to ship the beta on Friday."

    async def _extract(transcript_text: str, **kwargs: Any) -> str:
        return "- Ship the beta (Friday)"

    async def _structured(transcript_text: str, **kwargs: Any) -> dict[str, Any]:
        return {"insufficientData": False, "summary": "ok", "decisions": [], "actionItems": []}

    worker._require_assistant = lambda: SimpleNamespace(  # type: ignore[method-assign]
        summarize=_summarize,
        extract_action_items=_extract,
        generate_structured_summary=_structured,
    )
    return worker, published


@pytest.mark.asyncio
async def test_summary_and_action_items_on_the_global_stream_carry_meeting_id() -> None:
    worker, published = _worker()

    await worker._generate_summary(MEETING_ID)

    on_global = {data["type"]: data for stream, data in published if stream == GLOBAL_STREAM}
    assert set(on_global) == {"summary", "action_items"}
    for kind, data in on_global.items():
        assert data.get("meeting_id") == MEETING_ID, (
            f"{kind} reached {GLOBAL_STREAM} without meeting_id; the gateway cannot route "
            f"or ack it: {sorted(data)}"
        )


@pytest.mark.asyncio
async def test_the_per_room_copy_is_unchanged_apart_from_meeting_id() -> None:
    # The room stream already implied the meeting through its key. Adding the field must not
    # move anything else: same entries, same streams, same order.
    worker, published = _worker()

    await worker._generate_summary(MEETING_ID)

    room_stream = f"{GLOBAL_STREAM}:{MEETING_ID}"
    assert [(stream, data["type"]) for stream, data in published] == [
        (room_stream, "summary"),
        (GLOBAL_STREAM, "summary"),
        (room_stream, "action_items"),
        (GLOBAL_STREAM, "action_items"),
    ]
    for _, data in published:
        assert set(data) == {"meeting_id", "type", "content", "timestamp_ms"}
