"""Re-summarising a finished meeting reads the SAVED transcript, and reports why not.

The live accumulator AIAssistantWorker summarises from is gone the moment a meeting ends, so
a request that arrives afterwards has to go and fetch the stored segments — which is also
what keeps citations pointing at the segments the meeting page actually renders.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker.summary_template_worker import SummaryTemplateWorker
from shared.schemas import SummaryRequestMessage, SummaryResultMessage

ROOM = "019f6a39-a32c-7745-886e-1fe622c1f747"


def _worker() -> SummaryTemplateWorker:
    worker = SummaryTemplateWorker(transcript_base_url="http://transcript")
    worker.publish = AsyncMock()  # type: ignore[method-assign]
    worker.assistant = MagicMock()
    worker.assistant.generate_structured_summary = AsyncMock(
        return_value={"summary": "ok", "decisions": [], "templateKey": "standup"}
    )
    return worker


def _request(**over: Any) -> dict[bytes, bytes]:
    message = SummaryRequestMessage(request_id="req-1", room_id=ROOM, workspace_id="ws-1", **over)
    return {k.encode(): v.encode() for k, v in message.to_redis().items()}


def _published(worker: SummaryTemplateWorker) -> SummaryResultMessage:
    worker.publish.assert_awaited_once()  # type: ignore[attr-defined]
    args = worker.publish.await_args.args  # type: ignore[attr-defined]
    assert args[0] == "assistant:summary_results"
    return SummaryResultMessage.from_redis(args[2])


@pytest.mark.asyncio
async def test_a_meeting_with_no_saved_transcript_says_so() -> None:
    worker = _worker()
    worker._load_transcript = AsyncMock(return_value="")  # type: ignore[method-assign]

    await worker.process(b"1", _request(template_key="standup"))

    result = _published(worker)
    assert result.status == "failed"
    # The specific reason, not a generic failure — there is nothing wrong with the system.
    assert "no saved transcript" in result.error
    worker.assistant.generate_structured_summary.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_transcript_that_cannot_be_read_fails_loudly_not_silently() -> None:
    worker = _worker()
    worker._load_transcript = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]

    await worker.process(b"1", _request())

    result = _published(worker)
    assert result.status == "failed"
    assert result.error


@pytest.mark.asyncio
async def test_the_requested_template_reaches_the_assistant() -> None:
    worker = _worker()
    worker._load_transcript = AsyncMock(return_value="[t=0] [Tu] hello")  # type: ignore[method-assign]

    await worker.process(b"1", _request(template_key="standup"))

    kwargs = worker.assistant.generate_structured_summary.await_args.kwargs
    assert kwargs["template_key"] == "standup"

    result = _published(worker)
    assert result.status == "completed"
    assert json.loads(result.content_json)["summary"] == "ok"


@pytest.mark.asyncio
async def test_an_unknown_template_is_answered_in_the_general_shape() -> None:
    # A typo must not leave the requester with no summary at all.
    worker = _worker()
    worker._load_transcript = AsyncMock(return_value="[t=0] [Tu] hello")  # type: ignore[method-assign]

    await worker.process(b"1", _request(template_key="not-a-template"))

    assert (
        worker.assistant.generate_structured_summary.await_args.kwargs["template_key"] == "general"
    )
    assert _published(worker).template_key == "general"


@pytest.mark.asyncio
async def test_saved_segments_become_cited_transcript_lines() -> None:
    worker = _worker()
    lookup = MagicMock(status_code=200, json=lambda: {"id": "tr-1"})
    segments = MagicMock(
        status_code=200,
        json=lambda: {
            "items": [
                {"startTimeMs": 0, "speakerName": "Tu", "originalText": "hello"},
                {"startTimeMs": 90210, "speakerName": "Nhi", "originalText": "cap it"},
                # Blank text carries no evidence and would waste prompt budget.
                {"startTimeMs": 95000, "speakerName": "Ky", "originalText": "   "},
            ]
        },
    )
    lookup.raise_for_status = MagicMock()
    segments.raise_for_status = MagicMock()
    client = MagicMock()
    client.get = AsyncMock(side_effect=[lookup, segments])
    worker._transcript_client = client

    transcript = await worker._load_transcript(
        SummaryRequestMessage(request_id="r", room_id=ROOM, workspace_id="w")
    )

    assert transcript == "[t=0] [Tu] hello\n[t=90210] [Nhi] cap it"


@pytest.mark.asyncio
async def test_a_room_with_no_transcript_record_reads_as_empty_not_an_error() -> None:
    worker = _worker()
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock(status_code=404))
    worker._transcript_client = client

    assert (
        await worker._load_transcript(
            SummaryRequestMessage(request_id="r", room_id=ROOM, workspace_id="w")
        )
        == ""
    )


@pytest.mark.asyncio
async def test_a_supplied_transcript_is_used_and_the_service_is_never_called() -> None:
    """The system-initiated path: ArtifactsFinalizer already read the stored segments.

    The second assertion is the one that matters. A background finalization has no user and no
    bearer token, so any HTTP call here would either be unauthenticated or need a privileged
    bypass — precisely what this request's own contract refuses. Using what the publisher
    already lawfully holds is what makes the token unnecessary rather than merely optional.
    """
    worker = _worker()
    client = MagicMock()
    client.get = AsyncMock(side_effect=AssertionError("must not call the transcript service"))
    worker._transcript_client = client

    supplied = "[t=0] [Tu] hello\n[t=90210] [Nhi] cap it"
    transcript = await worker._load_transcript(
        SummaryRequestMessage(
            request_id="r", room_id=ROOM, workspace_id="w", transcript_text=supplied
        )
    )

    assert transcript == supplied
    client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_blank_supplied_transcript_falls_back_to_fetching() -> None:
    """Whitespace is not a transcript, and must not silently summarise nothing.

    Treating a string of newlines as "supplied" would skip the fetch and hand the model a blank
    page, which is the failure this whole change exists to stop.
    """
    worker = _worker()
    client = MagicMock()
    client.get = AsyncMock(return_value=MagicMock(status_code=404))
    worker._transcript_client = client

    result = await worker._load_transcript(
        SummaryRequestMessage(
            request_id="r", room_id=ROOM, workspace_id="w", transcript_text="   \n  "
        )
    )

    assert result == ""
    client.get.assert_awaited()


def test_a_supplied_transcript_survives_the_redis_round_trip() -> None:
    """It travels as a stream field, so it has to serialise like every other one."""
    message = SummaryRequestMessage(
        request_id="r", room_id=ROOM, workspace_id="w", transcript_text="[t=0] [Tu] hi"
    )

    assert SummaryRequestMessage.from_redis(message.to_redis()).transcript_text == "[t=0] [Tu] hi"

    # A message published before this field existed is the user-initiated shape, and has to keep
    # fetching rather than summarising an empty string.
    legacy = {k: v for k, v in message.to_redis().items() if k != "transcript_text"}
    assert SummaryRequestMessage.from_redis(legacy).transcript_text == ""


def test_the_worker_reads_and_writes_the_streams_the_backend_uses() -> None:
    # Renaming either side silently is how a request ends up with no consumer.
    assert SummaryTemplateWorker.input_stream == "assistant:summary_requests"
    assert SummaryTemplateWorker.consumer_group == "summary-template-workers"


@pytest.mark.asyncio
async def test_an_uninitialised_chat_worker_still_answers_before_it_raises() -> None:
    """Silence is the one outcome a mention must never produce.

    This check sits outside the try/except that publishes failures, so raising here used to
    publish nothing at all — and somebody who types @WarpBot and gets complete silence cannot
    tell "the assistant is misconfigured" from "my mention was never seen". That ambiguity is
    what made this bug take two rounds of investigation to place.
    """
    from ai_assistant_worker.chat_worker import ChatAssistantWorker
    from shared.schemas import ChatRequestMessage

    worker = ChatAssistantWorker()
    worker._publish_result = AsyncMock()  # type: ignore[method-assign]

    message = ChatRequestMessage(
        request_id="req-1",
        conversation_id=ROOM,
        workspace_id="ws-1",
        user_id="u-1",
        origin="meeting_chat",
    )
    payload = {k.encode(): v.encode() for k, v in message.to_redis().items()}

    with pytest.raises(RuntimeError):
        await worker.process(b"1", payload)

    worker._publish_result.assert_awaited_once()
    assert worker._publish_result.await_args.kwargs["type_"] == "failed"


@pytest.mark.asyncio
async def test_a_failed_generation_is_published_as_a_failure_not_a_completed_rewrite() -> None:
    """WT-530.

    The assistant returns a placeholder dict when the model call throws. It used to be
    indistinguishable from a real summary, so this worker published it as `completed` and the
    backend wrote it OVER the meeting's existing summary: a rewrite that failed AND destroyed
    what it was replacing, while the page waited ninety seconds for a template that never
    arrived and the console stayed clean.
    """
    worker = _worker()
    worker._load_transcript = AsyncMock(return_value="Alice: hello")  # type: ignore[method-assign]
    worker.assistant.generate_structured_summary = AsyncMock(
        return_value={
            "summary": "The AI assistant could not generate a structured summary for this meeting.",
            "decisions": [],
            "actionItems": [],
            "insufficientData": True,
            "generationFailed": True,
            "templateKey": "standup",
        }
    )

    await worker.process(b"1", _request(template_key="standup"))

    result = _published(worker)
    assert result.status == "failed", "a failed generation must not be reported as completed"
    # The consumer leaves the existing summary alone on a failure, so this sentence is what
    # stops a good summary being replaced by an apology.
    assert "previous one is unchanged" in result.error


@pytest.mark.asyncio
async def test_a_real_summary_is_still_published_as_completed() -> None:
    """The guard must key on the failure flag alone — insufficientData is a legitimate answer."""
    worker = _worker()
    worker._load_transcript = AsyncMock(return_value="Alice: hello")  # type: ignore[method-assign]
    worker.assistant.generate_structured_summary = AsyncMock(
        return_value={
            "summary": "Nobody said much.",
            "decisions": [],
            "actionItems": [],
            "insufficientData": True,
            "templateKey": "standup",
        }
    )

    await worker.process(b"1", _request(template_key="standup"))

    assert _published(worker).status == "completed"
