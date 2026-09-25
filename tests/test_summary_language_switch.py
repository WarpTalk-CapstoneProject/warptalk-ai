"""Switching a finished meeting's summary (and its biên bản) into another meeting language.

The shape here is production's on 2026-09-18: a published summary stamped general/en that
already carries `translations.vi`, on a room whose languages are en and vi. What a reader who
picks Vietnamese must get back is THAT summary in Vietnamese — same sections, same items, same
cited moments — stamped `vi`, and echoed with the pair they asked for so the backend files it
where the endpoint will look.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker.assistant import MeetingAssistant
from ai_assistant_worker.minutes_translation import render_in_language
from ai_assistant_worker.summary_template_worker import SummaryTemplateWorker
from shared.schemas import SummaryRequestMessage, SummaryResultMessage

ROOM = "01a0b374-527f-7774-9056-c9b1019e3777"


def _published() -> dict[str, Any]:
    return {
        "summary": "The team reviewed the AI customer system.",
        "decisions": [],
        "actionItems": [{"task": "Prepare the report", "owner": "Ky", "atMs": 97_470}],
        "openQuestions": [{"text": "Is latency acceptable?", "atMs": 12_000}],
        "citations": [{"key": "summary", "atMs": 1_200}],
        "templateKey": "general",
        "summaryLanguage": "en",
        "insufficientData": False,
        "translations": {
            "vi": {
                "summary": "Nhóm đã xem xét hệ thống AI.",
                "actionItems": [{"task": "Chuẩn bị báo cáo", "owner": "Ky", "atMs": 97_470}],
                "openQuestions": [{"text": "Độ trễ có chấp nhận được không?", "atMs": 12_000}],
            }
        },
    }


# ── render_in_language: complete or nothing ──────────────────────────────────────────────────


def test_a_carried_translation_becomes_the_whole_summary_in_that_language() -> None:
    source = _published()

    rendered = render_in_language(source, source["translations"]["vi"], "vi")

    assert rendered is not None
    assert rendered["summaryLanguage"] == "vi"
    assert rendered["templateKey"] == "general"
    assert rendered["summary"] == "Nhóm đã xem xét hệ thống AI."
    assert rendered["actionItems"][0] == {"task": "Chuẩn bị báo cáo", "owner": "Ky", "atMs": 97_470}
    # The source's evidence, not new claims.
    assert rendered["citations"] == source["citations"]
    assert "translations" not in rendered


def test_a_missing_section_is_not_passed_off_as_translated() -> None:
    source = _published()
    partial = dict(source["translations"]["vi"])
    del partial["openQuestions"]

    assert render_in_language(source, partial, "vi") is None


def test_a_blank_item_is_not_passed_off_as_translated() -> None:
    source = _published()
    partial = json.loads(json.dumps(source["translations"]["vi"]))
    partial["actionItems"][0]["task"] = ""

    assert render_in_language(source, partial, "vi") is None


def test_nothing_to_translate_from_is_none() -> None:
    assert render_in_language(_published(), None, "vi") is None


# ── MeetingAssistant.translate_summary ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_translation_the_summary_already_carries_costs_no_model_call() -> None:
    assistant = MeetingAssistant(api_key="test-key")
    assistant._translate_for_minutes = AsyncMock()  # type: ignore[method-assign]

    rendered = await assistant.translate_summary(_published(), "vi-VN")

    assert rendered is not None
    assert rendered["summaryLanguage"] == "vi"
    assistant._translate_for_minutes.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_language_the_summary_does_not_carry_is_translated_once() -> None:
    source = _published()
    del source["translations"]
    assistant = MeetingAssistant(api_key="test-key")
    assistant._translate_for_minutes = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "vi": {
                "summary": "Nhóm đã xem xét hệ thống AI.",
                "actionItems": [{"task": "Chuẩn bị báo cáo", "owner": "Ky", "atMs": 97_470}],
                "openQuestions": [{"text": "Độ trễ?", "atMs": 12_000}],
            }
        }
    )

    rendered = await assistant.translate_summary(source, "vi")

    assert rendered is not None
    assert rendered["summaryLanguage"] == "vi"
    assert rendered["openQuestions"][0]["atMs"] == 12_000
    args = assistant._translate_for_minutes.await_args.args
    assert args[1] == ["vi"]
    assert args[2] == "en"


@pytest.mark.asyncio
async def test_a_translation_the_model_could_not_complete_is_none() -> None:
    source = _published()
    del source["translations"]
    assistant = MeetingAssistant(api_key="test-key")
    assistant._translate_for_minutes = AsyncMock(return_value=None)  # type: ignore[method-assign]

    assert await assistant.translate_summary(source, "vi") is None


# ── SummaryTemplateWorker ────────────────────────────────────────────────────────────────────


def _worker() -> SummaryTemplateWorker:
    worker = SummaryTemplateWorker(transcript_base_url="http://transcript")
    worker.publish = AsyncMock()  # type: ignore[method-assign]
    worker._load_transcript = AsyncMock(return_value="[t=0] [Ky] hello")  # type: ignore[method-assign]
    assistant = MeetingAssistant(api_key="test-key")
    assistant._translate_for_minutes = AsyncMock(return_value=None)  # type: ignore[method-assign]
    worker.assistant = assistant
    return worker


def _request(**over: Any) -> dict[bytes, bytes]:
    message = SummaryRequestMessage(
        request_id="req-1", room_id=ROOM, workspace_id="ws-1", delivery="variant", **over
    )
    return {k.encode(): v.encode() for k, v in message.to_redis().items()}


def _result(worker: SummaryTemplateWorker) -> SummaryResultMessage:
    worker.publish.assert_awaited_once()  # type: ignore[attr-defined]
    return SummaryResultMessage.from_redis(worker.publish.await_args.args[2])  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_language_switch_translates_the_published_summary_without_the_transcript() -> None:
    worker = _worker()

    await worker.process(
        b"1",
        _request(
            template_key="general",
            summary_language="vi",
            mode="translate",
            source_content_json=json.dumps(_published(), ensure_ascii=False),
        ),
    )

    result = _result(worker)
    assert result.status == "completed", result.error
    content = json.loads(result.content_json)
    assert content["summaryLanguage"] == "vi"
    assert content["templateKey"] == "general"
    assert content["summary"] == "Nhóm đã xem xét hệ thống AI."
    # Echoed, so the backend can confirm it is filing this where the reader will look.
    assert (result.requested_template_key, result.summary_language) == ("general", "vi")
    worker._load_transcript.assert_not_awaited()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_translation_that_cannot_be_completed_is_a_failure_with_a_reason() -> None:
    worker = _worker()
    source = _published()
    del source["translations"]

    await worker.process(
        b"1",
        _request(
            summary_language="vi",
            mode="translate",
            source_content_json=json.dumps(source),
        ),
    )

    result = _result(worker)
    assert result.status == "failed"
    assert "Vietnamese" in result.error
    assert result.summary_language == "vi"


@pytest.mark.asyncio
async def test_a_rendering_in_a_language_nobody_can_name_is_refused_not_filed_as_spoken() -> None:
    worker = _worker()
    worker.assistant.generate_structured_summary = AsyncMock()  # type: ignore[method-assign, union-attr]

    await worker.process(b"1", _request(summary_language="klingon"))

    result = _result(worker)
    assert result.status == "failed"
    assert "klingon" in result.error
    worker.assistant.generate_structured_summary.assert_not_awaited()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_generated_rendering_is_stamped_and_echoed_in_the_requested_language() -> None:
    worker = _worker()
    worker.assistant = MagicMock()
    worker.assistant.generate_structured_summary = AsyncMock(
        return_value={"summary": "Xin chào", "templateKey": "standup", "summaryLanguage": "vi"}
    )

    await worker.process(b"1", _request(template_key="Standup", summary_language="vi-VN"))

    result = _result(worker)
    assert result.status == "completed"
    assert (
        worker.assistant.generate_structured_summary.await_args.kwargs["summary_language"] == "vi"
    )
    assert (result.requested_template_key, result.summary_language) == ("standup", "vi")


def test_the_new_request_and_result_fields_survive_the_redis_round_trip() -> None:
    request = SummaryRequestMessage(
        request_id="r",
        room_id=ROOM,
        workspace_id="w",
        mode="translate",
        source_content_json='{"summary":"x"}',
    )
    decoded = SummaryRequestMessage.from_redis(
        {k.encode(): v.encode() for k, v in request.to_redis().items()}
    )
    assert (decoded.mode, decoded.source_content_json) == ("translate", '{"summary":"x"}')

    # A request published before `mode` existed keeps generating.
    legacy = request.to_redis()
    del legacy["mode"]
    del legacy["source_content_json"]
    assert SummaryRequestMessage.from_redis(legacy).mode == "generate"

    result = SummaryResultMessage(
        request_id="r",
        room_id=ROOM,
        template_key="general",
        status="completed",
        requested_template_key="general",
        summary_language="vi",
    )
    back = SummaryResultMessage.from_redis(result.to_redis())
    assert (back.requested_template_key, back.summary_language) == ("general", "vi")
