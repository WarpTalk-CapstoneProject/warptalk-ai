"""Tests for WarpBot assistant configuration."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ai_assistant_worker.assistant import MeetingAssistant


async def test_assistant_requires_openai_api_key() -> None:
    assistant = MeetingAssistant(api_key="")

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await assistant.load()


def _make_assistant_with_fake_client(response_content: str) -> MeetingAssistant:
    assistant = MeetingAssistant(api_key="test-key")
    fake_message = SimpleNamespace(content=response_content)
    fake_choice = SimpleNamespace(message=fake_message)
    fake_response = SimpleNamespace(choices=[fake_choice])
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=fake_response))
        )
    )
    assistant._client = fake_client
    return assistant


async def test_generate_structured_summary_returns_insufficient_data_for_empty_transcript() -> None:
    assistant = MeetingAssistant(api_key="test-key")

    result = await assistant.generate_structured_summary("   ")

    assert result["insufficientData"] is True
    assert result["decisions"] == []
    assert result["actionItems"] == []


async def test_a_transcript_of_empty_segments_is_insufficient_not_summarised() -> None:
    """WT-478 — the bug, at the layer that decides.

    Timestamps and speaker labels with nothing said between them are truthy to `.strip()`,
    so this transcript used to reach the model. The model then reported the transcript was
    empty, the call SUCCEEDED, insufficientData stayed False, and the UI rendered that
    report as the meeting's summary. The fake client below would answer anything, so if the
    gate regresses this test fails on the assertion rather than on a missing client.
    """
    assistant = _make_assistant_with_fake_client('{"summary": "the model was asked anyway"}')

    result = await assistant.generate_structured_summary("[t=0] [Nhi] \n[t=1200] [Ky]    ")

    assert result["insufficientData"] is True
    assert result["summary"] == "No transcript content to summarize."


async def test_a_short_but_real_transcript_is_still_summarised() -> None:
    # The other half of the ticket: "kể cả khi nội dung ngắn". Two sentences is a meeting.
    payload = {"summary": "Nhi confirmed the Q3 receivables.", "decisions": [], "actionItems": []}
    assistant = _make_assistant_with_fake_client(json.dumps(payload))

    result = await assistant.generate_structured_summary("[t=0] [Nhi] chốt công nợ quý ba")

    assert result["insufficientData"] is False
    assert result["summary"] == payload["summary"]


async def test_generate_structured_summary_parses_model_json() -> None:
    payload = {
        "summary": "The team reviewed the Q3 roadmap.",
        "decisions": ["Ship the beta by August"],
        "actionItems": [{"owner": "Alice", "task": "Draft the release notes"}],
    }
    assistant = _make_assistant_with_fake_client(json.dumps(payload))

    result = await assistant.generate_structured_summary("Alice: let's ship the beta by August.")

    assert result["summary"] == payload["summary"]
    assert result["decisions"] == payload["decisions"]
    assert result["actionItems"] == payload["actionItems"]
    assert result["insufficientData"] is False


async def test_structured_summary_requests_bilingual_output() -> None:
    assistant = _make_assistant_with_fake_client(json.dumps({"summary": "ok"}))

    await assistant.generate_structured_summary(
        "Some transcript text.",
        target_languages=["en", "vi"],
    )

    call_kwargs = assistant._client.chat.completions.create.call_args.kwargs
    system_message = call_kwargs["messages"][0]["content"]
    assert "en, vi" in system_message
    assert call_kwargs["response_format"] == {"type": "json_object"}


async def test_generate_structured_summary_falls_back_gracefully_on_malformed_json() -> None:
    assistant = _make_assistant_with_fake_client("not valid json")

    result = await assistant.generate_structured_summary("Some transcript text.")

    assert result["insufficientData"] is True
    assert result["decisions"] == []
    assert result["actionItems"] == []


# ── WT-665: choosing a summary language must not cost the biên bản its other languages ──


def _assistant_answering(*response_contents: str) -> MeetingAssistant:
    """An assistant whose client answers each successive call with the next string.

    The bilingual path makes TWO calls — the summary, then the translation — so a single
    canned response cannot express what these tests are about.
    """
    assistant = MeetingAssistant(api_key="test-key")
    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        for content in response_contents
    ]
    assistant._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock(side_effect=responses)))
    )
    return assistant


_SUMMARY_JSON = json.dumps(
    {
        "summary": "Chốt công nợ quý ba.",
        "decisions": [{"text": "Duyệt ngân sách", "atMs": 0}],
        "actionItems": [{"task": "Gửi báo cáo", "owner": "Tú", "atMs": 0}],
    }
)


async def test_a_chosen_language_still_leaves_the_minutes_bilingual() -> None:
    # The defect: asking for the summary in Japanese used to null the translations map, and the
    # biên bản lost half of itself because of a choice made about a different document.
    assistant = _assistant_answering(
        _SUMMARY_JSON,
        json.dumps(
            {
                "en": {
                    "summary": "Q3 receivables settled.",
                    "decisions": ["Budget approved"],
                    "actionItems": ["Send the report"],
                }
            }
        ),
    )

    result = await assistant.generate_structured_summary(
        "[t=0] [Tú] chốt công nợ quý ba",
        target_languages=["vi", "en"],
        summary_language="ja",
    )

    assert result["summaryLanguage"] == "ja"
    assert result["translations"]["en"]["summary"] == "Q3 receivables settled."
    # The moment came from the source, not from the translating model — it was never shown one.
    assert result["translations"]["en"]["decisions"][0]["atMs"] == 0
    assert result["translations"]["en"]["actionItems"][0]["owner"] == "Tú"


async def test_the_translating_call_is_never_shown_a_moment() -> None:
    assistant = _assistant_answering(_SUMMARY_JSON, json.dumps({"en": {"summary": "x"}}))

    await assistant.generate_structured_summary(
        "[t=0] [Tú] chốt công nợ quý ba",
        target_languages=["vi", "en"],
        summary_language="ja",
    )

    second_call = assistant._client.chat.completions.create.call_args_list[1].kwargs
    sent = second_call["messages"][1]["content"]
    assert "atMs" not in sent
    assert "Duyệt ngân sách" in sent


async def test_one_language_means_there_is_nothing_to_translate_into() -> None:
    assistant = _assistant_answering(_SUMMARY_JSON)

    result = await assistant.generate_structured_summary(
        "[t=0] [Tú] chốt công nợ quý ba",
        target_languages=["vi"],
        summary_language="ja",
    )

    assert "translations" not in result
    assert assistant._client.chat.completions.create.await_count == 1


async def test_a_failed_translation_costs_the_minutes_nothing_else() -> None:
    # The summary itself succeeded and is about to be published. Losing the second language is
    # worth a warning; it is not a reason to throw away a summary that exists.
    assistant = _assistant_answering(_SUMMARY_JSON, "not json at all")

    result = await assistant.generate_structured_summary(
        "[t=0] [Tú] chốt công nợ quý ba",
        target_languages=["vi", "en"],
        summary_language="ja",
    )

    assert result["summary"] == "Chốt công nợ quý ba."
    assert result["insufficientData"] is False
    assert "translations" not in result


async def test_a_crash_in_the_grounding_pass_costs_the_citations_and_not_the_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check is not the generation, and a broken check must not destroy what it checked.

    `ground_summary` runs inside the same `try` that catches a malformed model response, so
    anything it raises used to come out as `generationFailed` — which the worker publishes as a
    failed rewrite, which the backend answers by leaving the OLD summary in place, which the
    browser never hears about. A bug in the checker would have read, from every side, as a model
    that could not write a summary.
    """

    def explode(_summary: object, _transcript: str) -> object:
        raise RuntimeError("the checker itself is broken")

    monkeypatch.setattr("ai_assistant_worker.assistant.ground_summary", explode)

    assistant = _make_assistant_with_fake_client(
        json.dumps(
            {
                "summary": "Nhóm chốt công nợ quý ba.",
                "narrative": [{"text": "Nhóm chốt công nợ.", "atMs": 0, "alsoAtMs": [12_000]}],
                "actionItems": [{"task": "Gửi hợp đồng.", "owner": "Ky", "atMs": 12_000}],
            }
        )
    )

    result = await assistant.generate_structured_summary(
        "[t=0] [Nhi] chốt công nợ quý ba", template_key="traceable"
    )

    # The summary survives, whole and readable...
    assert "generationFailed" not in result
    assert result["insufficientData"] is False
    assert result["summary"] == "Nhóm chốt công nợ quý ba."
    assert result["narrative"][0]["text"] == "Nhóm chốt công nợ."
    assert result["actionItems"][0]["owner"] == "Ky"
    # ...and claims nothing it could not confirm.
    assert result["narrative"][0]["atMs"] is None
    assert result["narrative"][0]["alsoAtMs"] == []
    assert result["actionItems"][0]["atMs"] is None
