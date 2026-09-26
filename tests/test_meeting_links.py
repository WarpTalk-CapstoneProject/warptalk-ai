"""A meeting WarpBot creates must reach the user as a link they can open.

Production, 17 Sep: "tạo 1 cuộc họp bằng @Google Meet" ended on "Bạn chọn các thông tin trên
thẻ…" with no card on screen and no link. These lock the three things that were missing: the
link is guaranteed in the answer, the two kinds of meeting are told apart, and the confirmation
card says what it will create.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_templates import GENERAL, build_system_prompt
from ai_assistant_worker.mcp_tools import build_mcp_confirmation_questions
from ai_assistant_worker.meeting_links import (
    MEETING_MARKER_PREFIX,
    MeetingLink,
    ensure_meeting_links,
    meet_code_from_url,
    meeting_link_from_tool_result,
    room_url,
    strip_meeting_markers,
)
from tests.test_chat_agent_loop import (
    _build_worker,
    _completed,
    _function_call,
    _message_item,
    _request,
    _text_delta,
)

MEET_URL = "https://meet.google.com/abc-defg-hij"


def _meet_result(**data: Any) -> str:
    return json.dumps(
        {
            "isSuccess": True,
            "data": {
                "provider": "google_meet",
                "summary": "Quick sync",
                "meetLink": MEET_URL,
                "meetingCode": "abc-defg-hij",
                **data,
            },
        }
    )


class TestMeetingLinkFromToolResult:
    def test_a_google_meet_result_is_a_google_meet_link(self) -> None:
        assert meeting_link_from_tool_result(_meet_result()) == MeetingLink(
            kind="google_meet", url=MEET_URL, title="Quick sync", code="abc-defg-hij"
        )

    def test_the_code_falls_back_to_the_url(self) -> None:
        link = meeting_link_from_tool_result(_meet_result(meetingCode=None))
        assert link is not None and link.code == "abc-defg-hij"

    def test_a_pending_meet_has_no_link_to_give(self) -> None:
        assert meeting_link_from_tool_result(_meet_result(meetLink=None)) is None

    def test_a_link_off_google_is_refused(self) -> None:
        assert meeting_link_from_tool_result(_meet_result(meetLink="https://evil.test/x")) is None

    def test_a_created_warptalk_room_is_a_room_link(self) -> None:
        result = json.dumps(
            {"status": "created", "id": "r-1", "title": "Sync team", "room_url": "/rooms/r-1"}
        )
        assert meeting_link_from_tool_result(result) == MeetingLink(
            kind="warptalk_room", url="/rooms/r-1", title="Sync team", code=None
        )

    def test_anything_else_is_nothing(self) -> None:
        for result in ("", "not json", "[]", json.dumps({"isSuccess": True, "data": {}})):
            assert meeting_link_from_tool_result(result) is None


class TestEnsureMeetingLinks:
    link = MeetingLink(
        kind="google_meet",
        url=MEET_URL,
        title="Quick sync",
        code="abc-defg-hij",
        start="2026-09-18T15:40:00+07:00",
        end="2026-09-18T16:10:00+07:00",
        calendar_url="https://www.google.com/calendar/event?eid=abc",
    )

    def _marker(self, answer: str) -> dict[str, Any]:
        line = next(line for line in answer.splitlines() if line.startswith(MEETING_MARKER_PREFIX))
        return json.loads(line[len(MEETING_MARKER_PREFIX) : -len(" -->")])

    def test_the_marker_is_appended_and_no_visible_link(self) -> None:
        answer = ensure_meeting_links("Đã tạo cuộc họp.", [self.link])
        assert answer.startswith(f"Đã tạo cuộc họp.\n\n{MEETING_MARKER_PREFIX}")
        assert f"[Quick sync]({MEET_URL})" not in answer
        assert self._marker(answer) == {
            "kind": "google_meet",
            "url": MEET_URL,
            "title": "Quick sync",
            "code": "abc-defg-hij",
            "start": "2026-09-18T15:40:00+07:00",
            "end": "2026-09-18T16:10:00+07:00",
            "calendarUrl": "https://www.google.com/calendar/event?eid=abc",
        }

    def test_one_marker_per_meeting(self) -> None:
        answer = ensure_meeting_links("ok", [self.link, self.link])
        assert answer.count(MEETING_MARKER_PREFIX) == 1

    def test_a_bridge_room_says_so(self) -> None:
        link = meeting_link_from_tool_result(
            json.dumps(
                {
                    "status": "created",
                    "title": "Japan client",
                    "room_url": "/rooms/r-1",
                    "room_type": "EXTERNAL_BRIDGE",
                }
            )
        )
        assert link is not None
        assert self._marker(ensure_meeting_links("", [link]))["roomType"] == "EXTERNAL_BRIDGE"

    def test_a_title_cannot_close_the_comment_early(self) -> None:
        link = MeetingLink(kind="warptalk_room", url="/rooms/r-1", title="a --> b")
        answer = ensure_meeting_links("ok", [link])
        marker_line = answer.splitlines()[-1]
        assert marker_line.count("-->") == 1 and marker_line.endswith(" -->")
        assert self._marker(answer)["title"] == "a --> b"

    def test_a_calendar_link_off_google_is_dropped(self) -> None:
        link = meeting_link_from_tool_result(
            _meet_result(calendarEventLink="https://evil.test/calendar")
        )
        assert link is not None and link.calendar_url is None


def test_helpers() -> None:
    assert room_url(" r-1 ") == "/rooms/r-1"
    assert meet_code_from_url(f"{MEET_URL}?authuser=0") == "abc-defg-hij"
    assert meet_code_from_url("https://meet.google.com/landing") is None


class TestPermissionPrompt:
    def _prompt(self) -> dict[str, Any]:
        payload = {"confirmationToken": "token-1", "message": "Confirm this action."}
        return build_mcp_confirmation_questions(
            payload,
            tool_name="google_calendar_create_meet_event",
            tool_label="Create Google Meet meeting",
        )["permission"]

    def test_it_names_the_action_and_nothing_else(self) -> None:
        # One prompt shape for every ask. It used to carry rows per tool — Title, When, Calendar —
        # built from the arguments, and those were the server's defaults as often as the user's
        # words, so the card described decisions nobody had made.
        prompt = self._prompt()
        assert prompt["kind"] == "tool"
        assert prompt["action"] == "Create Google Meet meeting"
        assert prompt["toolName"] == "google_calendar_create_meet_event"
        assert set(prompt) == {"kind", "action", "toolName", "options"}

    def test_the_answers_read_as_a_person_would_say_them(self) -> None:
        labels = [option["label"] for option in self._prompt()["options"]]
        assert labels == [
            "Yes",
            "Yes, and don't ask again for this tool",
            "No, and tell WarpBot what to do differently",
        ]

    def test_the_answer_leads_with_the_choice_and_keeps_the_token_for_the_model(self) -> None:
        value = self._prompt()["options"][0]["value"]
        human, machine = value.split("\n\n", 1)
        assert human == "Yes"
        assert "google_calendar_create_meet_event" in machine
        assert "confirmationToken: token-1" in machine

    def test_a_tool_with_no_label_is_named_by_its_own_name(self) -> None:
        prompt = build_mcp_confirmation_questions(
            {"confirmationToken": "t", "message": "Confirm this action."},
            tool_name="save_issue",
        )["permission"]
        assert prompt["action"] == "save_issue"


def test_the_prompt_separates_warptalk_rooms_from_google_meet() -> None:
    prompt = build_system_prompt(GENERAL)
    assert "A WARPTALK ROOM OR A GOOGLE MEET MEETING" in prompt
    assert "Do not create a WarpTalk room instead." in prompt
    # The marker is the worker's to write, never the model's.
    assert "Never" in prompt and "HTML comment of your own" in prompt
    # And the confirmation round-trip has to come back with the same arguments.
    assert "EXACTLY the arguments you sent the first time" in prompt


def test_the_history_the_model_sees_carries_no_markers() -> None:
    link = MeetingLink(kind="google_meet", url=MEET_URL, title="Quick sync", code="abc-defg-hij")
    answer = ensure_meeting_links("Đã tạo cuộc họp.", [link])
    assert MEETING_MARKER_PREFIX in answer
    assert strip_meeting_markers(answer) == "Đã tạo cuộc họp."
    assert strip_meeting_markers("plain answer") == "plain answer"


class TestAgentLoopGuaranteesTheLink:
    async def test_a_created_meet_link_reaches_an_answer_that_left_it_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(chat_worker_module, "TOOLS", [])
        monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", {})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "name": "google_calendar_create_meet_event",
                            "pluginKey": "google_meet",
                            "label": "Create Google Meet meeting",
                            "effect": "write",
                            "policy": "allow",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                )
            return httpx.Response(200, content=_meet_result())

        assistant_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://assistant-service"
        )
        worker, _ = _build_worker(
            [
                [_completed(_function_call("google_calendar_create_meet_event", "{}"))],
                [_text_delta("Đã tạo cuộc họp Google Meet."), _completed(_message_item())],
            ]
        )
        worker.chat_settings.web_search_enabled = False

        try:
            text, _ = await worker._run_agent_loop(
                _request(),
                [],
                SimpleNamespace(assistant_client=assistant_client, citations=None),
            )
        finally:
            await assistant_client.aclose()

        assert text.startswith(f"Đã tạo cuộc họp Google Meet.\n\n{MEETING_MARKER_PREFIX}")
        assert MEET_URL in text and "abc-defg-hij" in text

    async def test_the_confirmation_card_carries_the_meeting_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(chat_worker_module, "TOOLS", [])
        monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", {})

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "name": "google_calendar_create_meet_event",
                            "pluginKey": "google_meet",
                            "label": "Create Google Meet meeting",
                            "effect": "write",
                            "policy": "approval",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                )
            return httpx.Response(
                200,
                json={
                    "isSuccess": False,
                    "errorCode": "confirmation_required",
                    "message": "Confirm this action.",
                    "confirmationToken": "token-1",
                },
            )

        assistant_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://assistant-service"
        )
        worker, published = _build_worker(
            [
                [
                    _completed(
                        _function_call("google_calendar_create_meet_event", '{"summary":"Standup"}')
                    )
                ],
                [_text_delta("Bạn xác nhận để mình tạo nhé."), _completed(_message_item())],
            ]
        )
        worker.chat_settings.web_search_enabled = False

        try:
            await worker._run_agent_loop(
                _request(),
                [],
                SimpleNamespace(assistant_client=assistant_client, citations=None),
            )
        finally:
            await assistant_client.aclose()

        questions = [p for p in published if p.get("type_") == "question"]
        assert len(questions) == 1
        prompt = json.loads(questions[0]["tool_calls_json"])["permission"]
        assert prompt["kind"] == "tool"
        assert prompt["action"] == "Create Google Meet meeting"
