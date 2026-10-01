"""GMCAL1001: each Google plugin calls only its own API, and a WarpBot Meet lands on the calendars.

The Meet tool now creates the meeting through the Meet REST API, with no Calendar event. These
lock what follows from that: a Meet-only result is still a card; a Calendar event chained onto it
is folded onto that card rather than drawn beside it; and the worker itself files one
EXTERNAL_BRIDGE room per Meet so it shows on WarpTalk's Calendar page - once, quietly, and never
at the expense of the answer.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_templates import GENERAL, build_system_prompt
from ai_assistant_worker.chat_tools import ChatTool
from ai_assistant_worker.meet_rooms import (
    build_meet_room_payload,
    file_meet_rooms,
    scheduled_at_utc,
)
from ai_assistant_worker.meeting_links import (
    MEETING_MARKER_PREFIX,
    CalendarEvent,
    MeetingLink,
    bridged_meet_code_from_tool_result,
    calendar_event_from_tool_result,
    ensure_meeting_links,
    meeting_link_from_tool_result,
    merge_calendar_events,
)
from tests.test_chat_agent_loop import (
    _build_worker,
    _completed,
    _function_call,
    _message_item,
    _request,
    _text_delta,
)


def _tz_database_available() -> bool:
    try:
        ZoneInfo("Asia/Tokyo")
    except Exception:
        return False
    return True


#: Windows Python ships no IANA database (the worker runs on Linux, which has one).
_needs_tz_database = pytest.mark.skipif(
    not _tz_database_available(), reason="no IANA time zone database on this machine"
)

MEET_URL = "https://meet.google.com/abc-defg-hij"
CAL_URL = "https://www.google.com/calendar/event?eid=evt"


def _meet_result(**data: Any) -> str:
    return json.dumps(
        {
            "isSuccess": True,
            "data": {
                "provider": "google_meet",
                "summary": "Quick sync",
                "meetLink": MEET_URL,
                "meetingCode": "abc-defg-hij",
                "spaceName": "spaces/xyz",
                **data,
            },
        }
    )


def _calendar_result(**data: Any) -> str:
    return json.dumps(
        {
            "isSuccess": True,
            "data": {
                "provider": "google_calendar",
                "eventId": "evt-1",
                "htmlLink": CAL_URL,
                "hangoutLink": MEET_URL,
                "meetLink": MEET_URL,
                "summary": "Quick sync",
                "start": "2026-10-02T09:00:00+07:00",
                "end": "2026-10-02T09:30:00+07:00",
                "conferenceAttached": True,
                **data,
            },
        }
    )


def _marker(answer: str) -> dict[str, Any]:
    line = next(line for line in answer.splitlines() if line.startswith(MEETING_MARKER_PREFIX))
    return json.loads(line[len(MEETING_MARKER_PREFIX) : -len(" -->")])


class TestMeetOnlyCard:
    def test_a_meet_rest_result_with_no_time_or_calendar_is_still_a_card(self) -> None:
        link = meeting_link_from_tool_result(_meet_result(summary=None, start=None, end=None))
        assert link == MeetingLink(
            kind="google_meet", url=MEET_URL, title="Google Meet", code="abc-defg-hij"
        )
        assert _marker(ensure_meeting_links("ok", [link])) == {
            "kind": "google_meet",
            "url": MEET_URL,
            "title": "Google Meet",
            "code": "abc-defg-hij",
        }

    def test_echoed_times_reach_the_card_when_present(self) -> None:
        link = meeting_link_from_tool_result(
            _meet_result(start="2026-10-02T09:00:00", end="2026-10-02T09:30:00")
        )
        assert link is not None
        assert (link.start, link.end) == ("2026-10-02T09:00:00", "2026-10-02T09:30:00")


class TestCalendarMerge:
    def test_a_calendar_event_for_the_meet_is_parsed(self) -> None:
        assert calendar_event_from_tool_result(_calendar_result()) == CalendarEvent(
            meet_url=MEET_URL,
            meet_code="abc-defg-hij",
            event_id="evt-1",
            html_link=CAL_URL,
            start="2026-10-02T09:00:00+07:00",
            end="2026-10-02T09:30:00+07:00",
        )

    def test_hangout_link_alone_is_enough(self) -> None:
        event = calendar_event_from_tool_result(_calendar_result(meetLink=None))
        assert event is not None and event.meet_code == "abc-defg-hij"

    def test_google_time_objects_are_read(self) -> None:
        event = calendar_event_from_tool_result(
            _calendar_result(start={"dateTime": "2026-10-02T09:00:00+07:00"})
        )
        assert event is not None and event.start == "2026-10-02T09:00:00+07:00"

    def test_an_event_without_a_meet_link_or_a_failed_call_is_nothing(self) -> None:
        assert (
            calendar_event_from_tool_result(_calendar_result(meetLink=None, hangoutLink=None))
            is None
        )
        assert calendar_event_from_tool_result(_meet_result()) is None
        assert calendar_event_from_tool_result(json.dumps({"isSuccess": False})) is None
        assert calendar_event_from_tool_result("not json") is None

    def test_an_off_google_event_link_is_dropped(self) -> None:
        event = calendar_event_from_tool_result(_calendar_result(htmlLink="https://evil.test/x"))
        assert event is not None and event.html_link is None

    def test_the_event_folds_onto_the_meet_card(self) -> None:
        meet = meeting_link_from_tool_result(_meet_result())
        event = calendar_event_from_tool_result(_calendar_result())
        assert meet is not None and event is not None
        merged = merge_calendar_events([meet], [event])
        assert len(merged) == 1
        card = merged[0]
        assert card.calendar_url == CAL_URL
        assert card.calendar_event_id == "evt-1"
        assert (card.start, card.end) == (
            "2026-10-02T09:00:00+07:00",
            "2026-10-02T09:30:00+07:00",
        )
        answer = ensure_meeting_links("ok", merged)
        assert answer.count(MEETING_MARKER_PREFIX) == 1
        marker = _marker(answer)
        assert marker["calendarUrl"] == CAL_URL
        # The marker keeps its shape: the event id is the room's business, not the card's.
        assert set(marker) == {"kind", "url", "title", "code", "start", "end", "calendarUrl"}

    def test_an_event_for_another_meet_is_not_folded(self) -> None:
        meet = meeting_link_from_tool_result(_meet_result())
        other_url = "https://meet.google.com/zzz-zzzz-zzz"
        other = calendar_event_from_tool_result(
            _calendar_result(meetLink=other_url, hangoutLink=other_url)
        )
        assert meet is not None and other is not None
        assert merge_calendar_events([meet], [other]) == [meet]


class TestMeetRoomPayload:
    NOW = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)

    def test_a_meet_with_a_future_time_is_scheduled_and_linked_to_its_event(self) -> None:
        link = MeetingLink(
            kind="google_meet",
            url=f"{MEET_URL}?authuser=0",
            title="Quick sync",
            code="abc-defg-hij",
            start="2026-10-02T09:00:00+07:00",
            calendar_url=CAL_URL,
            calendar_event_id="evt-1",
        )
        assert build_meet_room_payload(link, "ws-1", now=self.NOW) == {
            "workspaceId": "ws-1",
            "title": "Quick sync",
            "translationRoomType": "EXTERNAL_BRIDGE",
            "externalProvider": "GOOGLE_MEET",
            "externalMeetingUrl": MEET_URL,
            "scheduledAt": "2026-10-02T02:00:00Z",
            "externalCalendarEventId": "evt-1",
            "externalCalendarEventUrl": CAL_URL,
        }

    def test_a_meet_for_now_is_not_scheduled_and_gets_a_default_title(self) -> None:
        link = MeetingLink(kind="google_meet", url=MEET_URL, title="Google Meet")
        payload = build_meet_room_payload(link, "ws-1", now=self.NOW)
        assert payload == {
            "workspaceId": "ws-1",
            "title": "Google Meet meeting",
            "translationRoomType": "EXTERNAL_BRIDGE",
            "externalProvider": "GOOGLE_MEET",
            "externalMeetingUrl": MEET_URL,
        }

    @_needs_tz_database
    def test_a_bare_time_is_read_in_the_calls_zone(self) -> None:
        assert (
            scheduled_at_utc("2026-10-02T09:00:00", "Asia/Ho_Chi_Minh", now=self.NOW)
            == "2026-10-02T02:00:00Z"
        )
        # An unknown zone falls back to the workspace default rather than to UTC.
        assert (
            scheduled_at_utc("2026-10-02T09:00:00", "Not/AZone", now=self.NOW)
            == "2026-10-02T02:00:00Z"
        )

    def test_times(self) -> None:
        assert scheduled_at_utc("2026-10-02T09:00:00+07:00", now=self.NOW) == (
            "2026-10-02T02:00:00Z"
        )
        assert scheduled_at_utc("2026-10-02T09:00:00Z", now=self.NOW) == "2026-10-02T09:00:00Z"
        # The room API refuses a ScheduledAt that is not in the future.
        assert scheduled_at_utc("2026-09-30T09:00:00Z", now=self.NOW) is None
        assert scheduled_at_utc("2026-10-01T00:00:30Z", now=self.NOW) is None
        assert scheduled_at_utc("not a time", now=self.NOW) is None
        assert scheduled_at_utc(None, now=self.NOW) is None

    def test_never_for_a_warptalk_room(self) -> None:
        link = MeetingLink(kind="warptalk_room", url="/rooms/r-1", title="Sync")
        assert build_meet_room_payload(link, "ws-1") is None


def _room_context(handler: Any) -> tuple[SimpleNamespace, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://translation-room"
    )
    ctx = SimpleNamespace(translation_room_client=client, workspace_id="ws-1", bearer_token="B t")
    return ctx, client


class TestFileMeetRooms:
    meet = MeetingLink(kind="google_meet", url=MEET_URL, title="Quick sync", code="abc-defg-hij")

    async def test_one_room_per_meet_as_the_user(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(201, json={"id": "room-1"})

        ctx, client = _room_context(handler)
        try:
            await file_meet_rooms(ctx, [self.meet, self.meet], already_bridged=set())
        finally:
            await client.aclose()
        assert len(calls) == 1
        assert calls[0].method == "POST"
        assert calls[0].url.path == "/api/v1/translation-rooms"
        assert calls[0].headers["Authorization"] == "B t"
        body = json.loads(calls[0].content)
        assert body["translationRoomType"] == "EXTERNAL_BRIDGE"
        assert body["externalMeetingUrl"] == MEET_URL

    async def test_none_when_the_model_already_bridged_it(self) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(201, json={"id": "room-1"})

        ctx, client = _room_context(handler)
        try:
            await file_meet_rooms(ctx, [self.meet], already_bridged={"ABC-DEFG-HIJ"})
        finally:
            await client.aclose()
        assert calls == []

    @pytest.mark.parametrize("failure", ["refused", "unreachable", "garbage"])
    async def test_a_failure_is_swallowed(self, failure: str) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if failure == "refused":
                return httpx.Response(403, json={"error": "nope"})
            if failure == "garbage":
                return httpx.Response(201, content=b"not json")
            raise httpx.ConnectError("down")

        ctx, client = _room_context(handler)
        try:
            await file_meet_rooms(ctx, [self.meet], already_bridged=set())
        finally:
            await client.aclose()

    async def test_no_client_no_room(self) -> None:
        await file_meet_rooms(SimpleNamespace(), [self.meet], already_bridged=set())


def test_the_bridged_code_of_a_created_room_is_read() -> None:
    created = {"status": "created", "room_url": "/rooms/r-1", "external_meeting_url": MEET_URL}
    assert bridged_meet_code_from_tool_result(json.dumps(created)) == "abc-defg-hij"
    assert bridged_meet_code_from_tool_result(json.dumps({"status": "created"})) is None
    assert bridged_meet_code_from_tool_result("nope") is None


def test_the_prompt_chains_calendar_and_leaves_the_room_to_the_worker() -> None:
    prompt = build_system_prompt(GENERAL)
    assert "call google_calendar_create_event in the same turn" in prompt
    assert "say nothing about Google Calendar" in prompt
    assert "Never say a calendar event was created" in prompt
    assert "Never call create_meeting for a Google Meet meeting you just created" in prompt
    # The old "Meet, then an EXTERNAL_BRIDGE room" rule is gone: the worker files that room.
    assert "then create a WarpTalk room of type EXTERNAL_BRIDGE" not in prompt
    # #464 stays: the Meet itself is created without asking.
    assert "do not ask questions first" in prompt


def _plugin_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "pluginKey": plugin,
            "label": label,
            "effect": "write",
            "policy": "allow",
            "parameters": {"type": "object", "properties": {}},
        }
        for name, plugin, label in (
            ("google_calendar_create_meet_event", "google_meet", "Create Google Meet meeting"),
            ("google_calendar_create_event", "google_calendar", "Create calendar event"),
        )
    ]


class TestAgentLoopFilesTheMeet:
    async def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        turns: list[list[Any]],
        room_handler: Any,
        extra_tools: dict[str, Any] | None = None,
    ) -> str:
        monkeypatch.setattr(chat_worker_module, "TOOLS", [])
        monkeypatch.setattr(chat_worker_module, "TOOLS_BY_NAME", extra_tools or {})

        def assistant(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json=_plugin_catalog())
            if json.loads(request.content).get("toolName") == "google_calendar_create_event":
                return httpx.Response(
                    200,
                    content=_calendar_result(
                        start="2099-10-02T09:00:00+07:00", end="2099-10-02T09:30:00+07:00"
                    ),
                )
            return httpx.Response(200, content=_meet_result(start="2099-10-02T09:00:00"))

        assistant_client = httpx.AsyncClient(
            transport=httpx.MockTransport(assistant), base_url="http://assistant-service"
        )
        room_client = httpx.AsyncClient(
            transport=httpx.MockTransport(room_handler), base_url="http://translation-room"
        )
        worker, _ = _build_worker(turns)
        worker.chat_settings.web_search_enabled = False
        ctx = SimpleNamespace(
            assistant_client=assistant_client,
            translation_room_client=room_client,
            workspace_id="ws-1",
            bearer_token="Bearer t",
            citations=None,
        )
        try:
            text, _ = await worker._run_agent_loop(_request(), [], ctx)
        finally:
            await assistant_client.aclose()
            await room_client.aclose()
        return text

    async def test_meet_then_calendar_is_one_card_and_one_room(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rooms: list[dict[str, Any]] = []

        def room_handler(request: httpx.Request) -> httpx.Response:
            rooms.append(json.loads(request.content))
            return httpx.Response(201, json={"id": "room-1"})

        meet_args = {
            "summary": "Quick sync",
            "start": "2099-10-02T09:00:00",
            "timeZone": "Asia/Ho_Chi_Minh",
        }
        text = await self._run(
            monkeypatch,
            [
                [
                    _completed(
                        _function_call(
                            "google_calendar_create_meet_event",
                            json.dumps(meet_args),
                            call_id="c1",
                        )
                    )
                ],
                [
                    _completed(
                        _function_call(
                            "google_calendar_create_event",
                            json.dumps({"summary": "Quick sync", "meetLink": MEET_URL}),
                            call_id="c2",
                        )
                    )
                ],
                [_text_delta("Đã tạo."), _completed(_message_item())],
            ],
            room_handler,
        )

        assert text.count(MEETING_MARKER_PREFIX) == 1
        marker = _marker(text)
        assert marker["calendarUrl"] == CAL_URL
        assert marker["start"] == "2099-10-02T09:00:00+07:00"
        assert rooms == [
            {
                "workspaceId": "ws-1",
                "title": "Quick sync",
                "translationRoomType": "EXTERNAL_BRIDGE",
                "externalProvider": "GOOGLE_MEET",
                "externalMeetingUrl": MEET_URL,
                "scheduledAt": "2099-10-02T02:00:00Z",
                "externalCalendarEventId": "evt-1",
                "externalCalendarEventUrl": CAL_URL,
            }
        ]

    @_needs_tz_database
    async def test_a_meet_only_turn_files_the_room_from_the_echoed_time_and_zone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rooms: list[dict[str, Any]] = []

        def room_handler(request: httpx.Request) -> httpx.Response:
            rooms.append(json.loads(request.content))
            return httpx.Response(201, json={"id": "room-1"})

        text = await self._run(
            monkeypatch,
            [
                [
                    _completed(
                        _function_call(
                            "google_calendar_create_meet_event",
                            json.dumps({"timeZone": "Asia/Tokyo"}),
                        )
                    )
                ],
                [_text_delta("Đã tạo."), _completed(_message_item())],
            ],
            room_handler,
        )
        assert "calendarUrl" not in _marker(text)
        assert len(rooms) == 1
        assert rooms[0]["scheduledAt"] == "2099-10-02T00:00:00Z"
        assert "externalCalendarEventId" not in rooms[0]

    async def test_a_room_failure_leaves_the_answer_and_the_card(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def room_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        text = await self._run(
            monkeypatch,
            [
                [_completed(_function_call("google_calendar_create_meet_event", "{}"))],
                [_text_delta("Đã tạo."), _completed(_message_item())],
            ],
            room_handler,
        )
        assert text.startswith(f"Đã tạo.\n\n{MEETING_MARKER_PREFIX}")
        assert MEET_URL in text

    async def test_no_second_room_when_the_model_bridged_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rooms: list[Any] = []

        def room_handler(request: httpx.Request) -> httpx.Response:
            rooms.append(request)
            return httpx.Response(201, json={"id": "room-x"})

        async def create_meeting(ctx: Any, arguments: dict[str, Any]) -> str:
            return json.dumps(
                {
                    "status": "created",
                    "room_url": "/rooms/r-1",
                    "title": "Quick sync",
                    "room_type": "EXTERNAL_BRIDGE",
                    "external_meeting_url": MEET_URL,
                }
            )

        tool = ChatTool(
            name="create_meeting",
            description="x",
            parameters={"type": "object", "properties": {}},
            handler=create_meeting,
        )
        await self._run(
            monkeypatch,
            [
                [
                    _completed(
                        _function_call("google_calendar_create_meet_event", "{}", call_id="c1")
                    )
                ],
                [_completed(_function_call("create_meeting", "{}", call_id="c2"))],
                [_text_delta("Đã tạo."), _completed(_message_item())],
            ],
            room_handler,
            extra_tools={"create_meeting": tool},
        )
        assert rooms == []
