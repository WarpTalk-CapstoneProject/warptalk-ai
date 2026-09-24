"""WarpBot acting, not only answering — and handing a meeting thread to the widget.

WHAT WAS REPORTED
    In the widget, somebody dictated "Action: Hỏi người phụ trách để xác nhận thông tin cần thiết —
    Người thực hiện: tôi — Deadline: chưa xác định". WarpBot replied "Đã ghi nhận" with a bullet
    list, and asked "đang lưu ở đâu", admitted it was saved nowhere. It had no tool that could
    save it. Separately: in a meeting, "chuyển qua widget để bàn tiếp" had nothing to call.

WHAT THESE PIN
    - create_action_item turns "owner: tôi / deadline: none" into the request the service needs,
      uses the meeting on screen, never invents a deadline, and returns a link.
    - Every write tool reports failure as failure — the model is told nothing was saved.
    - The handoff exists only on the meeting-chat surface, and its UI event is published.
    - The prompt says, in so many words, that prose is not an action.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_templates import (
    GENERAL,
    MEETING,
    MEETING_CHAT,
    build_system_prompt,
)
from ai_assistant_worker.chat_tools import (
    CONTINUE_IN_WIDGET_TOOL,
    TOOLS,
    TOOLS_BY_NAME,
    ToolContext,
    _add_glossary_term,
    _continue_in_widget,
    _create_action_item,
    _share_meeting_minutes,
    offered_on,
)
from ai_assistant_worker.chat_worker import ChatAssistantWorker
from ai_assistant_worker.tool_targets import describe_tool_target
from shared.config import ChatAssistantSettings

ROOM_ID = "0192f0d1-7a3b-7c4d-8e9f-0a1b2c3d4e5f"
USER_ID = "11111111-2222-3333-4444-555555555555"


def _response(status_code: int, payload: Any = None, text: str = "") -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if payload is None:
        response.json.side_effect = ValueError("no body")
    else:
        response.json.return_value = payload
    response.text = text
    return response


def _ctx(
    *,
    page_type: str | None = "in_meeting",
    page_entity_id: str | None = ROOM_ID,
    origin: str = "assistant",
    translation_room_client: Any = None,
    transcript_client: Any = None,
    workspace_client: Any = None,
) -> ToolContext:
    if workspace_client is None:
        workspace_client = AsyncMock()
        workspace_client.get.return_value = _response(200, {"id": "ws-1", "slug": "acme"})
    return ToolContext(
        workspace_id="ws-1",
        user_id=USER_ID,
        bearer_token="Bearer t",
        workspace_client=workspace_client,
        transcript_client=transcript_client or AsyncMock(),
        translation_room_client=translation_room_client or AsyncMock(),
        openai_client=None,
        model="gpt-5.6-luna",
        redis=MagicMock(),
        origin=origin,
        page_type=page_type,
        page_entity_id=page_entity_id,
    )


def _created_item(**overrides: Any) -> dict[str, Any]:
    item = {
        "id": "item-1",
        "translationRoomId": ROOM_ID,
        "roomTitle": "Weekly sync",
        "sourceMinutesId": None,
        "task": "Hỏi người phụ trách để xác nhận thông tin cần thiết",
        "ownerName": "Nhi",
        "assigneeUserId": USER_ID,
        "status": "OPEN",
        "dueDate": None,
        "source": "ASSISTANT",
    }
    item.update(overrides)
    return item


class TestCreateActionItem:
    async def test_the_dictated_item_from_the_report_is_saved_to_the_meeting_on_screen(
        self,
    ) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(201, _created_item())
        ctx = _ctx(translation_room_client=rooms)

        # Exactly what the model fills for the reported message: every property present, the
        # deadline and meeting as blanks.
        result = json.loads(
            await _create_action_item(
                ctx,
                {
                    "task": "Hỏi người phụ trách để xác nhận thông tin cần thiết",
                    "owner": "ME",
                    "owner_name": "",
                    "due_date": "",
                    "meeting_id": "",
                },
            )
        )

        rooms.post.assert_awaited_once()
        url = rooms.post.await_args.args[0]
        body = rooms.post.await_args.kwargs["json"]
        assert url == f"/api/v1/rooms/{ROOM_ID}/action-items"
        assert body == {
            "task": "Hỏi người phụ trách để xác nhận thông tin cần thiết",
            "assignToSelf": True,
            "ownerName": None,
            # "Deadline: chưa xác định" is no deadline, never today.
            "dueDate": None,
        }
        assert rooms.post.await_args.kwargs["headers"] == {"Authorization": "Bearer t"}

        assert result["status"] == "created"
        assert result["assigned_to_you"] is True
        assert "tasks_link" not in result, "the workspace task page is gone; link the meeting"
        assert result["meeting_link"] == f"/acme/rooms/{ROOM_ID}"

    async def test_a_named_owner_and_a_deadline_travel_as_said(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(
            201, _created_item(ownerName="chị Nhi", assigneeUserId="someone-else")
        )
        ctx = _ctx(translation_room_client=rooms)

        result = json.loads(
            await _create_action_item(
                ctx,
                {
                    "task": "Draft the release note",
                    "owner": "NAMED",
                    "owner_name": "chị Nhi",
                    "due_date": "2026-10-01",
                    "meeting_id": "",
                },
            )
        )

        body = rooms.post.await_args.kwargs["json"]
        assert body["assignToSelf"] is False
        assert body["ownerName"] == "chị Nhi"
        assert body["dueDate"] == "2026-10-01"
        assert result["assigned_to_you"] is False
        assert result["assigned"] is True

    async def test_named_with_no_name_is_the_filler_case_and_means_nobody(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(201, _created_item(assigneeUserId=None))
        await _create_action_item(
            _ctx(translation_room_client=rooms),
            {"task": "Decide the vendor", "owner": "NAMED", "owner_name": "", "due_date": ""},
        )

        body = rooms.post.await_args.kwargs["json"]
        assert body["assignToSelf"] is False
        assert body["ownerName"] is None

    async def test_an_unmatched_name_is_reported_so_the_model_can_ask(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(
            400, {"error": "Nobody in this meeting matches that name exactly once."}
        )

        result = json.loads(
            await _create_action_item(
                _ctx(translation_room_client=rooms),
                {"task": "Book the venue", "owner": "NAMED", "owner_name": "Kỳ"},
            )
        )

        assert result["status"] == "owner_not_found"
        assert "Nothing was saved" in result["instruction"]

    async def test_without_a_meeting_nothing_is_posted_and_the_model_is_told_to_find_one(
        self,
    ) -> None:
        rooms = AsyncMock()

        result = json.loads(
            await _create_action_item(
                _ctx(page_type="documents", page_entity_id=None, translation_room_client=rooms),
                {"task": "Send the deck", "owner": "ME", "meeting_id": ""},
            )
        )

        rooms.post.assert_not_awaited()
        assert result["status"] == "needs_more_information"
        assert result["missing"] == ["meeting"]

    async def test_an_explicit_meeting_id_wins_over_the_page(self) -> None:
        other = "0192f0d1-0000-7c4d-8e9f-0a1b2c3d4e5f"
        rooms = AsyncMock()
        rooms.post.return_value = _response(201, _created_item(translationRoomId=other))

        await _create_action_item(
            _ctx(translation_room_client=rooms),
            {"task": "x", "owner": "ME", "meeting_id": other},
        )

        assert rooms.post.await_args.args[0] == f"/api/v1/rooms/{other}/action-items"

    async def test_a_malformed_deadline_is_refused_before_the_network(self) -> None:
        rooms = AsyncMock()

        result = json.loads(
            await _create_action_item(
                _ctx(translation_room_client=rooms),
                {"task": "x", "owner": "ME", "due_date": "next Friday"},
            )
        )

        rooms.post.assert_not_awaited()
        assert result["status"] == "invalid"

    async def test_a_service_failure_is_a_failure_not_a_confirmation(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(500, {"error": "boom"})

        result = json.loads(
            await _create_action_item(
                _ctx(translation_room_client=rooms), {"task": "x", "owner": "ME"}
            )
        )

        assert result["status"] == "failed"

    async def test_no_slug_means_no_link_rather_than_a_broken_one(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(201, _created_item())
        workspaces = AsyncMock()
        workspaces.get.return_value = _response(403, {"error": "no"})

        result = json.loads(
            await _create_action_item(
                _ctx(translation_room_client=rooms, workspace_client=workspaces),
                {"task": "x", "owner": "ME"},
            )
        )

        assert result["status"] == "created"
        assert "meeting_link" not in result


class TestAddGlossaryTerm:
    @staticmethod
    def _transcript(glossaries: list[dict[str, Any]], post: MagicMock | None = None) -> AsyncMock:
        client = AsyncMock()
        client.get.return_value = _response(200, glossaries)
        client.post.return_value = post or _response(201)
        return client

    async def test_the_only_glossary_is_used(self) -> None:
        transcript = self._transcript(
            [{"id": "g1", "name": "Product", "isActive": True, "sourceLanguage": "vi"}]
        )

        result = json.loads(
            await _add_glossary_term(
                _ctx(transcript_client=transcript),
                {
                    "source_term": "WarpBot",
                    "target_term": "WarpBot",
                    "context": "",
                    "glossary_name": "",
                },
            )
        )

        assert transcript.post.await_args.args[0] == "/api/v1/glossaries/g1/terms"
        assert transcript.post.await_args.kwargs["json"]["sourceTerm"] == "WarpBot"
        assert transcript.post.await_args.kwargs["json"]["context"] is None
        assert result["status"] == "created"
        assert result["glossary_link"] == "/acme/glossary"

    async def test_several_glossaries_and_no_name_asks_rather_than_guesses(self) -> None:
        transcript = self._transcript(
            [{"id": "g1", "name": "Product"}, {"id": "g2", "name": "Legal"}]
        )

        result = json.loads(
            await _add_glossary_term(
                _ctx(transcript_client=transcript),
                {"source_term": "SLA", "target_term": "cam kết dịch vụ", "glossary_name": ""},
            )
        )

        transcript.post.assert_not_awaited()
        assert result["status"] == "choose_glossary"
        assert [g["name"] for g in result["glossaries"]] == ["Product", "Legal"]

    async def test_a_named_glossary_is_picked_by_name(self) -> None:
        transcript = self._transcript(
            [{"id": "g1", "name": "Product"}, {"id": "g2", "name": "Legal"}]
        )

        await _add_glossary_term(
            _ctx(transcript_client=transcript),
            {"source_term": "SLA", "target_term": "cam kết dịch vụ", "glossary_name": "legal"},
        )

        assert transcript.post.await_args.args[0] == "/api/v1/glossaries/g2/terms"

    async def test_a_duplicate_is_reported_as_already_there(self) -> None:
        transcript = self._transcript(
            [{"id": "g1", "name": "Product"}], post=_response(409, "Term already exists.")
        )

        result = json.loads(
            await _add_glossary_term(
                _ctx(transcript_client=transcript), {"source_term": "a", "target_term": "b"}
            )
        )

        assert result["status"] == "already_exists"

    async def test_no_glossary_saves_nothing(self) -> None:
        transcript = self._transcript([])

        result = json.loads(
            await _add_glossary_term(
                _ctx(transcript_client=transcript), {"source_term": "a", "target_term": "b"}
            )
        )

        transcript.post.assert_not_awaited()
        assert result["status"] == "no_glossary"


class TestShareMeetingMinutes:
    async def test_grants_access_and_hands_back_the_link_to_send(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(200, {"url": "https://app/minutes/s/tok", "people": []})

        result = json.loads(
            await _share_meeting_minutes(
                _ctx(translation_room_client=rooms), {"email": "an@example.com", "meeting_id": ""}
            )
        )

        assert rooms.post.await_args.args[0] == f"/api/v1/rooms/{ROOM_ID}/minutes/share/people"
        assert rooms.post.await_args.kwargs["json"] == {"email": "an@example.com"}
        assert result["status"] == "shared"
        assert result["share_url"] == "https://app/minutes/s/tok"
        assert "does not email" in result["instruction"]

    async def test_a_refusal_is_passed_through_in_the_services_words(self) -> None:
        rooms = AsyncMock()
        rooms.post.return_value = _response(403, {"error": "Only the host can share the minutes."})

        result = json.loads(
            await _share_meeting_minutes(
                _ctx(translation_room_client=rooms), {"email": "an@example.com"}
            )
        )

        assert result["status"] == "failed"
        assert result["reason"] == "Only the host can share the minutes."

    async def test_the_address_is_not_shown_in_the_trail(self) -> None:
        # In a meeting, the trail is read by the whole room.
        assert describe_tool_target("share_meeting_minutes", {"email": "an@example.com"}) == ""


class TestContinueInWidget:
    async def test_only_the_meeting_chat_offers_it(self) -> None:
        assert offered_on(CONTINUE_IN_WIDGET_TOOL, "meeting_chat")
        assert not offered_on(CONTINUE_IN_WIDGET_TOOL, "assistant")
        assert not offered_on(CONTINUE_IN_WIDGET_TOOL, None)
        assert all(
            offered_on(tool.name, "assistant")
            for tool in TOOLS
            if tool.name != CONTINUE_IN_WIDGET_TOOL
        )

    async def test_from_the_meeting_chat_it_agrees_to_hand_off(self) -> None:
        result = json.loads(await _continue_in_widget(_ctx(origin="meeting_chat"), {"topic": ""}))
        assert result["status"] == "handoff_requested"

    async def test_from_the_widget_there_is_nowhere_to_move(self) -> None:
        result = json.loads(await _continue_in_widget(_ctx(origin="assistant"), {}))
        assert result["status"] == "not_applicable"


class _FakeStream:
    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self):
        async def gen():
            for event in self._events:
                yield event

        return gen()


def _worker(turns: list[list[Any]]) -> tuple[Any, list[dict[str, Any]]]:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = ChatAssistantSettings(model="gpt-5.6-luna", max_tokens=256)
    worker.logger = MagicMock()
    worker._openai = MagicMock()
    worker._openai.responses.create = AsyncMock(side_effect=[_FakeStream(t) for t in turns])
    published: list[dict[str, Any]] = []

    async def publish(request: Any, **kwargs: Any) -> None:
        published.append(kwargs)

    worker._publish_result = publish
    return worker, published


def _request(origin: str) -> Any:
    return SimpleNamespace(
        request_id="req-1",
        conversation_id="room-1",
        workspace_id="ws-1",
        bearer_token="Bearer t",
        origin=origin,
        page_context_json=json.dumps({"pageType": "meeting_chat", "entityId": ROOM_ID}),
        mentions_json="",
        disabled_plugin_keys_json="",
        images_json="",
    )


def _call(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id="c1")


def _completed(*items: Any) -> SimpleNamespace:
    return SimpleNamespace(type="response.completed", response=SimpleNamespace(output=list(items)))


class TestHandoffThroughTheLoop:
    async def test_a_meeting_chat_handoff_publishes_the_ui_event(self) -> None:
        worker, published = _worker(
            [
                [_completed(_call(CONTINUE_IN_WIDGET_TOOL, '{"topic": "vendor risks"}'))],
                [
                    SimpleNamespace(type="response.output_text.delta", delta="Mở widget rồi nhé."),
                    _completed(SimpleNamespace(type="message")),
                ],
            ]
        )
        ctx = MagicMock()
        ctx.origin = "meeting_chat"

        text, _ = await worker._run_agent_loop(_request("meeting_chat"), [], ctx)

        handoffs = [p for p in published if p["type_"] == "handoff"]
        assert len(handoffs) == 1
        assert json.loads(handoffs[0]["tool_calls_json"]) == {
            "target": "widget",
            "topic": "vendor risks",
        }
        assert text == "Mở widget rồi nhé."

        offered = [
            tool["name"]
            for tool in worker._openai.responses.create.await_args_list[0].kwargs["tools"]
            if tool.get("type") == "function"
        ]
        assert CONTINUE_IN_WIDGET_TOOL in offered

    async def test_the_widget_is_never_offered_the_handoff(self) -> None:
        worker, published = _worker(
            [[SimpleNamespace(type="response.output_text.delta", delta="ok"), _completed()]]
        )
        ctx = MagicMock()
        ctx.origin = "assistant"

        await worker._run_agent_loop(_request("assistant"), [], ctx)

        offered = [
            tool["name"]
            for tool in worker._openai.responses.create.await_args_list[0].kwargs["tools"]
            if tool.get("type") == "function"
        ]
        assert CONTINUE_IN_WIDGET_TOOL not in offered
        assert "create_action_item" in offered
        assert not any(p["type_"] == "handoff" for p in published)

    async def test_a_hallucinated_handoff_from_the_widget_opens_nothing(self) -> None:
        worker, published = _worker(
            [
                [_completed(_call(CONTINUE_IN_WIDGET_TOOL, "{}"))],
                [_completed(SimpleNamespace(type="message"))],
            ]
        )
        ctx = MagicMock()
        ctx.origin = "assistant"

        await worker._run_agent_loop(_request("assistant"), [], ctx)

        assert not any(p["type_"] == "handoff" for p in published)


class TestPromptSaysProseIsNotAnAction:
    def test_every_template_names_the_write_tools_and_the_honesty_rule(self) -> None:
        for template in (GENERAL, MEETING, MEETING_CHAT):
            prompt = build_system_prompt(template)
            assert "create_action_item" in prompt
            assert "NEVER say something is saved" in prompt
            assert "add_glossary_term" in prompt

    def test_only_the_meeting_chat_is_told_about_the_handoff(self) -> None:
        assert CONTINUE_IN_WIDGET_TOOL in build_system_prompt(MEETING_CHAT)
        assert CONTINUE_IN_WIDGET_TOOL not in build_system_prompt(GENERAL)
        assert CONTINUE_IN_WIDGET_TOOL not in build_system_prompt(MEETING)

    def test_every_optional_enum_on_a_write_tool_has_a_way_to_say_none(self) -> None:
        # WT-399: the model fills every property, so an enum with no "nobody" forces a guess.
        owner = TOOLS_BY_NAME["create_action_item"].parameters["properties"]["owner"]
        assert "NONE" in owner["enum"]
