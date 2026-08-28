"""Tests for the chat assistant's situational prompt templates and context formatters.

The bug these exist for: the assistant answered "what did we just talk about?" out of the
chat history instead of reading the transcript. The pipeline was fine — the prompt named only
"members, terminology, recent meetings" as worth looking up, and never said that the entity id
it was handed on a meeting page is get_transcript's `meeting_id`. Both of those are prompt
content, so they are asserted as prompt content here.

The second half covers _format_mentions, which had no tests and silently dropped every
in-meeting @mention because the two publishers disagree on the mention shape.
"""

from __future__ import annotations

import json

from ai_assistant_worker.chat_templates import (
    DOCUMENT,
    GENERAL,
    MEETING,
    MEETING_CHAT,
    PERSONA,
    TEMPLATES,
    build_system_prompt,
    resolve_template,
)
from ai_assistant_worker.chat_worker import (
    SYSTEM_PROMPT,
    _format_mentions,
    _format_page_context,
    _page_type,
)


class TestResolveTemplate:
    def test_meeting_chat_origin_wins_over_page_context(self) -> None:
        """A @WarpBot turn is about the meeting whatever the browser last registered."""
        assert resolve_template(origin="meeting_chat", page_type="documents") is MEETING_CHAT

    def test_page_type_routes_when_origin_is_the_global_widget(self) -> None:
        assert resolve_template(origin="assistant", page_type="room_detail") is MEETING
        assert resolve_template(origin="assistant", page_type="in_meeting") is MEETING
        assert resolve_template(origin="assistant", page_type="document_detail") is DOCUMENT

    def test_unknown_and_missing_values_fall_back_to_general(self) -> None:
        """A pageType shipping in the web app before this module learns it must not raise."""
        assert resolve_template() is GENERAL
        assert resolve_template(origin="", page_type="") is GENERAL
        assert resolve_template(origin="assistant", page_type="some_new_page") is GENERAL

    def test_routing_is_case_and_whitespace_insensitive(self) -> None:
        assert resolve_template(origin=" Meeting_Chat ") is MEETING_CHAT
        assert resolve_template(page_type=" Room_Detail ") is MEETING


class TestBuildSystemPrompt:
    def test_every_template_embeds_the_shared_persona(self) -> None:
        for template in TEMPLATES.values():
            assert PERSONA in build_system_prompt(template)

    def test_system_prompt_export_is_the_persona(self) -> None:
        """chat_worker.SYSTEM_PROMPT stayed a real thing, not a stale leftover."""
        assert SYSTEM_PROMPT == PERSONA

    def test_meeting_templates_bind_the_entity_id_to_the_transcript_argument(self) -> None:
        """The line that turns transcript retrieval on: the bare uuid IS the meeting_id."""
        for template in (MEETING_CHAT, MEETING):
            prompt = build_system_prompt(template)
            assert "entity_id" in prompt
            assert "meeting_id of get_transcript" in prompt

    def test_document_template_binds_the_entity_id_to_the_document_argument(self) -> None:
        prompt = build_system_prompt(DOCUMENT)
        assert "document_id of get_document" in prompt

    def test_templates_without_an_entity_omit_the_binding_section(self) -> None:
        assert "THE ID YOU WERE GIVEN" not in build_system_prompt(GENERAL)

    def test_transcript_is_named_as_a_source_wherever_a_meeting_is_in_scope(self) -> None:
        """The original prompt never mentioned the transcript at all. That was the bug."""
        for template in (GENERAL, MEETING_CHAT, MEETING):
            assert "get_transcript" in build_system_prompt(template)

    def test_all_three_context_kinds_are_reachable_from_the_general_prompt(self) -> None:
        prompt = build_system_prompt(GENERAL)
        for tool in ("get_transcript", "get_document", "search_documents", "search_terminology"):
            assert tool in prompt

    def test_source_order_follows_the_template(self) -> None:
        """Preference order is data, so it has to survive into the generated text."""
        prompt = build_system_prompt(MEETING_CHAT)
        assert prompt.index("get_transcript") < prompt.index("semantic_search")

    def test_caveats_are_stated_so_an_empty_result_is_not_read_as_an_answer(self) -> None:
        assert "Limit:" in build_system_prompt(GENERAL)

    def test_meeting_chat_carries_its_style_and_others_do_not(self) -> None:
        assert "STYLE" in build_system_prompt(MEETING_CHAT)
        assert "STYLE" not in build_system_prompt(GENERAL)


class TestPageContext:
    def test_page_type_is_read_from_the_payload(self) -> None:
        assert _page_type(json.dumps({"pageType": "room_detail"})) == "room_detail"

    def test_page_type_tolerates_junk(self) -> None:
        for payload in ("", "not json", "[]", "{}", json.dumps({"pageType": ""})):
            assert _page_type(payload) is None

    def test_entity_and_snapshot_are_rendered(self) -> None:
        message = _format_page_context(
            json.dumps(
                {
                    "pageType": "in_meeting",
                    "entityId": "room-9",
                    "workspaceId": "ws-1",
                    "snapshot": {"title": "Sprint sync"},
                }
            )
        )
        assert message is not None
        assert "in_meeting" in message
        assert "entity_id=room-9" in message
        assert "Sprint sync" in message

    def test_malformed_context_is_ignored_rather_than_fatal(self) -> None:
        assert _format_page_context("") is None
        assert _format_page_context("{oops") is None
        assert _format_page_context(json.dumps(["nope"])) is None


class TestFormatMentions:
    def test_assistant_service_shape_is_read(self) -> None:
        message = _format_mentions(
            json.dumps([{"entityType": "document", "entityId": "doc-1", "label": "Spec"}])
        )
        assert message is not None
        assert 'document "Spec" (id=doc-1)' in message
        assert "get_document" in message

    def test_meeting_chat_shape_is_read_too(self) -> None:
        """ChatMentionDto is {Id, Display, Type}; reading only entityType dropped it whole."""
        message = _format_mentions(
            json.dumps([{"id": "room-9", "display": "Sprint sync", "type": "room"}])
        )
        assert message is not None
        assert 'room "Sprint sync" (id=room-9)' in message
        assert "get_transcript" in message

    def test_warpbot_itself_is_not_rendered_as_a_thing_to_look_up(self) -> None:
        """How the user summoned the assistant, not a reference — and no tool can fetch it."""
        assert (
            _format_mentions(
                json.dumps([{"id": "bot-warpbot", "display": "WarpBot", "type": "agent"}])
            )
            is None
        )

    def test_a_real_reference_survives_alongside_the_agent_mention(self) -> None:
        message = _format_mentions(
            json.dumps(
                [
                    {"id": "bot-warpbot", "display": "WarpBot", "type": "agent"},
                    {"id": "doc-1", "display": "Spec", "type": "document"},
                ]
            )
        )
        assert message is not None
        assert "WarpBot" not in message
        assert "doc-1" in message

    def test_mention_type_is_normalized_before_matching_a_tool_hint(self) -> None:
        message = _format_mentions(json.dumps([{"id": "doc-1", "type": " Document "}]))
        assert message is not None
        assert "get_document" in message

    def test_label_falls_back_to_the_id(self) -> None:
        message = _format_mentions(json.dumps([{"entityType": "member", "entityId": "user-3"}]))
        assert message is not None
        assert '"user-3"' in message

    def test_junk_is_ignored(self) -> None:
        for payload in ("", "not json", "{}", "[]", json.dumps(["string"]), json.dumps([{}])):
            assert _format_mentions(payload) is None

    def test_plugin_mention_points_at_its_own_tools_not_a_lookup(self) -> None:
        """WT-565: @Google Drive isn't a record to fetch - it's the user picking a capability."""
        message = _format_mentions(
            json.dumps(
                [{"entityType": "plugin", "entityId": "google_workspace:drive", "label": "Google Drive"}]
            )
        )
        assert message is not None
        assert 'plugin "Google Drive" (id=google_workspace:drive)' in message
        assert "prefer its tools" in message
        assert "look it up" not in message


class TestWebSearchIsTheLastResort:
    """WT — "nếu không có trong glossary thì tự search web".

    `web_search` was already in the tool schema OpenAI was handed, but it was named in no
    template, so the "WHERE TO GET CONTEXT" list never mentioned it and the ground rules read as
    a flat ban on outside knowledge. Production, in a live meeting: "@WarpBot c# là gì" →
    "Mình không tìm thấy 'C#' trong transcript cuộc họp hoặc glossary của workspace", followed by
    a definition it had not looked up. Both halves wrong — it should have searched.
    """

    def test_every_situation_can_reach_the_web(self) -> None:
        # The SOURCE LINE, not the bare word: "web_search" also appears in the ground rule
        # below, so asserting on the word alone passes even when the tool has no place in the
        # priority list — which is exactly the state this whole change fixes.
        for template in TEMPLATES.values():
            prompt = build_system_prompt(template, web_search_enabled=True)
            assert "- web_search: use when" in prompt, template.key

    def test_the_web_is_listed_after_every_workspace_source(self) -> None:
        """Position IS the priority — the list is rendered in order and called a preference."""
        for template in TEMPLATES.values():
            prompt = build_system_prompt(template, web_search_enabled=True)
            web_at = prompt.index("- web_search:")
            for source in template.sources:
                assert prompt.index(f"- {source.tool}:") < web_at, (template.key, source.tool)

    def test_the_glossary_still_outranks_the_web(self) -> None:
        prompt = build_system_prompt(MEETING_CHAT, web_search_enabled=True)
        assert "WORKSPACE FIRST, WEB LAST" in prompt
        assert "glossary" in prompt

    def test_a_disabled_web_search_is_not_offered(self) -> None:
        """A prompt that names a tool the model was not given plans a step it cannot take."""
        prompt = build_system_prompt(MEETING_CHAT, web_search_enabled=False)
        assert "web_search" not in prompt
        assert "WORKSPACE FIRST, WEB LAST" not in prompt

    def test_the_workspace_sources_are_unaffected_by_the_switch(self) -> None:
        on = build_system_prompt(MEETING_CHAT, web_search_enabled=True)
        off = build_system_prompt(MEETING_CHAT, web_search_enabled=False)
        for source in MEETING_CHAT.sources:
            assert f"- {source.tool}:" in on
            assert f"- {source.tool}:" in off

    def test_both_surfaces_get_the_same_rule(self) -> None:
        """The widget and the in-meeting chat run one agent; the escalation cannot differ."""
        widget = build_system_prompt(resolve_template(page_type="general"), web_search_enabled=True)
        in_meeting = build_system_prompt(
            resolve_template(origin="meeting_chat"), web_search_enabled=True
        )
        assert "WORKSPACE FIRST, WEB LAST" in widget
        assert "WORKSPACE FIRST, WEB LAST" in in_meeting
