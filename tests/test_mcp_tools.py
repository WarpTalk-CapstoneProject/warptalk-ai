import json

from ai_assistant_worker.mcp_tools import (
    MCP_MAX_DYNAMIC_TOOLS,
    build_mcp_confirmation_questions,
    build_mcp_plugin_connection_action,
    normalize_mcp_tool_payload,
    redact_mcp_tool_payload_for_model,
    select_mcp_tool_entries,
    split_mcp_tool_arguments,
    with_mcp_confirmation_parameter,
)


def test_connection_required_with_plugin_metadata_maps_to_plugin_connection_action() -> None:
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "connection_required",
            "message": "Your Google Drive connection has expired.",
            "pluginKey": "google_workspace",
            "pluginLabel": "Google Drive",
            "connectionStatus": "expired",
            "connectedAccountEmail": "user@example.test",
        }
    )

    assert payload["userAction"] == {
        "type": "plugin_connection_required",
        "pluginKey": "google_workspace",
        "pluginLabel": "Google Drive",
        "connectionStatus": "expired",
        "connectedAccountEmail": "user@example.test",
        "message": "Your Google Drive connection has expired.",
    }


def test_connection_required_without_plugin_metadata_keeps_legacy_connect_action() -> None:
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "connection_required",
            "message": "Connect your provider account first.",
        }
    )

    assert payload["userAction"]["type"] == "connect_plugin"


def test_plugin_connection_action_payload_can_be_forwarded_to_clients() -> None:
    payload = build_mcp_plugin_connection_action(
        {
            "userAction": {
                "type": "plugin_connection_required",
                "pluginKey": "google_workspace",
                "pluginLabel": "Google Calendar",
                "connectionStatus": "not_connected",
                "connectedAccountEmail": None,
                "message": "Connect Google Calendar before WarpBot can use it.",
            }
        }
    )

    assert payload == {
        "pluginConnection": {
            "type": "plugin_connection_required",
            "pluginKey": "google_workspace",
            "pluginLabel": "Google Calendar",
            "connectionStatus": "not_connected",
            "connectedAccountEmail": None,
            "message": "Connect Google Calendar before WarpBot can use it.",
        }
    }


def test_confirmation_required_keeps_the_token_out_of_the_user_action() -> None:
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "confirmation_required",
            "message": "Confirm first.",
            "confirmationToken": "token-1",
        }
    )

    assert payload["userAction"]["type"] == "confirm_write"
    assert "confirmationToken" not in payload["userAction"]
    # Still on the envelope, because build_mcp_confirmation_questions reads it from there to put
    # it in the card. Only redact_mcp_tool_payload_for_model strips it, on the way to the model.
    assert payload["confirmationToken"] == "token-1"


def test_redaction_removes_the_confirmation_token_the_model_could_spend() -> None:
    """The write gate only works if the token cannot be read by the thing it gates.

    The agent loop does not stop for the confirmation card, and every write tool carries a
    ``confirmationToken`` parameter, so a token visible in the tool output is a token the model
    can hand straight back on the next iteration - confirming the write on the user's behalf.
    """
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "confirmation_required",
            "message": "Confirm first.",
            "confirmationToken": "token-1",
        }
    )

    redacted = redact_mcp_tool_payload_for_model(payload)

    assert "token-1" not in json.dumps(redacted)
    assert redacted["userAction"]["type"] == "confirm_write"
    assert redacted["errorCode"] == "confirmation_required"
    assert payload["confirmationToken"] == "token-1", "must not mutate the caller's payload"


def test_redaction_leaves_an_ordinary_success_payload_alone() -> None:
    payload = {"isSuccess": True, "result": {"eventId": "abc"}}

    assert redact_mcp_tool_payload_for_model(payload) == payload


def test_confirmation_question_carries_hidden_token_value() -> None:
    question_payload = build_mcp_confirmation_questions(
        {
            "message": "Confirm first.",
            "confirmationToken": "token-1",
        },
        tool_name="google_calendar_create_event",
    )

    question = question_payload["questions"][0]
    confirm = question["options"][0]
    assert question["header"] == "Confirm plugin action"
    assert confirm["label"] == "Confirm"
    assert "token-1" in confirm["value"]


def test_split_mcp_tool_arguments_removes_confirmation_token_from_provider_args() -> None:
    arguments, token = split_mcp_tool_arguments(
        {"summary": "Roadmap review", "confirmationToken": " token-1 "}
    )

    assert arguments == {"summary": "Roadmap review"}
    assert token == "token-1"


def test_write_tool_schema_gets_optional_confirmation_token_parameter() -> None:
    parameters = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    updated = with_mcp_confirmation_parameter(parameters, effect="write")

    assert "confirmationToken" in updated["properties"]
    assert updated["required"] == ["summary"]
    assert "confirmationToken" not in parameters["properties"]


def test_client_registration_unsupported_does_not_offer_a_connect_action() -> None:
    """A provider that supports no registration mechanism cannot be fixed by connecting.

    Offering the connect card here is the failure this error code exists to prevent: the user
    clicks Connect, the ladder exhausts again, and nothing in the loop says an operator has to
    register an app. The action must name that instead.
    """
    normalized = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "client_registration_unsupported",
            "pluginKey": "remote_app",
            "pluginLabel": "Remote App",
            "message": "This provider needs an administrator to register an OAuth app.",
        }
    )

    action = normalized["userAction"]
    assert action["type"] == "plugin_needs_operator_setup"
    assert action["pluginLabel"] == "Remote App"
    assert action["message"] == "This provider needs an administrator to register an OAuth app."

    # And it must not be mistaken for a connect prompt by the card builder.
    assert build_mcp_plugin_connection_action(normalized) == {}


def _entry(name: str, plugin_key: str = "notion") -> dict[str, object]:
    return {"name": name, "pluginKey": plugin_key, "description": "does a thing"}


def test_selector_drops_names_the_responses_api_would_reject() -> None:
    """One bad name must cost one tool, not the whole turn.

    The API rejects the entire request when any function name is malformed, so an MCP server
    calling its tool "notion.search" would take the built-in tools down with it.
    """
    accepted, rejected = select_mcp_tool_entries(
        [_entry("notion.search"), _entry("x" * 65), _entry("notion_search")],
        reserved_names=set(),
    )

    assert [item["name"] for item in accepted] == ["notion_search"]
    assert [reason for reason, _ in rejected] == [
        "mcp_tool_name_rejected",
        "mcp_tool_name_rejected",
    ]


def test_selector_keeps_only_the_first_of_two_identically_named_tools() -> None:
    accepted, rejected = select_mcp_tool_entries(
        [_entry("search", "notion"), _entry("search", "linear")],
        reserved_names=set(),
    )

    assert len(accepted) == 1
    assert accepted[0]["pluginKey"] == "notion"
    assert rejected == [("mcp_tool_name_duplicate", "search")]


def test_selector_never_shadows_a_built_in_tool() -> None:
    accepted, rejected = select_mcp_tool_entries(
        [_entry("create_meeting"), _entry("notion_search")],
        reserved_names={"create_meeting"},
    )

    assert [item["name"] for item in accepted] == ["notion_search"]
    assert rejected == []


def test_selector_caps_how_many_tools_one_turn_will_carry() -> None:
    accepted, rejected = select_mcp_tool_entries(
        [_entry(f"tool_{index}") for index in range(MCP_MAX_DYNAMIC_TOOLS + 5)],
        reserved_names=set(),
    )

    assert len(accepted) == MCP_MAX_DYNAMIC_TOOLS
    assert rejected == [("mcp_tool_budget_exhausted", f"tool_{MCP_MAX_DYNAMIC_TOOLS}")]


def test_selector_tolerates_a_catalog_that_is_not_a_list() -> None:
    assert select_mcp_tool_entries({"tools": []}, reserved_names=set()) == ([], [])
