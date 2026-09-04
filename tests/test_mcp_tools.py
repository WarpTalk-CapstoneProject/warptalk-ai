from ai_assistant_worker.mcp_tools import (
    build_mcp_confirmation_questions,
    build_mcp_plugin_connection_action,
    normalize_mcp_tool_payload,
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


def test_confirmation_required_keeps_confirmation_token() -> None:
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "confirmation_required",
            "message": "Confirm first.",
            "confirmationToken": "token-1",
        }
    )

    assert payload["userAction"]["type"] == "confirm_write"
    assert payload["userAction"]["confirmationToken"] == "token-1"


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
