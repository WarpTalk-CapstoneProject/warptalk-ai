from ai_assistant_worker.mcp_tools import (
    build_mcp_confirmation_questions,
    normalize_mcp_tool_payload,
    split_mcp_tool_arguments,
    with_mcp_confirmation_parameter,
)


def test_connection_required_maps_to_connect_action() -> None:
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": False,
            "errorCode": "connection_required",
            "message": "Connect your provider account first.",
        }
    )

    assert payload["userAction"]["type"] == "connect_plugin"


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
