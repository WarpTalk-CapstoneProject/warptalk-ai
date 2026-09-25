import json

from ai_assistant_worker.mcp_tools import (
    MCP_MAX_DYNAMIC_TOOLS,
    build_mcp_confirmation_questions,
    build_mcp_plugin_connection_action,
    normalize_mcp_tool_payload,
    parse_disabled_plugin_keys,
    read_mcp_always_allow,
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
    assert question["header"] == "Allow plugin action"
    assert confirm["label"] == "Allow"
    assert "token-1" in confirm["value"]


def test_confirmation_question_offers_always_allow_with_the_token_and_the_flag() -> None:
    question_payload = build_mcp_confirmation_questions(
        {"message": "Confirm first.", "confirmationToken": "token-1"},
        tool_name="linear_save_issue",
    )

    labels = [option["label"] for option in question_payload["questions"][0]["options"]]
    always = question_payload["questions"][0]["options"][1]
    assert labels == ["Allow", "Always allow", "Cancel"]
    assert "token-1" in always["value"]
    assert "alwaysAllow: true" in always["value"]


def test_policy_decides_whether_a_tool_asks_and_effect_is_the_fallback() -> None:
    parameters = {"type": "object", "properties": {}}

    trusted_write = with_mcp_confirmation_parameter(parameters, effect="write", policy="allow")
    watched_read = with_mcp_confirmation_parameter(parameters, effect="read", policy="approval")
    legacy_write = with_mcp_confirmation_parameter(parameters, effect="write", policy="")

    assert "confirmationToken" not in trusted_write["properties"]
    assert {"confirmationToken", "alwaysAllow"} <= set(watched_read["properties"])
    assert "confirmationToken" in legacy_write["properties"]


def test_always_allow_is_read_only_from_a_real_true_and_never_reaches_the_provider() -> None:
    raw = {"title": "Bug", "confirmationToken": "t", "alwaysAllow": True}

    arguments, token = split_mcp_tool_arguments(raw)

    assert read_mcp_always_allow(raw) is True
    assert read_mcp_always_allow({"alwaysAllow": "true"}) is False
    assert arguments == {"title": "Bug"}
    assert token == "t"


def test_disabled_plugin_keys_tolerate_an_absent_or_malformed_field() -> None:
    assert parse_disabled_plugin_keys("") == []
    assert parse_disabled_plugin_keys("not json") == []
    assert parse_disabled_plugin_keys('{"linear": true}') == []
    assert parse_disabled_plugin_keys('[" linear ", 3, ""]') == ["linear"]


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


def test_selector_drops_a_name_two_plugins_claim_for_both_of_them() -> None:
    """A name two plugins claim is ambiguous, and an ambiguous name is not offered at all.

    The handler binds the name to one pluginKey, so keeping "the first" meant whichever plugin was
    listed first received every call - a private MCP server declaring ``google_drive_search`` got
    Drive's queries whenever it came first. This side cannot tell the trusted claimant from the
    other, so it keeps neither.
    """
    accepted, rejected = select_mcp_tool_entries(
        [
            _entry("google_drive_search", "ws_crm_1a2b3c4d"),
            _entry("google_drive_search", "google_drive"),
            _entry("notion_search", "notion"),
        ],
        reserved_names=set(),
    )

    assert [item["name"] for item in accepted] == ["notion_search"]
    assert rejected == [
        ("mcp_tool_name_ambiguous", "google_drive_search"),
        ("mcp_tool_name_ambiguous", "google_drive_search"),
    ]


def test_selector_treats_names_differing_only_in_case_as_one_name() -> None:
    accepted, rejected = select_mcp_tool_entries(
        [_entry("google_drive_search", "google_drive"), _entry("Google_Drive_Search", "ws_crm")],
        reserved_names=set(),
    )

    assert accepted == []
    assert [reason for reason, _ in rejected] == ["mcp_tool_name_ambiguous"] * 2


def test_selector_keeps_the_first_when_one_plugin_repeats_its_own_name() -> None:
    # Both entries would execute against the same plugin, so nothing is ambiguous; only the
    # Responses API's objection to a repeated function name has to be avoided.
    accepted, rejected = select_mcp_tool_entries(
        [_entry("search", "notion"), _entry("search", "notion")],
        reserved_names=set(),
    )

    assert len(accepted) == 1
    assert accepted[0]["pluginKey"] == "notion"
    assert rejected == [("mcp_tool_name_duplicate", "search")]


def test_selector_never_shadows_a_built_in_tool() -> None:
    accepted, rejected = select_mcp_tool_entries(
        [_entry("create_meeting"), _entry("Create_Meeting", "ws_crm"), _entry("notion_search")],
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


def test_always_allow_tells_the_model_the_card_is_gone_for_this_tool() -> None:
    """The one answer that changes something beyond this call has to reach the user.

    Pressing Always allow turns the confirmation card off for this tool. Nothing else on the way
    back says so, so a user who pressed it once finds out the next time WarpBot acts without
    asking - which is exactly the moment it should not be a surprise.
    """
    payload = normalize_mcp_tool_payload(
        {
            "isSuccess": True,
            "appliedToolPolicy": "allow",
            "data": {"provider": "google_meet"},
        }
    )

    assert "Always allow" in payload["instruction"]
    assert "not ask again" in payload["instruction"]
    # The result itself is untouched.
    assert payload["data"] == {"provider": "google_meet"}


def test_an_ordinary_result_carries_no_instruction() -> None:
    payload = normalize_mcp_tool_payload({"isSuccess": True, "data": {"files": []}})

    assert "instruction" not in payload
