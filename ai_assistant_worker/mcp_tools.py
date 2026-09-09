"""Helpers for AssistantService-backed MCP tool responses."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

#: What the Responses API accepts as a function name. An MCP server is free to call its tool
#: "notion.search" or to run past 64 characters; we are not, and the API rejects the whole
#: request rather than the one bad entry - which takes the built-in tools down with it.
MCP_TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

#: Upper bounds on what one turn will carry from third-party catalogs, so that a misbehaving
#: or hostile server cannot crowd out the rest of the prompt.
MCP_MAX_DYNAMIC_TOOLS = 64
MCP_MAX_DESCRIPTION_CHARS = 1024


def with_mcp_confirmation_parameter(
    parameters: dict[str, Any],
    *,
    effect: str | None,
) -> dict[str, Any]:
    if effect != "write":
        return parameters

    updated = deepcopy(parameters)
    properties = updated.setdefault("properties", {})
    if isinstance(properties, dict):
        properties.setdefault(
            "confirmationToken",
            {
                "type": "string",
                "description": "Confirmation token from WarpBot's previous confirmation card.",
            },
        )
    return updated


def split_mcp_tool_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    clean_arguments = dict(arguments)
    token = clean_arguments.pop("confirmationToken", None)
    if isinstance(token, str) and token.strip():
        return clean_arguments, token.strip()
    return clean_arguments, None


def normalize_mcp_tool_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"isSuccess": False, "error": "Plugin tool returned an invalid response."}

    error_code = payload.get("errorCode")
    if payload.get("isSuccess") is not False or not isinstance(error_code, str):
        return payload

    normalized = dict(payload)
    if error_code == "connection_required":
        plugin_key = payload.get("pluginKey")
        plugin_label = payload.get("pluginLabel")
        connection_status = payload.get("connectionStatus")
        if (
            isinstance(plugin_key, str)
            and plugin_key.strip()
            and isinstance(plugin_label, str)
            and plugin_label.strip()
            and isinstance(connection_status, str)
            and connection_status.strip()
        ):
            normalized["userAction"] = {
                "type": "plugin_connection_required",
                "pluginKey": plugin_key.strip(),
                "pluginLabel": plugin_label.strip(),
                "connectionStatus": connection_status.strip(),
                "connectedAccountEmail": payload.get("connectedAccountEmail"),
                "message": payload.get("message")
                or "Connect this plugin before WarpBot can use it for this request.",
            }
        else:
            normalized["userAction"] = {
                "type": "connect_plugin",
                "message": "Ask the user to connect this plugin from Personal Settings > Plugins.",
            }
    elif error_code == "client_registration_unsupported":
        # Deliberately not a connect action. The plugin's authorization server supports neither
        # metadata-document clients nor dynamic registration, so no amount of clicking Connect
        # helps - an operator has to register an app and supply a client id. Offering the connect
        # button here would send the user in a loop, which is the exact failure this error code
        # exists to avoid.
        normalized["userAction"] = {
            "type": "plugin_needs_operator_setup",
            "pluginKey": payload.get("pluginKey"),
            "pluginLabel": payload.get("pluginLabel"),
            "message": payload.get("message")
            or (
                "This app's provider needs credentials registered by an administrator "
                "before it can be connected."
            ),
        }
    elif error_code == "confirmation_required":
        # No token in here, deliberately. It reaches the user through the confirmation card and
        # comes back as their next message; see redact_mcp_tool_payload_for_model.
        normalized["userAction"] = {
            "type": "confirm_write",
            "message": (
                "WarpBot has shown the user a confirmation card. Wait for their answer - "
                "do not retry this action yourself."
            ),
        }
    elif error_code == "permission_denied":
        normalized["userAction"] = {
            "type": "workspace_policy_blocked",
            "message": "Tell the user this workspace does not allow personal plugins in WarpBot.",
        }

    return normalized


def select_mcp_tool_entries(
    tools_payload: Any,
    *,
    reserved_names: set[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Pick the third-party tool entries that can safely be offered to the model this turn.

    Returns the accepted entries and a list of ``(reason, name)`` rejections for the caller to
    log. Everything here is defence against a catalog we do not control: the name, the label,
    the description and the schema are all whatever some MCP server chose to return.

    Dropping one entry is always better than failing the turn. An unusable function name, or two
    plugins that both expose ``search``, makes the Responses API reject the *entire* request -
    so the user would lose the built-in tools as well, with the discovery error swallowed and no
    hint as to why WarpBot suddenly went quiet.
    """
    if not isinstance(tools_payload, list):
        return [], []

    accepted: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    for item in tools_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        plugin_key = str(item.get("pluginKey") or "").strip()
        if not name or not plugin_key or name in reserved_names:
            continue
        if not MCP_TOOL_NAME_PATTERN.fullmatch(name):
            rejected.append(("mcp_tool_name_rejected", name))
            continue
        if name in seen_names:
            rejected.append(("mcp_tool_name_duplicate", name))
            continue
        if len(accepted) >= MCP_MAX_DYNAMIC_TOOLS:
            rejected.append(("mcp_tool_budget_exhausted", name))
            break
        seen_names.add(name)
        accepted.append(item)

    return accepted, rejected


def redact_mcp_tool_payload_for_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip the confirmation token from the copy the model gets to read.

    The token *is* the write gate, and it is meant to make exactly one trip through a human: out
    to the confirmation card, back as the user's next message. Leaving it in the tool output
    skips that trip. ``with_mcp_confirmation_parameter`` has already given every write tool a
    ``confirmationToken`` parameter, and the agent loop deliberately does not block on the card,
    so the very next iteration can spend the token and land the write while the card is still
    sitting unanswered on screen.
    """
    redacted = {key: value for key, value in payload.items() if key != "confirmationToken"}
    user_action = redacted.get("userAction")
    if isinstance(user_action, dict):
        redacted["userAction"] = {
            key: value for key, value in user_action.items() if key != "confirmationToken"
        }
    return redacted


def build_mcp_plugin_connection_action(payload: dict[str, Any]) -> dict[str, Any]:
    user_action = payload.get("userAction")
    if not isinstance(user_action, dict):
        return {}
    if user_action.get("type") != "plugin_connection_required":
        return {}

    plugin_key = str(user_action.get("pluginKey") or "").strip()
    plugin_label = str(user_action.get("pluginLabel") or "").strip()
    connection_status = str(user_action.get("connectionStatus") or "").strip()
    if not plugin_key or not plugin_label or not connection_status:
        return {}

    return {
        "pluginConnection": {
            "type": "plugin_connection_required",
            "pluginKey": plugin_key,
            "pluginLabel": plugin_label,
            "connectionStatus": connection_status,
            "connectedAccountEmail": user_action.get("connectedAccountEmail"),
            "message": user_action.get("message"),
        }
    }


def build_mcp_operator_setup_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Surface a provider that no registration mechanism can reach.

    Kept separate from ``build_mcp_plugin_connection_action`` on purpose: the two look similar but
    mean opposite things to a user. One says "press Connect", the other says "no button here will
    help". Merging them would put a Connect button on a flow that has already exhausted the
    registration ladder.
    """
    user_action = payload.get("userAction")
    if not isinstance(user_action, dict):
        return {}
    if user_action.get("type") != "plugin_needs_operator_setup":
        return {}

    plugin_key = str(user_action.get("pluginKey") or "").strip()
    plugin_label = str(user_action.get("pluginLabel") or "").strip()
    if not plugin_key or not plugin_label:
        return {}

    return {
        "pluginOperatorSetup": {
            "type": "plugin_needs_operator_setup",
            "pluginKey": plugin_key,
            "pluginLabel": plugin_label,
            "message": user_action.get("message"),
        }
    }


def build_mcp_confirmation_questions(
    payload: dict[str, Any],
    *,
    tool_name: str,
) -> dict[str, Any]:
    token = str(payload.get("confirmationToken") or "").strip()
    message = str(
        payload.get("message")
        or "WarpBot wants to change data in a connected app. Confirm before it continues."
    )

    return {
        "questions": [
            {
                "header": "Confirm plugin action",
                "question": message,
                "options": [
                    {
                        "label": "Confirm",
                        "description": "Run this write action once.",
                        "value": (
                            f"Confirm the {tool_name} plugin action. confirmationToken: {token}"
                        ),
                    },
                    {
                        "label": "Cancel",
                        "description": "Do not run this action.",
                        "value": f"Do not run the {tool_name} plugin action.",
                    },
                ],
            }
        ]
    }
