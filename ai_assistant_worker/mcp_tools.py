"""Helpers for AssistantService-backed MCP tool responses."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


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
            or "This app's provider needs credentials registered by an administrator before it can be connected.",
        }
    elif error_code == "confirmation_required":
        normalized["userAction"] = {
            "type": "confirm_write",
            "confirmationToken": payload.get("confirmationToken"),
            "message": "Ask the user to confirm before retrying this write action with the token.",
        }
    elif error_code == "permission_denied":
        normalized["userAction"] = {
            "type": "workspace_policy_blocked",
            "message": "Tell the user this workspace does not allow personal plugins in WarpBot.",
        }

    return normalized


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
