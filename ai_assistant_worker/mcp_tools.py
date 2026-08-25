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
        normalized["userAction"] = {
            "type": "connect_plugin",
            "message": "Ask the user to connect this plugin from Personal Settings > Plugins.",
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
                            f"Confirm the {tool_name} plugin action. "
                            f"confirmationToken: {token}"
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
