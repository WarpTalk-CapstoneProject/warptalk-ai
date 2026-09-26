"""Helpers for AssistantService-backed MCP tool responses."""

from __future__ import annotations

import json
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


def mcp_tool_needs_confirmation(*, effect: str | None, policy: str | None = None) -> bool:
    """Whether AssistantService will ask before running this tool.

    WT-687: the user's per-tool ``policy`` decides when AssistantService sends one - a trusted
    write tool runs straight away, a read tool the user wants to see coming asks. A service older
    than the setting sends no policy, and then the old rule applies: writes ask.
    """
    if policy:
        return policy == "approval"
    return effect == "write"


def with_mcp_confirmation_parameter(
    parameters: dict[str, Any],
    *,
    effect: str | None,
    policy: str | None = None,
) -> dict[str, Any]:
    if not mcp_tool_needs_confirmation(effect=effect, policy=policy):
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
        properties.setdefault(
            "alwaysAllow",
            {
                "type": "boolean",
                "description": (
                    "True only when the user answered the confirmation card with Always allow."
                ),
            },
        )
    return updated


def split_mcp_tool_arguments(arguments: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    clean_arguments = dict(arguments)
    # Ours, not the provider's: AssistantService reads it off the request, and a server handed an
    # argument its schema never declared may refuse the call.
    clean_arguments.pop("alwaysAllow", None)
    token = clean_arguments.pop("confirmationToken", None)
    if isinstance(token, str) and token.strip():
        return clean_arguments, token.strip()
    return clean_arguments, None


def read_mcp_always_allow(arguments: dict[str, Any]) -> bool:
    """WT-687: the user chose Always allow on the card. Only a real ``true`` counts."""
    return arguments.get("alwaysAllow") is True


def parse_disabled_plugin_keys(disabled_plugin_keys_json: str) -> list[str]:
    """WT-687: the plugin keys switched off for this conversation, tolerating an absent field."""
    if not disabled_plugin_keys_json:
        return []
    try:
        raw = json.loads(disabled_plugin_keys_json)
    except ValueError:
        return []
    if not isinstance(raw, list):
        return []
    return [key.strip() for key in raw if isinstance(key, str) and key.strip()]


def normalize_mcp_tool_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"isSuccess": False, "error": "Plugin tool returned an invalid response."}

    result: dict[str, Any] = dict(payload)

    # "Always allow" is the one answer that changes something beyond this call: the card is gone
    # for this tool from now on. Nothing else on the way back says so, and a setting that changed
    # in silence is one the user finds out about the day WarpBot acts without asking.
    if result.get("appliedToolPolicy") == "allow":
        result["instruction"] = (
            "The user chose Always allow, so this tool now runs without a confirmation card. "
            "Tell them in one short clause that you will not ask again for this action, and "
            "that they can change it in the plugin's settings."
        )

    error_code = result.get("errorCode")
    if result.get("isSuccess") is not False or not isinstance(error_code, str):
        return result

    normalized = dict(result)
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

    A name two *different* plugins claim is dropped for both, not kept for whichever came first.
    The model calls a tool by name alone and the handler turns that name into the pluginKey it
    executes with, so "first wins" let a workspace Owner's private MCP server declare
    ``google_drive_search`` and receive the queries meant for Drive whenever it happened to be
    listed first. AssistantService now settles every such collision by trust before the list
    leaves it; this is the fail-closed backstop for a service that has not, since this side cannot
    tell which claimant is the trusted one. A plugin repeating its *own* name is harmless - both
    entries execute against the same plugin - and keeps its first entry as before.
    Names are compared case-folded throughout: two names that differ only in case are one name
    to the model reading the list.
    """
    if not isinstance(tools_payload, list):
        return [], []

    reserved_folded = {reserved.casefold() for reserved in reserved_names}

    # Pass one: every plugin claiming each name, before anything is accepted - the first claimant
    # must not be let through just because the second has not been seen yet.
    claimants: dict[str, set[str]] = {}
    for item in tools_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        plugin_key = str(item.get("pluginKey") or "").strip()
        if name and plugin_key:
            claimants.setdefault(name.casefold(), set()).add(plugin_key)

    accepted: list[dict[str, Any]] = []
    rejected: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    for item in tools_payload:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        plugin_key = str(item.get("pluginKey") or "").strip()
        folded = name.casefold()
        if not name or not plugin_key or folded in reserved_folded:
            continue
        if not MCP_TOOL_NAME_PATTERN.fullmatch(name):
            rejected.append(("mcp_tool_name_rejected", name))
            continue
        if len(claimants.get(folded, ())) > 1:
            rejected.append(("mcp_tool_name_ambiguous", name))
            continue
        if folded in seen_names:
            rejected.append(("mcp_tool_name_duplicate", name))
            continue
        if len(accepted) >= MCP_MAX_DYNAMIC_TOOLS:
            rejected.append(("mcp_tool_budget_exhausted", name))
            break
        seen_names.add(folded)
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


#: EVERY ask that stands between WarpBot and an action travels in this one shape.
#:
#: There were three: a confirmation card with per-tool rows invented from the arguments, a second
#: card for a plugin that needed connecting, and a third for a provider only an operator can
#: register. Three shapes for one question — "may I?" — and the rows described defaults nobody had
#: chosen, which reads as fact. A prompt now carries the action being asked about and the answers;
#: the detail of what happened belongs to the result, which the user reads afterwards.
def build_mcp_plugin_connection_action(payload: dict[str, Any]) -> dict[str, Any]:
    """The prompt for a plugin the user has not connected yet."""
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
        "permission": {
            "kind": "connect",
            "action": plugin_label,
            "pluginKey": plugin_key,
            "connectionStatus": connection_status,
            "connectedAccountEmail": user_action.get("connectedAccountEmail"),
            "message": user_action.get("message"),
        }
    }


def build_mcp_operator_setup_action(payload: dict[str, Any]) -> dict[str, Any]:
    """The prompt for a provider that no registration mechanism can reach.

    Its own kind rather than a variant of ``connect``: the two look alike and mean opposite things.
    One says "press Connect", this one says no button here will help, so it offers none.
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
        "permission": {
            "kind": "blocked",
            "action": plugin_label,
            "pluginKey": plugin_key,
            "message": user_action.get("message"),
        }
    }


def build_mcp_confirmation_questions(
    payload: dict[str, Any],
    *,
    tool_name: str,
    tool_label: str | None = None,
) -> dict[str, Any]:
    """The prompt for a write the user's policy wants confirmed.

    Every option's ``value`` is what comes back as the user's next message: the choice as a person
    would say it, then the machine line the model acts on (tool name and token). The web shows the
    first and hides the second, so the bubble reads "Yes" rather than a token.

    The prompt names the action and nothing else. It used to carry rows per tool - a title, a
    time, a calendar - built from the arguments, but those were the server's defaults as often as
    they were the user's words, and a row that says "Title: Google Meet meeting" reads as a
    decision somebody made. What was actually created is in the result, which the card under the
    answer shows afterwards.
    """
    token = str(payload.get("confirmationToken") or "").strip()
    action = (tool_label or "").strip() or tool_name

    return {
        "permission": {
            "kind": "tool",
            "action": action,
            "toolName": tool_name,
            "options": [
                {
                    "label": "Yes",
                    "value": (
                        f"Yes\n\nConfirm the {tool_name} plugin action. confirmationToken: {token}"
                    ),
                },
                {
                    # WT-687. Runs this call and stops asking for this tool. The token still has
                    # to validate before AssistantService records the choice.
                    "label": "Yes, and don't ask again for this tool",
                    "value": (
                        f"Always allow\n\nConfirm the {tool_name} plugin action and always "
                        f"allow it from now on. confirmationToken: {token} alwaysAllow: true"
                    ),
                },
                {
                    "label": "No, and tell WarpBot what to do differently",
                    "value": f"Cancel\n\nDo not run the {tool_name} plugin action.",
                },
            ],
        }
    }
