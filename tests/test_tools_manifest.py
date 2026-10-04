"""The built-in tool manifest the chat worker publishes for AssistantService.

The web's /{slug}/tools page lists what this manifest says, so a tool missing from it is a tool
the page does not show, and a wrong `effect` is a write tool presented as harmless.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_assistant_worker import tools_manifest
from ai_assistant_worker.chat_tools import (
    TOOL_AUDIENCES,
    TOOL_CATEGORIES,
    TOOL_EFFECTS,
    TOOLS,
    ChatTool,
)
from ai_assistant_worker.tools_manifest import (
    MANIFEST_KEY,
    REFRESH_INTERVAL_SECONDS,
    TTL_SECONDS,
    build_manifest,
    publish_manifest,
    run_manifest_publisher,
    web_search_available,
)
from shared.config import ChatAssistantSettings

#: The web catalog's category ids (warpbot-tools-catalog.ts), which the manifest must use.
WEB_CATEGORY_IDS = {
    "meetings",
    "knowledge",
    "documents",
    "glossary",
    "translation",
    "workspace",
    "conversation",
    "platform",
}


def _settings(
    *, api_key: str = "sk-test", web_search_enabled: bool = True
) -> ChatAssistantSettings:
    return ChatAssistantSettings(api_key=api_key, web_search_enabled=web_search_enabled)


def test_manifest_covers_every_registered_tool() -> None:
    manifest = build_manifest(_settings())

    assert [tool["name"] for tool in manifest["tools"]] == [
        tool.name for tool in TOOLS if tool.listed
    ]


def test_unlisted_tools_are_left_out() -> None:
    names = {tool["name"] for tool in build_manifest(_settings())["tools"]}

    assert "continue_in_widget" not in names
    assert [tool.name for tool in TOOLS if not tool.listed] == ["continue_in_widget"]


def test_every_tool_declares_valid_metadata() -> None:
    for tool in TOOLS:
        assert tool.category in TOOL_CATEGORIES, tool.name
        assert tool.effect in TOOL_EFFECTS, tool.name
        assert tool.audience in TOOL_AUDIENCES, tool.name


def test_manifest_values_are_valid_and_shaped_per_contract() -> None:
    manifest = build_manifest(_settings(), now=datetime(2026, 10, 1, 10, 0, 0, 123, tzinfo=UTC))

    assert manifest["version"] == 1
    assert manifest["generatedAt"] == "2026-10-01T10:00:00Z"
    assert set(manifest) == {"version", "generatedAt", "workerVersion", "webSearch", "tools"}
    for entry in manifest["tools"]:
        assert set(entry) == {"name", "category", "effect", "audience", "description"}
        assert entry["effect"] in {"read", "write"}
        assert entry["audience"] in {"member", "host", "platform_staff"}
        assert entry["category"] in WEB_CATEGORY_IDS | {"other"}
        assert entry["description"] and "\n" not in entry["description"]


def test_category_ids_match_the_web_catalog() -> None:
    assert set(TOOL_CATEGORIES) == WEB_CATEGORY_IDS | {"other"}


def test_markings_match_what_the_tools_do() -> None:
    by_name = {tool["name"]: tool for tool in build_manifest(_settings())["tools"]}

    writes = {name for name, tool in by_name.items() if tool["effect"] == "write"}
    assert writes == {
        "create_meeting",
        "create_action_item",
        "create_glossary",
        "add_glossary_term",
        "share_meeting_minutes",
    }
    assert by_name["share_meeting_minutes"]["audience"] == "host"
    assert by_name["get_platform_analytics"]["audience"] == "platform_staff"
    assert by_name["get_platform_analytics"]["category"] == "platform"


def test_invalid_metadata_is_refused() -> None:
    async def _handler(_ctx: object, _args: dict[str, object]) -> str:
        return ""

    bad = ChatTool(
        name="mystery",
        description="Does something.",
        parameters={"type": "object", "properties": {}},
        handler=_handler,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="mystery"):
        build_manifest(_settings(), tools=[bad])


@pytest.mark.parametrize(
    ("api_key", "enabled", "expected"),
    [
        ("sk-test", True, True),
        ("sk-test", False, False),
        ("", True, False),
    ],
)
def test_web_search_available_follows_env(
    monkeypatch: pytest.MonkeyPatch, api_key: str, enabled: bool, expected: bool
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = _settings(api_key=api_key, web_search_enabled=enabled)

    assert web_search_available(settings) is expected
    assert build_manifest(settings)["webSearch"] == {"available": expected}


def test_web_search_reads_the_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shared")
    monkeypatch.setenv("ASSISTANT_CHAT_WEB_SEARCH_ENABLED", "false")
    assert build_manifest(ChatAssistantSettings())["webSearch"]["available"] is False

    monkeypatch.setenv("ASSISTANT_CHAT_WEB_SEARCH_ENABLED", "true")
    assert build_manifest(ChatAssistantSettings())["webSearch"]["available"] is True


def test_worker_version_comes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tools_manifest.WORKER_VERSION_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    assert build_manifest(_settings())["workerVersion"] is None

    monkeypatch.setenv("WORKER_VERSION", "abc123")
    assert build_manifest(_settings())["workerVersion"] == "abc123"


async def test_publish_writes_json_with_ttl() -> None:
    redis = MagicMock()
    redis.set_with_ttl = AsyncMock()

    assert await publish_manifest(redis, _settings(), MagicMock()) is True

    key, value, ttl = redis.set_with_ttl.await_args.args
    assert key == MANIFEST_KEY == "assistant:tools:manifest"
    assert ttl == TTL_SECONDS == 1800
    assert len(json.loads(value)["tools"]) == len([tool for tool in TOOLS if tool.listed])


async def test_redis_failure_is_swallowed_and_logged() -> None:
    redis = MagicMock()
    redis.set_with_ttl = AsyncMock(side_effect=ConnectionError("redis down"))
    logger = MagicMock()

    assert await publish_manifest(redis, _settings(), logger) is False

    logger.exception.assert_called_once()
    assert logger.exception.call_args.args[0] == "tools_manifest_publish_failed"


async def test_publisher_keeps_running_after_a_failed_write() -> None:
    redis = MagicMock()
    calls = 0

    async def _set(*_args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("redis down")

    redis.set_with_ttl = AsyncMock(side_effect=_set)
    logger = MagicMock()

    task = asyncio.create_task(
        run_manifest_publisher(redis, _settings(), logger, interval_seconds=0)
    )
    for _ in range(50):
        if redis.set_with_ttl.await_count >= 3:
            break
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert redis.set_with_ttl.await_count >= 3
    logger.exception.assert_called_once()


def test_refresh_is_ten_minutes() -> None:
    assert REFRESH_INTERVAL_SECONDS == 600
