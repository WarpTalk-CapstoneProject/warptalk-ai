"""WarpBot's built-in tool manifest, published to Redis for AssistantService.

WHY REDIS
    This worker is the only process that knows its built-in tool registry and its env ceiling for
    web search, and it has no HTTP API. So it writes what it knows to one Redis key, and
    AssistantService serves it as GET /api/v1/assistant/tools. The web page lists what this says
    rather than a hand-copied catalog.

SHAPE (contract v1, key `assistant:tools:manifest`, a JSON string under a plain SET)
    {
      "version": 1,
      "generatedAt": "2026-10-01T10:00:00Z",
      "workerVersion": "string or null",
      "webSearch": {"available": true},
      "tools": [{"name", "category", "effect", "audience", "description"}]
    }

    The tools come from `TOOLS`, the same list the model is given, with the category / effect /
    audience each definition declares beside its schema. Plugin (MCP) tools are not included; the
    backend already knows those. Platform-scope tools (platform_tools.PLATFORM_TOOLS) are not
    included either: they are offered only on a platform-admin conversation, never to WarpBot in
    a workspace.

LIFETIME
    Written on startup, then every REFRESH_INTERVAL_SECONDS with a TTL_SECONDS expiry. If the
    worker dies the key expires and the backend reports the manifest as unavailable. A failed
    write is logged and never raised: the manifest is a listing, and the chat must keep serving
    without it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from ai_assistant_worker.chat_tools import (
    TOOL_AUDIENCES,
    TOOL_CATEGORIES,
    TOOL_EFFECTS,
    TOOLS,
    ChatTool,
)
from shared.config import ChatAssistantSettings, resolve_openai_api_key

MANIFEST_KEY = "assistant:tools:manifest"
MANIFEST_VERSION = 1
REFRESH_INTERVAL_SECONDS = 10 * 60
TTL_SECONDS = 30 * 60

#: Env vars a deployment may set to name the running build. Read in order; none set means null.
WORKER_VERSION_ENV_VARS = ("WORKER_VERSION", "IMAGE_TAG", "GIT_SHA")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def one_liner(description: str) -> str:
    """The first sentence of a tool's schema description, on one line."""
    flat = _WHITESPACE.sub(" ", description).strip()
    return _SENTENCE_END.split(flat, maxsplit=1)[0]


def web_search_available(settings: ChatAssistantSettings) -> bool:
    """The worker's env ceiling for OpenAI's hosted web_search.

    It runs on this worker's OpenAI credentials, so it is available when that key is present and
    ASSISTANT_CHAT_WEB_SEARCH_ENABLED is not off. The platform flag `flags.warpbot_web_search`
    can only narrow this, and the backend applies it; it is not read here.
    """
    return bool(settings.web_search_enabled) and bool(resolve_openai_api_key(settings.api_key))


def worker_version() -> str | None:
    for name in WORKER_VERSION_ENV_VARS:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _tool_entry(tool: ChatTool) -> dict[str, Any]:
    if tool.category not in TOOL_CATEGORIES:
        raise ValueError(f"built-in tool {tool.name!r} has invalid category {tool.category!r}")
    if tool.effect not in TOOL_EFFECTS:
        raise ValueError(f"built-in tool {tool.name!r} has invalid effect {tool.effect!r}")
    if tool.audience not in TOOL_AUDIENCES:
        raise ValueError(f"built-in tool {tool.name!r} has invalid audience {tool.audience!r}")
    return {
        "name": tool.name,
        "category": tool.category,
        "effect": tool.effect,
        "audience": tool.audience,
        "description": one_liner(tool.description),
    }


def build_manifest(
    settings: ChatAssistantSettings,
    *,
    tools: Iterable[ChatTool] = TOOLS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The manifest for this worker's built-in tools. Raises on a tool with invalid metadata."""
    moment = (now or datetime.now(UTC)).astimezone(UTC).replace(microsecond=0)
    return {
        "version": MANIFEST_VERSION,
        "generatedAt": moment.isoformat().replace("+00:00", "Z"),
        "workerVersion": worker_version(),
        "webSearch": {"available": web_search_available(settings)},
        "tools": [_tool_entry(tool) for tool in tools],
    }


async def publish_manifest(redis: Any, settings: ChatAssistantSettings, logger: Any) -> bool:
    """Write the manifest once. Never raises (except cancellation); a failure is logged."""
    try:
        manifest = build_manifest(settings)
        await redis.set_with_ttl(MANIFEST_KEY, json.dumps(manifest), TTL_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("tools_manifest_publish_failed", key=MANIFEST_KEY)
        return False
    logger.info(
        "tools_manifest_published",
        key=MANIFEST_KEY,
        tools=len(manifest["tools"]),
        web_search_available=manifest["webSearch"]["available"],
    )
    return True


async def run_manifest_publisher(
    redis: Any,
    settings: ChatAssistantSettings,
    logger: Any,
    *,
    interval_seconds: float = REFRESH_INTERVAL_SECONDS,
) -> None:
    """Publish now, then every `interval_seconds` until cancelled."""
    while True:
        await publish_manifest(redis, settings, logger)
        await asyncio.sleep(interval_seconds)
