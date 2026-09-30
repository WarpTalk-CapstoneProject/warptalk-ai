"""WarpBot reads `flags.warpbot_web_search` live when it builds a turn's tool list.

Driven through the real `_run_agent_loop` -> `_run_tool_loop` -> `responses.create`, and asserted
on what was actually sent to OpenAI: the hosted tool in `tools`, and the prompt naming it or not.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared import platform_settings as ps
from shared.platform_settings import PLATFORM_HASH, PlatformSettings
from tests.conftest import FakeClock
from tests.test_chat_agent_loop import (
    _build_worker,
    _completed,
    _message_item,
    _request,
    _text_delta,
)

WS_IN = "ffffffff-ffff-ffff-ffff-ffffffffffff"  # flags.warpbot_web_search bucket 4
WS_OUT = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"  # bucket 99


class _Redis:
    def __init__(self) -> None:
        self.platform: dict[str, str] = {ps.VERSION_FIELD: "1"}

    def put(self, key: str, value: Any) -> None:
        self.platform[key] = json.dumps(value)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.platform) if key == PLATFORM_HASH else {}

    async def get(self, key: str) -> None:
        return None


def _turn() -> list[Any]:
    return [_text_delta("ok"), _completed(_message_item())]


def _worker(clock: FakeClock, turns: int, *, env: bool = True) -> tuple[Any, _Redis]:
    worker, _ = _build_worker([_turn() for _ in range(turns)])
    worker.chat_settings.web_search_enabled = env
    redis = _Redis()
    worker.redis = redis
    worker._platform_settings = PlatformSettings(redis, clock=clock)  # type: ignore[arg-type]
    return worker, redis


def _offered(worker: Any, call: int) -> bool:
    kwargs = worker._openai.responses.create.call_args_list[call].kwargs
    offered = {"type": "web_search"} in kwargs["tools"]
    # The prompt must agree with the tool list — never name a tool the model was not given.
    assert ("web_search" in kwargs["instructions"]) is offered
    return offered


async def test_nothing_stored_means_the_env_switch_decides(fake_clock: FakeClock) -> None:
    worker, _ = _worker(fake_clock, 1, env=True)
    await worker._run_agent_loop(_request(workspace_id=WS_OUT), [], MagicMock())
    assert _offered(worker, 0) is True

    worker, _ = _worker(fake_clock, 1, env=False)
    await worker._run_agent_loop(_request(workspace_id=WS_OUT), [], MagicMock())
    assert _offered(worker, 0) is False


async def test_the_flag_is_read_live(fake_clock: FakeClock) -> None:
    worker, redis = _worker(fake_clock, 3)
    redis.put(ps.FLAG_WARPBOT_WEB_SEARCH, {"enabled": False})
    await worker._run_agent_loop(_request(workspace_id=WS_OUT), [], MagicMock())
    assert _offered(worker, 0) is False

    redis.put(ps.FLAG_WARPBOT_WEB_SEARCH, {"enabled": True})
    fake_clock.advance(11)
    await worker._run_agent_loop(_request(workspace_id=WS_OUT), [], MagicMock())
    assert _offered(worker, 1) is True

    redis.put(ps.FLAG_WARPBOT_WEB_SEARCH, {"enabled": True, "denyWorkspaces": [WS_OUT]})
    fake_clock.advance(11)
    await worker._run_agent_loop(_request(workspace_id=WS_OUT), [], MagicMock())
    assert _offered(worker, 2) is False


@pytest.mark.parametrize(("workspace", "expected"), [(WS_IN, True), (WS_OUT, False)])
async def test_a_partial_rollout_uses_the_requests_workspace(
    fake_clock: FakeClock, workspace: str, expected: bool
) -> None:
    worker, redis = _worker(fake_clock, 1)
    redis.put(ps.FLAG_WARPBOT_WEB_SEARCH, {"enabled": True, "rolloutPercent": 10})
    await worker._run_agent_loop(_request(workspace_id=workspace), [], MagicMock())
    assert _offered(worker, 0) is expected


async def test_the_flag_cannot_turn_on_what_the_deployment_turned_off(
    fake_clock: FakeClock,
) -> None:
    worker, redis = _worker(fake_clock, 1, env=False)
    redis.put(ps.FLAG_WARPBOT_WEB_SEARCH, {"enabled": True, "allowWorkspaces": [WS_IN]})
    await worker._run_agent_loop(_request(workspace_id=WS_IN), [], MagicMock())
    assert _offered(worker, 0) is False
