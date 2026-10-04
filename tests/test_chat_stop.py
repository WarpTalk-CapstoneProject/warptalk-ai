"""Stop: the user ends a WarpBot turn, and the worker actually stops (3 Oct 2026).

AssistantService sets assistant:chat_cancel:{request_id} when Stop is pressed. The worker looks
for it before every model call, every half second while streaming, and before any tool runs. What
was already written is kept as the answer; a turn stopped before it wrote anything is failed, so
it never joins the next turn's history as an empty assistant message.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker import chat_worker as chat_worker_module
from ai_assistant_worker.chat_worker import (
    CHAT_CANCEL_KEY_PREFIX,
    STOPPED_BEFORE_ANSWER,
    ChatAssistantWorker,
    TurnStopped,
)
from shared.config import ChatAssistantSettings


class _Stream:
    """A Responses stream: text deltas, then (optionally) a function call."""

    def __init__(self, deltas: list[str], function_call: bool = False) -> None:
        self._deltas = deltas
        self._function_call = function_call
        self.closed = False

    def __aiter__(self):
        async def gen():
            for delta in self._deltas:
                yield SimpleNamespace(type="response.output_text.delta", delta=delta)
            output = (
                [
                    SimpleNamespace(
                        type="function_call", name="list_issues", arguments="{}", call_id="c1"
                    )
                ]
                if self._function_call
                else [SimpleNamespace(type="message")]
            )
            yield SimpleNamespace(
                type="response.completed", response=SimpleNamespace(output=output)
            )

        return gen()

    async def close(self) -> None:
        self.closed = True


class _Redis:
    """Redis whose cancel key appears after `after` reads (None: never)."""

    def __init__(self, after: int | None) -> None:
        self.after = after
        self.reads = 0

    async def get(self, key: str) -> bytes | None:
        assert key == f"{CHAT_CANCEL_KEY_PREFIX}req-1"
        self.reads += 1
        return b"1" if self.after is not None and self.reads > self.after else None


def _worker(redis: Any, *streams: _Stream) -> Any:
    worker = ChatAssistantWorker.__new__(ChatAssistantWorker)
    worker.chat_settings = ChatAssistantSettings(model="gpt-5.6-luna", max_tokens=256)
    worker.chat_settings.chunk_flush_chars = 1
    worker.logger = MagicMock()
    worker.redis = redis
    worker._openai = MagicMock()
    worker._openai.responses.create = AsyncMock(side_effect=list(streams))
    worker._publish_result = AsyncMock()
    return worker


def _request() -> Any:
    return SimpleNamespace(request_id="req-1", conversation_id="conv-1", workspace_id="ws-1")


async def _loop(worker: Any, tool: Any = None) -> Any:
    lookup = {"list_issues": tool} if tool else {}
    return await worker._run_tool_loop(
        _request(),
        [],
        MagicMock(citations=None),
        instructions="",
        tool_lookup=lookup,
        tool_schemas=[],
    )


async def test_a_stop_pressed_before_the_turn_runs_never_calls_the_model() -> None:
    worker = _worker(_Redis(after=0), _Stream(["never"]))

    try:
        await _loop(worker)
        raise AssertionError("expected TurnStopped")
    except TurnStopped as stopped:
        assert stopped.partial == ""
    worker._openai.responses.create.assert_not_awaited()


async def test_a_stop_while_streaming_keeps_what_was_written_and_closes_the_stream(
    monkeypatch,
) -> None:
    monkeypatch.setattr(chat_worker_module, "STOP_POLL_SECONDS", 0.0)
    stream = _Stream(["Hello ", "there ", "never"])
    # Read 1 is the iteration check; reads 2 and 3 run inside the stream, after one delta each.
    worker = _worker(_Redis(after=2), stream)

    try:
        await _loop(worker)
        raise AssertionError("expected TurnStopped")
    except TurnStopped as stopped:
        assert stopped.partial == "Hello "
    assert stream.closed


async def test_no_tool_runs_once_stop_was_pressed() -> None:
    tool = MagicMock()
    tool.handler = AsyncMock(return_value="{}")
    # Read 1: before the model call. The stream never polls (0.5s has not passed). Read 2: before
    # the tool would run - that is where the stop lands.
    worker = _worker(_Redis(after=1), _Stream(["Checking"], function_call=True))

    try:
        await _loop(worker, tool)
        raise AssertionError("expected TurnStopped")
    except TurnStopped:
        pass
    tool.handler.assert_not_awaited()


async def test_a_redis_failure_never_stops_a_turn() -> None:
    class _Broken:
        async def get(self, key: str) -> None:
            raise ConnectionError("redis down")

    worker = _worker(_Broken(), _Stream(["All good"]))

    final_text, _ = await _loop(worker)

    assert final_text == "All good"


def _process_ready(worker: Any) -> None:
    for name in (
        "_workspace_client",
        "_assistant_client",
        "_transcript_client",
        "_translation_room_client",
    ):
        setattr(worker, name, MagicMock())
    for name in ("_billing_client", "_auth_client"):
        setattr(worker, name, None)


def _raw_request() -> dict[bytes, bytes]:
    return {
        b"request_id": b"req-1",
        b"conversation_id": b"conv-1",
        b"workspace_id": b"ws-1",
        b"user_id": b"u-1",
    }


async def test_process_keeps_a_partial_answer_as_completed() -> None:
    worker = _worker(_Redis(after=None))
    _process_ready(worker)
    worker._run_agent_loop = AsyncMock(side_effect=TurnStopped("Half an answer"))

    await worker.process(b"1-0", _raw_request())

    kwargs = worker._publish_result.await_args.kwargs
    assert kwargs["type_"] == "completed"
    assert kwargs["content"] == "Half an answer"


async def test_process_fails_a_turn_stopped_before_it_wrote_anything() -> None:
    worker = _worker(_Redis(after=None))
    _process_ready(worker)
    worker._run_agent_loop = AsyncMock(side_effect=TurnStopped(""))

    await worker.process(b"1-0", _raw_request())

    kwargs = worker._publish_result.await_args.kwargs
    assert kwargs["type_"] == "failed"
    assert kwargs["content"] == STOPPED_BEFORE_ANSWER
