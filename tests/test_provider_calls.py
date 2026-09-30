"""Provider call outcomes: which vendor failed, how, and when — for the admin Providers page.

The billing service copies `warptalk:provider_calls:{day}` into Postgres and builds the 90-day
uptime bars and the live success rate from it, so the field layout below is a cross-repo
contract (billing-service ProviderCallStatsParser). These tests pin it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from shared import provider_calls
from shared.config import RedisSettings
from shared.provider_calls import (
    PROVIDER_CALL_OUTCOMES,
    PROVIDER_CALLS_KEY_PREFIX,
    ObservedTransport,
    bind_provider_calls,
    classify_exception,
    classify_status,
    latency_bucket,
    model_of_request,
    provider_call_fields,
    provider_calls_key,
    record_provider_call,
)
from shared.redis_client import RedisStreamClient

pytestmark = pytest.mark.asyncio

AT = datetime(2026, 9, 25, 7, 42, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _unbound() -> Any:
    bind_provider_calls(None)
    yield
    bind_provider_calls(None)


# ── classification ───────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (200, "ok"),
        (204, "ok"),
        (302, "ok"),
        (400, "client_error"),
        (401, "auth"),
        (402, "quota"),
        (403, "auth"),
        (404, "client_error"),
        (422, "client_error"),
        (429, "rate_limited"),
        (500, "server_error"),
        (503, "server_error"),
    ],
)
async def test_a_status_names_whose_problem_it_is(status: int, outcome: str) -> None:
    assert classify_status(status) == outcome


class _StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"boom {status_code}")
        self.status_code = status_code


async def test_an_sdk_error_is_classified_by_its_status() -> None:
    assert classify_exception(_StatusError(402)) == "quota"
    assert classify_exception(_StatusError(429)) == "rate_limited"
    assert classify_exception(_StatusError(502)) == "server_error"


async def test_timeouts_and_connection_failures_are_their_own_outcomes() -> None:
    assert classify_exception(TimeoutError()) == "timeout"
    assert classify_exception(httpx.ReadTimeout("slow")) == "timeout"
    assert classify_exception(httpx.ConnectError("refused")) == "network_error"
    assert classify_exception(ConnectionResetError()) == "network_error"


async def test_a_websocket_error_that_only_says_its_status_in_prose_is_still_read() -> None:
    assert (
        classify_exception(RuntimeError("server rejected WebSocket connection: HTTP 402"))
        == "quota"
    )
    assert classify_exception(RuntimeError("status code 429 from upstream")) == "rate_limited"
    assert classify_exception(RuntimeError("Payment Required")) == "quota"


async def test_an_exception_with_nothing_to_go_on_is_an_unclassified_error_not_a_guess() -> None:
    assert classify_exception(ValueError("bad json")) == "error"


# ── the Redis contract ───────────────────────────────────────────────────────────────────────


async def test_one_call_is_one_outcome_count_one_latency_bucket_and_the_latency_sum() -> None:
    fields = provider_call_fields("OpenAI", "translation", "gpt-4.1", "ok", 730, AT)

    assert fields == [
        ("openai|07|translation|gpt-4.1|ok", 1),
        ("openai|07|translation|gpt-4.1|lat:1000", 1),
        ("openai|07|translation|gpt-4.1|lat_sum", 730),
    ]


async def test_the_key_is_the_utc_day_of_the_call() -> None:
    late_in_hanoi = datetime(2026, 9, 25, 23, 30, tzinfo=UTC)
    assert provider_calls_key(late_in_hanoi) == f"{PROVIDER_CALLS_KEY_PREFIX}2026-09-25"


async def test_an_unknown_model_and_a_separator_in_a_label_cannot_break_the_field_layout() -> None:
    fields = provider_call_fields("cartesia", "tts|x", None, "quota", None, AT)

    assert fields == [("cartesia|07|tts-x|-|quota", 1)]
    assert all(field.count("|") == 4 for field, _ in fields)


async def test_an_outcome_outside_the_vocabulary_is_counted_as_error() -> None:
    fields = provider_call_fields("openai", "stt", None, "exploded", None, AT)
    assert fields == [("openai|07|stt|-|error", 1)]
    assert "error" in PROVIDER_CALL_OUTCOMES


async def test_latency_past_the_last_edge_lands_in_inf() -> None:
    assert latency_bucket(90) == "100"
    assert latency_bucket(100) == "100"
    assert latency_bucket(20001) == "+Inf"


def _client() -> tuple[RedisStreamClient, MagicMock]:
    client = RedisStreamClient(RedisSettings())
    redis = MagicMock()
    pipeline = MagicMock()
    pipeline.execute = AsyncMock(return_value=[1, 1, 1, True])
    redis.pipeline = MagicMock(return_value=pipeline)
    client._redis = redis
    return client, pipeline


async def test_a_recorded_call_is_one_pipeline_of_increments_with_a_ttl() -> None:
    client, pipeline = _client()
    bind_provider_calls(client)

    await record_provider_call("cartesia", "tts", "ok", 420, "sonic-3.5", AT)

    key = f"{PROVIDER_CALLS_KEY_PREFIX}2026-09-25"
    pipeline.hincrby.assert_any_call(key, "cartesia|07|tts|sonic-3.5|ok", 1)
    pipeline.hincrby.assert_any_call(key, "cartesia|07|tts|sonic-3.5|lat:500", 1)
    pipeline.hincrby.assert_any_call(key, "cartesia|07|tts|sonic-3.5|lat_sum", 420)
    pipeline.expire.assert_called_once_with(key, provider_calls.PROVIDER_CALLS_KEY_TTL_SECONDS)


async def test_recording_before_redis_is_bound_is_a_no_op() -> None:
    await record_provider_call("openai", "stt", "ok", 10)


async def test_a_redis_failure_never_reaches_the_call_it_measures() -> None:
    client, pipeline = _client()
    pipeline.execute = AsyncMock(side_effect=ConnectionError("redis gone"))
    bind_provider_calls(client)

    await record_provider_call("openai", "stt", "ok", 10, at=AT)


# ── the httpx transport ──────────────────────────────────────────────────────────────────────


class _Inner(httpx.AsyncBaseTransport):
    def __init__(self, respond: Any) -> None:
        self._respond = respond

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        result = self._respond(request)
        if isinstance(result, BaseException):
            raise result
        return result


async def _drain() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


async def test_the_transport_records_each_response_with_the_model_from_the_body() -> None:
    client, pipeline = _client()
    bind_provider_calls(client)
    transport = ObservedTransport("openai", "translation", _Inner(lambda _: httpx.Response(429)))

    async with httpx.AsyncClient(transport=transport) as http:
        response = await http.post(
            "https://api.openai.com/v1/chat/completions", json={"model": "gpt-4.1"}
        )
    await _drain()

    assert response.status_code == 429
    fields = [call.args[1] for call in pipeline.hincrby.call_args_list]
    assert any(field.endswith("|translation|gpt-4.1|rate_limited") for field in fields)


async def test_the_transport_records_a_failure_and_still_raises_it() -> None:
    client, pipeline = _client()
    bind_provider_calls(client)
    transport = ObservedTransport(
        "openai", "embedding", _Inner(lambda _: httpx.ConnectError("refused"))
    )

    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(httpx.ConnectError):
            await http.post(
                "https://api.openai.com/v1/embeddings", json={"model": "text-embedding-3-small"}
            )
    await _drain()

    fields = [call.args[1] for call in pipeline.hincrby.call_args_list]
    assert any(
        field.endswith("|embedding|text-embedding-3-small|network_error") for field in fields
    )


async def test_our_own_cancellation_is_not_blamed_on_the_vendor() -> None:
    client, pipeline = _client()
    bind_provider_calls(client)
    transport = ObservedTransport("openai", "assistant", _Inner(lambda _: asyncio.CancelledError()))

    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(asyncio.CancelledError):
            await http.post("https://api.openai.com/v1/responses", json={"model": "x"})
    await _drain()

    pipeline.hincrby.assert_not_called()


async def test_model_sniffing_reads_json_only_and_both_spellings() -> None:
    openai = httpx.Request(
        "POST",
        "https://x",
        content=json.dumps({"model": "gpt-4.1"}),
        headers={"content-type": "application/json"},
    )
    cartesia = httpx.Request(
        "POST",
        "https://x",
        content=json.dumps({"model_id": "sonic-3.5"}),
        headers={"content-type": "application/json"},
    )
    audio = httpx.Request(
        "POST", "https://x", content=b"\x00\x01", headers={"content-type": "audio/wav"}
    )

    assert model_of_request(openai) == "gpt-4.1"
    assert model_of_request(cartesia) == "sonic-3.5"
    assert model_of_request(audio) is None


async def test_every_openai_client_in_the_workers_is_observed() -> None:
    """An AsyncOpenAI(...) without the observed client is a provider the page cannot see."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    unobserved = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel.startswith((".venv/", "tests/", "tools/", "benchmarks/")):
            continue
        source = path.read_text(encoding="utf-8")
        for constructor in ("AsyncOpenAI(", "AsyncCartesia("):
            start = 0
            while (index := source.find(constructor, start)) != -1:
                start = index + 1
                call = source[index : source.find(")", index) + 80]
                if "import" in source[max(0, index - 30) : index]:
                    continue
                if "http_client=observed_" not in call:
                    unobserved.append(f"{rel}: {constructor}")
    assert unobserved == []
