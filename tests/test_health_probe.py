import json
from unittest.mock import AsyncMock, MagicMock, patch

from shared.health_probe import check_worker, heartbeat_key, heartbeat_keys


def test_heartbeat_key_matches_worker_runtime_identity() -> None:
    assert (
        heartbeat_key("translation", "container-abc")
        == "warptalk:worker:heartbeat:translation:container-abc"
    )


def test_heartbeat_keys_require_every_worker_in_a_combined_process() -> None:
    assert heartbeat_keys("embedding, embedding-search", "host-1") == [
        "warptalk:worker:heartbeat:embedding:host-1",
        "warptalk:worker:heartbeat:embedding-search:host-1",
    ]


async def test_health_probe_uses_shared_sentinel_aware_client(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_HEALTH_NAME", "stt")
    redis = AsyncMock()
    redis.mget = AsyncMock(
        return_value=[
            json.dumps(
                {
                    "worker": "stt",
                    "timestamp_unix_ms": 1_000_000,
                    "last_progress_unix_ms": 1_000_000,
                }
            ).encode()
        ]
    )
    client = MagicMock()
    client.redis = redis
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    with (
        patch("shared.health_probe.RedisStreamClient", return_value=client),
        patch("shared.health_probe.time.time", return_value=1000),
    ):
        assert await check_worker() is True

    client.connect.assert_awaited_once()
    client.disconnect.assert_awaited_once()


async def test_health_probe_stays_healthy_when_idle_worker_heartbeat_is_fresh(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_HEALTH_NAME", "tts")
    monkeypatch.setenv("WORKER_HEALTH_MAX_HEARTBEAT_AGE_SECONDS", "30")
    redis = AsyncMock()
    redis.mget = AsyncMock(
        return_value=[
            json.dumps(
                {
                    "worker": "tts",
                    "timestamp_unix_ms": 1_000_000,
                    "last_progress_unix_ms": 700_000,
                }
            ).encode()
        ]
    )
    client = MagicMock()
    client.redis = redis
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    with (
        patch("shared.health_probe.RedisStreamClient", return_value=client),
        patch("shared.health_probe.time.time", return_value=1000),
    ):
        assert await check_worker() is True


async def test_health_probe_fails_when_heartbeat_is_stale(monkeypatch) -> None:
    monkeypatch.setenv("WORKER_HEALTH_NAME", "tts")
    monkeypatch.setenv("WORKER_HEALTH_MAX_HEARTBEAT_AGE_SECONDS", "30")
    redis = AsyncMock()
    redis.mget = AsyncMock(
        return_value=[
            json.dumps(
                {
                    "worker": "tts",
                    "timestamp_unix_ms": 900_000,
                    "last_progress_unix_ms": 900_000,
                }
            ).encode()
        ]
    )
    client = MagicMock()
    client.redis = redis
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    with (
        patch("shared.health_probe.RedisStreamClient", return_value=client),
        patch("shared.health_probe.time.time", return_value=1000),
    ):
        assert await check_worker() is False


# ── a traceback that reaches the log ─────────────────────────────────────────────────────────


def test_logging_is_configured_to_render_tracebacks() -> None:
    """`logger.exception(...)` has to produce a stack, not the word `true`.

    structlog only MARKS an event as carrying an exception; a processor has to turn that mark
    into text. With none, the JSON renderer serialised the flag — production logs read
    `"exc_info": true` and the exception, its type and its stack were gone. tts_worker logged
    exactly that on every sentence for two releases (WT-400) while the reason stayed invisible,
    and finding it in the end needed a probe against the live vendor API.
    """
    import structlog

    from shared.logger import setup_logging

    setup_logging("INFO")
    processors = structlog.get_config()["processors"]

    assert structlog.processors.format_exc_info in processors, (
        "no processor renders exc_info — every traceback in every worker is discarded"
    )
    renderers = [structlog.processors.JSONRenderer, structlog.dev.ConsoleRenderer]
    assert processors.index(structlog.processors.format_exc_info) < min(
        i for i, p in enumerate(processors) if isinstance(p, tuple(renderers))
    ), "exc_info is rendered after the output is already serialised"
