import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from shared.health_probe import (
    check_worker,
    heartbeat_file,
    heartbeat_key,
    heartbeat_keys,
    touch_heartbeat_file,
)


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


def test_heartbeat_file_lives_where_the_kubelet_probe_reads_it(monkeypatch, tmp_path) -> None:
    # The chart's probe builds "$WORKER_HEALTH_DIR/warptalk-heartbeat-$name"; the two must agree.
    monkeypatch.setenv("WORKER_HEALTH_DIR", str(tmp_path))
    assert heartbeat_file("tts") == tmp_path / "warptalk-heartbeat-tts"


def test_heartbeat_file_defaults_to_tmp(monkeypatch) -> None:
    monkeypatch.delenv("WORKER_HEALTH_DIR", raising=False)
    assert str(heartbeat_file("stt")) == "/tmp/warptalk-heartbeat-stt"


def test_touch_heartbeat_file_creates_then_refreshes_mtime(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORKER_HEALTH_DIR", str(tmp_path))
    touch_heartbeat_file("translation")
    path = tmp_path / "warptalk-heartbeat-translation"
    assert path.exists()
    os.utime(path, (1, 1))
    touch_heartbeat_file("translation")
    assert path.stat().st_mtime > 1


def test_touch_heartbeat_file_never_raises(monkeypatch, tmp_path) -> None:
    # A missing or read-only directory must not take the Redis heartbeat down with it.
    monkeypatch.setenv("WORKER_HEALTH_DIR", str(tmp_path / "does-not-exist"))
    touch_heartbeat_file("tts")


async def test_base_worker_stamps_the_file_only_after_redis_took_the_heartbeat(
    monkeypatch, tmp_path
) -> None:
    from shared.base_worker import BaseWorker

    class _Worker(BaseWorker):
        async def load_model(self) -> None: ...
        async def process(self, message_id, data) -> None: ...

    monkeypatch.setenv("WORKER_HEALTH_DIR", str(tmp_path))
    worker = _Worker.__new__(_Worker)
    worker.worker_name = "tts"
    worker._consumer_name = "tts-host-1"
    worker.input_stream = "translate:results"
    worker.consumer_group = "tts-workers"
    worker._report_integrations = AsyncMock()
    worker.redis = MagicMock()
    worker.redis.set_with_ttl = AsyncMock(side_effect=ConnectionError("redis down"))

    try:
        await worker._publish_heartbeat()
    except ConnectionError:
        pass
    assert not (tmp_path / "warptalk-heartbeat-tts").exists()

    worker.redis.set_with_ttl = AsyncMock()
    await worker._publish_heartbeat()
    assert (tmp_path / "warptalk-heartbeat-tts").exists()
