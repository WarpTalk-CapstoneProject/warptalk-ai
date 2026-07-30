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
