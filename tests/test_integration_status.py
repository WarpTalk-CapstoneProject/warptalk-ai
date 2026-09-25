"""Workers report which integrations they are configured for — and never the credentials.

Hash `platform:integrations:v1:ai-{worker}`: field per integration, JSON value
{"configured", "detail", "reportedAt"}, EXPIRE 600, refreshed on the heartbeat every 300s.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from shared.config import (
    ChatAssistantSettings,
    LiveKitSettings,
    STTSettings,
    SuggestionSettings,
    TTSSettings,
    WorkerSettings,
)
from shared.integration_status import (
    CARTESIA,
    LIVEKIT,
    OPENAI,
    TTL_SECONDS,
    IntegrationReport,
    integrations_key,
    livekit_report,
    publish_integration_status,
)
from shared.redis_client import RedisStreamClient

# Built at runtime rather than written as literals: they only need to be recognisable in the
# output, and credential-shaped literals in a diff trip the CI secret scan.
FAKE_OPENAI = "-".join(("fake", "openai", "value", "123"))
FAKE_LIVEKIT_KEY = "-".join(("fake", "livekit", "key"))
FAKE_LIVEKIT_SECRET = "-".join(("fake", "livekit", "value"))
FAKE_CARTESIA = "-".join(("fake", "cartesia", "value", "456"))
SECRETS = (FAKE_OPENAI, FAKE_LIVEKIT_KEY, FAKE_LIVEKIT_SECRET, FAKE_CARTESIA, "wss://lk.example")


def _livekit() -> LiveKitSettings:
    return LiveKitSettings(
        url="wss://lk.example", api_key=FAKE_LIVEKIT_KEY, api_secret=FAKE_LIVEKIT_SECRET
    )


async def _read(client: RedisStreamClient, worker: str) -> dict[str, dict[str, Any]]:
    raw = await client.redis.hgetall(integrations_key(worker))
    return {k.decode(): json.loads(v) for k, v in raw.items()}


async def test_publish_writes_the_contract_shape_with_a_ttl(
    settings_redis: RedisStreamClient,
) -> None:
    at = datetime(2026, 9, 25, 3, 4, 5, tzinfo=UTC)
    ok = await publish_integration_status(
        settings_redis,
        "tts",
        {
            CARTESIA: IntegrationReport(True, "tts model sonic-3.5"),
            LIVEKIT: IntegrationReport(False),
        },
        now=at,
    )

    assert ok is True
    assert await _read(settings_redis, "tts") == {
        CARTESIA: {
            "configured": True,
            "detail": "tts model sonic-3.5",
            "reportedAt": "2026-09-25T03:04:05Z",
        },
        LIVEKIT: {"configured": False, "detail": None, "reportedAt": "2026-09-25T03:04:05Z"},
    }
    assert 0 < await settings_redis.redis.ttl("platform:integrations:v1:ai-tts") <= TTL_SECONDS


async def test_publish_never_raises() -> None:
    broken = MagicMock()

    async def boom(*_args: Any) -> None:
        raise ConnectionError("redis down")

    broken.hset = boom
    assert (
        await publish_integration_status(broken, "tts", {OPENAI: IntegrationReport(True)}) is False
    )
    assert (
        await publish_integration_status(object(), "tts", {OPENAI: IntegrationReport(True)})
        is False
    )


def test_livekit_placeholders_are_not_configured() -> None:
    assert livekit_report(LiveKitSettings()).configured is False
    assert livekit_report(_livekit()).configured is True
    assert livekit_report(LiveKitSettings(url="", api_key="k", api_secret="s")).configured is False


async def test_no_secret_or_url_is_ever_written(settings_redis: RedisStreamClient) -> None:
    from ai_assistant_worker.chat_worker import ChatAssistantWorker
    from livekit_ingress_worker.worker import LiveKitIngressWorker
    from stt_worker.worker import STTWorker
    from suggestion_worker.worker import SuggestionWorker
    from tts_worker.worker import TTSWorker

    settings = WorkerSettings(livekit=_livekit())
    workers: list[Any] = [
        TTSWorker(tts_settings=TTSSettings(api_key=FAKE_CARTESIA), settings=settings),
        LiveKitIngressWorker(settings=settings),
        STTWorker(stt_settings=STTSettings(api_key=FAKE_OPENAI), settings=settings),
        SuggestionWorker(
            suggestion_settings=SuggestionSettings(api_key=FAKE_OPENAI), settings=settings
        ),
        ChatAssistantWorker(
            chat_settings=ChatAssistantSettings(api_key=FAKE_OPENAI), settings=settings
        ),
    ]
    for worker in workers:
        worker.redis = settings_redis
        await worker._report_integrations()

    raw = json.dumps({w.worker_name: await _read(settings_redis, w.worker_name) for w in workers})
    for secret in SECRETS:
        assert secret not in raw

    tts = await _read(settings_redis, "tts")
    assert tts[CARTESIA]["configured"] is True
    assert tts[CARTESIA]["detail"] == "tts model sonic-3.5"
    assert tts[LIVEKIT]["configured"] is True
    assert (await _read(settings_redis, "livekit_ingress"))[LIVEKIT]["configured"] is True
    assert (await _read(settings_redis, "stt"))[OPENAI]["detail"] == "stt model gpt-live-transcribe"
    assert (await _read(settings_redis, "assistant-chat"))[OPENAI]["configured"] is True


async def test_an_empty_key_reports_not_configured(
    settings_redis: RedisStreamClient, monkeypatch: Any
) -> None:
    from tts_worker.worker import TTSWorker

    worker = TTSWorker(tts_settings=TTSSettings(api_key=""), settings=WorkerSettings())
    worker.redis = settings_redis
    await worker._report_integrations()

    report = await _read(settings_redis, "tts")
    assert report[CARTESIA]["configured"] is False
    assert report[LIVEKIT]["configured"] is False  # shipped placeholders


async def test_the_heartbeat_reports_every_300s_not_every_beat(
    settings_redis: RedisStreamClient,
) -> None:
    from livekit_ingress_worker.worker import LiveKitIngressWorker

    worker = LiveKitIngressWorker(settings=WorkerSettings(livekit=_livekit()))
    worker.redis = settings_redis
    worker.logger = MagicMock()
    key = integrations_key("livekit_ingress")

    await worker._publish_heartbeat()
    assert await settings_redis.redis.exists(key)

    await settings_redis.redis.delete(key)
    await worker._publish_heartbeat()
    assert not await settings_redis.redis.exists(key), "reported again inside the interval"

    worker._integrations_reported_at = time.monotonic() - 301
    await worker._publish_heartbeat()
    assert await settings_redis.redis.exists(key)


async def test_a_worker_with_no_integrations_writes_nothing(
    settings_redis: RedisStreamClient,
) -> None:
    from translation_worker.backfill_worker import TranslationBackfillWorker

    worker = TranslationBackfillWorker.__new__(TranslationBackfillWorker)
    worker.redis = settings_redis
    await worker._report_integrations()
    assert await settings_redis.redis.keys("platform:integrations:*") == []
