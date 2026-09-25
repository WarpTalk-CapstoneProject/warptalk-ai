"""The ingress worker reads `meetings.flash_mode_default` and `meetings.chunk_duration_ms` live.

Each test publishes a value the way the workspace service does, drives the real code path, then
changes the value inside the same test and moves the reader's clock past its TTL — the worker is
never rebuilt, which is the property: a console change reaches a running worker.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import livekit_ingress_worker.worker as ingress
from livekit_ingress_worker.worker import _FLASH_MODE_DEFAULT_KEY, LiveKitIngressWorker
from shared import platform_settings as ps
from shared.config import LiveKitSettings, WorkerSettings
from shared.platform_settings import PLATFORM_HASH, PlatformSettings
from shared.redis_client import RedisStreamClient
from tests.conftest import FakeClock

ROOM = "01a0015d-c945-758d-a622-8794cb537dfb"
SPEAKER = "019f0d00-0de0-7000-9000-000000000003"


async def put(client: RedisStreamClient, key: str, value: Any) -> None:
    await client.redis.hset(PLATFORM_HASH, mapping={ps.VERSION_FIELD: "1", key: json.dumps(value)})


def _worker(
    redis: RedisStreamClient, clock: FakeClock, *, streaming_env: bool, chunk_env: int = 6000
) -> LiveKitIngressWorker:
    settings = WorkerSettings(
        stt_streaming_enabled=streaming_env,
        chunk_duration_ms=chunk_env,
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret"),
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = redis
    worker.logger = MagicMock()
    worker._platform_settings = PlatformSettings(redis, clock=clock)
    return worker


@pytest.fixture(autouse=True)
def _no_flash_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # The per-room 3s cache is the ingress worker's own and has its own tests; here only the
    # platform reader's TTL is under test.
    monkeypatch.setattr(ingress, "_FLASH_MODE_CACHE_SECONDS", 0.0)


# ── flash mode default ──────────────────────────────────────────────────────────────────────


async def test_nothing_stored_means_the_env_default_decides(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    assert (
        await _worker(settings_redis, fake_clock, streaming_env=False)._flash_mode_enabled(ROOM)
        is False
    )
    assert (
        await _worker(settings_redis, fake_clock, streaming_env=True)._flash_mode_enabled(ROOM)
        is True
    )


async def test_the_platform_default_is_read_live(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    worker = _worker(settings_redis, fake_clock, streaming_env=False)
    await put(settings_redis, ps.FLASH_MODE_DEFAULT, True)
    assert await worker._flash_mode_enabled(ROOM) is True

    await put(settings_redis, ps.FLASH_MODE_DEFAULT, False)
    assert await worker._flash_mode_enabled(ROOM) is True  # still inside the reader's TTL
    fake_clock.advance(11)
    assert await worker._flash_mode_enabled(ROOM) is False


async def test_a_hosts_per_room_choice_still_wins_over_the_platform_default(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    worker = _worker(settings_redis, fake_clock, streaming_env=True)
    await put(settings_redis, ps.FLASH_MODE_DEFAULT, False)
    await settings_redis.redis.set(f"translationRoom:{ROOM}:flash_mode", "on")

    assert await worker._flash_mode_enabled(ROOM) is True
    assert await worker._flash_mode_enabled("another-room") is False


async def test_the_heartbeat_publishes_the_effective_default(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    worker = _worker(settings_redis, fake_clock, streaming_env=True)

    await worker._publish_flash_mode_default()
    assert await settings_redis.redis.get(_FLASH_MODE_DEFAULT_KEY) == b"on"

    await put(settings_redis, ps.FLASH_MODE_DEFAULT, False)
    fake_clock.advance(11)
    await worker._publish_flash_mode_default()
    assert await settings_redis.redis.get(_FLASH_MODE_DEFAULT_KEY) == b"off"

    await put(settings_redis, ps.FLASH_MODE_DEFAULT, True)
    fake_clock.advance(11)
    await worker._publish_heartbeat()
    assert await settings_redis.redis.get(_FLASH_MODE_DEFAULT_KEY) == b"on"


# ── chunk cap ───────────────────────────────────────────────────────────────────────────────


async def test_the_chunk_cap_is_read_live_with_the_env_as_fallback(
    settings_redis: RedisStreamClient, fake_clock: FakeClock
) -> None:
    worker = _worker(settings_redis, fake_clock, streaming_env=False, chunk_env=6000)
    assert await worker._max_chunk_ms() == 6000

    await put(settings_redis, ps.CHUNK_DURATION_MS, 3000)
    fake_clock.advance(11)
    assert await worker._max_chunk_ms() == 3000

    await put(settings_redis, ps.CHUNK_DURATION_MS, 1000)  # below the catalog's min: ignored
    fake_clock.advance(11)
    assert await worker._max_chunk_ms() == 6000


class _Stream:
    """`seconds` of continuous 16 kHz mono "speech", in 100ms frames, then the end of the track."""

    def __init__(self, seconds: float) -> None:
        samples = np.full(1600, 8000, dtype=np.int16).tobytes()
        self._events = [
            SimpleNamespace(frame=SimpleNamespace(data=samples, sample_rate=16000, num_channels=1))
            for _ in range(int(seconds * 10))
        ]

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


async def _first_chunk_ms(worker: LiveKitIngressWorker, monkeypatch: pytest.MonkeyPatch) -> float:
    published: list[int] = []

    async def capture(
        room: str, speaker: str, buffer: bytearray, *args: Any, **kwargs: Any
    ) -> None:
        published.append(len(buffer))

    worker._publish_speech_chunk = capture  # type: ignore[method-assign]
    worker._publish_speech_frame = AsyncMock()  # type: ignore[method-assign]
    worker._vad_model = MagicMock()
    # Every window is speech: the only thing that can cut the turn is the chunk cap.
    worker._score_vad_frames = lambda pcm, vad_model=None: [1.0] * 3  # type: ignore[method-assign]
    monkeypatch.setattr(ingress.rtc, "AudioStream", MagicMock(return_value=_Stream(8.0)))
    resampler = MagicMock()
    resampler.push = lambda frame: [SimpleNamespace(data=frame.data)]
    monkeypatch.setattr(ingress.rtc, "AudioResampler", MagicMock(return_value=resampler))
    track = MagicMock()
    track.sid = "TR_1"

    await worker.process_audio_track(ROOM, SPEAKER, track)
    assert published, "no chunk was cut"
    return published[0] / 2 / 16000 * 1000


async def test_a_running_track_cuts_at_the_live_chunk_cap(
    settings_redis: RedisStreamClient, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _worker(settings_redis, fake_clock, streaming_env=False, chunk_env=6000)

    await put(settings_redis, ps.CHUNK_DURATION_MS, 2000)
    first = await _first_chunk_ms(worker, monkeypatch)
    assert 2000 <= first < 2200

    await put(settings_redis, ps.CHUNK_DURATION_MS, 4000)
    fake_clock.advance(11)
    second = await _first_chunk_ms(worker, monkeypatch)
    assert 4000 <= second < 4200
