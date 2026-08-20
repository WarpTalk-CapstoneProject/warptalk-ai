"""Flash mode — streaming audio during speech — is a property of the ROOM, not the deployment.

`STT_STREAMING_ENABLED` alone made this all-or-nothing for the whole platform, which is the trap
WT-427 already documented for denoising: whichever way the variable is set, half the estate is
configured for the other half. It is also the only way to A/B the feature honestly — one room on,
one room off, same build, same meeting conditions.

Two properties here are load-bearing and neither is obvious from reading the call site:

  * The value is read at SPEECH ONSET, so a person who flips the switch mid-meeting affects the
    next thing they say. Captured once per track it would do nothing until they rejoined, which
    reads as a dead switch.
  * The value is HELD for the rest of the utterance, so a turn that started streaming finishes
    streaming. A turn whose frames stop partway is one the STT side's gap check throws away.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from livekit_ingress_worker.worker import (
    _FLASH_MODE_DEFAULT_KEY,
    LiveKitIngressWorker,
)
from shared.config import WorkerSettings

ROOM = "room-1"


class _Redis:
    def __init__(self, value: str | None = None) -> None:
        self.value = value
        self.gets = 0
        self.written: dict[str, tuple[str, int]] = {}

    async def get(self, key: str) -> str | None:
        self.gets += 1
        return self.value

    async def set_with_ttl(self, key: str, value: bytes | str, ttl_seconds: int) -> None:
        self.written[key] = (
            value.decode() if isinstance(value, bytes) else value,
            ttl_seconds,
        )


def _worker(redis: _Redis, *, default: bool = False) -> LiveKitIngressWorker:
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    settings = WorkerSettings()
    settings.stt_streaming_enabled = default
    worker.settings = settings
    worker.logger = MagicMock()
    worker.redis = redis  # type: ignore[assignment]
    return worker


@pytest.mark.asyncio
async def test_a_room_can_turn_flash_mode_on_while_the_deployment_default_is_off() -> None:
    """The whole point. Shipping default is off, and a single room must still be able to opt in."""
    worker = _worker(_Redis("on"), default=False)

    assert await worker._flash_mode_enabled(ROOM) is True


@pytest.mark.asyncio
async def test_a_room_can_turn_flash_mode_off_while_the_deployment_default_is_on() -> None:
    worker = _worker(_Redis("off"), default=True)

    assert await worker._flash_mode_enabled(ROOM) is False


@pytest.mark.asyncio
async def test_a_room_that_never_set_it_uses_the_deployment_default() -> None:
    assert await _worker(_Redis(None), default=False)._flash_mode_enabled(ROOM) is False
    assert await _worker(_Redis(None), default=True)._flash_mode_enabled(ROOM) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("spelling", ["on", "true", "1", "enabled", "yes", "ON", " On "])
async def test_generous_about_how_on_is_spelled(spelling: str) -> None:
    """This key is written by the backend today and by a person with redis-cli during an incident
    tomorrow. A switch that silently ignores "true" is worse than one that accepts it."""
    assert await _worker(_Redis(spelling), default=False)._flash_mode_enabled(ROOM) is True


@pytest.mark.asyncio
async def test_an_unrecognised_value_falls_back_rather_than_guessing() -> None:
    worker = _worker(_Redis("maybe"), default=False)

    assert await worker._flash_mode_enabled(ROOM) is False
    worker.logger.warning.assert_called()


@pytest.mark.asyncio
async def test_redis_being_unreachable_never_stops_audio_being_processed() -> None:
    class _Broken(_Redis):
        async def get(self, key: str) -> str | None:
            raise RuntimeError("redis is down")

    worker = _worker(_Broken(), default=True)

    assert await worker._flash_mode_enabled(ROOM) is True


@pytest.mark.asyncio
async def test_repeated_reads_inside_the_window_do_not_hammer_redis() -> None:
    # A six-person room asks once per speaker per sentence. Without the cache that is a Redis
    # round trip on the latency path this feature exists to shorten.
    redis = _Redis("on")
    worker = _worker(redis, default=False)

    for _ in range(5):
        assert await worker._flash_mode_enabled(ROOM) is True

    assert redis.gets == 1


@pytest.mark.asyncio
async def test_a_toggle_is_picked_up_without_rejoining() -> None:
    """The property that makes it a usable switch rather than a deploy-time constant.

    Simulated by expiring the cache rather than sleeping: the test pins that the value is
    re-read, not how many seconds the cache happens to hold.
    """
    redis = _Redis("off")
    worker = _worker(redis, default=False)
    assert await worker._flash_mode_enabled(ROOM) is False

    redis.value = "on"
    worker._flash_mode_cache.clear()

    assert await worker._flash_mode_enabled(ROOM) is True


@pytest.mark.asyncio
async def test_two_rooms_are_answered_independently() -> None:
    class _PerRoom(_Redis):
        async def get(self, key: str) -> str | None:
            self.gets += 1
            return "on" if key.endswith("room-a:flash_mode") else "off"

    worker = _worker(_PerRoom(), default=False)

    assert await worker._flash_mode_enabled("room-a") is True
    assert await worker._flash_mode_enabled("room-b") is False


def test_the_frame_consumer_is_not_gated_on_the_deployment_default() -> None:
    """The bug this closes.

    While the consumer was started only when STT_STREAMING_ENABLED was true, the shipping default
    of false was a hard ceiling: a room could set its key to "on", the ingress would publish
    frames, and nothing would ever read them. A per-room switch whose consumer does not exist is
    a switch wired to nothing.

    Asserted over the AST rather than the text. The first version of this test looked for "if" on
    the same LINE as the call, which a re-gate does not put there — it passed against the very
    regression it was written to catch. What matters is whether the call is nested inside a
    conditional at all, and that is a question about the tree, not the characters.
    """
    import ast
    import inspect
    import textwrap

    from stt_worker import worker as stt_worker_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(stt_worker_module.STTWorker.load_model)))

    def starts_frame_consumer(node: ast.AST) -> bool:
        return any(
            isinstance(inner, ast.Attribute) and inner.attr == "_consume_speech_frames"
            for inner in ast.walk(node)
        )

    assert starts_frame_consumer(tree), "load_model no longer starts the frame consumer at all"

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Try, ast.While)) and starts_frame_consumer(node):
            raise AssertionError(
                "the frame consumer is started inside a conditional; a per-room switch cannot "
                "turn streaming on when the loop that consumes the frames does not exist"
            )


# ---------------------------------------------------------------------------
# Publishing the deployment default, so the host's switch can describe reality
# ---------------------------------------------------------------------------
#
# The backend owns only the per-room OVERRIDE key. Asked about a room that never set one it
# answered "off", which was true of the override and false of the room — harmless while the
# deployment default was also off, and actively misleading the day that default became on. It
# reads this key to tell the difference, and this worker is the only thing that knows the answer.


@pytest.mark.asyncio
async def test_the_deployment_default_is_published_for_the_backend_to_read() -> None:
    redis = _Redis()
    await _worker(redis, default=True)._publish_flash_mode_default()

    value, ttl = redis.written[_FLASH_MODE_DEFAULT_KEY]
    assert value == "on"
    # Long enough to outlive a deploy window, and not immortal: this Redis runs allkeys-lru.
    # It does not need to be longer because the heartbeat republishes it every interval — which
    # is the property the next test pins.
    assert 10 * 60 <= ttl <= 24 * 60 * 60


@pytest.mark.asyncio
async def test_an_off_deployment_publishes_off_rather_than_nothing() -> None:
    """Absent and "off" are different answers, and the backend renders them differently."""
    redis = _Redis()
    await _worker(redis, default=False)._publish_flash_mode_default()

    assert redis.written[_FLASH_MODE_DEFAULT_KEY][0] == "off"


@pytest.mark.asyncio
async def test_a_redis_that_refuses_the_write_does_not_stop_the_worker_starting() -> None:
    """This key exists so a switch can describe reality. Failing to describe reality must never
    stop audio being processed — the backend simply keeps reporting "unknown"."""

    class _Refuses(_Redis):
        async def set_with_ttl(self, key: str, value: bytes | str, ttl_seconds: int) -> None:
            raise ConnectionError("redis is down")

    await _worker(_Refuses(), default=True)._publish_flash_mode_default()


@pytest.mark.asyncio
async def test_the_default_rides_the_heartbeat_so_an_eviction_heals_itself() -> None:
    """Not a one-shot at startup.

    On a Redis running allkeys-lru a startup-only write is gone until the next deploy, and the
    backend would spend that whole window telling every host "unknown". Riding the heartbeat
    caps that at one interval. It also keeps load_model doing one thing — a Redis write bolted
    in there broke a contract test that builds this worker with __new__ purely to assert the
    Silero release is pinned.
    """
    redis = _Redis()
    worker = _worker(redis, default=True)
    worker._consumer_name = "livekit-ingress-worker-test"
    worker.worker_name = "livekit-ingress-worker"

    await worker._publish_heartbeat()

    assert _FLASH_MODE_DEFAULT_KEY in redis.written
