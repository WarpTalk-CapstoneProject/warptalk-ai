"""The first sentence of a turn should not pay for dialling Cartesia.

MEASURED, which is why this exists (tools/probe_tts_first_audio.py, 2026-08-18):

    websocket_connect().enter()   p50 427.6ms
    connection.context(...)       p50   0.1ms
    cold time-to-first-audio      p50 0.669s
    warm time-to-first-audio      p50 0.180s

So roughly three quarters of the wait before a listener hears anything is a handshake that
could have happened at any earlier moment. These pin the pool that moves it earlier, and in
particular pin the two ways a pool makes things WORSE than no pool at all: handing over a
socket that has gone stale, and filling once so that only the first few turns benefit.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from tts_worker import synthesizer as synth_mod
from tts_worker.synthesizer import CartesiaSynthesizer


class _Connection:
    def __init__(self, index: int) -> None:
        self.index = index
        self.closed = False
        self.contexts: list[dict[str, Any]] = []

    def context(self, **kwargs: Any) -> Any:
        self.contexts.append(kwargs)
        return MagicMock()

    async def close(self) -> None:
        self.closed = True


class _FakeCartesia:
    """Counts dials. Everything this test cares about is how many happen and when."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.dials = 0
        self.connections: list[_Connection] = []
        self._fail_after = fail_after
        self.tts = self

    def websocket_connect(self) -> Any:
        outer = self

        class _Manager:
            async def enter(self) -> _Connection:
                if outer._fail_after is not None and outer.dials >= outer._fail_after:
                    raise RuntimeError("cartesia refused")
                outer.dials += 1
                connection = _Connection(outer.dials)
                outer.connections.append(connection)
                return connection

        return _Manager()


def _synth(fake: _FakeCartesia) -> CartesiaSynthesizer:
    s = CartesiaSynthesizer(api_key="k", model="sonic-3.5", sample_rate=44100)
    s._client = fake  # type: ignore[assignment]
    return s


@pytest.mark.asyncio
async def test_warm_up_dials_ahead_of_the_first_sentence() -> None:
    fake = _FakeCartesia()
    s = _synth(fake)

    await s.warm_up(pool_size=2)

    assert fake.dials == 2, "the pool is filled before anybody speaks"


@pytest.mark.asyncio
async def test_a_turn_takes_from_the_pool_instead_of_dialling() -> None:
    fake = _FakeCartesia()
    s = _synth(fake)
    await s.warm_up(pool_size=2)

    await s.open_prosody_context(context_id="c1", language="vi")

    # The claim itself must not dial. The refill below may, in the background.
    assert len(s._warm_connections) == 1


@pytest.mark.asyncio
async def test_the_pool_refills_so_it_is_not_a_one_shot_allocation() -> None:
    """The lesson OpenAISTT.warm_up already learned: a pool filled exactly once helps the first
    few speakers a process ever sees and nobody after them."""
    fake = _FakeCartesia()
    s = _synth(fake)
    await s.warm_up(pool_size=2)

    for index in range(3):
        await s.open_prosody_context(context_id=f"c{index}", language="vi")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert fake.dials > 2, "claims are replaced, not merely spent"


@pytest.mark.asyncio
async def test_an_empty_pool_falls_back_to_dialling_rather_than_failing() -> None:
    fake = _FakeCartesia()
    s = _synth(fake)  # never warmed

    context, connection = await s.open_prosody_context(context_id="c1", language="vi")

    assert context is not None
    assert connection is not None
    assert fake.dials >= 1, "behaves exactly as it did before the pool existed"


@pytest.mark.asyncio
async def test_a_stale_connection_is_discarded_rather_than_handed_to_a_turn() -> None:
    """Handing over a dead socket costs a spoken sentence — worse than no pool, where the dial
    is at least fresh. Age is checked on the way OUT, which is the whole staleness policy."""
    fake = _FakeCartesia()
    s = _synth(fake)
    await s.warm_up(pool_size=1)
    stale = fake.connections[0]

    # Backdate the only pooled entry past the idle bound.
    connection, opened_at = s._warm_connections[0]
    s._warm_connections[0] = (
        connection,
        opened_at - synth_mod.WARM_CONNECTION_MAX_IDLE_SECONDS - 1.0,
    )

    _context, handed_over = await s.open_prosody_context(context_id="c1", language="vi")

    assert handed_over is not stale, "the stale one was not reused"
    assert handed_over.index > stale.index, "a fresh connection was dialled instead"

    # And the stale socket is actually closed, not merely dropped on the floor — the close is
    # fire-and-forget precisely so it does not sit on the hot path of a spoken sentence.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert stale.closed


@pytest.mark.asyncio
async def test_a_provider_outage_during_warm_up_is_survivable() -> None:
    """A pool that cannot be filled must leave a worker that still speaks — cold, but alive."""
    fake = _FakeCartesia(fail_after=0)
    s = _synth(fake)

    await s.warm_up(pool_size=2)

    assert len(s._warm_connections) == 0


@pytest.mark.asyncio
async def test_refill_stops_rather_than_spinning_when_the_provider_is_down() -> None:
    fake = _FakeCartesia(fail_after=1)
    s = _synth(fake)
    await s.warm_up(pool_size=2)

    await s._refill_warm_connections()

    # One successful dial in warm_up, then refusal — and the loop returned instead of looping.
    assert fake.dials == 1


@pytest.mark.asyncio
async def test_close_drains_the_pool() -> None:
    fake = _FakeCartesia()
    s = _synth(fake)
    await s.warm_up(pool_size=2)

    await s.close()

    assert all(connection.closed for connection in fake.connections)
    assert len(s._warm_connections) == 0


@pytest.mark.asyncio
async def test_close_stops_the_refill_before_draining() -> None:
    """Otherwise the refill races shutdown and reopens sockets nobody will ever close."""
    fake = _FakeCartesia()
    s = _synth(fake)
    await s.warm_up(pool_size=2)
    await s.open_prosody_context(context_id="c1", language="vi")

    await s.close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert s._warm_target == 0
    assert len(s._warm_connections) == 0
