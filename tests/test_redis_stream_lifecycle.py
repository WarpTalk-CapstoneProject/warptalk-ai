"""Why a production Redis filled up, and what now stops it.

THE INCIDENT (2026-08-14)
    A tester reported that a meeting's transcript "runs for a while then freezes and stops being
    recorded", and that the cloned voice arrived late. Every container was healthy, every
    /health/ready returned 200, and no service logged anything at all.

    Redis was at 716.77 MB of 768 MB with `maxmemory-policy allkeys-lru` and 472 keys already
    evicted. Under that policy a full Redis does not fail a write — it silently deletes whichever
    keys look least recently used. During a meeting those include that meeting's own streams and
    consumer groups, so the pipeline stops mid-session with nothing to log. The same eviction
    empties `tts:cache:*`, which is why the dubbed voice had to be synthesised again and arrived
    late. One cause, both symptoms.

    Three separate defects filled it, and each of these tests pins one of them:

      1. Nothing ever deleted a room's streams. 70 abandoned room streams, 284 MB, oldest
         untouched for 10 days.
      2. `stream_maxlen` bounds a COUNT while the entries are ~113 KB of audio, so 1000 entries
         is a 113 MB ceiling — per stream, per room.
      3. A consumer group that stopped reading pinned the trim floor forever.
         `billing-stt-workers` last read `stt:results` on 2026-08-10 and still held its floor on
         2026-08-14, so four days of entries were untrimmable by anyone.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import RedisSettings
from shared.redis_client import RedisStreamClient

# Applied per-test rather than module-wide: one check below is synchronous.
async_test = pytest.mark.asyncio


def _client(**overrides: object) -> RedisStreamClient:
    settings = RedisSettings(**overrides)  # type: ignore[arg-type]
    client = RedisStreamClient(settings)
    redis = MagicMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    redis.xlen = AsyncMock(return_value=0)
    redis.xtrim = AsyncMock()
    redis.xinfo_groups = AsyncMock(return_value=[])
    redis.xpending = AsyncMock(return_value={"pending": 0})
    redis.expire = AsyncMock()

    pipeline = MagicMock()
    pipeline.xadd = MagicMock()
    pipeline.expire = MagicMock()
    pipeline.execute = AsyncMock(return_value=[b"1-0", True])
    redis.pipeline = MagicMock(return_value=pipeline)

    client._redis = redis
    client._pipeline = pipeline  # type: ignore[attr-defined]
    return client


def _stream_id(seconds_ago: float) -> bytes:
    return f"{int((time.time() - seconds_ago) * 1000)}-0".encode()


# ── 1. A stream that stops being written stops existing ──────────────────────────────────────


@async_test
async def test_publishing_refreshes_the_streams_expiry() -> None:
    client = _client(stream_ttl_seconds=3600)

    await client.publish("audio:chunks:room1", {"k": "v"})

    client._pipeline.expire.assert_called_once_with("audio:chunks:room1", 3600)


@async_test
async def test_the_expiry_rides_with_the_xadd_rather_than_costing_a_round_trip() -> None:
    # This is the hot path — one publish per speech chunk. A separate EXPIRE call would add
    # latency to the pipeline whose latency was already the other half of the report.
    client = _client(stream_ttl_seconds=3600)

    await client.publish("audio:chunks:room1", {"k": "v"})

    client._redis.pipeline.assert_called_once_with(transaction=False)
    client._pipeline.execute.assert_awaited_once()
    client._redis.xadd.assert_not_awaited()
    client._redis.expire.assert_not_awaited()


@async_test
async def test_publish_still_returns_the_message_id_through_the_pipeline() -> None:
    # The id is the first pipeline result, not the EXPIRE's boolean. Returning the wrong element
    # would break every caller that tracks its own message.
    client = _client(stream_ttl_seconds=3600)

    assert await client.publish("audio:chunks:room1", {"k": "v"}) == b"1-0"


@async_test
async def test_a_ttl_of_zero_leaves_streams_permanent_and_skips_the_pipeline() -> None:
    client = _client(stream_ttl_seconds=0)

    await client.publish("audio:chunks:room1", {"k": "v"})

    client._redis.xadd.assert_awaited_once()
    client._redis.pipeline.assert_not_called()


# ── 2. The bound must fit what the entries actually weigh ────────────────────────────────────


def test_the_default_bound_accounts_for_audio_sized_entries() -> None:
    """1000 x ~113 KB is 113 MB for one stream, and production runs a set per room inside a
    768 MB Redis. The number is not arbitrary — it is the ceiling, and it has to be survivable
    when several meetings run at once."""
    maxlen = RedisSettings().stream_maxlen

    assert maxlen <= 250, f"{maxlen} entries of ~113 KB audio is {maxlen * 113 / 1024:.0f} MB"


# ── 3. A dead consumer group must not hold the floor forever ─────────────────────────────────


@async_test
async def test_a_group_that_stopped_reading_days_ago_no_longer_pins_the_trim_floor() -> None:
    client = _client(stream_maxlen=100, stream_group_stale_after_seconds=3600)
    client._redis.xlen = AsyncMock(return_value=1_000)
    client._redis.xinfo_groups = AsyncMock(
        return_value=[
            {"name": b"live-workers", "last-delivered-id": _stream_id(10)},
            # billing-stt-workers, four days behind.
            {"name": b"dead-workers", "last-delivered-id": _stream_id(4 * 86_400)},
        ]
    )
    client._redis.xpending = AsyncMock(return_value={"pending": 0})

    await client.publish("stt:results", {"k": "v"})

    trimmed_to = client._redis.xtrim.await_args.kwargs["minid"]
    assert trimmed_to != _stream_id(4 * 86_400)
    # The floor is the live group's position, so everything the dead group never read is freed.
    assert int(trimmed_to.split(b"-")[0]) > int(time.time() - 60) * 1000


@async_test
async def test_a_merely_slow_group_still_holds_the_floor() -> None:
    # The distinction is the whole point. A consumer that is behind but still advancing must not
    # have its unread entries deleted — that was the original reason for this trim logic.
    client = _client(stream_maxlen=100, stream_group_stale_after_seconds=3600)
    client._redis.xlen = AsyncMock(return_value=1_000)
    slow = _stream_id(120)
    client._redis.xinfo_groups = AsyncMock(
        return_value=[
            {"name": b"fast", "last-delivered-id": _stream_id(1)},
            {"name": b"slow", "last-delivered-id": slow},
        ]
    )
    client._redis.xpending = AsyncMock(return_value={"pending": 0})

    await client.publish("stt:results", {"k": "v"})

    assert client._redis.xtrim.await_args.kwargs["minid"] == slow


@async_test
async def test_when_every_group_is_stale_the_stream_falls_back_to_the_count_bound() -> None:
    # Growing without limit to protect readers that have all stopped reading is how 768 MB
    # filled. With nobody consuming, the count bound is the safer of the two.
    client = _client(stream_maxlen=100, stream_group_stale_after_seconds=3600)
    client._redis.xlen = AsyncMock(return_value=1_000)
    client._redis.xinfo_groups = AsyncMock(
        return_value=[{"name": b"dead", "last-delivered-id": _stream_id(4 * 86_400)}]
    )

    await client.publish("stt:results", {"k": "v"})

    assert client._redis.xtrim.await_args.kwargs["maxlen"] == 100


@async_test
async def test_staleness_can_be_switched_off() -> None:
    client = _client(stream_maxlen=100, stream_group_stale_after_seconds=0)
    client._redis.xlen = AsyncMock(return_value=1_000)
    ancient = _stream_id(4 * 86_400)
    client._redis.xinfo_groups = AsyncMock(
        return_value=[{"name": b"dead", "last-delivered-id": ancient}]
    )
    client._redis.xpending = AsyncMock(return_value={"pending": 0})

    await client.publish("stt:results", {"k": "v"})

    assert client._redis.xtrim.await_args.kwargs["minid"] == ancient


@async_test
async def test_a_group_that_has_never_read_anything_still_blocks_trimming_entirely() -> None:
    """0-0 is not staleness, it is a group that was created moments ago and has not had its
    first read yet. Trimming past it would delete entries it is about to consume — and the
    staleness check must not reclassify it, because a 0-0 id looks infinitely old."""
    client = _client(stream_maxlen=100, stream_group_stale_after_seconds=3600)
    client._redis.xlen = AsyncMock(return_value=1_000)
    client._redis.xinfo_groups = AsyncMock(
        return_value=[{"name": b"brand-new", "last-delivered-id": b"0-0"}]
    )

    await client.publish("stt:results", {"k": "v"})

    client._redis.xtrim.assert_not_awaited()
