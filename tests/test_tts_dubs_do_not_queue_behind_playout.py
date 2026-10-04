"""Overlapping speakers must not wait for each other's dubs to finish PLAYING.

Production (1 Oct 2026): dubs ran p50 ~2s / p90 ~5s behind, the worst room p50 8.2s and max 24s.
Two things made a dub wait for an unrelated one to be heard, not merely generated:

  * The Cartesia slot (2 per process, the plan's limit) was held around the whole streamed
    sentence, and a streamed sentence does not finish until it has PLAYED — the track
    back-pressures to real time. A third speaker, language or voice waited out somebody else's
    dub before Cartesia was even asked.
  * The consume loop read 8 messages and waited for all 8 before reading again, so a new
    sentence from speaker B sat unread behind speaker A's whole batch, playout included.

The slot now bounds what Cartesia counts — a one-shot request in flight, or a context from its
first push until its `done` — and never playout; with nothing queued behind it, a sentence's
context is ended at its flush_done (see test_tts_context_closes_on_drain.py). The reader keeps
reading while earlier messages play. Per-key order — the reason the per-key lock exists — must
survive both.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.config import TTSSettings, WorkerSettings
from shared.redis_client import RedisStreamClient
from shared.schemas import TranslationResultMessage
from tts_worker.synthesizer import GenerationLease
from tts_worker.worker import TTSWorker

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Turn:
    def __init__(self, log: list[str], speaker: str) -> None:
        self._log = log
        self._speaker = speaker
        self.is_closed = False

    async def speak(self, text: str, generation_config: Any = None, on_pcm: Any = None) -> Any:
        self._log.append(f"generated:{self._speaker}")
        pcm = b"\x01\x02" * 100
        if on_pcm is not None:
            await on_pcm(pcm)
        return b"\x00" * 44 + pcm, 12

    async def aclose(self) -> None:
        self.is_closed = True

    async def abandon(self) -> None:
        self.is_closed = True


class _Connection:
    async def close(self) -> None:
        return None


class _Synth:
    """One slot only — stricter than production's three, so a slot held across playout shows."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._slot = asyncio.Semaphore(1)

    def generation_slot(self) -> asyncio.Semaphore:
        return self._slot

    async def open_prosody_context(
        self, *, context_id: str, language: str, voice_id: str | None
    ) -> tuple[_Turn, _Connection]:
        return _Turn(self._log, context_id.split(":")[0]), _Connection()

    async def synthesize(self, **kwargs: Any) -> tuple[bytes, int, str]:
        raise AssertionError("the streaming path must not fall back here")


class _Track:
    def __init__(self) -> None:
        self.fed = 0
        self.first_audio_at: float | None = None

    async def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    @property
    def spoken_bytes(self) -> int:
        return self.fed


class _SlowPublisher:
    """Leaving `stream()` waits for the sentence to finish playing — as the real pump does."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self.playing: dict[str, asyncio.Event] = {}

    @asynccontextmanager
    async def stream(self, meeting_id: str, speaker_id: str, *args: Any, **kwargs: Any) -> Any:
        track = _Track()
        finished = self.playing.setdefault(speaker_id, asyncio.Event())
        try:
            yield track
        finally:
            self._log.append(f"playing:{speaker_id}")
            await finished.wait()
            self._log.append(f"played:{speaker_id}")

    def retire_voice_variants(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def publish_pcm(self, *args: Any, **kwargs: Any) -> None:
        return None


def _msg(speaker: str, lang: str = "vi") -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id=f"seg-{speaker}",
        meeting_id="m1",
        speaker_id=speaker,
        original_text="src",
        translated_text=f"sentence from {speaker}",
        source_lang="en",
        target_lang=lang,
    )


def _streaming_worker(redis: RedisStreamClient) -> tuple[TTSWorker, list[str], _SlowPublisher]:
    log: list[str] = []
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(
        prosody_continuity=True, stream_to_livekit=True, cache_enabled=False
    )
    worker.logger = MagicMock()
    worker.redis = redis
    worker._contexts = {}
    publisher = _SlowPublisher(log)
    worker.livekit_publisher = publisher  # type: ignore[assignment]
    worker.cartesia = _Synth(log)  # type: ignore[assignment]
    return worker, log, publisher


async def _until(predicate: Any, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# (a) The slot bounds generation, not playout
# ---------------------------------------------------------------------------


async def test_a_second_track_generates_while_the_first_is_still_playing(
    mock_redis_client: RedisStreamClient,
) -> None:
    worker, log, publisher = _streaming_worker(mock_redis_client)

    first = asyncio.create_task(
        worker._synthesize_and_publish(_msg("s1"), "one", "voice-1", "default", "")
    )
    await _until(lambda: "playing:s1" in log)

    # Speaker 1's sentence is generated and still PLAYING. With one slot, a slot held across
    # playout would keep speaker 2 from reaching Cartesia at all until speaker 1 is heard.
    second = asyncio.create_task(
        worker._synthesize_and_publish(_msg("s2", "ja"), "two", "voice-2", "default", "")
    )
    await _until(lambda: "generated:s2" in log)
    assert "played:s1" not in log

    publisher.playing["s1"].set()
    publisher.playing.setdefault("s2", asyncio.Event()).set()
    await asyncio.gather(first, second)

    assert log.index("generated:s2") < log.index("played:s1")


async def test_the_slot_is_free_while_a_sentence_plays_and_taken_while_it_generates(
    mock_redis_client: RedisStreamClient,
) -> None:
    worker, log, publisher = _streaming_worker(mock_redis_client)
    slot = worker._require_cartesia().generation_slot()
    held_during_generation: list[bool] = []

    original_open = worker._require_cartesia().open_prosody_context

    async def observing_open(**kwargs: Any) -> Any:
        turn, connection = await original_open(**kwargs)
        original_speak = turn.speak

        async def speak(*args: Any, **inner: Any) -> Any:
            held_during_generation.append(slot.locked())
            return await original_speak(*args, **inner)

        turn.speak = speak  # type: ignore[method-assign]
        return turn, connection

    worker._require_cartesia().open_prosody_context = observing_open  # type: ignore[method-assign]

    task = asyncio.create_task(
        worker._synthesize_and_publish(_msg("s1"), "one", "voice-1", "default", "")
    )
    await _until(lambda: "playing:s1" in log)

    assert held_during_generation == [True]
    # Back before the sentence finishes playing: nothing is queued behind it, so its context is
    # ended at flush_done and the slot follows once the close is confirmed — off this path.
    await _until(lambda: not slot.locked())
    assert "played:s1" not in log, "the slot must be back before the sentence finishes playing"

    publisher.playing["s1"].set()
    await task
    assert not slot.locked()


async def test_a_cancelled_sentence_gives_its_slot_back(
    mock_redis_client: RedisStreamClient,
) -> None:
    """The processing timeout cancels a wedged attempt mid-generation; that must not leak a
    slot, or two wedged sentences would silence the whole process."""
    worker, log, _publisher = _streaming_worker(mock_redis_client)
    slot = worker._require_cartesia().generation_slot()
    wedged = asyncio.Event()

    async def open_wedged(**kwargs: Any) -> Any:
        turn = _Turn(log, "s1")

        async def speak(*args: Any, **inner: Any) -> Any:
            await wedged.wait()

        turn.speak = speak  # type: ignore[method-assign]
        return turn, _Connection()

    worker._require_cartesia().open_prosody_context = open_wedged  # type: ignore[method-assign]
    task = asyncio.create_task(
        worker._synthesize_and_publish(_msg("s1"), "one", "voice-1", "default", "")
    )
    await _until(slot.locked)
    task.cancel()
    # Back even while whatever had reached the track is still draining.
    await _until(lambda: not slot.locked())
    assert not task.done()

    _publisher.playing.setdefault("s1", asyncio.Event()).set()
    await asyncio.gather(task, return_exceptions=True)
    assert not slot.locked()


async def test_the_lease_is_one_claim_however_often_it_is_taken_or_returned() -> None:
    slot = asyncio.Semaphore(1)
    lease = GenerationLease(slot)
    await lease.acquire()
    await lease.acquire()  # the fallback re-acquiring a lease it still holds: no second claim
    assert slot.locked()
    lease.release()
    lease.release()  # the caller's safety-net release after the early one: not an over-release
    assert not slot.locked()
    await slot.acquire()
    assert slot.locked(), "an over-release would have left room for a second holder"


# ---------------------------------------------------------------------------
# (b) The reader keeps reading while earlier messages play
# ---------------------------------------------------------------------------


def _entry(message_id: bytes, speaker: str, lang: str = "vi") -> tuple[bytes, dict[bytes, bytes]]:
    return message_id, {
        b"meeting_id": b"m1",
        b"speaker_id": speaker.encode(),
        b"target_lang": lang.encode(),
    }


def _scripted_reads(
    client: RedisStreamClient, batches: list[list[tuple[bytes, dict[bytes, bytes]]]]
) -> list[int]:
    """xreadgroup hands out one scripted batch per call, then blocks briefly and returns empty."""
    reads: list[int] = []

    async def xreadgroup(**kwargs: Any) -> Any:
        reads.append(kwargs["count"])
        if batches:
            return [(b"translate:results", batches.pop(0))]
        await asyncio.sleep(0.005)
        return []

    client._redis.xreadgroup = AsyncMock(side_effect=xreadgroup)
    return reads


async def test_the_reader_fetches_a_new_message_while_an_earlier_one_is_still_playing(
    mock_redis_client: RedisStreamClient,
) -> None:
    _scripted_reads(mock_redis_client, [[_entry(b"1-0", "s1")], [_entry(b"2-0", "s2", "ja")]])
    playing = asyncio.Event()
    started: list[bytes] = []
    running = True

    async def handler(message_id: bytes, data: dict[bytes, bytes]) -> None:
        started.append(message_id)
        if message_id == b"1-0":
            await playing.wait()  # speaker 1's dub is still playing

    consume = asyncio.create_task(
        mock_redis_client.consume_pipelined(
            "translate:results",
            "tts-workers",
            handler,
            keep_running=lambda: running,
            block_ms=5,
        )
    )
    # The old loop could not read message 2 until message 1 had finished.
    await _until(lambda: b"2-0" in started)
    assert not playing.is_set()

    playing.set()
    running = False
    await consume

    acked = [call.args[2] for call in mock_redis_client._redis.xack.call_args_list]
    assert sorted(acked) == [b"1-0", b"2-0"]


async def test_a_failed_handler_is_not_acked_and_does_not_stop_the_reader(
    mock_redis_client: RedisStreamClient,
) -> None:
    _scripted_reads(mock_redis_client, [[_entry(b"1-0", "s1")], [_entry(b"2-0", "s2")]])
    running = True
    seen: list[bytes] = []

    async def handler(message_id: bytes, data: dict[bytes, bytes]) -> None:
        seen.append(message_id)
        if message_id == b"1-0":
            raise RuntimeError("boom")

    consume = asyncio.create_task(
        mock_redis_client.consume_pipelined(
            "s", "g", handler, keep_running=lambda: running, block_ms=5
        )
    )
    await _until(lambda: b"2-0" in seen)
    running = False
    await consume

    acked = [call.args[2] for call in mock_redis_client._redis.xack.call_args_list]
    assert acked == [b"2-0"], "a failed message stays pending for reclaim/dead-letter"


async def test_at_capacity_the_reader_waits_instead_of_reading_more(
    mock_redis_client: RedisStreamClient,
) -> None:
    reads = _scripted_reads(
        mock_redis_client,
        [[_entry(b"1-0", "s1"), _entry(b"2-0", "s2")], [_entry(b"3-0", "s3")]],
    )
    release = asyncio.Event()
    in_flight_ids: set[bytes] = set()
    running = True

    async def handler(message_id: bytes, data: dict[bytes, bytes]) -> None:
        await release.wait()

    consume = asyncio.create_task(
        mock_redis_client.consume_pipelined(
            "s",
            "g",
            handler,
            keep_running=lambda: running,
            block_ms=5,
            count=8,
            max_in_flight=2,
            in_flight_ids=in_flight_ids,
        )
    )
    await _until(lambda: in_flight_ids == {b"1-0", b"2-0"})
    await asyncio.sleep(0.05)
    # Full: exactly one read, asking for no more than the capacity.
    assert reads == [2]

    release.set()
    await _until(lambda: len(reads) >= 2)
    running = False
    await consume
    assert in_flight_ids == set()


async def test_reclaim_skips_a_message_that_is_only_queued_here(
    mock_redis_client: RedisStreamClient,
) -> None:
    """A sentence waiting behind its key's playout looks idle to XAUTOCLAIM. Reclaiming it would
    run it twice — or, worse, ack it while the first run is still to come."""
    worker = TTSWorker.__new__(TTSWorker)
    worker.logger = MagicMock()
    worker.redis = mock_redis_client
    worker._consumer_name = "tts-test"
    worker.input_stream = "translate:results"
    worker.consumer_group = "tts-workers"
    worker._key_locks = {}
    worker._in_flight_message_ids().add(b"1-0")
    mock_redis_client._redis.xautoclaim.return_value = [b"0-0", [_entry(b"1-0", "s1")], []]
    worker._process_and_log_errors = AsyncMock()  # type: ignore[method-assign]

    await worker._recover_stale_messages()

    worker._process_and_log_errors.assert_not_awaited()
    mock_redis_client._redis.xack.assert_not_awaited()


# ---------------------------------------------------------------------------
# (c) Per-key order survives the pipelined reader
# ---------------------------------------------------------------------------


async def test_per_key_order_holds_while_other_keys_overtake(
    mock_redis_client: RedisStreamClient,
) -> None:
    _scripted_reads(
        mock_redis_client,
        [
            [_entry(b"1-0", "s1")],
            [_entry(b"2-0", "s2", "ja")],
            [_entry(b"3-0", "s1")],
            [_entry(b"4-0", "s1")],
        ],
    )
    worker = TTSWorker.__new__(TTSWorker)
    worker.logger = MagicMock()
    worker.redis = mock_redis_client
    worker._shutdown_event = asyncio.Event()
    worker._consumer_name = "tts-test"
    worker.input_stream = "translate:results"
    worker.consumer_group = "tts-workers"
    worker._key_locks = {}
    worker._recover_stale_messages = AsyncMock()  # type: ignore[method-assign]

    events: list[tuple[str, bytes]] = []
    first_playing = asyncio.Event()

    async def process(message_id: bytes, data: dict[bytes, bytes]) -> None:
        events.append(("start", message_id))
        if message_id == b"1-0":
            await first_playing.wait()
        else:
            await asyncio.sleep(0.01)
        events.append(("end", message_id))

    worker.process = process  # type: ignore[method-assign]

    loop = asyncio.create_task(worker._consume_loop())
    # Another speaker overtakes speaker 1's long sentence...
    await _until(lambda: ("end", b"2-0") in events)
    # ...but speaker 1's own later sentences wait, even though they have been read.
    await _until(lambda: b"4-0" in worker._in_flight_message_ids())
    assert ("start", b"3-0") not in events

    first_playing.set()
    await _until(lambda: ("end", b"4-0") in events)
    worker._shutdown_event.set()
    await asyncio.wait_for(loop, timeout=2.0)

    speaker_1 = [event for event in events if event[1] in (b"1-0", b"3-0", b"4-0")]
    assert speaker_1 == [
        ("start", b"1-0"),
        ("end", b"1-0"),
        ("start", b"3-0"),
        ("end", b"3-0"),
        ("start", b"4-0"),
        ("end", b"4-0"),
    ]
    acked = [call.args[2] for call in mock_redis_client._redis.xack.call_args_list]
    assert sorted(acked) == [b"1-0", b"2-0", b"3-0", b"4-0"]
