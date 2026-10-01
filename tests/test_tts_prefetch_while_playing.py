"""A speaker's next sentence is generated while the previous one is still playing (prefetch).

Without it, tts_worker held a key's lock until the sentence had been HEARD, so the next sentence
of the same speaker and language could not even ask Cartesia for audio until then. Every
sentence queued behind another therefore paid Cartesia's whole time-to-first-audio again after
the previous one had ended: an audible gap per sentence, and a backlog that grew through a long
turn (prod: tts_first_audio p50 0.6 s, p90 1.8-3.7 s; chunk-publish -> TTS start p90 7-10 s).

`LiveKitTTSPublisher.stream_ahead` returns once generation is done and queues the playout behind
the previous one. What must NOT change, and what these pin:
  * a track still plays its lines strictly in order, never interleaved;
  * a one-shot publish for the same track queues behind a prefetched line, not in front of it;
  * a sentence the track could not speak at all is still heard (the fallback rule), and a
    partially spoken one is not repeated;
  * `tts_first_audio` keeps measuring what a listener waits through.
Behind TTS_PREFETCH_WHILE_PLAYING, off by default.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from shared.config import TTSSettings, WorkerSettings
from shared.redis_client import RedisStreamClient
from shared.schemas import TranslationResultMessage
from tests.test_tts_stream_to_livekit import SAMPLE_RATE, FakeSource, _Publisher, _tone
from tts_worker.worker import TTSWorker

M, S, LANG = "m1", "s1", "vi"


def _runs(source: FakeSource) -> list[int]:
    """The sequence of distinct tones heard, read from each frame's middle sample."""
    runs: list[int] = []
    for frame in source.frames:
        samples = np.frombuffer(frame, dtype=np.int16)
        level = int(samples[len(samples) // 2])
        if not runs or runs[-1] != level:
            runs.append(level)
    return runs


async def _say(publisher: _Publisher, pcm: bytes) -> Any:
    async with publisher.stream_ahead(M, S, LANG, SAMPLE_RATE) as track:
        await track.feed(pcm)
    return track


async def test_the_next_line_generates_while_the_previous_one_plays_and_plays_after_it() -> None:
    source = FakeSource(delay_s=0.004)  # each 20 ms frame takes 4 ms to "play"
    publisher = _Publisher(source)

    first = await _say(publisher, _tone(8000, amplitude=1000))  # 25 frames, ~100 ms to play
    # Generation is over and the block has returned while the line is still being heard.
    assert len(source.frames) < 25
    second = await _say(publisher, _tone(4000, amplitude=2000))
    assert not first.playout.done(), "the second line had to wait for the first to be heard"

    await asyncio.wait_for(second.playout, timeout=2.0)

    assert first.playout.done()
    assert _runs(source) == [1000, 2000], "lines interleaved or swapped on the track"
    assert second.pump_started_at >= first.pump_started_at


async def test_a_one_shot_publish_queues_behind_a_prefetched_line() -> None:
    source = FakeSource(delay_s=0.004)
    publisher = _Publisher(source)

    queued = await _say(publisher, _tone(8000, amplitude=1000))
    await publisher.publish_pcm(M, S, LANG, _tone(4000, amplitude=3000), SAMPLE_RATE)

    assert queued.playout.done()
    assert _runs(source) == [1000, 3000]


async def test_other_tracks_do_not_queue_behind_this_one() -> None:
    slow, fast = FakeSource(delay_s=0.01), FakeSource()
    publisher = _Publisher(slow, fast)

    async with publisher.stream_ahead(M, S, LANG, SAMPLE_RATE) as busy:
        await busy.feed(_tone(8000, amplitude=1000))
    async with publisher.stream_ahead(M, "s2", LANG, SAMPLE_RATE) as other:
        await other.feed(_tone(4000, amplitude=2000))

    await asyncio.wait_for(other.playout, timeout=2.0)
    assert not busy.playout.done(), "another speaker waited for this speaker's dub"
    await asyncio.wait_for(busy.playout, timeout=2.0)


async def test_a_line_the_track_could_not_speak_at_all_is_still_heard() -> None:
    # Both connections refuse every frame: the streamed copy is lost entirely, so the playout
    # plays the whole sentence on a fresh bot — what the caller's own fallback used to do.
    dead_1, dead_2, alive = (
        FakeSource(fail_at_frame=0),
        FakeSource(fail_at_frame=0),
        FakeSource(),
    )
    publisher = _Publisher(dead_1, dead_2, alive)

    async with publisher.stream_ahead(M, S, LANG, SAMPLE_RATE) as track:
        await track.feed(_tone(4000, amplitude=1000))
        track.fallback_audio = _tone(4000, amplitude=1000)
    await asyncio.wait_for(track.playout, timeout=2.0)

    assert track.spoken_bytes == 0
    assert _runs(alive) == [1000]


async def test_a_line_that_was_partly_heard_is_not_repeated() -> None:
    source = FakeSource()
    publisher = _Publisher(source)

    async with publisher.stream_ahead(M, S, LANG, SAMPLE_RATE) as track:
        await track.feed(_tone(4000, amplitude=1000))
        track.fallback_audio = _tone(4000, amplitude=5000)
    await asyncio.wait_for(track.playout, timeout=2.0)

    assert _runs(source) == [1000]


# ---------------------------------------------------------------------------
# Through the worker
# ---------------------------------------------------------------------------


class _Turn:
    is_closed = False

    async def speak(self, text: str, generation_config: Any = None, on_pcm: Any = None) -> Any:
        pcm = _tone(8000, amplitude=1000 if text == "one" else 2000)
        if on_pcm is not None:
            # In frame-sized pieces, like Cartesia's chunks: one 16 000-byte feed would be
            # captured in a single call and stamp first_audio_at only once all of it had played.
            for start in range(0, len(pcm), 640):
                await on_pcm(pcm[start : start + 640])
        return b"\x00" * 44 + pcm, 500

    async def aclose(self) -> None:
        self.is_closed = True


class _Synth:
    def __init__(self) -> None:
        self._slot = asyncio.Semaphore(2)

    def generation_slot(self) -> asyncio.Semaphore:
        return self._slot

    async def open_prosody_context(self, **_kwargs: Any) -> Any:
        connection = MagicMock()
        connection.close = AsyncMock()
        return _Turn(), connection

    async def synthesize(self, **kwargs: Any) -> Any:
        raise AssertionError("the streaming path must not fall back here")


def _worker(redis: RedisStreamClient, source: FakeSource, *, prefetch: bool) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(
        prosody_continuity=True,
        stream_to_livekit=True,
        cache_enabled=False,
        sample_rate=SAMPLE_RATE,
        prefetch_while_playing=prefetch,
    )
    worker.logger = MagicMock()
    worker.redis = redis
    worker._turns = {}
    worker._turn_connections = {}
    worker._dub_fits = {}
    worker._turn_dub_ms = {}
    worker.livekit_publisher = _Publisher(source)  # type: ignore[assignment]
    worker.cartesia = _Synth()  # type: ignore[assignment]
    worker._publish_result = AsyncMock()  # type: ignore[method-assign]
    return worker


def _msg(text: str) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id=f"seg-{text}",
        meeting_id=M,
        speaker_id=S,
        original_text=text,
        translated_text=text,
        source_lang="en",
        target_lang=LANG,
    )


async def test_with_prefetch_the_worker_is_free_before_the_line_is_heard(
    mock_redis_client: RedisStreamClient,
) -> None:
    source = FakeSource(delay_s=0.004)
    worker = _worker(mock_redis_client, source, prefetch=True)
    recorded: list[tuple[str, int]] = []

    async def record_latency(stage: str, ms: int) -> None:
        recorded.append((stage, ms))

    worker.redis.record_latency = record_latency  # type: ignore[method-assign]

    await worker._synthesize_and_publish(_msg("one"), "one", "voice-1", "default", "")
    assert len(source.frames) < 25, "synthesis waited for the playout"
    await worker._synthesize_and_publish(_msg("two"), "two", "voice-1", "default", "")

    # Neither was published a second time: the track owns both.
    for call in worker._publish_result.await_args_list:  # type: ignore[attr-defined]
        assert call.kwargs["publish_to_livekit"] is False

    async with asyncio.timeout(2.0):
        while sum(1 for stage, _ in recorded if stage == "tts_first_audio") < 2:
            await asyncio.sleep(0.01)
    assert _runs(source) == [1000, 2000]
    # The second line waited for the first to finish PLAYING, which is not a delay a listener
    # experiences: measured from when it got the track, it is near zero, not ~100 ms.
    first_audio = [ms for stage, ms in recorded if stage == "tts_first_audio"]
    assert first_audio[1] < 60, first_audio


async def test_without_prefetch_the_worker_waits_for_the_line_to_be_heard(
    mock_redis_client: RedisStreamClient,
) -> None:
    source = FakeSource(delay_s=0.002)
    worker = _worker(mock_redis_client, source, prefetch=False)

    await worker._synthesize_and_publish(_msg("one"), "one", "voice-1", "default", "")

    assert len(source.frames) == 25
