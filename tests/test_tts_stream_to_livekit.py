"""Audio reaches the listener while Cartesia is still generating it — WT-397.

`publish_pcm` takes a finished sentence, so the first sample only ever left for the room after
the last one had been synthesized. `TrackStream` hands each chunk on as it lands, which means
three things that were free when the buffer was whole now have to be arranged deliberately:

    * a Cartesia chunk is not a whole 20ms frame, and `_capture_from` DROPS the trailing
      partial. Per chunk that is up to one frame of audio thrown away, silently, dozens of
      times per sentence.
    * the 8ms anti-click fade belongs to the two ends of the SENTENCE. Applied per chunk it
      becomes a tremolo.
    * `capture_frame` back-pressures to real time, and it is being called from inside
      ProsodyContext's 6-second sentence timeout.

Each of those is one test below.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import numpy as np
import pytest
from livekit import rtc

from tts_worker.livekit_publisher import FRAME_MS, LiveKitTTSPublisher, TrackStream

SAMPLE_RATE = 16000
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 640
FADE_SAMPLES = int(SAMPLE_RATE * 8 / 1000)  # 128
KEY = ("m1", "s1", "vi", "")


class FakeSource:
    """Records frames; can fail at a chosen frame, or stall to simulate back-pressure."""

    def __init__(self, fail_at_frame: int | None = None, delay_s: float = 0.0) -> None:
        self.frames: list[bytes] = []
        self.fail_at_frame = fail_at_frame
        self.delay_s = delay_s
        self._seen = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail_at_frame is not None and self._seen == self.fail_at_frame:
            self._seen += 1
            raise RuntimeError("InvalidState")
        self._seen += 1
        self.frames.append(bytes(frame.data))

    @property
    def heard(self) -> bytes:
        return b"".join(self.frames)


class FakeRoom:
    async def disconnect(self) -> None:
        return None


class _Publisher(LiveKitTTSPublisher):
    """The real capture/retry machinery, with the LiveKit connection replaced.

    Subclassed rather than mocked so `_capture_from` — the method that decides what a partial
    frame means — is the production one.
    """

    def __init__(self, *sources: FakeSource) -> None:
        self._bots: dict[Any, Any] = {}
        self._locks: dict[Any, asyncio.Lock] = {}
        self._reaper = None
        self._sources = list(sources)
        self.connects = 0

    async def _get_or_create_bot(
        self, meeting_id: str, speaker_id: str, target_lang: str, voice_key: str, sample_rate: int
    ) -> dict[str, Any]:
        key = (meeting_id, speaker_id, target_lang, voice_key)
        cached = self._bots.get(key)
        if cached is not None:
            return cached
        if not self._sources:
            raise RuntimeError("livekit unreachable")
        self.connects += 1
        bot = {"room": FakeRoom(), "source": self._sources.pop(0), "last_used": 0.0}
        self._bots[key] = bot
        return bot


def _track(publisher: LiveKitTTSPublisher) -> TrackStream:
    return TrackStream(publisher, cast(Any, KEY), SAMPLE_RATE)


def _tone(samples: int, amplitude: int = 1000) -> bytes:
    return np.full(samples, amplitude, dtype=np.int16).tobytes()


def _chunks(pcm: bytes, size: int) -> list[bytes]:
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]


# ── the partial-frame carry ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_audio_is_lost_between_chunks() -> None:
    """The failure that would be inaudible-but-real: a shortened, clicking dub.

    Cartesia chunks are not frame multiples. Handing each one straight to `_capture_from`
    discards its trailing partial frame — here that would be ~24 separate drops in one
    sentence. The stream must carry the remainder across the boundary and lose only the one
    partial frame at the very end, exactly as the whole-buffer path does.
    """
    source = FakeSource()
    publisher = _Publisher(source)
    stream = _track(publisher)
    audio = _tone(4000)  # 8000 bytes = 12.5 frames

    stream.start()
    for chunk in _chunks(audio, 331):  # deliberately coprime with the 640-byte frame
        await stream.feed(chunk)
    await stream.close()

    assert len(source.heard) == len(audio) - (len(audio) % FRAME_BYTES)
    assert len(source.heard) == 7680, (
        "the chunk remainders were dropped instead of being carried to the next frame"
    )


@pytest.mark.asyncio
async def test_a_sentence_shorter_than_one_frame_is_still_spoken() -> None:
    # Nothing ever reaches the non-final drain, so both fades and the only capture happen at
    # close. Under a frame there is nothing capture_frame can take, but the stream must not
    # deadlock or report audio that never played.
    source = FakeSource()
    stream = _track(_Publisher(source))

    stream.start()
    await stream.feed(_tone(100))
    await stream.close()

    assert source.heard == b""


# ── the fade ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_sentence_is_faded_at_its_two_ends_not_at_every_chunk() -> None:
    """A fade per chunk is a tremolo.

    The 8ms ramp exists because each sentence starts and ends at whatever amplitude Cartesia
    happened to render, and splicing those cuts onto a continuous track clicks. That is a
    property of the sentence's edges. Streaming would otherwise apply it to each of the ~24
    slices this sentence is delivered in, modulating the audio all the way through.
    """
    source = FakeSource()
    stream = _track(_Publisher(source))
    audio = _tone(4000, amplitude=1000)

    stream.start()
    for chunk in _chunks(audio, 331):
        await stream.feed(chunk)
    await stream.close()

    heard = np.frombuffer(source.heard, dtype=np.int16)
    attenuated = int(np.count_nonzero(heard != 1000))

    assert 0 < attenuated <= 2 * FADE_SAMPLES, (
        f"{attenuated} attenuated samples — a ramp was applied per chunk, not per sentence"
    )
    assert heard[0] == 0, "no fade-in at the start of the sentence"
    assert abs(int(heard[-1])) < 1000, "no fade-out at the end of the sentence"
    assert np.all(heard[FADE_SAMPLES:-FADE_SAMPLES] == 1000), "the middle of the dub was ramped"


# ── back-pressure ───────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_feeding_never_waits_for_the_track() -> None:
    """Why there is a queue rather than a direct await.

    `feed` is called from inside ProsodyContext's 6-second sentence timeout, and capture_frame
    returns only once the track has room — roughly real time. Awaiting the track there would
    make every dub longer than 6 seconds time out, drop to the one-shot fallback, and be spoken
    twice. Feeding a whole sentence must cost ~nothing regardless of how slow the track is.
    """
    source = FakeSource(delay_s=0.05)  # 50ms per 20ms frame — slower than real time
    stream = _track(_Publisher(source))
    audio = _tone(4000)

    stream.start()
    started = asyncio.get_running_loop().time()
    for chunk in _chunks(audio, 331):
        await stream.feed(chunk)
    fed_in = asyncio.get_running_loop().time() - started

    assert fed_in < 0.2, f"feeding blocked on the track for {fed_in:.2f}s"

    await stream.close()
    assert len(source.heard) == 7680, "the pump did not finish the sentence"


@pytest.mark.asyncio
async def test_close_gives_up_on_a_wedged_track_instead_of_holding_the_lock_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # close() runs while this key's publisher lock AND tts_worker's per-(speaker, language)
    # lock are held. An unbounded wait here stops that speaker's dub for the rest of the
    # meeting — the same failure SENTENCE_TIMEOUT_SECONDS exists to prevent, one layer down.
    monkeypatch.setattr("tts_worker.livekit_publisher._DRAIN_TIMEOUT_S", 0.05)
    source = FakeSource(delay_s=3600)
    stream = _track(_Publisher(source))

    stream.start()
    await stream.feed(_tone(4000))
    await asyncio.wait_for(stream.close(), timeout=2.0)


# ── failures, and what the caller is told about them ────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_track_that_never_connects_reports_nothing_spoken() -> None:
    """`spoken_bytes == 0` is what tells the worker the one-shot fallback is still safe to
    play. Reporting anything else here would turn a failed connection into silence."""
    stream = _track(_Publisher())  # no sources: every connect raises

    stream.start()
    await stream.feed(_tone(4000))
    await stream.close()

    assert stream.spoken_bytes == 0


@pytest.mark.asyncio
async def test_a_capture_failure_resumes_on_a_fresh_bot_without_repeating() -> None:
    # Same guarantee test_tts_publish_resume.py pins for the whole-buffer path: InvalidState
    # costs the frames it broke on, not the sentence so far.
    first = FakeSource(fail_at_frame=2)
    second = FakeSource()
    publisher = _Publisher(first, second)
    stream = _track(publisher)

    stream.start()
    await stream.feed(_tone(4000))
    await stream.close()

    assert publisher.connects == 2, "the broken bot was reused"
    assert first.heard and second.heard
    assert stream.spoken_bytes > 0
