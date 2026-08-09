"""A dub that breaks mid-sentence resumes; it does not start over.

capture_frame() fails sporadically with InvalidState. The publisher used to answer that by
replaying the whole line on a fresh connection, so a failure nine tenths of the way through
meant the listener heard nine tenths of the line and then the entire line again. These tests
pin the two halves of the fix: nothing is spoken twice, and nothing is dropped.
"""

from typing import Any, cast

import pytest
from livekit import rtc

from tts_worker.livekit_publisher import LiveKitTTSPublisher

SAMPLE_RATE = 48000


class FakeSource:
    """Captures frames, and fails once at a chosen frame to simulate InvalidState."""

    def __init__(self, fail_at_frame: int | None = None) -> None:
        self.frames: list[bytes] = []
        self.fail_at_frame = fail_at_frame
        self._seen = 0

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        if self.fail_at_frame is not None and self._seen == self.fail_at_frame:
            self._seen += 1
            raise RuntimeError("InvalidState")
        self._seen += 1
        self.frames.append(bytes(frame.data))

    @property
    def captured_bytes(self) -> bytes:
        return b"".join(self.frames)


def publisher() -> LiveKitTTSPublisher:
    return LiveKitTTSPublisher.__new__(LiveKitTTSPublisher)


def as_source(fake: FakeSource) -> rtc.AudioSource:
    """The fake is the point of these tests — it fails on demand, which a real AudioSource
    cannot be made to do. Cast once here rather than at every call site."""
    return cast(rtc.AudioSource, cast(Any, fake))


def pcm(frames: int) -> bytes:
    frame_bytes = int(SAMPLE_RATE * 10 / 1000) * 2
    return bytes(range(256)) * ((frames * frame_bytes) // 256 + 1)


@pytest.mark.asyncio
async def test_a_clean_run_reports_the_whole_buffer() -> None:
    audio = pcm(5)
    source = FakeSource()
    sent = await publisher()._capture_from(as_source(source), audio, SAMPLE_RATE)
    assert sent == len(audio)


@pytest.mark.asyncio
async def test_a_break_reports_where_it_stopped_not_failure() -> None:
    # The old answer was False, which told the caller nothing about how much had already
    # been heard — which is exactly what it needed to know to avoid repeating it.
    audio = pcm(10)
    source = FakeSource(fail_at_frame=4)
    sent = await publisher()._capture_from(as_source(source), audio, SAMPLE_RATE)

    assert 0 < sent < len(audio)
    assert sent == len(source.captured_bytes)


@pytest.mark.asyncio
async def test_resuming_speaks_the_line_once_and_whole() -> None:
    audio = pcm(10)

    first = FakeSource(fail_at_frame=4)
    sent = await publisher()._capture_from(as_source(first), audio, SAMPLE_RATE)

    second = FakeSource()
    sent += await publisher()._capture_from(as_source(second), audio[sent:], SAMPLE_RATE)

    heard = first.captured_bytes + second.captured_bytes
    frame_bytes = int(SAMPLE_RATE * 10 / 1000) * 2
    usable = len(audio) - (len(audio) % frame_bytes)

    # Every captured byte, in order, exactly once — no gap at the break, no repeat before it.
    assert heard == audio[:usable]
    assert sent == len(audio)
