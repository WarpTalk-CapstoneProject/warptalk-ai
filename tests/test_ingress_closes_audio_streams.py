"""Every audio reader closes its rtc.AudioStream when it ends — the ingress OOM, prod 3 Oct 2026.

A stopped reader (mute, replaced by a republished track, cancelled) left its stream open: the
native side kept delivering 48 kHz frames into a queue that is unbounded by default and that
nobody read any more. Memory rose 300-500 MiB per meeting and never came back, until the pod was
OOMKilled in the middle of a bridge meeting.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import livekit.rtc as rtc
import pytest

import livekit_ingress_worker.worker as worker_module
from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings
from tests.conftest import FakeSharedRedis

ROOM = "01a1010c-62db-7d7a-9a94-3bdbebfe3a3b"
SPEAKER = "019f0d00-0de0-7000-9000-000000000001"


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    worker = LiveKitIngressWorker(settings=settings)
    worker.redis = FakeSharedRedis()
    worker.logger = MagicMock()
    worker._publish_speech_chunk = AsyncMock()  # type: ignore[method-assign]
    worker._vad_model = MagicMock()
    return worker


def _track() -> MagicMock:
    track = MagicMock()
    track.sid = "TR_AjeGxWwytrLWz"
    track.kind = rtc.TrackKind.KIND_AUDIO
    return track


class _Stream:
    def __init__(self, *, ends: bool) -> None:
        self.ends = ends
        self.aclose = AsyncMock()

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> object:
        if self.ends:
            raise StopAsyncIteration
        await asyncio.Event().wait()  # a live track: frames would keep coming
        raise AssertionError("unreachable")


async def _drain(worker: LiveKitIngressWorker) -> None:
    for _ in range(5):
        if worker._event_tasks:
            await asyncio.gather(*list(worker._event_tasks), return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_stream_that_ends_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = _Stream(ends=True)
    factory = MagicMock(return_value=stream)
    monkeypatch.setattr(rtc, "AudioStream", factory)
    worker = _worker()

    await worker.process_audio_track(ROOM, SPEAKER, _track())
    await _drain(worker)

    stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_cancelled_reader_closes_its_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """The case that leaked: mute, replace and teardown all stop a reader by cancelling it."""
    stream = _Stream(ends=False)
    monkeypatch.setattr(rtc, "AudioStream", MagicMock(return_value=stream))
    worker = _worker()

    reader = asyncio.create_task(worker.process_audio_track(ROOM, SPEAKER, _track()))
    await asyncio.sleep(0.01)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    await _drain(worker)

    stream.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_frame_queue_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = MagicMock(return_value=_Stream(ends=True))
    monkeypatch.setattr(rtc, "AudioStream", factory)
    worker = _worker()

    await worker.process_audio_track(ROOM, SPEAKER, _track())
    await _drain(worker)

    assert factory.call_args.kwargs["capacity"] == worker_module._AUDIO_STREAM_CAPACITY_FRAMES
    assert worker_module._AUDIO_STREAM_CAPACITY_FRAMES > 0


@pytest.mark.asyncio
async def test_a_close_that_never_returns_does_not_leak_a_waiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_module, "_AUDIO_STREAM_CLOSE_TIMEOUT_S", 0.01)
    worker = _worker()
    run_task = asyncio.create_task(asyncio.Event().wait())
    stream = MagicMock()
    stream._task = run_task

    async def _hang() -> None:
        await asyncio.Event().wait()

    stream.aclose = _hang

    await worker._close_audio_stream(stream, "TR_x")
    await asyncio.sleep(0)

    assert run_task.cancelled()
    warned = [c.args[0] for c in worker.logger.warning.call_args_list]
    assert "audio_stream_close_timed_out" in warned


@pytest.mark.asyncio
async def test_livekit_resamples_natively_to_what_the_pipeline_uses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48 kHz frames crossed into Python 3x larger than needed and back out through a second
    FFI resample call, per frame per speaker. LiveKit is asked for 16 kHz mono instead."""
    factory = MagicMock(return_value=_Stream(ends=True))
    monkeypatch.setattr(rtc, "AudioStream", factory)
    worker = _worker()

    await worker.process_audio_track(ROOM, SPEAKER, _track())
    await _drain(worker)

    kwargs = factory.call_args.kwargs
    assert kwargs["sample_rate"] == LiveKitIngressWorker.SAMPLE_RATE
    assert kwargs["num_channels"] == 1


@pytest.mark.asyncio
async def test_a_frame_already_at_16k_mono_skips_the_python_resampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = rtc.AudioFrame(
        data=b"\x00\x00" * 160, sample_rate=16000, num_channels=1, samples_per_channel=160
    )

    class _OneFrame(_Stream):
        def __init__(self) -> None:
            super().__init__(ends=True)
            self.sent = False

        async def __anext__(self) -> object:
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            event = MagicMock()
            event.frame = frame
            return event

    monkeypatch.setattr(rtc, "AudioStream", MagicMock(return_value=_OneFrame()))
    resampler = MagicMock()
    monkeypatch.setattr(rtc, "AudioResampler", resampler)
    worker = _worker()

    await worker.process_audio_track(ROOM, SPEAKER, _track())
    await _drain(worker)

    resampler.assert_not_called()
