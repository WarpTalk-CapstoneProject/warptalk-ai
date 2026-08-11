from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from redis.exceptions import TimeoutError as RedisTimeoutError

from livekit_ingress_worker.worker import (
    VAD_WINDOW_MS,
    VAD_WINDOW_SAMPLES,
    LiveKitIngressWorker,
)
from shared.config import WorkerSettings


def test_ingress_uses_fast_vad_with_semantic_max_chunk_context() -> None:
    settings = WorkerSettings()

    assert VAD_WINDOW_MS <= 100
    assert VAD_WINDOW_SAMPLES == LiveKitIngressWorker.VAD_FRAME_SIZE * 3
    # Two 96ms windows cut the final syllables from a production LiveKit replay
    # ("Kubernetes" became "Kuber"). Six windows preserve normal Vietnamese
    # intra-utterance pauses while keeping the added flush delay below 600ms.
    # Keep natural Vietnamese micro-pauses inside one utterance so STT receives enough
    # phonetic context instead of a series of ambiguous ~1 second fragments.
    assert settings.vad_silence_hangover_ms == 576
    # Short turns still flush on VAD silence. Only continuous speech reaches this cap;
    # production replay showed short hard cuts losing "review pull request và deploy".
    # Six seconds preserves a natural sentence; VAD still flushes ordinary short turns.
    assert settings.chunk_duration_ms == 6000
    assert settings.vad_min_speech_ms <= 300


async def test_silero_model_uses_an_immutable_release() -> None:
    model = object()
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()

    torch = MagicMock()
    torch.hub.load.return_value = (model, [])
    with patch("livekit_ingress_worker.worker.torch", torch):
        await worker.load_model()

    assert worker._vad_model is model
    torch.hub.load.assert_called_once_with(
        "snakers4/silero-vad:v6.2.1",
        "silero_vad",
        trust_repo=True,
    )


def test_vad_rejects_an_isolated_probability_spike() -> None:
    """One noisy 32ms frame must not classify a ~96ms window as speech."""
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    probabilities = iter([0.99, 0.01, 0.01])
    worker._vad_model = MagicMock(
        side_effect=lambda *_: SimpleNamespace(item=lambda: next(probabilities))
    )
    torch = MagicMock()

    with patch("livekit_ingress_worker.worker.torch", torch):
        score = worker._run_vad_on_window(
            np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32), threshold=0.5
        )

    assert score == 0.0


def test_vad_accepts_sustained_speech_frames() -> None:
    """At least three positive frames (~96ms) retain the strongest speech score."""
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    probabilities = iter([0.91, 0.82, 0.73])
    worker._vad_model = MagicMock(
        side_effect=lambda *_: SimpleNamespace(item=lambda: next(probabilities))
    )
    torch = MagicMock()

    with patch("livekit_ingress_worker.worker.torch", torch):
        score = worker._run_vad_on_window(
            np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32), threshold=0.5
        )

    assert score == 0.91


def test_vad_window_uses_the_track_specific_model_state() -> None:
    """Concurrent tracks must not interleave Silero's recurrent VAD state."""
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    shared_model = MagicMock(
        side_effect=AssertionError("shared model must not process track audio")
    )
    track_probabilities = iter([0.91, 0.82, 0.73])
    track_model = MagicMock(
        side_effect=lambda *_: SimpleNamespace(item=lambda: next(track_probabilities))
    )
    worker._vad_model = shared_model
    torch = MagicMock()

    with patch("livekit_ingress_worker.worker.torch", torch):
        score = worker._run_vad_on_window(
            np.zeros(VAD_WINDOW_SAMPLES, dtype=np.float32),
            threshold=0.5,
            vad_model=track_model,
        )

    assert score == 0.91
    assert track_model.call_count == 3
    shared_model.assert_not_called()


async def test_transient_language_lookup_timeout_does_not_kill_mic_track() -> None:
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker._route_states = {"room-1": "IN_PROGRESS"}
    worker._paused_rooms = set()
    worker.redis = SimpleNamespace(hget=AsyncMock(side_effect=RedisTimeoutError()))
    worker.publish = AsyncMock()

    await worker._publish_speech_chunk(
        "room-1",
        "speaker-1",
        bytearray(np.full(16000, 3000, dtype=np.int16).tobytes()),
        0,
        16000,
    )

    worker.publish.assert_awaited_once()
    worker.logger.warning.assert_any_call(
        "speak_language_lookup_failed",
        room="room-1",
        speaker_id="speaker-1",
        exc_info=True,
    )


async def test_transient_chunk_publish_timeout_retries_without_killing_mic_track() -> None:
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker._route_states = {"room-1": "IN_PROGRESS"}
    worker._paused_rooms = set()
    worker.redis = SimpleNamespace(hget=AsyncMock(return_value=b"vi"))
    worker.publish = AsyncMock(side_effect=[RedisTimeoutError(), None])

    with patch("livekit_ingress_worker.worker.asyncio.sleep", new=AsyncMock()):
        await worker._publish_speech_chunk(
            "room-1",
            "speaker-1",
            bytearray(np.full(16000, 3000, dtype=np.int16).tobytes()),
            0,
            16000,
        )

    assert worker.publish.await_count == 2
    worker.logger.info.assert_any_call(
        "speech_chunk_published",
        room="room-1",
        speaker_id="speaker-1",
        chunk_index=0,
        duration_ms=1000,
        raw_rms=round(3000 / 32768, 6),
        raw_peak=round(3000 / 32768, 6),
    )


async def test_ingress_transcribes_before_translation_is_started() -> None:
    """Transcription must not wait for translation.

    This asserted the opposite until a live meeting produced no transcript at all: the
    gate discarded every chunk until the room reported translation active, a state only
    reached once somebody pressed Start Translation. Reading a published microphone is
    what makes a transcript, and that is a different feature from translating it — the
    translation worker now owns that decision.
    """
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._paused_rooms = set()
    worker.redis = SimpleNamespace(hget=AsyncMock(return_value=b"vi"))
    worker.publish = AsyncMock()

    await worker._publish_speech_chunk(
        "room-1",
        "speaker-1",
        bytearray(np.full(16000, 3000, dtype=np.int16).tobytes()),
        0,
        16000,
    )

    worker.publish.assert_awaited_once()
    assert worker.publish.await_args.args[0] == "audio:chunks"


async def test_ingress_still_discards_speech_while_the_room_is_paused() -> None:
    """A pause is a deliberate "stop listening", and it still means stop."""
    worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
    worker.logger = MagicMock()
    worker._route_states = {"room-1": "PAUSED"}
    worker._paused_rooms = {"room-1"}
    worker.redis = SimpleNamespace(hget=AsyncMock(return_value=b"vi"))
    worker.publish = AsyncMock()

    await worker._publish_speech_chunk(
        "room-1",
        "speaker-1",
        bytearray(np.full(16000, 3000, dtype=np.int16).tobytes()),
        0,
        16000,
    )

    worker.redis.hget.assert_not_awaited()
    worker.publish.assert_not_awaited()
