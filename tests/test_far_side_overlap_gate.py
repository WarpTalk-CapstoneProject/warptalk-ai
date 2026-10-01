"""The ingress time gate: zero the stand-in's copy of a WarpTalk participant's own voice.

A WarpTalk user who is also in the Meet reaches ingress twice — on their own mic, and a few
hundred ms later inside the stand-in's mixed Meet feed. Every stand-in frame whose arrival,
shifted back by the configurable lag window, overlaps a participant's VAD speech is zeroed; the
rest of the stand-in turn is left alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from livekit_ingress_worker.far_side_gate import (
    FRAME_S,
    FarSideGateConfig,
    FarSideOverlapGate,
    lagged_overlap_ms,
    zero_frames,
)
from livekit_ingress_worker.worker import LiveKitIngressWorker, _frames_to_ms
from shared.config import WorkerSettings

ROOM = "room-1"
STAND_IN = "00000000-0000-0000-0000-00000000b21d"


class TestLaggedOverlap:
    def test_speech_inside_the_lag_window_overlaps(self) -> None:
        # Human spoke 10.0-11.0s. A stand-in frame at 10.4-10.432 carries what arrived during
        # [10.4-0.6, 10.432-0.3] = [9.8, 10.132] -> 132ms of that speech.
        assert lagged_overlap_ms(10.4, 10.432, [(10.0, 11.0)], 300, 600) == pytest.approx(132)

    def test_speech_closer_than_lag_min_does_not_overlap(self) -> None:
        # Frame at 10.0-10.032 looks at [9.4, 9.732]; speech only began at 9.8.
        assert lagged_overlap_ms(10.0, 10.032, [(9.8, 11.0)], 300, 600) == 0.0

    def test_speech_older_than_lag_max_does_not_overlap(self) -> None:
        # Frame at 12.0 looks at [11.4, 11.732]; speech ended at 11.0.
        assert lagged_overlap_ms(12.0, 12.032, [(10.0, 11.0)], 300, 600) == 0.0

    def test_partial_overlap_is_measured_not_rounded_up(self) -> None:
        # Looks at [9.4, 9.732]; speech covers only its last 32ms.
        assert lagged_overlap_ms(10.0, 10.032, [(9.7, 9.9)], 300, 600) == pytest.approx(32)

    def test_multiple_intervals_add(self) -> None:
        overlap = lagged_overlap_ms(10.0, 10.032, [(9.4, 9.45), (9.6, 9.65)], 300, 600)
        assert overlap == pytest.approx(100)

    def test_zero_lag_window_is_a_plain_overlap(self) -> None:
        assert lagged_overlap_ms(10.0, 10.1, [(10.05, 10.2)], 0, 0) == pytest.approx(50)


class TestSuppressionMask:
    def _gate(self, **overrides: Any) -> FarSideOverlapGate:
        return FarSideOverlapGate(FarSideGateConfig(**overrides), clock=lambda: 0.0)

    def test_no_other_human_means_nothing_is_suppressed(self) -> None:
        gate = self._gate()
        assert gate.suppression_mask(ROOM, 10.0, 3, exclude=(STAND_IN,)) == [False] * 3

    def test_frames_after_lag_are_suppressed_and_only_those(self) -> None:
        gate = self._gate(lag_min_ms=300, lag_max_ms=600)
        gate.note_speech(ROOM, "alice", 10.0, 10.2)
        # Speech arrived 10.0-10.2, so its copy can arrive 10.3-10.8. A window ending at 10.33
        # holds frames ending 10.266, 10.298 and 10.33: the first two look at ranges ending
        # before 10.0 (frame end - lag_min), the last one reaches 30ms into the speech.
        assert gate.suppression_mask(ROOM, 10.33, 3) == [False, False, True]
        # Well inside the echo span: everything.
        assert gate.suppression_mask(ROOM, 10.6, 3) == [True, True, True]
        # Past lag_max after the speech ended: nothing.
        assert gate.suppression_mask(ROOM, 10.9, 3) == [False, False, False]

    def test_the_stand_in_never_gates_itself(self) -> None:
        gate = self._gate()
        gate.note_speech(ROOM, STAND_IN, 10.0, 11.0)
        assert gate.suppression_mask(ROOM, 10.6, 3, exclude=(STAND_IN,)) == [False] * 3

    def test_other_rooms_do_not_leak(self) -> None:
        gate = self._gate()
        gate.note_speech("other-room", "alice", 10.0, 11.0)
        assert gate.suppression_mask(ROOM, 10.6, 3) == [False] * 3

    def test_min_overlap_threshold(self) -> None:
        gate = self._gate(lag_min_ms=300, lag_max_ms=600, min_overlap_ms=20)
        gate.note_speech(ROOM, "alice", 10.0, 10.01)  # 10ms of speech
        assert not any(gate.suppression_mask(ROOM, 10.5, 3))
        gate.note_speech(ROOM, "bob", 10.05, 10.1)  # 50ms
        assert any(gate.suppression_mask(ROOM, 10.5, 3))

    def test_disabled_gate_suppresses_nothing(self) -> None:
        gate = self._gate(enabled=False)
        gate.note_speech(ROOM, "alice", 10.0, 11.0)
        assert gate.suppression_mask(ROOM, 10.6, 3) == [False] * 3

    def test_note_frames_records_only_speech_frames(self) -> None:
        gate = self._gate(lag_min_ms=0, lag_max_ms=0)
        # Window ending at 10.096: frames 10.0-10.032 (speech), 10.032-10.064 (silence),
        # 10.064-10.096 (speech).
        gate.note_frames(ROOM, "alice", 10.096, [0.9, 0.1, 0.8], threshold=0.5)
        assert gate.suppression_mask(ROOM, 10.096, 3) == [True, False, True]

    def test_contiguous_frames_merge_and_history_is_pruned(self) -> None:
        gate = self._gate(history_ms=1000)
        for i in range(10):
            gate.note_speech(ROOM, "alice", 10.0 + i * FRAME_S, 10.0 + (i + 1) * FRAME_S)
        assert len(gate._speech[ROOM]["alice"]) == 1
        gate.note_speech(ROOM, "alice", 20.0, 20.032)
        assert list(gate._speech[ROOM]["alice"]) == [(20.0, 20.032)]

    def test_forget_speaker_and_room(self) -> None:
        gate = self._gate()
        gate.note_speech(ROOM, "alice", 10.0, 11.0)
        gate.forget_speaker(ROOM, "alice")
        assert not any(gate.suppression_mask(ROOM, 10.6, 3))
        gate.note_speech(ROOM, "alice", 10.0, 11.0)
        gate.forget_room(ROOM)
        assert not any(gate.suppression_mask(ROOM, 10.6, 3))


class TestConfigAndZeroing:
    def test_config_from_settings_defaults(self) -> None:
        config = FarSideGateConfig.from_settings(WorkerSettings())
        assert config.enabled is True
        assert (config.lag_min_ms, config.lag_max_ms) == (300, 600)
        assert config.history_ms >= config.lag_max_ms + 1000

    def test_config_swaps_an_inverted_window(self) -> None:
        settings = WorkerSettings(far_side_gate_lag_min_ms=700, far_side_gate_lag_max_ms=200)
        config = FarSideGateConfig.from_settings(settings)
        assert (config.lag_min_ms, config.lag_max_ms) == (200, 700)

    def test_zero_frames_only_touches_masked_frames(self) -> None:
        frame_bytes = 512 * 2
        pcm = (np.ones(512 * 3, dtype=np.int16) * 1000).tobytes()
        out = np.frombuffer(zero_frames(pcm, [False, True, False], frame_bytes), dtype=np.int16)
        assert out[:512].tolist() == [1000] * 512
        assert out[512:1024].tolist() == [0] * 512
        assert out[1024:].tolist() == [1000] * 512

    def test_zero_frames_without_mask_is_identity(self) -> None:
        pcm = b"\x01\x02" * 1536
        assert zero_frames(pcm, [False, False, False], 1024) is pcm

    def test_frames_to_ms(self) -> None:
        assert _frames_to_ms(3) == 96


class TestWorkerWiring:
    def test_worker_built_with_new_gets_a_gate_lazily(self) -> None:
        worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
        worker.settings = WorkerSettings()
        gate = worker._far_side_overlap_gate()
        assert gate is not None
        assert worker._far_side_overlap_gate() is gate

    def test_disabled_by_env_returns_none(self) -> None:
        worker = LiveKitIngressWorker.__new__(LiveKitIngressWorker)
        worker.settings = WorkerSettings(far_side_gate_enabled=False)
        assert worker._far_side_overlap_gate() is None
