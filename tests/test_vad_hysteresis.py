"""Leaving speech is a different decision from entering it, and the loop used to make it twice
with the same number.

One threshold governed both, so the moment a trailing syllable faded under 0.35 the hangover
countdown began — while the speaker was still finishing the word. Vietnamese ends a great many
words on an unstressed vowel or a nasal; an unvoiced final consonant carries almost no energy at
all. Those are exactly the frames that fall under an entry bar tuned to keep a speaker across a
room audible, and it is the mechanism behind the measurement already recorded beside
`vad_silence_hangover_ms`: a production replay that cut "Kubernetes" to "Kuber".

The second half of the same change is resolution. Silero decides per 32ms frame; this loop
collapsed three of them into one 96ms verdict and then counted whole 96ms windows of silence, so
the start of a pause was known only to ±96ms and the silent tail inside a majority-speech window
counted for nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

from livekit_ingress_worker.worker import (
    LiveKitIngressWorker,
    _hangover_frames,
    _trailing_silent_frames,
)
from shared.config import WorkerSettings

WORKER_SOURCE = (
    Path(__file__).resolve().parents[1] / "livekit_ingress_worker" / "worker.py"
).read_text(encoding="utf-8")

SAMPLE_RATE = 16000

# A word fading out: still voiced, no longer confident. Under one threshold this is the start of
# a pause; under two it is the speaker finishing a word.
FADING_FINAL_SYLLABLE = [0.30, 0.25, 0.22]


class TestHysteresis:
    def test_a_fading_final_syllable_is_silence_on_the_way_in_and_speech_on_the_way_out(
        self,
    ) -> None:
        settings = WorkerSettings()
        verdict = LiveKitIngressWorker._window_verdict

        assert verdict(FADING_FINAL_SYLLABLE, settings.vad_threshold) == 0.0, (
            "this must not be loud enough to START a turn — that is the entry bar doing its job, "
            "and lowering it is what let ambiguous noise hallucinate into sentences"
        )
        assert verdict(FADING_FINAL_SYLLABLE, settings.vad_release_threshold) > 0.0, (
            "the same audio must be enough to STAY in a turn; if it is not, the hangover starts "
            "counting while the speaker is still saying the word"
        )

    def test_the_release_bar_is_below_the_entry_bar(self) -> None:
        settings = WorkerSettings()
        assert settings.vad_release_threshold < settings.vad_threshold, (
            "equal thresholds are no hysteresis at all, and a higher release bar is a turn that "
            "can never close"
        )

    def test_a_release_bar_above_the_entry_bar_cannot_reach_the_loop(self) -> None:
        """Not left to the deployment to get right. VAD_RELEASE_THRESHOLD is overridable, and the
        one setting of it that breaks the product is the one that never closes a turn."""
        assert (
            "release_threshold = min(self.settings.vad_release_threshold, vad_threshold)"
            in WORKER_SOURCE
        ), "the release threshold is used unclamped, so a misconfigured deployment hangs every turn"

    def test_both_halves_of_the_decision_use_the_same_active_threshold(self) -> None:
        """Scoring against the release bar and then comparing against the entry bar would leave
        hysteresis half-applied — and silently, because the window verdict already returns 0.0 for
        anything it rejects."""
        assert (
            "active_threshold = release_threshold if is_speaking else vad_threshold"
            in WORKER_SOURCE
        )
        assert "vad_prob = self._window_verdict(frame_probabilities, active_threshold)" in (
            WORKER_SOURCE
        )
        assert "if vad_prob >= active_threshold:" in WORKER_SOURCE

    def test_a_deployment_can_tune_it_without_a_rebuild(self, monkeypatch) -> None:
        monkeypatch.setenv("VAD_RELEASE_THRESHOLD", "0.28")
        assert WorkerSettings().vad_release_threshold == 0.28


class TestFrameResolution:
    def test_a_hangover_is_whole_silero_frames(self) -> None:
        # 576ms and 864ms are exact; 250ms is not, and rounds UP so a rung never fires early.
        assert _hangover_frames(int(SAMPLE_RATE * 0.576)) == 18
        assert _hangover_frames(int(SAMPLE_RATE * 0.864)) == 27
        assert _hangover_frames(int(SAMPLE_RATE * 0.250)) == 8

    def test_a_hangover_is_never_zero(self) -> None:
        """A zero-frame hangover would close a turn on the first silent frame — 32ms, which is
        inside a plosive."""
        assert _hangover_frames(0) == 1

    def test_the_silent_tail_of_a_speech_window_still_counts(self) -> None:
        """The defect this fixes. A window is speech on a MAJORITY, so it can be speech and still
        end silent — the fade at the end of a word. Restarting the run at zero threw those frames
        away and gave back up to 64ms of hangover every time a pause began mid-window."""
        assert _trailing_silent_frames([0.91, 0.88, 0.10], 0.20) == 1
        assert _trailing_silent_frames([0.91, 0.10, 0.05], 0.20) == 2

    def test_speech_all_the_way_to_the_edge_starts_the_run_at_zero(self) -> None:
        assert _trailing_silent_frames([0.91, 0.88, 0.77], 0.20) == 0

    def test_a_fully_silent_window_contributes_every_frame(self) -> None:
        assert _trailing_silent_frames([0.02, 0.01, 0.03], 0.20) == 3

    def test_the_loop_counts_frames_rather_than_windows(self) -> None:
        assert "silence_frames += len(frame_probabilities)" in WORKER_SOURCE, (
            "a silent window must contribute every frame it holds, not one tick"
        )
        assert "silence_frames = _trailing_silent_frames(" in WORKER_SOURCE, (
            "a speech window restarts the silence run at zero again, discarding its silent tail"
        )
        # Precisely at the site that regressed: the speech branch, right after the buffer and the
        # speech count are advanced. Elsewhere — initialisation, pause/resume, turn close — zero
        # is correct, so a blanket search would pin the wrong thing.
        speech_branch = re.search(
            r"speech_samples \+= len\(window_data\) // 2\n(?:\s*#.*\n)*\s*silence_frames = (.+)",
            WORKER_SOURCE,
        )
        assert speech_branch is not None and "_trailing_silent_frames" in speech_branch.group(1), (
            "the speech branch restarts the silence run at zero again, discarding the silent tail "
            "of a majority-speech window"
        )


class TestTheScorerContractIsUnchanged:
    def test_the_window_verdict_still_rejects_an_isolated_spike(self) -> None:
        """Splitting the scorer in two must not quietly re-litigate MIN_VAD_SPEECH_FRAMES: one
        loud frame in an otherwise quiet window is a door or a keyboard."""
        assert LiveKitIngressWorker._window_verdict([0.97, 0.02, 0.03], 0.35) == 0.0

    def test_it_still_reports_the_strongest_frame(self) -> None:
        assert LiveKitIngressWorker._window_verdict([0.42, 0.05, 0.77], 0.35) == 0.77

    def test_an_empty_window_is_not_speech(self) -> None:
        assert LiveKitIngressWorker._window_verdict([], 0.35) == 0.0
