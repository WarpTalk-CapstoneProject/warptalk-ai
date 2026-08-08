"""The speech gate must favour recall over precision, and stay tunable per deployment.

This threshold has been moved in both directions. Too low and distant or ambiguous sound
reaches an STT model with no confidence signal of its own, which hallucinates it into a
fluent sentence. Too high and the speaker has to lean into the microphone to register at
all — which is what a room-distance test reported, with most of what was said never
appearing in the transcript.

For this product the second failure is worse: an imperfect sentence can be read and
corrected, a missing one cannot, and the speaker has no way to tell it happened.
"""

from __future__ import annotations

from shared.config import WorkerSettings


def test_the_gate_is_open_enough_for_someone_across_a_room() -> None:
    settings = WorkerSettings()
    assert settings.vad_threshold <= 0.4, (
        f"0.5 made a speaker at any distance shout to be transcribed; got {settings.vad_threshold}"
    )


def test_it_has_not_gone_back_to_the_value_that_hallucinated() -> None:
    settings = WorkerSettings()
    assert settings.vad_threshold > 0.3, (
        "0.3 let ambiguous noise through and STT invented sentences from it; "
        f"got {settings.vad_threshold}"
    )


def test_a_deployment_can_tune_it_without_a_rebuild(monkeypatch) -> None:
    # A close-mic studio should be able to raise it and a hall to lower it, without this
    # default having to move a third time.
    monkeypatch.setenv("VAD_THRESHOLD", "0.62")
    assert WorkerSettings().vad_threshold == 0.62


def test_word_onsets_and_final_syllables_are_still_padded() -> None:
    # Lowering the gate must not be paired with trimming the padding — a production replay
    # once cut "Kubernetes" to "Kuber" on a shorter hangover.
    settings = WorkerSettings()
    assert settings.vad_pre_speech_ms >= 192
    assert settings.vad_silence_hangover_ms >= 576
