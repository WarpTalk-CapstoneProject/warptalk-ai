"""Tests for the livekit_ingress_worker per-track near-field energy gate."""

from __future__ import annotations

import pytest

from livekit_ingress_worker.near_field_gate import NearFieldGate
from shared.config import WorkerSettings


@pytest.fixture
def enabled_settings(worker_settings: WorkerSettings) -> WorkerSettings:
    return worker_settings.model_copy(
        update={
            "near_field_gate_enabled": True,
            "near_field_gate_relative_floor": 0.35,
            "near_field_gate_min_baseline_chunks": 2,
            "near_field_gate_baseline_ema_alpha": 0.3,
        }
    )


def test_disabled_gate_always_accepts(worker_settings: WorkerSettings):
    settings = worker_settings.model_copy(update={"near_field_gate_enabled": False})
    gate = NearFieldGate(settings)

    assert gate.accept(0.01) is True  # would be rejected as "far" once a baseline exists
    assert gate.accept(0.01) is True
    assert gate.accept(0.01) is True


def test_near_field_gate_is_disabled_by_default_to_preserve_quiet_words():
    """Peak amplitude cannot reliably distinguish background speech from a quiet
    syllable in the primary speaker's own turn.

    The production regression dropped the middle of a clearly detected sentence
    after a louder first chunk established the relative baseline. Semantic
    relevance filtering belongs after STT; ingress must preserve recall.
    """
    assert WorkerSettings().near_field_gate_enabled is False


def test_baseline_bootstrap_chunks_are_always_accepted(enabled_settings: WorkerSettings):
    gate = NearFieldGate(enabled_settings)

    assert gate.accept(0.5) is True  # baseline chunk #1
    assert gate.accept(0.5) is True  # baseline chunk #2 — baseline now established


def test_quiet_chunk_is_rejected_after_baseline_established(enabled_settings: WorkerSettings):
    gate = NearFieldGate(enabled_settings)
    gate.accept(0.5)
    gate.accept(0.5)

    # 0.1 is well below 35% of a ~0.5 baseline peak (floor ~0.175) — a distant/muffled voice.
    assert gate.accept(0.1) is False


def test_loud_chunk_is_never_rejected_and_raises_baseline(enabled_settings: WorkerSettings):
    gate = NearFieldGate(enabled_settings)
    gate.accept(0.2)
    gate.accept(0.2)

    # A louder, closer utterance than the initial (possibly wrong/quiet) bootstrap must
    # always be accepted — the gate is one-directional, never blocks "louder than baseline".
    assert gate.accept(0.8) is True

    # The baseline moved up toward 0.8 (EMA, so 0.2 -> 0.38 after this one update) —
    # a chunk far below even that new floor is now correctly treated as far-field.
    assert gate.accept(0.05) is False


def test_clipped_agc_bootstrap_does_not_block_normal_follow_up_speech(
    enabled_settings: WorkerSettings,
):
    """Production capture can clip the first AGC-adjusted chunks near 1.0.

    That transient must not anchor the relative floor so high that ordinary clear
    follow-up speech around 0.22 peak disappears from the transcript.
    """
    gate = NearFieldGate(enabled_settings)
    assert gate.accept(0.92) is True
    assert gate.accept(0.79) is True

    assert gate.accept(0.22) is True


def test_self_corrects_when_bootstrap_was_a_distant_voice(enabled_settings: WorkerSettings):
    """Regression scenario: the first two chunks captured on a track happen to be a
    distant bleed-through voice (quiet), before the real near speaker ever talks. The
    gate must not get permanently stuck rejecting the real speaker's much louder voice."""
    gate = NearFieldGate(enabled_settings)
    gate.accept(0.05)  # bootstrap #1 — distant voice
    gate.accept(0.05)  # bootstrap #2 — distant voice, baseline now ~0.05

    # The real, close speaker talks — much louder than the wrongly-bootstrapped baseline.
    assert gate.accept(0.6) is True
