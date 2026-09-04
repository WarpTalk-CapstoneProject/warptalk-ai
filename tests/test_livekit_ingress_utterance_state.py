"""Two things the audio loop must get right between one utterance and the next (WT-371 #7).

Both are about the same reported symptom: speech that registers late, or not at all, and
registers better when there is background noise in the room.

Source-level assertions rather than a driven loop. `process_audio_track` consumes a live
`rtc.AudioStream` and a real Silero model; standing that up would test the mocks. What actually
regressed here is a single statement in a 200-line loop, and that is what these pin.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKER_SOURCE = (
    Path(__file__).resolve().parents[1] / "livekit_ingress_worker" / "worker.py"
).read_text(encoding="utf-8")


def test_the_vad_is_not_reset_between_utterances() -> None:
    """Silero is recurrent, and its first frames after a reset are its least reliable — which is
    exactly the moment the next utterance begins.

    Resetting after EVERY utterance meant every sentence in a conversation was judged by a cold
    model, so the first word registered late or not at all, and registered better when background
    noise kept the probabilities up.

    Exactly two resets are legitimate, and both sit where the audio genuinely discontinues: the
    start of a track, and a pause/resume (which discards the buffers for the same reason). A pause
    between two sentences is not a discontinuity — it is the signal Silero exists to model.
    """
    resets = re.findall(r"^\s*track_vad_model\.reset_states\(\)", WORKER_SOURCE, re.MULTILINE)

    assert len(resets) == 2, (
        f"expected exactly 2 VAD resets (track start, pause/resume); found {len(resets)}. "
        "A third is almost certainly the per-utterance reset that made every sentence after "
        "the first one start on a cold model."
    )


def test_the_minimum_speech_gate_does_not_count_padding_as_speech() -> None:
    """The gate measured `len(speech_buffer)`, which by then also holds the pre-speech padding and
    the entire 576ms hangover.

    A 100ms cough therefore arrived as ~870ms and sailed past a 288ms minimum, sending a fragment
    of non-speech to a model with no confidence signal of its own — the input it invents fluent
    sentences from (see test_vad_threshold_default).

    The padding must still SHIP, so the fix is a separate count of what VAD actually called
    speech, not a smaller buffer.
    """
    assert "speech_samples += len(window_data) // 2" in WORKER_SOURCE, (
        "nothing accumulates a speech-only sample count, so the gate can only be measuring the "
        "whole padded buffer again"
    )

    # The two questions must stay distinct: chunk SIZE weighs the buffer, the speech gate weighs
    # the speech. Re-deriving speech_samples from the buffer collapses them back together.
    assert "speech_samples = len(speech_buffer)" not in WORKER_SOURCE, (
        "speech_samples is being re-derived from the padded buffer, which is the defect"
    )
    assert "if len(speech_buffer) // 2 >= max_chunk_samples:" in WORKER_SOURCE, (
        "the max-chunk check must weigh the whole buffer — it is about how much audio gets sent"
    )
    assert "speech_samples >= min_speech_samples" in WORKER_SOURCE, (
        "the minimum-speech gate must weigh the speech-only count"
    )


def test_a_continuation_is_not_held_to_the_minimum_speech_gate() -> None:
    """The gate exists to stop a cough reaching the model. A leftover tail is not a cough.

    After a cap or boundary-seek cut, `speech_samples` restarts at zero while the SPEAKER has
    not stopped. If they then pause with less than `vad_min_speech_ms` of new speech buffered,
    the old code dropped it — silently, at debug level — and those words never existed. With a
    6000ms cap the vulnerable window is the 288ms immediately after each cut, and "speak just
    past the cap, then pause" is exactly what expressive delivery sounds like.

    So the gate is skipped when this turn has already published: what is in the buffer is the
    remainder of an utterance the model has already been given the front of.
    """
    assert "published_this_turn" in WORKER_SOURCE, (
        "nothing tracks whether this turn already sent a chunk, so a leftover tail cannot be "
        "told apart from a fresh cough"
    )
    assert "speech_samples >= min_speech_samples or published_this_turn" in WORKER_SOURCE, (
        "the minimum-speech gate is being applied to continuations again"
    )


def test_a_long_turn_takes_the_next_real_pause_instead_of_the_hard_cap() -> None:
    """Boundary seeking — the reviewer's "cắt ngữ nghĩa khi nói câu dài", in code.

    `chunk_duration_ms` is a hard cap that cuts wherever the speaker happens to be, and SHAS
    (Interspeech 2022) measured pause-based segmentation retaining only 81.3% of
    manual-segmentation BLEU — below naive fixed-length chunking — with a 2025 replication
    finding a 7.3 BLEU spread at constant latency purely from where cuts land.

    The only place this loop can cut without severing a word is a window VAD already called
    silence, so seeking means accepting a SHORTER pause once the turn has run long. Short
    turns must keep at least the full hangover, or ordinary conversation fragments.

    This used to pin the ternary that chose between two thresholds. There are three now — a
    turn that has barely spoken waits LONGER still, see test_vad_hangover_ladder.py — so what
    is pinned here is the part this test is actually about: seeking is the exception, reached
    only by a long turn, and everything else falls through to at least the full hangover.
    """
    assert "seek_hangover_frames" in WORKER_SOURCE, "boundary seeking is not wired at all"
    assert "if speech_samples >= seek_after_samples" in WORKER_SOURCE, (
        "the seek threshold must be measured on SPEECH — the buffer also holds padding and "
        "every internal pause, so a hesitant speaker would trip it having said very little"
    )
    assert "hangover = silence_hangover_frames" in WORKER_SOURCE, (
        "nothing falls back to the full end-of-sentence hangover any more, so an ordinary "
        "clause is being cut on a threshold meant for long turns"
    )
    assert "silence_frames >= hangover" in WORKER_SOURCE, (
        "the hangover check must read the chosen threshold, not the fixed one"
    )
