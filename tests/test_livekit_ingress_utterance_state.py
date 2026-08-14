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
    assert "if speech_samples >= min_speech_samples:" in WORKER_SOURCE, (
        "the minimum-speech gate must weigh the speech-only count"
    )
