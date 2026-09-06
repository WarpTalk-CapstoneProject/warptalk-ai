"""A pause inside a sentence must not end the turn (WT-573/WT-576, "transcript bubble vụn").

The reported symptom is transcript arriving as two- and three-word bubbles, each one
separately transcribed, translated and dubbed.

WT-576 named boundary seeking as the likely source. It cannot be: `speech_samples` restarts
at zero at every cut, so a seek cut cannot happen again until the speaker has produced another
`vad_seek_boundary_after_ms` of speech — and a chunk holding four seconds of speech is not a
fragment. What is left is the ordinary hangover, and 576ms sits inside the clause-internal
pause distribution of spontaneous speech.

Source-level assertions, matching test_livekit_ingress_utterance_state.py and for the same
reason: `process_audio_track` consumes a live `rtc.AudioStream` and a real Silero model, so a
driven loop would be testing the mocks. What regressed here is a handful of statements inside
a 200-line loop, and that is what these pin.
"""

from __future__ import annotations

import re
from pathlib import Path

from shared.config import WorkerSettings

WORKER_SOURCE = (
    Path(__file__).resolve().parents[1] / "livekit_ingress_worker" / "worker.py"
).read_text(encoding="utf-8")


class TestTheLadder:
    def test_a_turn_that_has_barely_spoken_waits_longer(self) -> None:
        """The rung that fixes the fragmentation, and the only one that adds any latency."""
        settings = WorkerSettings()

        assert settings.vad_short_turn_hangover_ms > settings.vad_silence_hangover_ms, (
            "the short-turn rung must wait LONGER than an ordinary clause, or it changes "
            "nothing about where a sentence gets cut"
        )
        assert settings.vad_short_turn_speech_ms < settings.vad_seek_boundary_after_ms, (
            "the short-turn rung and the seek rung would overlap, and a turn would qualify "
            "for both the longest and the shortest hangover at once"
        )

    def test_the_rungs_are_ordered_longest_speech_first(self) -> None:
        """One rule: the more a speaker has already said, the readier the loop is to cut.

        Written as an if/elif chain, so the order is the logic. Seeking must be tested first —
        it is the highest speech threshold, and an earlier short-turn branch would be
        unreachable for it only because of where it sits.
        """
        ladder = re.search(
            r"if speech_samples >= seek_after_samples:.*?"
            r"hangover = seek_hangover_frames.*?"
            r"elif speech_samples < short_turn_samples:.*?"
            r"hangover = short_turn_hangover_frames.*?"
            r"else:.*?"
            r"hangover = silence_hangover_frames",
            WORKER_SOURCE,
            re.DOTALL,
        )
        assert ladder is not None, (
            "the three-rung hangover ladder is gone or reordered; a short-turn branch placed "
            "before the seek branch would hold long turns open too"
        )

    def test_the_hangover_is_still_decided_on_speech_not_on_the_buffer(self) -> None:
        """The buffer holds the padding and every internal pause, so a hesitant speaker who has
        said very little would look like they had said plenty — and would be cut where the
        fragment rung exists to hold them open."""
        assert "speech_samples >= seek_after_samples" in WORKER_SOURCE
        assert "speech_samples < short_turn_samples" in WORKER_SOURCE
        assert "len(speech_buffer) < short_turn_samples" not in WORKER_SOURCE


class TestTheHoldOpenShipsNoExtraSilence:
    def test_silence_beyond_the_ordinary_hangover_is_trimmed_before_publish(self) -> None:
        """Waiting longer is a DECISION, not padding.

        Silence counted past `vad_silence_hangover_ms` was time spent waiting to see whether the
        speaker would resume. They did not, so it is evidence of nothing — and trailing silence
        is exactly what a Whisper-family model invents text over. A chunk that waited longer
        must still ship the tail it always shipped.
        """
        assert "excess_frames = silence_frames - silence_hangover_frames" in WORKER_SOURCE, (
            "nothing computes how much silence the hold-open added, so the extra wait is being "
            "sent to STT as audio"
        )

        trim = re.search(
            r"if excess_frames > 0:\s*\n\s*"
            r"del speech_buffer\[-excess_frames \* VAD_FRAME_BYTES\s*:\]",
            WORKER_SOURCE,
        )
        assert trim is not None, "the excess silence is computed and then not removed"

        # It has to happen BEFORE the chunk goes out, or it removes nothing that mattered.
        trim_at = WORKER_SOURCE.index("excess_frames = silence_frames")
        publish_at = WORKER_SOURCE.index("if speech_samples >= min_speech_samples")
        assert trim_at < publish_at, "the trim runs after the publish decision, so it is dead"


class TestSpeechDurationReachesSTT:
    def test_the_chunk_says_how_much_of_it_was_speech(self) -> None:
        """The STT side has guards that ask "was there enough audio to justify this text".

        Handed the chunk's PCM duration they can never fire — every chunk carries roughly a
        second of pre-speech and hangover padding. This is the number that question is about.
        """
        assert "speech_ms=(speech_samples * 1000 // sample_rate) if speech_samples else 0" in (
            WORKER_SOURCE
        ), "the published chunk no longer carries a speech-only duration"
