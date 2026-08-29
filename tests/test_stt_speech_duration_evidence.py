"""A guard that asks "was there enough audio for this?" must be shown the audio, not the padding.

Production runs `gpt-live-transcribe`, which returns NO token logprobs, so `min_avg_logprob`
and every per-language floor beneath it are skipped by construction. What is left standing
between marginal audio and a fluent invented caption is the small set of guards that reason
about the audio itself — and the strongest of them, "no human says this many characters in
this little audio", was being handed the wrong number.

Every chunk the ingress publishes is wrapped in `vad_pre_speech_ms` + `vad_silence_hangover_ms`
of padding, about a second of it. Against a 0.5s threshold, the padded duration can never be
small enough for the guard to fire: it was live in the source and dead in the room. This is
the same defect the minimum-speech gate had one stage upstream (WT-371 #7).
"""

from __future__ import annotations

from pathlib import Path

from shared.schemas import STT_UNKNOWN_CONFIDENCE, AudioChunkMessage
from stt_worker.model import _filter_segments

# Long enough to be impossible in half a second — 37 characters is over 70 chars/sec, and no
# human clears ~30 in any language.
INVENTED = "Chúng ta sẽ bàn về kiến trúc hệ thống"


def _raw(text: str) -> list[dict[str, object]]:
    """One completed segment as the production model returns it: no confidence at all."""
    return [
        {
            "text": text,
            "start": 0.0,
            "end": 1.0,
            "avg_logprob": STT_UNKNOWN_CONFIDENCE,
            "no_speech_prob": 0.0,
        }
    ]


class TestTheGuardCanFinallyFire:
    def test_a_sentence_invented_over_a_third_of_a_second_is_dropped(self) -> None:
        segments = _filter_segments(
            _raw(INVENTED),
            "vi",
            0,
            {"vi"},
            # What the chunk actually measures: 0.3s of speech wrapped in ~0.96s of padding.
            real_duration_s=1.26,
            speech_duration_s=0.3,
        )
        assert segments == [], (
            "37 characters over 300ms of speech is the classic hallucination onto near-silence "
            "and must not reach the transcript"
        )

    def test_the_padded_duration_alone_could_never_have_caught_it(self) -> None:
        """The regression this fixes, pinned as its own case.

        Same audio, same text — but judged the way it used to be judged, on the padded chunk
        duration. It survives, which is exactly why the guard never fired in production.
        """
        segments = _filter_segments(_raw(INVENTED), "vi", 0, {"vi"}, real_duration_s=1.26)
        assert len(segments) == 1, (
            "if this now drops, the guard is reading the padded duration again and the shortest "
            "real utterance in the product is at risk"
        )

    def test_real_speech_behind_the_words_survives(self) -> None:
        segments = _filter_segments(
            _raw(INVENTED),
            "vi",
            0,
            {"vi"},
            real_duration_s=3.0,
            speech_duration_s=2.0,
        )
        assert [seg.text for seg in segments] == [INVENTED]

    def test_a_short_acknowledgement_is_not_collateral(self) -> None:
        """The guard is about characters per second, not about being brief. A real 300ms "Vâng"
        has to survive the same measurement that drops the invented sentence above."""
        segments = _filter_segments(
            _raw("Vâng"),
            "vi",
            0,
            {"vi"},
            real_duration_s=1.26,
            speech_duration_s=0.3,
        )
        assert [seg.text for seg in segments] == ["Vâng"]


class TestUnknownStaysUnknown:
    def test_a_publisher_that_says_nothing_gets_the_old_behaviour(self) -> None:
        """`speech_ms` defaults to 0 — an older ingress mid-rolling-deploy — and 0 must read as
        "did not say", never as "no speech at all"."""
        assert (
            AudioChunkMessage(
                meeting_id="m", speaker_id="s", chunk_index=0, audio_data=b""
            ).speech_ms
            == 0
        )

        segments = _filter_segments(
            _raw(INVENTED), "vi", 0, {"vi"}, real_duration_s=1.26, speech_duration_s=None
        )
        assert len(segments) == 1, "an unknown speech duration must not be treated as silence"

    def test_the_early_path_no_longer_claims_a_perfect_score(self) -> None:
        """A delta carries no logprob. Claiming 0.0 — the best possible value — exempted early
        sentences from every confidence-shaped guard below, and in flash mode most production
        segments arrive down that path."""
        source = (Path(__file__).resolve().parents[1] / "stt_worker" / "model.py").read_text(
            encoding="utf-8"
        )
        assert '"avg_logprob": 0.0' not in source, (
            "an early/speculative sentence is fabricating a perfect confidence again; the "
            "sentinel is this codebase's word for 'unknown'"
        )


class TestTheChunkCarriesIt:
    def test_speech_ms_survives_the_redis_round_trip(self) -> None:
        message = AudioChunkMessage(
            meeting_id="m", speaker_id="s", chunk_index=0, audio_data=b"\x00\x01", speech_ms=864
        )
        assert AudioChunkMessage.from_redis(message.to_redis()).speech_ms == 864
