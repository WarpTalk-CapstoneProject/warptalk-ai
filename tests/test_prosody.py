"""Tests for shared/prosody.py.

The pitch tests use synthetic tones because a known answer is the only way to tell a working
pitch detector from one that returns plausible numbers. The rest pin the decisions that keep a
wrong reading from becoming a wrong performance: two features must agree before an utterance is
called activated, an unestablished speaker is always neutral, and the output stays inside the
range Cartesia's API actually accepts.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from shared.prosody import (
    _EMOTION_LADDERS,
    MIN_BASELINE_SAMPLES,
    NEUTRAL_DELIVERY,
    SPEED_MAX,
    SPEED_MIN,
    VOLUME_MAX,
    Delivery,
    ProsodyFeatures,
    SpeakerBaseline,
    measure,
    pcm16_to_float,
    to_delivery,
    to_generation_config,
    update_baseline,
)

SAMPLE_RATE = 16000


def tone(hz: float, seconds: float = 1.0, amplitude: float = 0.5) -> npt.NDArray[np.float32]:
    """A voiced-sounding signal at a known pitch. Harmonics are included because a pure sine is
    easier to track than speech, and a detector that only works on sines would pass a test it
    should not."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    wave = np.sin(2 * np.pi * hz * t)
    wave += 0.5 * np.sin(2 * np.pi * 2 * hz * t)
    wave += 0.25 * np.sin(2 * np.pi * 3 * hz * t)
    return (amplitude * wave / np.max(np.abs(wave))).astype(np.float32)


class TestMeasure:
    @pytest.mark.parametrize("hz", [95.0, 120.0, 180.0, 240.0])
    def test_finds_the_pitch_it_was_given(self, hz: float) -> None:
        features = measure(tone(hz), SAMPLE_RATE)
        # Within 5%: the frame length quantises the lag, so exactness is not on offer and not
        # needed — every downstream comparison is a ratio against the same measurement.
        assert features.pitch_median_hz == pytest.approx(hz, rel=0.05)
        assert features.is_usable

    def test_silence_is_not_usable(self) -> None:
        features = measure(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)
        assert not features.is_usable
        assert features.pitch_median_hz == 0.0

    def test_noise_is_not_mistaken_for_voice(self) -> None:
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 0.2, SAMPLE_RATE).astype(np.float32)
        features = measure(noise, SAMPLE_RATE)
        # White noise has no periodicity to lock onto. If this ever starts returning a confident
        # pitch, the peak threshold in _frame_pitch has been loosened too far and room tone is
        # about to start setting people's baselines.
        assert not features.is_usable

    def test_an_animated_voice_has_a_wider_spread_than_a_monotone(self) -> None:
        monotone = measure(tone(140.0), SAMPLE_RATE)
        varied = measure(
            np.concatenate([tone(120.0, 0.4), tone(200.0, 0.4), tone(150.0, 0.4)]), SAMPLE_RATE
        )
        assert varied.pitch_iqr_hz > monotone.pitch_iqr_hz

    def test_empty_input_does_not_raise(self) -> None:
        assert measure(np.zeros(0, dtype=np.float32), SAMPLE_RATE).duration_ms == 0


class TestBaseline:
    def test_unusable_utterances_do_not_move_it(self) -> None:
        baseline = SpeakerBaseline(
            pitch_median_hz=120, pitch_iqr_hz=20, rms=0.1, speech_rate=0.6, sample_count=5
        )
        silence = ProsodyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 500)
        assert update_baseline(baseline, silence) == baseline

    def test_first_sample_becomes_the_baseline(self) -> None:
        features = measure(tone(150.0), SAMPLE_RATE)
        baseline = update_baseline(SpeakerBaseline(), features)
        assert baseline.sample_count == 1
        assert baseline.pitch_median_hz == features.pitch_median_hz

    def test_one_loud_sentence_does_not_redefine_normal(self) -> None:
        quiet = measure(tone(120.0, amplitude=0.2), SAMPLE_RATE)
        baseline = SpeakerBaseline()
        for _ in range(5):
            baseline = update_baseline(baseline, quiet)

        shout = measure(tone(230.0, amplitude=0.9), SAMPLE_RATE)
        after = update_baseline(baseline, shout)

        # It moves toward the outlier but stays much closer to the speaker's normal — which is
        # what makes the NEXT loud sentence still register as loud.
        assert after.pitch_median_hz < (baseline.pitch_median_hz + shout.pitch_median_hz) / 2


class TestDelivery:
    def _baseline(self) -> SpeakerBaseline:
        calm = measure(tone(120.0, amplitude=0.3), SAMPLE_RATE)
        baseline = SpeakerBaseline()
        for _ in range(MIN_BASELINE_SAMPLES):
            baseline = update_baseline(baseline, calm)
        return baseline

    def test_a_speaker_with_no_baseline_is_always_neutral(self) -> None:
        loud = measure(tone(240.0, amplitude=0.9), SAMPLE_RATE)
        # The fail-safe that matters most: the first thing anyone says is dubbed plainly rather
        # than dramatically, because there is nothing yet to compare it against.
        assert to_delivery(loud, SpeakerBaseline()) == NEUTRAL_DELIVERY

    def test_pitch_alone_is_not_enough_to_call_someone_activated(self) -> None:
        # Same loudness, higher pitch — a question, not excitement.
        question = measure(tone(150.0, amplitude=0.3), SAMPLE_RATE)
        delivery = to_delivery(question, self._baseline())
        assert delivery.pitch_lift > 1.0
        assert delivery.arousal == "neutral"

    def test_pitch_and_energy_together_are(self) -> None:
        excited = measure(tone(160.0, amplitude=0.75), SAMPLE_RATE)
        assert to_delivery(excited, self._baseline()).arousal == "high"

    def test_lower_and_quieter_reads_as_subdued(self) -> None:
        subdued = measure(tone(105.0, amplitude=0.15), SAMPLE_RATE)
        assert to_delivery(subdued, self._baseline()).arousal == "low"

    def test_ordinary_speech_stays_ordinary(self) -> None:
        same = measure(tone(121.0, amplitude=0.3), SAMPLE_RATE)
        assert to_delivery(same, self._baseline()).arousal == "neutral"


class TestGenerationConfig:
    def test_neutral_delivery_sends_no_emotion(self) -> None:
        config = to_generation_config(NEUTRAL_DELIVERY, "neutral")
        # Sending "neutral" would be a claim where there is none, and it is what the model does
        # anyway.
        assert "emotion" not in config
        assert config["speed"] == 1.0
        assert config["volume"] == 1.0

    def test_arousal_and_valence_choose_the_label_together(self) -> None:
        high = Delivery(1.3, 1.5, 1.4, 1.2, "high")
        assert to_generation_config(high, "positive")["emotion"] == "excited"
        assert to_generation_config(high, "negative")["emotion"] == "frustrated"
        # The same sound, opposite meanings. This is why valence cannot come from audio.
        assert (
            to_generation_config(high, "positive")["emotion"]
            != to_generation_config(high, "negative")["emotion"]
        )

    def test_speed_stays_inside_what_the_api_accepts(self) -> None:
        frantic = Delivery(1.0, 1.0, 1.0, 9.0, "neutral")
        crawling = Delivery(1.0, 1.0, 1.0, 0.01, "neutral")
        assert to_generation_config(frantic)["speed"] == SPEED_MAX
        assert to_generation_config(crawling)["speed"] == SPEED_MIN

    def test_a_whisper_lowers_the_volume(self) -> None:
        whisper = Delivery(0.9, 0.8, 0.35, 0.9, "low")
        assert float(to_generation_config(whisper)["volume"]) < 1.0

    def test_undetermined_valence_is_not_neutral_valence(self) -> None:
        """The pipeline's real state today: nothing has read the words for sentiment.

        Collapsing that into "neutral" would reach ("high", "neutral") in the table and label
        an emphatic speaker "surprised" — a claim about their feelings inferred from loudness
        alone. Unknown must produce no label at all.
        """
        emphatic = Delivery(1.3, 1.5, 1.4, 1.2, "high")

        assert "emotion" not in to_generation_config(emphatic)
        assert "emotion" not in to_generation_config(emphatic, None)
        assert to_generation_config(emphatic, "neutral")["emotion"] == "surprised"

    def test_delivery_still_carries_without_any_valence(self) -> None:
        # The half that needs no reading of the words must not be held hostage to the half
        # that does — an unlabelled utterance is still dubbed faster and louder if that is how
        # it was said.
        config = to_generation_config(Delivery(1.3, 1.5, 1.4, 1.25, "high"))

        assert config["speed"] == pytest.approx(1.25)
        assert config["volume"] == pytest.approx(1.4)


class TestPcmConversion:
    def test_round_trips_amplitude(self) -> None:
        raw = np.array([0, 16384, -16384], dtype="<i2").tobytes()
        assert pcm16_to_float(raw) == pytest.approx([0.0, 0.5, -0.5], abs=1e-4)

    def test_empty_is_empty(self) -> None:
        assert pcm16_to_float(b"").size == 0

    def test_measuring_real_pcm_bytes_works_end_to_end(self) -> None:
        pcm = (tone(130.0) * 32767).astype("<i2").tobytes()
        features = measure(pcm16_to_float(pcm), SAMPLE_RATE)
        assert features.pitch_median_hz == pytest.approx(130.0, rel=0.05)


class TestEmotionLadders:
    """Two coarse three-way axes can only name six feelings. The pipeline measures pitch and
    energy continuously and then throws the magnitude away by bucketing; the ladders spend that
    magnitude instead of inventing a third axis to justify more labels.
    """

    def _high(self, pitch_lift: float, energy_ratio: float) -> Delivery:
        return Delivery(pitch_lift, 1.5, energy_ratio, 1.0, "high")

    def test_the_ladder_only_adds_to_what_was_there_before(self) -> None:
        """Rung 0 of every cell is the label the flat table produced.

        This is the property that makes the change safe to ship: an ordinary emphatic sentence
        keeps exactly the word it had, and only a delivery that is extreme by this speaker's own
        standard reaches a stronger one.
        """
        ordinary = self._high(1.30, 1.40)

        assert to_generation_config(ordinary, "positive")["emotion"] == "excited"
        assert to_generation_config(ordinary, "negative")["emotion"] == "frustrated"
        assert to_generation_config(ordinary, "neutral")["emotion"] == "surprised"

    def test_a_far_stronger_delivery_reaches_a_stronger_word(self) -> None:
        mild = self._high(1.15, 1.28)
        extreme = self._high(1.60, 1.75)

        assert to_generation_config(mild, "negative")["emotion"] == "frustrated"
        assert to_generation_config(extreme, "negative")["emotion"] == "outraged"

    def test_climbing_is_monotone(self) -> None:
        # A stronger delivery must never produce a milder word. Without this the ladder could
        # be non-monotone at a boundary and nobody would notice.
        ladder = ("frustrated", "angry", "outraged")
        seen = [
            ladder.index(
                str(
                    to_generation_config(self._high(1.12 + step, 1.25 + step), "negative")[
                        "emotion"
                    ]
                )
            )
            for step in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6)
        ]

        assert seen == sorted(seen), f"rung went backwards as delivery got stronger: {seen}"

    def test_an_error_in_intensity_never_changes_the_feeling(self) -> None:
        """The safety property. Every rung within a cell is the same feeling at a different
        strength, so a mis-tiered label is one step along one scale — never 'angry' where
        'happy' was meant, which would take an error in valence that no intensity can produce.
        """
        positive_words = {
            str(to_generation_config(self._high(1.12 + s, 1.25 + s), "positive")["emotion"])
            for s in (0.0, 0.2, 0.4, 0.8)
        }
        negative_words = {
            str(to_generation_config(self._high(1.12 + s, 1.25 + s), "negative")["emotion"])
            for s in (0.0, 0.2, 0.4, 0.8)
        }

        assert positive_words & negative_words == set()

    def test_every_label_is_one_cartesia_actually_accepts(self) -> None:
        """Cartesia says an emotion outside its list is 'not supported, and results are not
        guaranteed' — so an invented name fails silently and strangely rather than loudly.
        The SDK's own Literal is the authority here, not the docs page."""
        from cartesia.types.generation_config import GenerationConfig

        accepted = set(GenerationConfig.model_fields["emotion"].annotation.__args__[0].__args__)
        for ladder in _EMOTION_LADDERS.values():
            for word in ladder:
                assert word in accepted, f"{word!r} is not in Cartesia's emotion vocabulary"

    def test_a_neutral_delivery_still_gets_no_label_at_all(self) -> None:
        # The ladders must not have accidentally given the neutral cells a rung.
        assert "emotion" not in to_generation_config(
            Delivery(1.0, 1.0, 1.0, 1.0, "neutral"), "positive"
        )


class TestVolumeRange:
    def test_volume_may_use_the_range_the_api_actually_accepts(self) -> None:
        """VOLUME_MAX was 1.5 — SPEED_MAX copied one line down. The verified rejection quoted in
        this module is about `speed`; Cartesia accepts volume in [0.5, 2.0], and the measured
        table at the top of the module records a successful render AT 2.0."""
        assert VOLUME_MAX == 2.0

        shouted = Delivery(1.3, 1.5, 3.0, 1.0, "high")
        assert float(to_generation_config(shouted)["volume"]) == 2.0
