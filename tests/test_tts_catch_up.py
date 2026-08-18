"""A dub that has fallen behind is read faster, so the gap closes instead of persisting.

WT-528. Two different faults put the dub behind the conversation and both were reported as one
symptom — "voice clone đọc không liền mạch", "đoạn dưới trước rồi ngược lên đoạn trên":

    * OUT OF ORDER. The per-key lock preserves the order messages arrive in, and that order is
      decided by when each translation finished upstream. Translation runs concurrently, so a
      long sentence loses the race to a short one spoken after it and the dub says them
      backwards.
    * SIMPLY SLOW. Translation took a long time, the dub starts well after the speaker stopped,
      and because the key is serialized every sentence behind it inherits the delay.

The product decision was that a late sentence is still SPOKEN — dropping it loses content the
speaker actually said — but read faster, because reading a backlog at normal pace never catches
up with anything.

The tests that matter most here are the ones asserting nothing changes: a meeting that is
keeping up must make byte-for-byte the Cartesia call it made before any of this existed.
"""

from __future__ import annotations

import pytest

from shared.prosody import SPEED_MAX
from shared.schemas import TranslationResultMessage
from tts_worker.worker import (
    _CATCH_UP_FULL_LAG_MS,
    _CATCH_UP_MAX_SPEED,
    _CATCH_UP_ON_TIME_MS,
    TTSWorker,
)


def _worker() -> TTSWorker:
    """Built without __init__, the way the rest of this suite builds one."""
    return TTSWorker.__new__(TTSWorker)


def _sentence(
    *, start_ms: int, latency_ms: int | None = None, target_lang: str = "vi"
) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id=f"seg-{start_ms}",
        meeting_id="meeting-1",
        speaker_id="speaker-1",
        original_text="xin chao",
        translated_text="hello",
        source_lang="en",
        target_lang=target_lang,
        start_ms=start_ms,
        latency_ms=latency_ms,
    )


class TestLag:
    def test_first_sentence_is_never_behind(self) -> None:
        assert _worker()._catch_up_lag_ms(_sentence(start_ms=1000)) == 0

    def test_in_order_and_quick_is_not_behind(self) -> None:
        worker = _worker()
        worker._catch_up_lag_ms(_sentence(start_ms=1000, latency_ms=400))
        assert worker._catch_up_lag_ms(_sentence(start_ms=5000, latency_ms=400)) == 0

    def test_a_sentence_that_belongs_earlier_is_behind_by_the_gap(self) -> None:
        worker = _worker()
        worker._catch_up_lag_ms(_sentence(start_ms=5000))
        assert worker._catch_up_lag_ms(_sentence(start_ms=3000)) == 2000

    def test_only_the_overrun_of_a_slow_translation_counts(self) -> None:
        """A sentence inside the budget was never behind; charging its whole duration as lag
        would speed up every dub in a healthy meeting."""
        worker = _worker()
        lag = worker._catch_up_lag_ms(
            _sentence(start_ms=1000, latency_ms=_CATCH_UP_ON_TIME_MS + 900)
        )
        assert lag == 900

    def test_the_two_signals_are_maxed_not_summed(self) -> None:
        """A slow translation is usually also what pushed a sentence out of order — adding them
        would count the same delay twice."""
        worker = _worker()
        worker._catch_up_lag_ms(_sentence(start_ms=5000))
        lag = worker._catch_up_lag_ms(
            _sentence(start_ms=3000, latency_ms=_CATCH_UP_ON_TIME_MS + 500)
        )
        assert lag == 2000

    def test_each_target_language_keeps_its_own_timeline(self) -> None:
        """One listener falling behind must not make every other listener's dub race."""
        worker = _worker()
        worker._catch_up_lag_ms(_sentence(start_ms=9000, target_lang="vi"))
        assert worker._catch_up_lag_ms(_sentence(start_ms=1000, target_lang="ja")) == 0

    def test_ending_a_room_forgets_its_timeline(self) -> None:
        """A room that ends and is somehow seen again must not judge its first sentence late
        against the previous meeting's timeline."""
        worker = _worker()
        # The attributes _cleanup_room touches on the way through BaseWorker. __new__ leaves
        # them unset, and the ones this test is not about only need to exist.
        worker._key_locks = {}
        worker._route_states = {}
        worker._translation_active = {}
        worker._paused_rooms = set()
        worker._room_routes = {}
        worker._catch_up_lag_ms(_sentence(start_ms=9000))

        worker._cleanup_room("meeting-1")

        assert worker._catch_up_lag_ms(_sentence(start_ms=1000)) == 0


class TestSpeed:
    def test_no_lag_leaves_the_config_exactly_alone(self) -> None:
        """Including None. Prosody's rule is that silence is the honest thing to say about
        delivery nobody measured, and keeping up must not break it."""
        assert TTSWorker._with_catch_up(None, 0) is None

        measured: dict[str, float | str] = {"speed": 0.9, "volume": 1.1, "emotion": "sad"}
        assert TTSWorker._with_catch_up(measured, 0) == measured

    def test_lag_speaks_faster_even_with_nothing_measured(self) -> None:
        """Being behind is known whether or not the speaker's prosody was measurable, so this is
        the one case allowed to create a config out of None."""
        config = TTSWorker._with_catch_up(None, _CATCH_UP_FULL_LAG_MS)
        assert config is not None
        assert config["speed"] == pytest.approx(_CATCH_UP_MAX_SPEED)

    def test_the_ramp_is_gradual(self) -> None:
        """A step change would be audible as the pace jumping mid-conversation."""
        half = TTSWorker._with_catch_up(None, _CATCH_UP_FULL_LAG_MS // 2)
        full = TTSWorker._with_catch_up(None, _CATCH_UP_FULL_LAG_MS)
        assert half is not None and full is not None
        assert 1.0 < float(half["speed"]) < float(full["speed"])

    def test_it_multiplies_the_speakers_own_rate(self) -> None:
        """A slow talker who is behind still sounds like a slow talker, just less far behind."""
        config = TTSWorker._with_catch_up({"speed": 0.8, "volume": 1.0}, _CATCH_UP_FULL_LAG_MS)
        assert config is not None
        assert config["speed"] == pytest.approx(0.8 * _CATCH_UP_MAX_SPEED, abs=0.001)
        assert config["volume"] == 1.0

    def test_it_never_exceeds_what_cartesia_accepts(self) -> None:
        config = TTSWorker._with_catch_up({"speed": SPEED_MAX}, _CATCH_UP_FULL_LAG_MS * 10)
        assert config is not None
        assert config["speed"] <= SPEED_MAX

    def test_a_dub_that_outruns_comprehension_has_caught_up_with_nothing(self) -> None:
        """However far behind it gets, the boost stays bounded."""
        config = TTSWorker._with_catch_up(None, _CATCH_UP_FULL_LAG_MS * 100)
        assert config is not None
        assert config["speed"] == pytest.approx(_CATCH_UP_MAX_SPEED)


class TestCachedDuration:
    """A cache hit plays for as long as the render that filled the cache. WT-528.

    `duration_ms=0` was hardcoded on that path, so the number reached
    transcript.audio_dubbings.duration_ms as a fact: a third of one production evening's rows
    read 0 with status 'done'. That is indistinguishable from audio that failed to synthesize,
    and it is what sent an investigation hunting for silence that was never there.

    It also feeds _observe_dub_fit, which sums these to learn how much longer a dub runs than the
    speech it replaces — isochrony then centres every later sentence's speed on that fit. Zeros
    taught it that dubs take no time at all.
    """

    @staticmethod
    def _wav(pcm_samples: int, sample_rate: int) -> bytes:
        return b"\x00" * 44 + b"\x00\x00" * pcm_samples

    def test_a_cached_second_reports_a_second(self) -> None:
        worker = _worker()
        worker.cartesia = type("_C", (), {"sample_rate": 44100})()
        assert worker._wav_duration_ms(self._wav(44100, 44100)) == 1000

    def test_it_follows_the_configured_sample_rate(self) -> None:
        """A wrong rate would misreport every cached line by the ratio between the two."""
        worker = _worker()
        worker.cartesia = type("_C", (), {"sample_rate": 22050})()
        assert worker._wav_duration_ms(self._wav(22050, 22050)) == 1000

    def test_header_only_audio_is_zero_not_negative(self) -> None:
        worker = _worker()
        worker.cartesia = type("_C", (), {"sample_rate": 44100})()
        assert worker._wav_duration_ms(b"\x00" * 20) == 0
