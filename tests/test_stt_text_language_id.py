"""Room-restricted text language-ID: a line's label follows what was said, between Latin languages.

Bridge room 01a10069 (3 Oct, text-only bridge). The Realtime model returns no language, and
between vi and en the text rarely proves anything to the script / Vietnamese-unique rules:

* B1 — "Anh làm gì?" from the Vietnamese host, whose picker said English, was labelled en
  (à and ì are shared with Romance languages, so not evidence).
* B2 — "AI opportunities?" / "Morning is great." from the Meet side, whose stand-in was seeded
  vi, were labelled vi and "translated" vi -> en; ASCII is never evidence, so nothing could ever
  re-pin that speaker toward English either.
"""

from __future__ import annotations

from stt_worker.model import OpenAISTT, TranscribedSegment, _filter_segments
from stt_worker.text_language_id import MIN_CONFIDENCE, identify_room_language

ROOM = {"vi", "en"}


def _raw(text: str) -> dict:
    return {"text": text, "start": 0.0, "end": 1.5, "avg_logprob": -0.2, "no_speech_prob": 0.01}


def _label(
    text: str, declared: str, room: set[str] | None = ROOM, floor: float | None = MIN_CONFIDENCE
):
    segments = _filter_segments(
        [_raw(text)], declared, 0, allowed_languages=room, text_language_id_min_confidence=floor
    )
    assert len(segments) == 1, segments
    return segments[0].language, segments[0].language_source


class TestIdentifyRoomLanguage:
    def test_vietnamese_without_unique_letters(self):
        assert identify_room_language("Anh làm gì?", ROOM) == "vi"

    def test_plain_english(self):
        assert identify_room_language("Morning is great.", ROOM) == "en"
        assert identify_room_language("AI opportunities?", ROOM) == "en"

    def test_too_short_proves_nothing(self):
        assert identify_room_language("OK", ROOM) is None
        assert identify_room_language("Thanks", ROOM) is None

    def test_code_switching_below_the_floor_is_left_alone(self):
        # Mostly English words in a Vietnamese frame: en ~0.86, under the floor.
        assert identify_room_language("Anh deploy cái backend API nha", ROOM) is None

    def test_needs_two_latin_candidates(self):
        # One Latin language: nothing to choose between.
        assert identify_room_language("Morning is great.", {"vi"}) is None
        assert identify_room_language("Morning is great.", set()) is None

    def test_never_answers_outside_the_candidates(self):
        assert identify_room_language("Good morning everyone", {"vi", "fr"}) in {None, "vi", "fr"}

    def test_regional_tags_are_reduced(self):
        assert identify_room_language("Morning is great.", {"vi-VN", "en-US"}) == "en"


class TestFilterSegmentsLabels:
    def test_b1_vietnamese_host_declared_english(self):
        assert _label("Anh làm gì?", "en") == ("vi", "text_id")

    def test_b1_before_this_fix_the_declaration_won(self):
        assert _label("Anh làm gì?", "en", floor=None) == ("en", "declared")

    def test_b2_english_meet_side_declared_vietnamese(self):
        assert _label("Morning is great.", "vi") == ("en", "text_id")
        assert _label("AI opportunities?", "vi") == ("en", "text_id")

    def test_unpinned_speaker_is_identified_rather_than_guessed(self):
        # "auto" / unknown used to fall to the weak guess, which prefers en whatever was said.
        assert _label("Anh làm gì?", "unknown") == ("vi", "text_id")

    def test_matching_declaration_keeps_its_label(self):
        assert _label("Morning is great.", "en")[0] == "en"
        assert _label("Anh làm gì?", "vi")[0] == "vi"

    def test_unique_letters_are_still_evidence_first(self):
        assert _label("Đổi tên không có lưu được ấy", "en") == ("vi", "evidence")

    def test_short_lines_keep_the_declaration(self):
        assert _label("OK", "vi") == ("vi", "declared")

    def test_room_that_declared_nothing_is_untouched(self):
        # Native rooms whose language set is not known yet behave exactly as before.
        assert _label("Morning is great.", "vi", room=None) == ("vi", "declared")

    def test_language_outside_the_room_is_never_produced(self):
        # vi + ja room: one Latin language, so English text keeps the vi declaration.
        assert _label("Morning is great.", "vi", room={"vi", "ja"}) == ("vi", "declared")

    def test_switched_off(self):
        assert _label("Morning is great.", "vi", floor=None) == ("vi", "declared")


def _seg(text: str, language: str, source: str) -> TranscribedSegment:
    return TranscribedSegment(
        text=text,
        language=language,
        confidence=-0.2,
        start_ms=0,
        end_ms=1000,
        language_source=source,
    )


class TestLearningBothWays:
    def _model(self) -> OpenAISTT:
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._language_override = {}
        return model

    def test_vi_declared_speaker_speaking_english_is_re_pinned_to_english(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "standin"),
            "vi",
            [
                _seg("Morning is great.", "en", "text_id"),
                _seg("Good morning everyone", "en", "text_id"),
            ],
        )
        assert model._language_override[("m", "standin")] == ("en", "vi")

    def test_short_identified_lines_do_not_move_the_pin(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "s"),
            "vi",
            [_seg("AI opportunities?", "en", "text_id"), _seg("Thank you", "en", "text_id")],
        )
        assert model._language_override == {}

    def test_short_identified_line_does_not_reset_the_count_either(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "s"),
            "vi",
            [
                _seg("Morning is great.", "en", "text_id"),
                _seg("AI opportunities?", "en", "text_id"),
                _seg("Sounds good to me", "en", "text_id"),
            ],
        )
        assert model._language_override[("m", "s")] == ("en", "vi")

    def test_identified_declared_language_releases_the_override(self):
        model = self._model()
        model._language_override[("m", "host")] = ("en", "vi")
        model._learn_language_evidence(
            ("m", "host"),
            "en",
            # Neither line carries a Vietnamese-unique letter: _release_verdict alone says None.
            [_seg("Anh làm gì?", "vi", "text_id"), _seg("Chào anh, em là Nam", "vi", "text_id")],
        )
        assert ("m", "host") not in model._language_override

    def test_without_the_identifier_those_lines_do_not_release(self):
        model = self._model()
        model._language_override[("m", "host")] = ("en", "vi")
        model._learn_language_evidence(
            ("m", "host"),
            "en",
            [_seg("Anh làm gì?", "en", "declared"), _seg("Chào anh, em là Nam", "en", "declared")],
        )
        assert model._language_override[("m", "host")] == ("en", "vi")
