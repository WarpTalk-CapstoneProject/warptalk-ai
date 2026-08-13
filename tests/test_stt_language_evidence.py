"""The speaker's own speech is allowed to correct the language they declared.

Grounded in a real production meeting: a participant whose profile said ``en`` spoke
Vietnamese for the whole call. STT is pinned to the declared language on the session, so every
segment came back fluent-looking and wrong, and the translator then ran en->vi over text that
was already Vietnamese.

The evidence that separates vi from en existed in this module and was unreachable: script
detection returns None for Latin text by design, and the Vietnamese-unique character class only
ran on the no-declaration fallback path.
"""

from stt_worker.model import (
    OpenAISTT,
    TranscribedSegment,
    _detect_unambiguous_language,
)


def _segment(text: str, language: str) -> TranscribedSegment:
    return TranscribedSegment(text=text, language=language, confidence=0.9, start_ms=0, end_ms=1000)


class TestUnambiguousLanguage:
    def test_vietnamese_diacritics_are_evidence(self):
        assert _detect_unambiguous_language("Đổi tên không có lưu được ấy") == "vi"

    def test_non_latin_scripts_still_win(self):
        assert _detect_unambiguous_language("こんにちは") == "ja"
        assert _detect_unambiguous_language("안녕하세요") == "ko"

    def test_plain_english_proves_nothing(self):
        # None, not "en": the declaration must keep winning where there is no evidence.
        assert _detect_unambiguous_language("Let us finish this part today") is None

    def test_romance_accents_are_not_vietnamese(self):
        # The bug this character class was narrowed to fix. An accent is not evidence.
        assert _detect_unambiguous_language("Deberíamos terminar esta parte") is None
        assert _detect_unambiguous_language("Nous avons regardé le rapport") is None
        assert _detect_unambiguous_language("Però non è così") is None


class TestLearnedOverride:
    def _model(self) -> OpenAISTT:
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._language_override = {}
        return model

    def test_one_contradicting_segment_is_not_enough(self):
        model = self._model()
        model._learn_language_evidence(("m", "s"), "en", [_segment("Chào anh", "vi")])
        assert model._language_override == {}

    def test_two_consecutive_contradictions_re_pin_the_session(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "s"), "en", [_segment("Chào anh", "vi"), _segment("Đổi tên đi", "vi")]
        )
        assert model._language_override[("m", "s")] == "vi"

    def test_agreement_resets_the_count(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "s"),
            "en",
            [_segment("Chào anh", "vi"), _segment("okay sure", "en"), _segment("Đổi tên", "vi")],
        )
        assert model._language_override == {}

    def test_a_speaker_who_matches_their_profile_is_never_overridden(self):
        model = self._model()
        model._learn_language_evidence(
            ("m", "s"), "vi", [_segment("Chào anh", "vi"), _segment("Đổi tên", "vi")]
        )
        assert model._language_override == {}

    def test_no_declaration_means_nothing_to_contradict(self):
        model = self._model()
        model._learn_language_evidence(("m", "s"), None, [_segment("Chào anh", "vi")] * 3)
        assert model._language_override == {}
