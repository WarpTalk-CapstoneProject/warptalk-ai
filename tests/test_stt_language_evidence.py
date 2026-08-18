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
        # The declaration travels with the override: it is the claim being corrected, and the
        # override lives exactly as long as that claim does.
        assert model._language_override[("m", "s")] == ("vi", "en")

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


class TestOverrideRelease:
    """A learned override lives exactly as long as the declaration it corrected.

    Production meeting 01a00a34 (16 Aug): the override was permanent, so a speaker who joined
    declared-en/speaking-vi (correctly re-pinned to vi) and then DELIBERATELY picked English in
    the meeting bar could never get their microphone back. Every English sentence stayed
    labelled vi, and their vi-listening partner got no translation — source vi, target vi,
    dropped as same-language. Nothing the person did was recoverable, because the learning loop
    returns early while an override exists and plain English text carries no unambiguous
    evidence to contradict it. The only signal strong enough to release it is the one this
    class pins: the person declaring something new.
    """

    def _model_with_override(self) -> OpenAISTT:
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._language_override = {}
        model._learn_language_evidence(
            ("m", "s"), "en", [_segment("Chào anh", "vi"), _segment("Đổi tên đi", "vi")]
        )
        assert model._language_override[("m", "s")] == ("vi", "en")
        return model

    def test_the_override_corrects_the_declaration_it_was_learned_against(self):
        model = self._model_with_override()
        assert model._apply_language_override(("m", "s"), "en") == "vi"
        # Still in force: the declaration has not changed.
        assert ("m", "s") in model._language_override

    def test_a_new_declaration_takes_the_microphone_back(self):
        model = self._model_with_override()
        # The production sequence: pinned to vi while declared en, then the speaker picks vi
        # themselves (declaration now matches what they speak)...
        assert model._apply_language_override(("m", "s"), "vi") == "vi"
        assert ("m", "s") not in model._language_override
        # ...and later picks en again and actually speaks English. With the override released,
        # the fresh declaration wins — this exact call returned "vi" in production forever.
        assert model._apply_language_override(("m", "s"), "en") == "en"

    def test_release_also_resets_the_evidence_count(self):
        model = self._model_with_override()
        model._language_evidence[("m", "s")] = ("vi", 1)
        model._apply_language_override(("m", "s"), "vi")
        # A half-accumulated count from the old declaration must not carry over: the next
        # override has to be earned against the NEW declaration from zero.
        assert model._language_evidence == {}

    def test_still_speaking_the_other_language_relearns_the_override(self):
        model = self._model_with_override()
        model._apply_language_override(("m", "s"), "ja")  # released
        # The person declared ja but keeps audibly speaking Vietnamese: same two-segment bar
        # as the first time, and the override comes back — scoped to the new declaration.
        model._learn_language_evidence(
            ("m", "s"), "ja", [_segment("Chào anh", "vi"), _segment("Đổi tên đi", "vi")]
        )
        assert model._language_override[("m", "s")] == ("vi", "ja")

    def test_an_uncontradicted_speaker_is_untouched(self):
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._language_override = {}
        assert model._apply_language_override(("m", "s"), "en") == "en"
