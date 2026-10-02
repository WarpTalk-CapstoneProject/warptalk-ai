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
    _release_verdict,
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


class TestSpeakingTheDeclarationReleasesTheOverride:
    """Production meeting 01a0fbe6 (2 Oct 2026).

    Tú declared en, said two short Vietnamese lines, and the override en -> vi was learned at
    16:18:19 — correctly. Tú then spoke English for the rest of the meeting, and every sentence
    was stored as vi, so the vi listener got no translation of any of it. The only exit was to
    re-declare, which nothing on screen told anyone to do. These are the lines Tú actually said.
    """

    KEY = ("01a0fbe6", "019f0d00-0de0-7000-9000-000000000001")

    def _model(self) -> OpenAISTT:
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._release_evidence = {}
        model._language_override = {self.KEY: ("vi", "en")}
        return model

    def _hear(self, model: OpenAISTT, *texts: str) -> list[TranscribedSegment]:
        # Under the vi pin, Latin text with no evidence comes back labelled vi — that is the bug.
        segments = [_segment(text, "vi") for text in texts]
        pinned = model._apply_language_override(self.KEY, "en")
        model._learn_language_evidence(self.KEY, pinned, segments)
        return segments

    def test_two_english_sentences_take_the_microphone_back(self):
        model = self._model()
        segments = self._hear(model, "Good morning.", "AI is great.", "I use it for reports.")

        assert self.KEY not in model._language_override
        assert model._apply_language_override(self.KEY, "en") == "en"
        # This chunk's own lines ship under the language they proved, not the one they disproved.
        assert [s.language for s in segments] == ["en", "en", "en"]

    def test_the_evidence_may_span_chunks(self):
        model = self._model()
        self._hear(model, "I use it for reports.")
        assert self.KEY in model._language_override

        self._hear(model, "AI can be wrong.")
        assert self.KEY not in model._language_override

    def test_one_sentence_is_not_enough(self):
        model = self._model()
        segments = self._hear(model, "I use it for reports.")

        assert model._language_override[self.KEY] == ("vi", "en")
        assert segments[0].language == "vi"

    def test_vietnamese_in_between_resets_the_count(self):
        model = self._model()
        self._hear(model, "I use it for reports.", "Rồi, bắt đầu đi.", "AI can be wrong.")

        assert model._language_override[self.KEY] == ("vi", "en")

    def test_short_acknowledgements_prove_nothing(self):
        model = self._model()
        self._hear(model, "Yeah.", "OK, hey", "So", "It is", "We must")

        assert model._language_override[self.KEY] == ("vi", "en")

    def test_a_short_line_does_not_break_a_run_either(self):
        model = self._model()
        self._hear(model, "I use it for reports.", "Yeah.", "AI can be wrong.")

        assert self.KEY not in model._language_override

    def test_vietnamese_without_its_unique_letters_is_still_not_english(self):
        # "Tôi là Nam" carries none of the Vietnamese-unique class — that class is narrow on
        # purpose — but it is not ASCII, and that is what keeps it from counting as English.
        model = self._model()
        self._hear(model, "Tôi là Nam", "Em là ai vậy", "Cho nên là")

        assert model._language_override[self.KEY] == ("vi", "en")

    def test_a_hallucinated_third_language_is_neutral(self):
        model = self._model()
        self._hear(model, "I use it for reports.", "嗯", "AI can be wrong.")

        assert self.KEY not in model._language_override

    def test_a_vietnamese_speaker_keeps_their_override(self):
        model = self._model()
        self._hear(
            model,
            "Hôm nay chúng ta sẽ bàn về AI.",
            "Mọi người đã trải nghiệm AI như thế nào rồi?",
        )

        assert model._language_override[self.KEY] == ("vi", "en")

    def test_released_speaker_can_earn_the_override_again(self):
        model = self._model()
        self._hear(model, "AI is great.", "I use it for reports.")
        assert self.KEY not in model._language_override

        model._learn_language_evidence(
            self.KEY, "en", [_segment("Chào anh", "vi"), _segment("Đổi tên đi", "vi")]
        )
        assert model._language_override[self.KEY] == ("vi", "en")

    def test_redeclaring_clears_half_collected_release_evidence(self):
        model = self._model()
        self._hear(model, "I use it for reports.")
        model._apply_language_override(self.KEY, "ja")

        assert self.KEY not in model._release_evidence


class TestReleaseVerdict:
    def test_script_evidence_for_the_declaration_supports_release(self):
        # declared ja, re-pinned to vi, now audibly Japanese again.
        assert _release_verdict("今日はいい天気です", "vi", "ja") is True

    def test_proof_of_the_learned_language_opposes_release(self):
        assert _release_verdict("Đổi tên đi", "vi", "en") is False

    def test_ascii_latin_cannot_speak_for_a_non_latin_declaration(self):
        assert _release_verdict("I use it for reports.", "vi", "ja") is None

    def test_ascii_latin_cannot_speak_for_a_vietnamese_declaration(self):
        # Declared vi, re-pinned to ja: plain Latin text does not prove Vietnamese.
        assert _release_verdict("I use it for reports.", "ja", "vi") is None

    def test_a_regional_declaration_is_compared_on_its_base(self):
        assert _release_verdict("I use it for reports.", "vi", "en-US") is True
