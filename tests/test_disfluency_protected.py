"""`protected_indices` — what tier 1 refuses to delete on principle, for tier 2 to respect (WT-716).

This is a narrower set than `Token.protected` inside the prepass: a self-repair marker ("à
không", "I mean", "じゃなくて") is also `.protected` there, so the prepass's OWN filler/stutter
passes leave it alone, but it is deliberately absent from `protected_indices` — resolving the
repair by deleting the marker is the LLM tier's job, and protecting it here would make that
impossible. See `shared/disfluency/prepass.py::_Run.protect_for_llm` for the one-paragraph ruling.
"""

from __future__ import annotations

from shared.disfluency import protected_indices, tokenize


class TestVietnamese:
    def test_tu_tu_reduplication_is_protected(self):
        text = "mình cứ từ từ làm, không vội"
        tokens = tokenize(text, "vi")
        assert tokens[2] == "từ" and tokens[3] == "từ"
        assert protected_indices(text, "vi") == frozenset({2, 3})

    def test_ba_ba_reduplication_is_protected(self):
        text = "con ba ba to quá"
        tokens = tokenize(text, "vi")
        assert tokens[1] == "ba" and tokens[2] == "ba"
        assert protected_indices(text, "vi") == frozenset({1, 2})

    def test_vocative_a_after_a_kinship_term_is_protected(self):
        text = "chị à, cho em hỏi chút"
        tokens = tokenize(text, "vi")
        assert tokens[1] == "à"
        assert protected_indices(text, "vi") == frozenset({1})

    def test_a_bare_a_with_no_kinship_term_before_it_is_not_protected(self):
        # Nothing licenses treating this "à" as a vocative, so it is not in the protected set —
        # it is still up to the fillers/self-repair rules elsewhere to decide its fate.
        text = "sáng nay à, tôi bận"
        assert protected_indices(text, "vi") == frozenset()

    def test_the_self_repair_marker_is_not_protected(self):
        # "à không" is a correction marker: tier 2 has to be free to delete it to resolve the
        # repair it was escalated for, so it must NOT show up here even though the prepass marks
        # it `.protected` internally.
        text = "họp thứ hai, à không, thứ ba"
        assert protected_indices(text, "vi") == frozenset()

    def test_a_nham_self_repair_marker_is_not_protected(self):
        text = "gửi cho anh Nam, à nhầm, anh Nam Anh"
        assert protected_indices(text, "vi") == frozenset()

    def test_dạ_vâng_are_protected(self):
        text = "dạ, em hiểu rồi"
        tokens = tokenize(text, "vi")
        assert tokens[0] == "dạ"
        assert protected_indices(text, "vi") == frozenset({0})


class TestEnglish:
    def test_very_very_is_protected(self):
        text = "it was very very expensive"
        tokens = tokenize(text, "en")
        assert tokens[2] == "very" and tokens[3] == "very"
        assert protected_indices(text, "en") == frozenset({2, 3})

    def test_uh_huh_is_protected(self):
        text = "uh-huh that makes sense"
        tokens = tokenize(text, "en")
        assert tokens[0] == "uh-huh"
        assert protected_indices(text, "en") == frozenset({0})

    def test_a_single_very_is_not_protected(self):
        # Only an actual repeated run is reduplication; a lone intensifier is an ordinary word
        # and outside this function's job entirely.
        text = "it was very expensive"
        assert protected_indices(text, "en") == frozenset()

    def test_the_i_mean_self_repair_marker_is_not_protected(self):
        text = "We ship on Monday, I mean Tuesday"
        assert protected_indices(text, "en") == frozenset()


class TestJapanese:
    def test_mada_mada_reduplication_is_protected(self):
        text = "まだまだ頑張ります"
        tokens = tokenize(text, "ja")
        assert tokens[0] == "まだまだ"
        assert protected_indices(text, "ja") == frozenset({0})

    def test_ano_as_a_demonstrative_is_protected(self):
        text = "あの資料はもう送りました。"
        tokens = tokenize(text, "ja")
        assert tokens[0] == "あの"
        assert protected_indices(text, "ja") == frozenset({0})

    def test_the_elongated_filler_form_is_not_protected(self):
        # "あのー" (the elongated form) is a hesitation, unlike bare "あの" — it is tier 1's own
        # A1 filler list, deleted outright rather than protected.
        text = "あのー、明日でお願いします"
        assert protected_indices(text, "ja") == frozenset()

    def test_the_janakute_self_repair_marker_is_not_protected(self):
        text = "月曜、じゃなくて火曜です"
        assert protected_indices(text, "ja") == frozenset()

    def test_a_real_contrast_is_untouched_by_this_function(self):
        # Not a repair test — just confirming protected_indices does not error or over-claim on
        # the ticket's own contrast example.
        text = "赤じゃなくて青がいいです。"
        assert protected_indices(text, "ja") == frozenset()


class TestEdgeCases:
    def test_empty_text_is_protected_nowhere(self):
        assert protected_indices("", "en") == frozenset()
        assert protected_indices("   ", "vi") == frozenset()

    def test_an_unsupported_language_protects_nothing(self):
        assert protected_indices("hello", "ko") == frozenset()

    def test_indices_line_up_with_lexical_tokens(self):
        # The whole point of the function: its indices must be usable directly against
        # `tokenize`/`lexical_tokens`, with no separate re-numbering needed by the caller.
        text = "it was very very expensive, believe me"
        tokens = tokenize(text, "en")
        for index in protected_indices(text, "en"):
            assert tokens[index] == "very"
