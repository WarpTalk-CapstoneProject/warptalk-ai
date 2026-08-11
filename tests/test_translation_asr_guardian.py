"""The translator's ASR-repair behaviour: what it is allowed to fix, and what it must not.

Speech recognition hands the translator text that may contain misheard words. Left alone,
the translator faithfully translates the mistake into a confident wrong sentence. These
tests pin the two halves of the guard that stops that:

  * a glossary term is still offered when the utterance merely SOUNDS like it, which is
    exactly the moment literal matching stops working, and
  * it is offered as a suspicion rather than a fact, because a fluent invention is worse
    than a visible mistranscription — a reader spots "cô đích" as nonsense at once, but
    cannot tell a confidently repaired sentence from a real one.

The similarity numbers here were measured against real Vietnamese sentences; see the
comment on _MISHEARD_SIMILARITY for the table they came from.
"""

from __future__ import annotations

from translation_worker.translator import (
    _ASR_REPAIR_INSTRUCTION,
    _BATCH_SYSTEM_PROMPT,
    _MIN_MISHEARD_SKELETON,
    _SYSTEM_PROMPT,
    _build_glossary_block,
    _match_skeleton,
    _select_relevant_glossary_terms,
    _sounds_like,
)


class TestMatchSkeleton:
    def test_vietnamese_folds_toward_the_english_it_renders(self) -> None:
        """Diacritics and spacing are what separate a mishearing from its source."""
        assert _match_skeleton("Warp Talk") == _match_skeleton("WarpTalk")
        assert _match_skeleton("xì ta ging") == "xitaging"

    def test_bar_d_is_mapped_by_hand(self) -> None:
        """đ has no canonical decomposition, so NFD alone leaves it intact."""
        assert _match_skeleton("đích") == "dich"


class TestSoundsLike:
    def test_recognises_a_mangled_term(self) -> None:
        assert _sounds_like("WarpTalk", "Bên Warp Talk có tính năng đó")
        assert _sounds_like("staging", "Deploy lên xì ta ging trước")
        assert _sounds_like("Kubernetes", "Con cu bơ nét tự restart pod")

    def test_leaves_unrelated_speech_alone(self) -> None:
        """A matcher that fires on ordinary speech would poison every prompt."""
        assert not _sounds_like("WarpTalk", "Chúng ta cần chốt deadline")
        assert not _sounds_like("staging", "Anh gửi lại tài liệu cho em sau buổi họp nhé")
        assert not _sounds_like("Kubernetes", "Em gửi tài liệu cho anh sau nhé")
        assert not _sounds_like("repository", "Cảm ơn mọi người đã tham gia")

    def test_short_terms_are_refused_rather_than_guessed(self) -> None:
        """The measured limit, stated as a test so nobody quietly lowers it.

        A five-character skeleton scores the same against a genuine mishearing as against
        unrelated Vietnamese, so no threshold separates them. "Codex" is therefore out of
        reach here and has to be fixed in the recogniser's keyword list instead.
        """
        assert len(_match_skeleton("Codex")) < _MIN_MISHEARD_SKELETON
        assert not _sounds_like("Codex", "Mình dùng cô đích để review lại")
        assert not _sounds_like("Codex", "Hôm nay trời đẹp quá mọi người ạ")


class TestGlossarySelection:
    def test_literal_match_is_reported_as_exact(self) -> None:
        terms = [{"source": "deadline", "target": "hạn chót"}]
        selected = _select_relevant_glossary_terms("Chốt deadline tuần này", terms)
        assert [(t["source"], t["match"]) for t in selected] == [("deadline", "exact")]

    def test_mangled_match_is_reported_as_possible(self) -> None:
        """The case the whole feature exists for: literal matching finds nothing."""
        terms = [{"source": "staging", "target": "staging"}]
        selected = _select_relevant_glossary_terms("Deploy lên xì ta ging trước", terms)
        assert [(t["source"], t["match"]) for t in selected] == [("staging", "possible")]

    def test_exact_matches_keep_their_slots(self) -> None:
        """Suspicions must never crowd out terms that are actually present."""
        terms = [
            {"source": "staging", "target": "staging"},
            {"source": "deadline", "target": "hạn chót"},
        ]
        selected = _select_relevant_glossary_terms(
            "Deploy lên xì ta ging trước khi chốt deadline", terms, limit=1
        )
        assert [t["source"] for t in selected] == ["deadline"]

    def test_nothing_relevant_sends_nothing(self) -> None:
        terms = [{"source": "Kubernetes", "target": "Kubernetes"}]
        assert _select_relevant_glossary_terms("Em gửi tài liệu cho anh nhé", terms) == []


class TestGlossaryBlock:
    def test_suspicions_are_worded_as_suspicions(self) -> None:
        """Rendering a guess as an instruction is how a translator invents content."""
        block = _build_glossary_block(
            [{"source": "staging", "target": "staging", "match": "possible"}]
        )
        assert "SOUNDS like" in block
        assert "ONLY if that makes the sentence coherent" in block
        assert "translate what was actually said and change nothing" in block

    def test_exact_terms_stay_imperative(self) -> None:
        block = _build_glossary_block(
            [{"source": "deadline", "target": "hạn chót", "match": "exact"}]
        )
        assert '"deadline" → "hạn chót"' in block
        assert "SOUNDS like" not in block

    def test_the_two_kinds_are_kept_apart(self) -> None:
        block = _build_glossary_block(
            [
                {"source": "deadline", "target": "hạn chót", "match": "exact"},
                {"source": "staging", "target": "staging", "match": "possible"},
            ]
        )
        assert block.index('"deadline"') < block.index("SOUNDS like")


class TestSystemPrompts:
    def test_both_prompts_declare_the_input_is_asr(self) -> None:
        """Without this the translator treats a mishearing as authoritative text."""
        for prompt in (_SYSTEM_PROMPT, _BATCH_SYSTEM_PROMPT):
            assert "automatic speech recognition" in prompt

    def test_repair_is_permitted_only_on_evidence(self) -> None:
        """A blanket 'fix ASR errors' rewrites sentences that were already right."""
        assert "glossary or the meeting context" in _ASR_REPAIR_INSTRUCTION
        assert "translate exactly what is written and invent nothing" in _ASR_REPAIR_INSTRUCTION
        assert "Never add, drop" in _ASR_REPAIR_INSTRUCTION
