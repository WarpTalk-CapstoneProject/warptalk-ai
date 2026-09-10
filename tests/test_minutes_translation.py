"""WT-665: the biên bản keeps its other languages when a summary language is chosen.

The pairing these tests defend is positional — MinutesBilingualPairing walks the source
sections and the translated sections together — so the only thing that can go wrong quietly is
a translated list that no longer lines up with the one it is paired against. A reader who
speaks one of the two languages cannot see it; the document just attributes the wrong sentence
to the wrong moment.

So the interesting cases here are all about refusing, not about translating.
"""

from __future__ import annotations

from typing import Any

from ai_assistant_worker.minutes_translation import collect_translatable, merge_translation


def _summary() -> dict[str, Any]:
    return {
        "summary": "Chốt công nợ quý ba.",
        "decisions": [
            {"text": "Duyệt ngân sách", "atMs": 12_000, "alsoAtMs": [15_000]},
            {"text": "Hoãn tuyển dụng", "atMs": 40_000},
        ],
        "actionItems": [
            {"task": "Gửi báo cáo", "owner": "Tú", "atMs": 61_000},
        ],
        "citations": [{"id": "S1"}],
        "templateKey": "general",
        "summaryLanguage": "vi",
        "insufficientData": False,
    }


def test_collects_only_the_words_a_reader_sees() -> None:
    payload = collect_translatable(_summary())

    assert payload == {
        "summary": "Chốt công nợ quý ba.",
        "decisions": ["Duyệt ngân sách", "Hoãn tuyển dụng"],
        "actionItems": ["Gửi báo cáo"],
    }
    # Moments and bookkeeping never reach the model — it cannot return a moment it was never
    # shown, which is what lets translations skip the citation check honestly.
    assert "citations" not in payload
    assert "templateKey" not in payload
    assert "summaryLanguage" not in payload


def test_merged_items_keep_the_source_moments_and_owner() -> None:
    merged = merge_translation(
        _summary(),
        {
            "summary": "Q3 receivables settled.",
            "decisions": ["Budget approved", "Hiring deferred"],
            "actionItems": ["Send the report"],
        },
    )

    assert merged is not None
    assert merged["summary"] == "Q3 receivables settled."
    assert merged["decisions"][0] == {
        "text": "Budget approved",
        "atMs": 12_000,
        "alsoAtMs": [15_000],
    }
    assert merged["decisions"][1]["atMs"] == 40_000
    # A person's name is not a word with a translation. Whatever the model returned for the
    # task, the owner is the one the meeting said.
    assert merged["actionItems"][0] == {
        "task": "Send the report",
        "owner": "Tú",
        "atMs": 61_000,
    }


def test_a_shorter_list_is_refused_rather_than_zipped() -> None:
    # Two decisions in, one back. Zipping would leave "Hiring deferred" paired with nothing, or
    # worse, shift every later item onto the moment before it.
    assert merge_translation(_summary(), {"summary": "x", "decisions": ["Budget approved"]}) is None


def test_a_longer_list_is_refused_too() -> None:
    assert (
        merge_translation(
            _summary(),
            {"decisions": ["a", "b", "c"], "actionItems": ["d"]},
        )
        is None
    )


def test_a_section_the_model_ignored_is_dropped_not_faked() -> None:
    # `actionItems` absent from the answer: the language keeps the sections it did translate.
    # The alternative — copying the source text in — would put Vietnamese under an English
    # heading and look like a translation that had been checked.
    merged = merge_translation(
        _summary(),
        {"summary": "Q3 receivables settled.", "decisions": ["Budget approved", "Hiring deferred"]},
    )

    assert merged is not None
    assert "actionItems" not in merged
    assert len(merged["decisions"]) == 2


def test_an_empty_string_leaves_the_source_line_standing() -> None:
    # A blank in one position is not a reason to lose the item: the lists must stay the same
    # length or the pairing breaks, and the source words are more use than an empty bullet.
    merged = merge_translation(
        _summary(),
        {"decisions": ["Budget approved", "   "], "actionItems": ["Send the report"]},
    )

    assert merged is not None
    assert merged["decisions"][1]["text"] == "Hoãn tuyển dụng"


def test_nothing_usable_produces_nothing() -> None:
    assert merge_translation(_summary(), None) is None
    assert merge_translation(_summary(), "not an object") is None
    assert merge_translation(_summary(), {}) is None


def test_bare_string_items_survive_the_round_trip() -> None:
    # Items predating citations are plain strings. `ReadItems` on the backend still accepts
    # them, so this module has to as well.
    source = {"summary": "s", "decisions": ["một", "hai"]}

    merged = merge_translation(source, {"summary": "s2", "decisions": ["one", "two"]})

    assert merged == {"summary": "s2", "decisions": ["one", "two"]}


def test_a_summary_with_no_prose_asks_for_nothing() -> None:
    assert collect_translatable({"templateKey": "general", "insufficientData": True}) == {}
