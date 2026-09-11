"""A cited moment is only worth anything if somebody checks it. Nobody did.

`atMs` is what makes a summary item checkable: click it and the meeting page scrolls the
transcript to the moment the item came from. But the page's lookup is "the last segment at or
before this moment, otherwise the first one after", so ANY number resolves to a segment. A
moment the model invented does not fail — it scrolls the reader somewhere plausible and wrong,
which is the one outcome worse than no citation.

The traceable template lets a sentence rest on up to three moments, so it triples that
exposure. These tests are the check that ships with it.
"""

from __future__ import annotations

from typing import Any

from ai_assistant_worker.summary_grounding import ground_summary, strip_all_citations
from ai_assistant_worker.summary_templates import format_transcript_line

# The moments this meeting actually has. Built through `format_transcript_line` so the tests
# check citations against the same string the model is shown, not against a hand-written
# imitation of it that could drift.
_TRANSCRIPT = "\n".join(
    [
        format_transcript_line(0, "Nhi", "chốt công nợ quý ba"),
        format_transcript_line(12_000, "Ky", "gửi hợp đồng chiều nay"),
        format_transcript_line(30_500, "Tuan", "ai lo phần thuế"),
    ]
)


def _narrative(*items: dict[str, Any]) -> dict[str, Any]:
    return {"narrative": list(items), "templateKey": "traceable", "insufficientData": False}


def _first(summary: dict[str, Any], section: str = "narrative") -> dict[str, Any]:
    item: dict[str, Any] = summary[section][0]
    return item


def test_a_moment_the_transcript_carries_survives_and_an_invented_one_does_not() -> None:
    grounded = ground_summary(
        _narrative(
            {"text": "Nhi chốt công nợ.", "atMs": 12_000},
            {"text": "Và rồi ai đó nói điều này.", "atMs": 999_999},
        ),
        _TRANSCRIPT,
    )

    kept, invented = grounded.summary["narrative"]
    assert kept["atMs"] == 12_000
    assert invented["atMs"] is None
    assert grounded.moments_dropped == 1


def test_a_near_miss_is_dropped_because_snapping_it_would_manufacture_a_citation() -> None:
    """Rounding 11_999 to 12_000 would be a bug, not a kindness.

    A one-millisecond miss is not a transcript that moved; it is a number the model did not
    read off the prompt. Snapping it produces a jump that lands on a real line and therefore
    looks verified — which is precisely the failure this module exists to expose. The model
    was handed these integers and asked to repeat one, so there is nothing to forgive.
    """
    grounded = ground_summary(
        _narrative(
            {"text": "Một mili giây lệch.", "atMs": 11_999, "alsoAtMs": [30_501]},
        ),
        _TRANSCRIPT,
    )

    item = _first(grounded.summary)
    assert item["atMs"] is None, "a near miss was snapped to the nearest real moment"
    assert item["alsoAtMs"] == []
    assert grounded.moments_dropped == 2


def test_one_bad_moment_does_not_cost_a_sentence_its_good_ones() -> None:
    grounded = ground_summary(
        _narrative(
            {"text": "Câu này dựa vào ba chỗ.", "atMs": 0, "alsoAtMs": [12_000, 777_000]},
        ),
        _TRANSCRIPT,
    )

    item = _first(grounded.summary)
    assert item["atMs"] == 0
    assert item["alsoAtMs"] == [12_000]
    assert (grounded.moments_checked, grounded.moments_dropped) == (3, 1)
    assert grounded.items_uncited == 0


def test_an_unusable_atms_is_replaced_by_the_earliest_moment_that_survives() -> None:
    # The traceable template anchors a merged sentence to the EARLIEST moment it rests on, so
    # that the summary reads in meeting order. A promotion has to honour the same rule.
    grounded = ground_summary(
        _narrative(
            {"text": "Neo hỏng, phần còn lại thì không.", "atMs": 5, "alsoAtMs": [30_500, 12_000]},
        ),
        _TRANSCRIPT,
    )

    item = _first(grounded.summary)
    assert item["atMs"] == 12_000
    assert item["alsoAtMs"] == [30_500]
    assert grounded.items_uncited == 0


def test_an_item_with_nothing_left_to_point_at_is_kept_rather_than_deleted() -> None:
    """Deleting it would shorten the summary without saying so.

    An uncited line admits it cannot be checked; a line that vanished tells the reader
    nothing at all, and the web already renders a null `atMs` as text instead of a control.
    """
    grounded = ground_summary(
        _narrative({"text": "Không neo được vào đâu.", "atMs": 4_000, "alsoAtMs": [5_000]}),
        _TRANSCRIPT,
    )

    item = _first(grounded.summary)
    assert item["text"] == "Không neo được vào đâu."
    assert item["atMs"] is None
    assert item["alsoAtMs"] == []
    assert grounded.items_uncited == 1


def test_alsoatms_is_sorted_deduplicated_and_never_echoes_atms() -> None:
    grounded = ground_summary(
        _narrative(
            {"text": "Lặp và lộn xộn.", "atMs": 12_000, "alsoAtMs": [30_500, 0, 12_000, 30_500]},
        ),
        _TRANSCRIPT,
    )

    item = _first(grounded.summary)
    assert item["atMs"] == 12_000
    assert item["alsoAtMs"] == [0, 30_500]
    # A moment written twice is still a real moment; normalising it away is not a drop.
    assert grounded.moments_dropped == 0


def test_every_section_is_checked_and_action_items_keep_their_own_fields() -> None:
    grounded = ground_summary(
        {
            "decisions": [{"text": "Chốt ngày ship.", "atMs": 0}, {"text": "Bịa.", "atMs": 61}],
            "actionItems": [
                {"task": "Gửi hợp đồng", "owner": "Ky", "atMs": 12_000},
                {"task": "Hỏi thuế", "owner": "", "atMs": 1},
            ],
            "openQuestions": [{"text": "Ai lo phần thuế?", "atMs": 30_500}],
        },
        _TRANSCRIPT,
    )

    assert [item["atMs"] for item in grounded.summary["decisions"]] == [0, None]
    assert grounded.summary["openQuestions"][0]["atMs"] == 30_500

    sent, tax = grounded.summary["actionItems"]
    assert sent == {"task": "Gửi hợp đồng", "owner": "Ky", "atMs": 12_000}
    assert (tax["task"], tax["owner"]) == ("Hỏi thuế", "")
    assert tax["atMs"] is None
    assert grounded.items_uncited == 2


def test_the_fields_that_carry_no_citation_pass_through_untouched() -> None:
    payload: dict[str, Any] = {
        "summary": "Cuộc họp chốt công nợ quý ba.",
        "templateKey": "traceable",
        "insufficientData": False,
        "translations": {"en": {"summary": "Q3 receivables", "decisions": [{"atMs": 999_999}]}},
    }

    grounded = ground_summary(payload, _TRANSCRIPT)

    assert grounded.summary == payload
    assert grounded.moments_checked == 0


def test_a_citation_that_points_at_nothing_is_removed_rather_than_blanked() -> None:
    # A citation entry is nothing but a link, so a null one is a chip pointing nowhere. No
    # text is lost by removing it — which is exactly why an ITEM is treated the other way.
    grounded = ground_summary(
        {
            "summary": "Overview.",
            "citations": [{"key": "summary", "atMs": 0}, {"key": "summary", "atMs": 42}],
        },
        _TRANSCRIPT,
    )

    assert grounded.summary["citations"] == [{"key": "summary", "atMs": 0}]
    assert grounded.moments_dropped == 1


def test_an_empty_transcript_leaves_everything_uncited_instead_of_raising() -> None:
    grounded = ground_summary(
        _narrative(
            {"text": "Một.", "atMs": 0},
            {"text": "Hai.", "atMs": 12_000, "alsoAtMs": [30_500]},
        ),
        "",
    )

    assert [item["atMs"] for item in grounded.summary["narrative"]] == [None, None]
    assert grounded.items_uncited == 2
    assert grounded.moments_dropped == 3


def test_a_value_that_was_never_a_moment_is_not_a_moment() -> None:
    # `True == 1` in Python, so a JSON `true` would validate against a 1 ms offset if the
    # boolean were not ruled out before the integer check. A whole-numbered float is the same
    # integer written differently and is kept; a fraction and a string are not moments.
    grounded = ground_summary(
        _narrative(
            {"text": "Số thực.", "atMs": 12_000.0},
            {"text": "Chuỗi.", "atMs": "12000"},
            {"text": "Đúng.", "atMs": True},
            {"text": "Lẻ.", "atMs": 12_000.5},
            {"text": "Rỗng.", "atMs": None},
        ),
        "\n".join([format_transcript_line(1, "Nhi", "một"), *_TRANSCRIPT.splitlines()]),
    )

    assert [item["atMs"] for item in grounded.summary["narrative"]] == [
        12_000,
        None,
        None,
        None,
        None,
    ]


def test_an_item_that_never_claimed_a_moment_is_left_exactly_as_it_was() -> None:
    # A section the model answered without citing, or an item from an older payload. Neither
    # is a failed citation, so neither is counted as one — and nothing is added to it.
    payload = {"decisions": ["Ship the beta by August"], "actionItems": [{"owner": "Alice"}]}

    grounded = ground_summary(payload, _TRANSCRIPT)

    assert grounded.summary == payload
    assert (grounded.moments_checked, grounded.items_uncited) == (0, 0)


def test_the_summary_handed_in_is_never_modified() -> None:
    # The caller keeps the parsed response; grounding must not rewrite it underneath them.
    payload = _narrative({"text": "Bịa.", "atMs": 999_999, "alsoAtMs": [0, 888]})
    before = {"atMs": 999_999, "alsoAtMs": [0, 888]}

    ground_summary(payload, _TRANSCRIPT)

    item = _first(payload)
    assert (item["atMs"], item["alsoAtMs"]) == (before["atMs"], before["alsoAtMs"])


def test_every_moment_the_model_wrote_is_counted_once() -> None:
    grounded = ground_summary(
        {
            "narrative": [{"text": "Một.", "atMs": 0, "alsoAtMs": [12_000, 999]}],
            "decisions": [{"text": "Hai.", "atMs": 77}],
            "citations": [{"key": "summary", "atMs": 30_500}],
        },
        _TRANSCRIPT,
    )

    # 3 in the narrative item, 1 decision, 1 citation. The dropped pair is the narrative's
    # 999 and the decision's 77; the decision is the one item left uncited, because the
    # citation entry was removed instead.
    assert grounded.moments_checked == 5
    assert grounded.moments_dropped == 2
    assert grounded.items_uncited == 1


# --- When the CHECK is what breaks -------------------------------------------------------
#
# `ground_summary` used to run inside the same `try` that catches a malformed model response,
# so an exception raised by the checker was reported as `generationFailed`: the summary thrown
# away, the previous one left standing, and — because a failed rewrite has no path back to the
# browser — nothing said to the person who asked. These pin the third answer.


def test_stripping_keeps_every_word_and_removes_every_link() -> None:
    payload = {
        "narrative": [
            {"text": "Một.", "atMs": 0, "alsoAtMs": [12_000]},
            {"text": "Hai.", "atMs": 30_500, "alsoAtMs": []},
        ],
        "actionItems": [{"task": "Gửi hợp đồng.", "owner": "Ky", "atMs": 12_000}],
        "citations": [{"key": "summary", "atMs": 0}],
        "summary": "Nội dung.",
        "templateKey": "traceable",
        "insufficientData": False,
    }

    stripped = strip_all_citations(payload)

    assert [item["text"] for item in stripped["narrative"]] == ["Một.", "Hai."]
    assert all(item["atMs"] is None for item in stripped["narrative"])
    assert all(item["alsoAtMs"] == [] for item in stripped["narrative"])
    # The owner and the words of an action item are not provenance and must survive.
    assert stripped["actionItems"][0] == {
        "task": "Gửi hợp đồng.",
        "owner": "Ky",
        "atMs": None,
        "alsoAtMs": [],
    }
    # A citation entry is a link and nothing else, so it has no remainder worth keeping.
    assert stripped["citations"] == []
    assert stripped["summary"] == "Nội dung."
    assert stripped["templateKey"] == "traceable"


def test_stripping_does_not_mutate_what_it_was_given() -> None:
    payload = _narrative({"text": "Một.", "atMs": 0, "alsoAtMs": [12_000]})
    before = dict(_first(payload))

    strip_all_citations(payload)

    assert _first(payload) == before


def test_stripping_leaves_an_item_that_never_claimed_a_moment_alone() -> None:
    payload = {"decisions": [{"text": "Không mốc."}], "templateKey": "general"}

    assert strip_all_citations(payload)["decisions"] == [{"text": "Không mốc."}]
