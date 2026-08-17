"""What counts as somebody speaking, when a meeting is summarised. WT-478.

A meeting with a full transcript on screen reported a summary of "The transcript is empty and
contains no substantive meeting content". Nothing in the code says that — the model wrote it,
because the transcript it was handed was scaffolding: timestamps and speaker labels with
nothing between them, plus the ``__MEETING_END__`` marker that had triggered the summary in
the first place. The formatted string was truthy, so the emptiness check passed and the call
succeeded, and a refusal was stored as the summary.
"""

from __future__ import annotations

from ai_assistant_worker.summary_templates import format_transcript_line, spoken_text_only
from ai_assistant_worker.worker import substantive_segments
from shared.control_markers import MEETING_END_MARKER


def test_a_segment_with_no_text_is_not_speech() -> None:
    kept = substantive_segments(
        [
            ("Nhi", "chốt công nợ quý ba", 1_000),
            ("Ky", "", 2_000),
            ("Tuan", "   ", 3_000),
            ("Nhi", "gửi hợp đồng chiều nay", 4_000),
        ]
    )

    assert [text for _, text, _ in kept] == ["chốt công nợ quý ba", "gửi hợp đồng chiều nay"]


def test_the_end_of_meeting_marker_is_not_summarised() -> None:
    # `process` appends every segment BEFORE testing for the marker, so the marker that
    # triggers summarisation is in the list being summarised.
    kept = substantive_segments(
        [
            ("Nhi", "chốt công nợ quý ba", 1_000),
            ("system", MEETING_END_MARKER, 2_000),
        ]
    )

    assert [text for _, text, _ in kept] == ["chốt công nợ quý ba"]


def test_a_meeting_of_only_scaffolding_and_a_marker_has_nothing_to_summarise() -> None:
    """The reported case, end to end at this layer.

    Every one of these produced a formatted line, so the old code saw a non-empty transcript
    and asked the model to summarise punctuation. Now it collapses to nothing and the caller
    returns before the model is involved.
    """
    segments = [
        ("Nhi", "", 1_000),
        ("Ky", "  ", 2_000),
        ("system", MEETING_END_MARKER, 3_000),
    ]

    assert substantive_segments(segments) == []

    # And the same judgement holds after formatting, which is what the assistant checks.
    formatted = "\n".join(
        format_transcript_line(ts, speaker, text) for speaker, text, ts in segments
    )
    assert formatted.strip(), "precondition: the formatted transcript is not blank"
    assert spoken_text_only(formatted) == MEETING_END_MARKER


def test_one_short_sentence_is_still_a_meeting() -> None:
    # No minimum length: "kể cả khi nội dung ngắn" is the ticket's own wording.
    assert substantive_segments([("Nhi", "đồng ý", 1_000)]) == [("Nhi", "đồng ý", 1_000)]
