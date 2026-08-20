"""Putting per-speaker transcriptions back together as one meeting.

Two things are being defended. Speaker attribution must stay a FACT — it comes from which file a
segment was in, and nothing here may turn it back into a guess. And a model that filled a silence
must not get its invention into a transcript people will read, quote and sign a biên bản from.
"""

from __future__ import annotations

from retranscribe_worker.merge import (
    SpeakerSegment,
    SpeechSpan,
    is_within_speech,
    load_spans,
    merge_speakers,
)


def seg(start: int, end: int, text: str = "xin chào") -> SpeakerSegment:
    return SpeakerSegment(start_ms=start, end_ms=end, text=text)


def test_a_segments_speaker_is_the_file_it_came_from():
    # The half that costs every other notetaker accuracy, and costs this one nothing.
    merged = merge_speakers({"tu": [seg(0, 1000, "một")], "nhi": [seg(2000, 3000, "hai")]})

    assert [(m.speaker_id, m.text) for m in merged] == [("tu", "một"), ("nhi", "hai")]


def test_the_meeting_is_ordered_by_when_each_line_started():
    merged = merge_speakers(
        {
            "tu": [seg(5000, 6000, "sau"), seg(0, 1000, "trước")],
            "nhi": [seg(2500, 3000, "giữa")],
        }
    )

    assert [m.text for m in merged] == ["trước", "giữa", "sau"]


def test_two_lines_starting_at_the_same_instant_order_stably():
    # Arbitrary, but it must not change between runs: two runs of one meeting producing different
    # orders would read as a diff in a transcript nobody edited.
    first = merge_speakers({"b": [seg(1000, 2000, "B")], "a": [seg(1000, 2000, "A")]})
    second = merge_speakers({"a": [seg(1000, 2000, "A")], "b": [seg(1000, 2000, "B")]})

    assert [m.speaker_id for m in first] == [m.speaker_id for m in second] == ["a", "b"]


def test_overlapping_speech_is_kept_not_resolved():
    # Two people talking at once is a thing that happened, and a transcript that silently picked
    # one of them would be editing the meeting.
    merged = merge_speakers(
        {"tu": [seg(1000, 4000, "tôi nghĩ")], "nhi": [seg(2000, 5000, "đồng ý")]}
    )

    assert len(merged) == 2


def test_a_segment_the_model_invented_over_silence_is_dropped():
    # The failure this filter exists for: the archive is mostly silence by construction, and
    # Whisper-family models fill silence rather than skipping it.
    spans = {"tu": [SpeechSpan(0, 1000)]}
    merged = merge_speakers({"tu": [seg(0, 1000, "thật"), seg(30_000, 33_000, "bịa")]}, spans)

    assert [m.text for m in merged] == ["thật"]


def test_a_real_utterance_slightly_outside_vads_opinion_survives():
    # Speech routinely starts a little before VAD notices, so "entirely inside" would delete real
    # lines. More inside than outside is the bar.
    spans = [SpeechSpan(1000, 3000)]

    assert is_within_speech(seg(800, 3000), spans) is True


def test_a_segment_that_merely_touches_the_end_of_real_speech_is_not_believed():
    # A filled silence often begins exactly where the previous utterance ended, so any-overlap
    # would wave most inventions through.
    spans = [SpeechSpan(0, 1000)]

    assert is_within_speech(seg(900, 5000), spans) is False


def test_overlap_is_counted_across_every_span_not_just_one():
    spans = [SpeechSpan(0, 500), SpeechSpan(600, 1100)]

    # 500 + 400 of a 1000ms segment is inside speech.
    assert is_within_speech(seg(100, 1100), spans) is True


def test_no_span_index_means_believe_everything():
    # An archive written before the sidecar, or one whose write failed. Losing the index must not
    # silently throw the meeting's transcript away.
    merged = merge_speakers({"tu": [seg(0, 1000, "một"), seg(90_000, 91_000, "hai")]}, {})

    assert len(merged) == 2


def test_a_malformed_index_reads_as_no_index_rather_than_as_no_speech():
    assert load_spans(None) == []
    assert load_spans({"spans": "nonsense"}) == []
    assert load_spans({"spans": [{"startMs": "a", "endMs": 5}, {"startMs": 5, "endMs": 5}]}) == []
    assert load_spans({"spans": [{"startMs": 0, "endMs": 10}]}) == [SpeechSpan(0, 10)]


def test_empty_text_never_reaches_the_transcript():
    merged = merge_speakers({"tu": [seg(0, 1000, "   "), seg(2000, 3000, "thật")]})

    assert [m.text for m in merged] == ["thật"]


def test_text_is_trimmed_but_not_otherwise_touched():
    merged = merge_speakers({"tu": [seg(0, 1000, "  Xin chào các bạn.  ")]})

    assert merged[0].text == "Xin chào các bạn."


def test_a_zero_length_segment_is_kept():
    # No evidence either way, and dropping it would lose a real one-word utterance the model
    # happened to time badly.
    assert is_within_speech(seg(5000, 5000), [SpeechSpan(0, 1000)]) is True
