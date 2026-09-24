"""A clean transcript line is a SENTENCE, and these are the rules that decide where one ends.

The reference case is the ticket's: Japanese STT cuts "えーと、あのー、来週の会議ですが" and
"火曜日に変更になりました" apart on a 500 ms breath. They are one sentence, they must come out
as one line, and that line has to carry BOTH segment ids — corrections and the
transcript→video mark point are keyed on them.
"""

from __future__ import annotations

from shared.disfluency import prepass
from transcript_clean_worker.segmenter import (
    CleanSegment,
    SentenceSegmenter,
    is_backchannel,
    is_finished,
    join_texts,
)


def segment(
    text: str,
    *,
    language: str = "en",
    speaker: str = "alice",
    start_ms: int = 0,
    end_ms: int | None = None,
    segment_id: str | None = None,
    arrived_at_ms: int | None = None,
) -> CleanSegment:
    """A segment as the worker builds one: raw text plus its deterministic prepass."""
    result = prepass(text, language)
    return CleanSegment(
        segment_id=segment_id or f"seg-{start_ms}-{speaker}",
        speaker_id=speaker,
        language=language,
        raw_text=text,
        clean_text=result.clean_text,
        flags=result.flags,
        start_ms=start_ms,
        end_ms=start_ms + 1000 if end_ms is None else end_ms,
        arrived_at_ms=start_ms if arrived_at_ms is None else arrived_at_ms,
    )


def make(**kwargs: int) -> SentenceSegmenter:
    return SentenceSegmenter(
        merge_gap_ms=int(kwargs.get("merge_gap_ms", 1200)),
        max_sentence_ms=int(kwargs.get("max_sentence_ms", 30000)),
        idle_flush_ms=int(kwargs.get("idle_flush_ms", 4000)),
    )


class TestTheTicketExample:
    def test_two_japanese_segments_five_hundred_ms_apart_are_one_sentence(self):
        segmenter = make()
        assert (
            segmenter.add(
                segment(
                    "えーと、あのー、来週の会議ですが",
                    language="ja",
                    start_ms=0,
                    end_ms=2000,
                    segment_id="a",
                )
            )
            == []
        )
        assert (
            segmenter.add(
                segment(
                    "火曜日に変更になりました",
                    language="ja",
                    start_ms=2500,
                    end_ms=4000,
                    segment_id="b",
                )
            )
            == []
        )

        [sentence] = segmenter.flush()
        assert sentence.prepass_text == "来週の会議ですが、火曜日に変更になりました。"
        assert sentence.segment_ids == ["a", "b"]
        assert sentence.raw_text == "えーと、あのー、来週の会議ですが、火曜日に変更になりました"


class TestMerging:
    def test_a_short_gap_continues_an_unfinished_line(self):
        segmenter = make()
        assert segmenter.add(segment("I think that", start_ms=0, end_ms=1000)) == []
        sentences = segmenter.add(segment("we should ship it.", start_ms=1500, end_ms=3000))
        assert sentences == []
        [sentence] = segmenter.flush()
        assert sentence.raw_text == "I think that we should ship it."

    def test_a_long_gap_ends_the_line_even_when_it_is_unfinished(self):
        segmenter = make(merge_gap_ms=1200)
        segmenter.add(segment("I think that", start_ms=0, end_ms=1000))
        [first] = segmenter.add(segment("we should ship it.", start_ms=4000, end_ms=5000))
        assert first.raw_text == "I think that"
        assert first.reason == "finished"
        [second] = segmenter.flush()
        assert second.raw_text == "we should ship it."

    def test_a_finished_line_is_not_merged_with_the_next_one(self):
        segmenter = make()
        segmenter.add(segment("Okay.", start_ms=0, end_ms=500))
        [first] = segmenter.add(segment("Let's start.", start_ms=700, end_ms=1500))
        assert first.raw_text == "Okay."
        [second] = segmenter.flush()
        assert second.raw_text == "Let's start."

    def test_a_segment_holding_two_sentences_stays_one_line(self):
        # Never split inside a segment: segment_ids must map cleanly back to the raw record.
        segmenter = make()
        segmenter.add(segment("Okay. Let's start.", start_ms=0, end_ms=1500))
        [sentence] = segmenter.flush()
        assert sentence.raw_text == "Okay. Let's start."
        assert sentence.segment_ids == ["seg-0-alice"]

    def test_the_recognisers_full_stop_after_a_continuation_word_does_not_end_the_line(self):
        segmenter = make()
        segmenter.add(segment("We will ship it and.", start_ms=0, end_ms=1000))
        assert segmenter.add(segment("Then tell the team.", start_ms=1200, end_ms=2500)) == []
        [sentence] = segmenter.flush()
        assert sentence.raw_text == "We will ship it and then tell the team."


class TestContinuationMarkersPerLanguage:
    def test_english(self):
        assert not is_finished("we need to ship it because.", "en")
        assert is_finished("we need to ship it.", "en")

    def test_vietnamese(self):
        assert not is_finished("Mình deploy chiều nay để.", "vi")
        assert is_finished("Mình deploy chiều nay.", "vi")

    def test_japanese(self):
        assert not is_finished("来週の会議ですが。", "ja")
        assert is_finished("火曜日に変更になりました。", "ja")

    def test_vietnamese_segments_merge_over_a_continuation_word(self):
        segmenter = make()
        segmenter.add(segment("Mình deploy chiều nay để", language="vi", start_ms=0, end_ms=900))
        assert (
            segmenter.add(
                segment("kịp demo ngày mai.", language="vi", start_ms=1300, end_ms=2500)
            )
            == []
        )
        [sentence] = segmenter.flush()
        assert sentence.raw_text == "Mình deploy chiều nay để kịp demo ngày mai."


class TestBackchannels:
    def test_a_backchannel_from_another_speaker_does_not_split_the_sentence(self):
        segmenter = make()
        segmenter.add(segment("I think that", start_ms=0, end_ms=1000))
        [interjection] = segmenter.add(
            segment("Yeah.", speaker="bob", start_ms=1100, end_ms=1300, segment_id="bob-1")
        )
        assert interjection.speaker_id == "bob"
        assert interjection.segment_ids == ["bob-1"]
        assert segmenter.add(segment("we should ship it.", start_ms=1400, end_ms=2500)) == []

        [sentence] = segmenter.flush()
        assert sentence.raw_text == "I think that we should ship it."
        assert sentence.speaker_id == "alice"

    def test_japanese_and_vietnamese_backchannels_are_recognised(self):
        assert is_backchannel("はい", "ja")
        assert is_backchannel("うん", "ja")
        assert is_backchannel("ừ", "vi")
        assert is_backchannel("mm-hmm", "en")
        assert not is_backchannel("Yeah, and that is why we should wait a week", "en")

    def test_a_real_turn_from_another_speaker_ends_the_open_line(self):
        segmenter = make()
        segmenter.add(segment("I think that", start_ms=0, end_ms=1000))
        [sentence] = segmenter.add(
            segment("Can you repeat that?", speaker="bob", start_ms=1100, end_ms=2000)
        )
        assert sentence.speaker_id == "alice"
        assert sentence.reason == "turn_change"

    def test_a_filler_only_interjection_is_dropped_entirely(self):
        segmenter = make()
        segmenter.add(segment("I think that", start_ms=0, end_ms=1000))
        assert segmenter.add(segment("Ummm", speaker="bob", start_ms=1100, end_ms=1400)) == []


class TestFlushing:
    def test_idle_closes_a_line_whose_speaker_has_stopped(self):
        segmenter = make(idle_flush_ms=4000)
        segmenter.add(segment("I think that", start_ms=0, end_ms=1000, arrived_at_ms=1000))
        assert segmenter.flush_idle(now_ms=3000) == []
        [sentence] = segmenter.flush_idle(now_ms=5200)
        assert sentence.reason == "idle"
        assert sentence.raw_text == "I think that"
        assert segmenter.flush_idle(now_ms=9000) == []

    def test_max_length_cuts_a_line_and_adds_no_words(self):
        segmenter = make(max_sentence_ms=3000)
        segmenter.add(segment("and we also", start_ms=0, end_ms=1500))
        [sentence] = segmenter.add(segment("need to think about", start_ms=2000, end_ms=3500))
        assert sentence.reason == "max_length"
        # Cut, not completed: no "…", no invented full stop in the RAW record.
        assert sentence.raw_text == "and we also need to think about"

    def test_meeting_end_publishes_the_open_line(self):
        segmenter = make()
        segmenter.add(segment("We will ship it and", start_ms=0, end_ms=1000))
        [sentence] = segmenter.flush()
        assert sentence.reason == "meeting_end"
        assert sentence.raw_text == "We will ship it and"
        assert segmenter.is_empty

    def test_a_filler_only_line_is_reported_as_empty(self):
        segmenter = make()
        segmenter.add(segment("Ummm", start_ms=0, end_ms=500))
        [sentence] = segmenter.flush()
        assert sentence.is_empty
        assert sentence.prepass_text == ""


class TestJoining:
    def test_japanese_joins_with_a_comma_after_a_conjunctive_particle(self):
        assert (
            join_texts(["来週の会議ですが", "火曜日に変更になりました。"], "ja")
            == "来週の会議ですが、火曜日に変更になりました。"
        )

    def test_japanese_joins_without_a_comma_otherwise(self):
        assert join_texts(["来週の", "会議です。"], "ja") == "来週の会議です。"

    def test_english_lowercases_a_capital_the_merge_moved_mid_sentence(self):
        assert join_texts(["I think that.", "We should ship it."], "en") == (
            "I think that we should ship it."
        )

    def test_a_name_keeps_its_capital(self):
        assert join_texts(["I think that.", "Tuan should ship it."], "en") == (
            "I think that Tuan should ship it."
        )

    def test_a_question_mark_is_never_dropped_by_a_join(self):
        assert join_texts(["Are you sure?", "Because we can wait."], "en") == (
            "Are you sure? Because we can wait."
        )

    def test_empty_pieces_are_skipped(self):
        assert join_texts(["", "We should ship it."], "en") == "We should ship it."
