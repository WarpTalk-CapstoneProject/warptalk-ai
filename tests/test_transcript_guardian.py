"""The guardian may reformat a same-language transcript. It may not rewrite one.

The instruction in `guardian_instruction` asks for that. These tests cover the part that
ENFORCES it, because an instruction is not a guarantee: a model asked to tidy a garbled line
will happily produce a fluent, confident, invented one, and that invention would flow on into
the summary, the knowledge index and the meeting record.

The reference failure is a real production transcript in which a speaker's language was
mis-declared, so their words came back as nonsense — "A lô, rồi hỏi của cọp đầu tiên rồi". No
amount of language modelling can recover what the recogniser never heard, and the only safe
answer for text like that is to leave it exactly as it is.
"""

from translation_worker.transcript_guardian import (
    choose_transcript,
    guardian_instruction,
    is_faithful,
)


class TestFormattingIsAllowed:
    def test_punctuation_and_casing(self):
        original = "ê hồi nãy anh bị cái bug gì không có lưu được tên hả"
        polished = "Ê, hồi nãy anh bị cái bug gì? Không có lưu được tên hả?"
        assert is_faithful(original, polished, "vi")

    def test_vietnamese_fillers_may_be_dropped(self):
        original = "ờ anh xin lỗi à giờ để anh lướt"
        polished = "Anh xin lỗi, giờ để anh lướt."
        assert is_faithful(original, polished, "vi")

    def test_a_filler_no_list_would_have_contained_may_be_dropped(self):
        # The point of dropping the dictionary: "kiểu", "thì", "á" are ordinary Vietnamese
        # hesitations and were on no list. Under the old similarity check this looked like a
        # rewrite; under the subsequence rule it is plainly a deletion.
        original = "thì cái này kiểu là mình deploy chiều nay á"
        polished = "Cái này mình deploy chiều nay."
        assert is_faithful(original, polished, "vi")

    def test_english_false_starts_may_be_dropped(self):
        original = "so i i i think we we should ship it"
        polished = "So I think we should ship it."
        assert is_faithful(original, polished, "en")

    def test_english_fillers_may_be_dropped(self):
        original = "um so i think uh we should ship it"
        polished = "So I think we should ship it."
        assert is_faithful(original, polished, "en")

    def test_already_clean_text_passes_unchanged(self):
        text = "Chúng ta sẽ deploy vào chiều nay."
        assert is_faithful(text, text, "vi")


class TestRewritingIsRefused:
    def test_a_paraphrase_is_not_formatting(self):
        original = "ê hồi nãy anh bị cái bug gì không có lưu được tên hả"
        polished = "Xin chào, tôi muốn hỏi về lỗi lưu tên mà anh gặp phải lúc nãy."
        assert not is_faithful(original, polished, "vi")

    def test_garbled_text_is_not_repaired_into_fluent_text(self):
        # The line that motivated the whole module. Whatever was really said is gone; a
        # plausible reconstruction is the dangerous outcome, not the good one.
        original = "A lô rồi hỏi của cọp đầu tiên rồi đừng có bấm chân ship vào đây nha"
        polished = "Alô, hỏi câu đầu tiên rồi, đừng bấm nút ship vào đây nhé."
        assert not is_faithful(original, polished, "vi")

    def test_invented_continuation_is_refused(self):
        original = "Mình deploy chiều nay"
        polished = (
            "Mình deploy chiều nay. Sau khi deploy xong thì cả nhóm sẽ họp lại để "
            "review kết quả và lên kế hoạch cho sprint tiếp theo."
        )
        assert not is_faithful(original, polished, "vi")

    def test_a_translation_is_refused(self):
        original = "Chúng ta sẽ deploy vào chiều nay."
        polished = "We will deploy this afternoon."
        assert not is_faithful(original, polished, "vi")

    def test_empty_output_is_refused(self):
        assert not is_faithful("Mình deploy chiều nay", "   ", "vi")

    def test_dropping_meaningful_words_is_refused(self):
        original = "Anh Tú sẽ sửa bug billing và Nhi sẽ test lại toàn bộ luồng thanh toán"
        polished = "Anh Tú sẽ sửa bug billing."
        assert not is_faithful(original, polished, "vi")

    def test_reordering_is_refused(self):
        # Impossible to accept by construction: order is part of the subsequence rule.
        original = "Nhi test rồi Tú deploy"
        polished = "Tú deploy rồi Nhi test."
        assert not is_faithful(original, polished, "vi")

    def test_a_single_inserted_word_is_refused(self):
        original = "mình deploy chiều nay"
        polished = "Mình sẽ deploy chiều nay."
        assert not is_faithful(original, polished, "vi")


class TestJapanese:
    """Japanese has no spaces, so until WT-716 this module could not clean it at all.

    Splitting on whitespace made a whole sentence one token: deleting a filler from the middle
    of it changed that one token, the subsequence rule saw an entirely different word, and the
    polish was refused every time. The tokenizer is morphemes now (shared.disfluency.tokenize).
    """

    def test_a_filler_inside_a_phrase_may_be_dropped(self):
        assert is_faithful("明日えーとリリースします", "明日リリースします。", "ja")

    def test_a_leading_filler_and_punctuation_may_be_dropped(self):
        assert is_faithful("えーと、あのー、来週の会議ですが", "来週の会議ですが", "ja")

    def test_a_repeated_restart_may_be_collapsed(self):
        assert is_faithful("その、その件は明日話します", "その件は明日話します。", "ja")

    def test_a_rewrite_is_still_refused(self):
        assert not is_faithful("明日リリースします", "明日デプロイします。", "ja")

    def test_deleting_the_negation_is_refused(self):
        # A subsequence, and the opposite statement: "we will not release tomorrow" becoming
        # "we will release tomorrow" is the failure a reader cannot see.
        assert not is_faithful("明日はリリースしません", "明日はリリースします。", "ja")

    def test_deleting_a_number_is_refused(self):
        assert not is_faithful("三時に始めます", "始めます。", "ja")

    def test_dropping_the_question_particle_is_refused(self):
        assert not is_faithful("明日リリースしますか", "明日リリースします。", "ja")

    def test_an_already_clean_line_passes(self):
        assert is_faithful("明日リリースします。", "明日リリースします。", "ja")


class TestMeaningIsNotFormatting:
    """I2: a deletion can be a perfect subsequence and still change what was said."""

    def test_dropping_an_english_negation_is_refused(self):
        assert not is_faithful("we should not ship it today", "We should ship it today.", "en")

    def test_dropping_a_vietnamese_negation_is_refused(self):
        assert not is_faithful("mình không deploy chiều nay", "Mình deploy chiều nay.", "vi")

    def test_dropping_a_number_is_refused(self):
        assert not is_faithful("we ship on the 15th", "We ship.", "en")

    def test_a_question_may_gain_its_mark_but_not_lose_it(self):
        assert is_faithful("anh gửi báo cáo chưa", "Anh gửi báo cáo chưa?", "vi")
        assert not is_faithful("anh gửi báo cáo chưa?", "Anh gửi báo cáo.", "vi")


class TestDiacriticsAreWords:
    def test_stripping_vietnamese_tone_marks_is_a_rewrite(self):
        # "được" and "duoc" are not the same word, and normalising them together would let a
        # model quietly strip the diacritics off an entire meeting.
        original = "Không có lưu được tên"
        polished = "Khong co luu duoc ten."
        assert not is_faithful(original, polished, "vi")


class TestChooseTranscript:
    def test_keeps_the_polish_when_it_is_faithful(self):
        original = "mình deploy chiều nay"
        polished = "Mình deploy chiều nay."
        assert choose_transcript(original, polished, "vi") == polished

    def test_falls_back_to_the_original_when_it_is_not(self):
        original = "A lô rồi hỏi của cọp đầu tiên rồi"
        polished = "Alô, câu hỏi đầu tiên là gì vậy anh?"
        assert choose_transcript(original, polished, "vi") == original


class TestInstruction:
    def test_forbids_translating(self):
        assert "do NOT translate" in guardian_instruction("vi")

    def test_forbids_guessing_at_garbled_passages(self):
        assert "guess what a garbled passage" in guardian_instruction("vi")

    def test_states_the_boundary_rather_than_listing_fillers(self):
        # No dictionary. A per-language filler list told a model that already knows them far
        # more than it needed, and read as an exhaustive permission — real hesitations are not
        # a closed set. The structural check in is_faithful is what makes the boundary true.
        instruction = guardian_instruction("vi")
        assert "filler sounds, stutters" in instruction
        assert "ờ" not in instruction
