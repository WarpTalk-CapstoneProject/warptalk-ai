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
