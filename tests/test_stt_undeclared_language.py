"""WT-928: a room's transcript stays inside the languages the room declared.

Production room 01a0fb2e (2 Oct) declared vi, en and ja. Its transcript filter offered
Chinese: 5 of 68 lines, among them 嗯, 可能, 都 and "嗯 Chào mừng mọi người đến với" from a
Vietnamese speaker. Japanese is written in Han too, so the script allow-list let the Han
through, script detection labelled it `zh`, and two such lines in a row re-pinned the speaker
to Chinese for the rest of the meeting.
"""

from stt_worker.model import OpenAISTT, _filter_segments, _without_undeclared_han

ROOM = {"vi", "en", "ja"}


def _raw(text: str) -> dict:
    return {"text": text, "start": 0.0, "end": 1.0, "avg_logprob": -0.2, "no_speech_prob": 0.01}


class TestUndeclaredHan:
    def test_a_lone_chinese_filler_from_a_vietnamese_speaker_is_dropped(self) -> None:
        assert _filter_segments([_raw("嗯")], "vi", 0, allowed_languages=ROOM) == []
        assert _filter_segments([_raw("都")], "vi", 0, allowed_languages=ROOM) == []

    def test_the_vietnamese_around_a_stray_han_filler_survives_as_vietnamese(self) -> None:
        [segment] = _filter_segments(
            [_raw("嗯 Chào mừng mọi người đến với")], "vi", 0, allowed_languages=ROOM
        )
        assert segment.text == "Chào mừng mọi người đến với"
        assert segment.language == "vi"

    def test_kanji_from_a_japanese_speaker_is_japanese_not_chinese(self) -> None:
        # Han with no kana is legitimate Japanese from a Japanese speaker. It is kept, and it
        # is labelled with the room's language rather than the `zh` script detection guesses.
        [segment] = _filter_segments([_raw("可能")], "ja", 0, allowed_languages=ROOM)
        assert segment.text == "可能"
        assert segment.language == "ja"

    def test_text_with_kana_is_left_alone(self) -> None:
        [segment] = _filter_segments([_raw("我会今この 部分")], "vi", 0, allowed_languages=ROOM)
        assert segment.text == "我会今この 部分"
        assert segment.language == "ja"

    def test_a_room_that_declared_chinese_keeps_chinese(self) -> None:
        [segment] = _filter_segments([_raw("可能")], "vi", 0, allowed_languages={"vi", "zh"})
        assert segment.language == "zh"

    def test_a_room_that_declared_nothing_is_unchanged(self) -> None:
        # No declared set: nothing to hold the text to, exactly as before.
        [segment] = _filter_segments([_raw("可能")], "unknown", 0)
        assert segment.language == "zh"

    def test_the_helper_only_touches_han_without_kana(self) -> None:
        assert _without_undeclared_han("Xin chào", ROOM, "vi") == "Xin chào"
        assert _without_undeclared_han("说  mình xác định được dữ liệu", ROOM, "vi") == (
            "mình xác định được dữ liệu"
        )
        assert _without_undeclared_han("可能", ROOM, "ko") == "可能"


class TestNoOverrideOutsideTheRoom:
    def test_two_han_lines_no_longer_re_pin_a_japanese_speaker_to_chinese(self) -> None:
        model = OpenAISTT.__new__(OpenAISTT)
        model._language_evidence = {}
        model._language_override = {}

        segments = _filter_segments([_raw("可能"), _raw("部分")], "ja", 0, allowed_languages=ROOM)
        assert [segment.language for segment in segments] == ["ja", "ja"]

        model._learn_language_evidence(("m", "s"), "ja", segments)
        assert model._language_override == {}
