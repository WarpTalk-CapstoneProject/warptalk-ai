"""WT-716: split_into_sentences recognises full-width 。！？ without a following space."""

from __future__ import annotations

import pytest

from shared.text_utils import split_into_sentences


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "来週の会議です。資料は届きましたか？はい！",
            ["来週の会議です。", "資料は届きましたか？", "はい！"],
        ),
        ("終わり。 次。", ["終わり。", "次。"]),
        ("本当！？はい", ["本当！？", "はい"]),
        ("「はい。」と言った。次です。", ["「はい。」と言った。", "次です。"]),
        ("会議です。\n次", ["会議です。", "次"]),
        ("你好。我们开始吧！", ["你好。", "我们开始吧！"]),
    ],
)
def test_cjk_sentence_ends(text: str, expected: list[str]) -> None:
    assert split_into_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Hello world. It costs 3.5 dollars! Ok?\nNext",
            ["Hello world.", "It costs 3.5 dollars!", "Ok?", "Next"],
        ),
        ("e.g.this stays whole", ["e.g.this stays whole"]),
        ("Xin chào. Mai họp nhé.", ["Xin chào.", "Mai họp nhé."]),
        ("", []),
    ],
)
def test_latin_behaviour_is_unchanged(text: str, expected: list[str]) -> None:
    assert split_into_sentences(text) == expected
