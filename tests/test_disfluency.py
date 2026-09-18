"""WT-716 deterministic disfluency prepass: fixtures, question punctuation, invariants.

Every fixture states the exact prepass output. "Unchanged" fixtures still get the terminal
punctuation policy and a sentence-initial capital (en/vi) — the only edits the prepass may make
to a line it does not clean.
"""

from __future__ import annotations

import random
import sys

import pytest

from shared.disfluency import (
    FLAG_ESCALATE,
    FLAG_FILLER_ONLY,
    check_invariants,
    detect_question,
    normalize_key,
    normalize_terminal_punctuation,
    prepass,
    tokenize,
)
from shared.disfluency.tokenize import ja_morphology_available

requires_morphology = pytest.mark.skipif(
    not ja_morphology_available(), reason="fugashi/unidic-lite not installed"
)

AFTER_QUESTION = {"prev_turn_is_question": True, "standalone_turn": True}

# (id, raw, language, prepass kwargs, expected clean_text, must escalate)
FIXTURES: list[tuple[str, str, str, dict[str, bool], str, bool]] = [
    # --- en -------------------------------------------------------------------------------
    (
        "E1",
        "um so we uh we need to finalize the budget",
        "en",
        {},
        "So we need to finalize the budget.",
        False,
    ),
    ("E2", "Uh-huh.", "en", AFTER_QUESTION, "Uh-huh.", False),
    ("E3", "I like the new layout", "en", {}, "I like the new layout.", False),
    ("E4", "It was very very expensive", "en", {}, "It was very very expensive.", False),
    ("E5", "the the con- contract is signed", "en", {}, "The contract is signed.", False),
    (
        "E6",
        "We should meet on Monday, I mean Tuesday.",
        "en",
        {},
        "We should meet on Monday, I mean Tuesday.",
        True,
    ),
    (
        "E7",
        "What I mean is we're over budget",
        "en",
        {},
        "What I mean is we're over budget.",
        False,
    ),
    ("E8", "No no no, don't merge it yet", "en", {}, "No no no, don't merge it yet.", False),
    ("E9", "What he had had was sold", "en", {}, "What he had had was sold.", False),
    (
        "E10",
        "Let's cover pre- and post-launch metrics",
        "en",
        {},
        "Let's cover pre- and post-launch metrics.",
        False,
    ),
    (
        "E11",
        "we need to, we need to go through the list",
        "en",
        {},
        "We need to go through the list.",
        False,
    ),
    ("E12", "Ummm", "en", {}, "", False),
    ("E13", "Hmm.", "en", AFTER_QUESTION, "Hmm.", False),
    # --- vi -------------------------------------------------------------------------------
    (
        "V1",
        "ờ thì cái dự án này ừm tuần sau mới xong",
        "vi",
        {},
        "Thì cái dự án này tuần sau mới xong.",
        False,
    ),
    ("V2", "Ờ.", "vi", AFTER_QUESTION, "Ờ.", False),
    ("V3", "Dạ vâng, em gửi rồi ạ", "vi", {}, "Dạ vâng, em gửi rồi ạ.", False),
    ("V4", "mai anh đi Hà Nội à", "vi", {}, "Mai anh đi Hà Nội à?", False),
    ("V5", "Chị à, mai họp nhé", "vi", {}, "Chị à, mai họp nhé.", False),
    (
        "V6",
        "tôi tôi nghĩ là là mình nên từ từ làm",
        "vi",
        {},
        "Tôi nghĩ là mình nên từ từ làm.",
        False,
    ),
    ("V7", "con diều bay là là trên ruộng", "vi", {}, "Con diều bay là là trên ruộng.", False),
    ("V8", "Đi đi, muộn rồi", "vi", {}, "Đi đi, muộn rồi.", False),
    ("V9", "ai ai cũng đồng ý", "vi", {}, "Ai ai cũng đồng ý.", False),
    ("V10", "họp thứ hai, à không, thứ ba", "vi", {}, "Họp thứ hai, à không, thứ ba.", True),
    ("V11", "nhà mình có nuôi ba ba", "vi", {}, "Nhà mình có nuôi ba ba.", False),
    ("V12", "ừmmm ờ để em xem", "vi", {}, "Để em xem.", False),
    ("V13", "kiểu dáng cái áo này đẹp", "vi", {}, "Kiểu dáng cái áo này đẹp.", False),
    # --- ja -------------------------------------------------------------------------------
    # J1: "ですが" ends on a conjunctive particle — unfinished, so no 。 is added.
    ("J1", "えーと、あのー、来週の会議ですが", "ja", {}, "来週の会議ですが", False),
    ("J2", "あの資料はもう送りました", "ja", {}, "あの資料はもう送りました。", False),
    ("J3", "はい。", "ja", {"standalone_turn": True}, "はい。", False),
    ("J4", "ええと、ええ、そうです。", "ja", {}, "ええ、そうです。", False),
    ("J5", "まだまだ時間があります", "ja", {}, "まだまだ時間があります。", False),
    ("J6", "どんどん進めましょう", "ja", {}, "どんどん進めましょう。", False),
    ("J7", "わ、私が担当します。", "ja", {}, "私が担当します。", False),
    ("J8", "月曜、じゃなくて火曜に出します。", "ja", {}, "月曜、じゃなくて火曜に出します。", True),
    ("J9", "赤じゃなくて青がいいです。", "ja", {}, "赤じゃなくて青がいいです。", True),
    ("J10", "ちょっと待ってください", "ja", {}, "ちょっと待ってください。", False),
    ("J11", "なんか飲みますか", "ja", {}, "なんか飲みますか？", False),
    ("J12", "もしもし、聞こえますか", "ja", {}, "もしもし、聞こえますか？", False),
    ("J13", "うーん。", "ja", {"standalone_turn": True}, "うーん。", False),
    ("J14", "その、その件は、えー、確認します。", "ja", {}, "その件は、確認します。", False),
]


def _marks(case: tuple[str, str, str, dict[str, bool], str, bool]) -> list[pytest.MarkDecorator]:
    return [requires_morphology] if case[2] == "ja" else []


@pytest.mark.parametrize(
    ("raw", "language", "kwargs", "expected", "must_escalate"),
    [pytest.param(*c[1:], id=c[0], marks=_marks(c)) for c in FIXTURES],
)
def test_prepass_fixture(
    raw: str, language: str, kwargs: dict[str, bool], expected: str, must_escalate: bool
) -> None:
    result = prepass(raw, language, **kwargs)
    assert result.clean_text == expected
    if must_escalate:
        assert FLAG_ESCALATE in result.flags
        assert result.escalate_reasons
    assert not check_invariants(raw, result.clean_text, language)


def test_filler_only_turn_is_empty_and_flagged() -> None:
    result = prepass("Ummm", "en")
    assert result.clean_text == ""
    assert FLAG_FILLER_ONLY in result.flags
    assert result.removed_spans == [(0, 4)]


@pytest.mark.parametrize(
    ("raw", "language", "expected"),
    [
        ("Hmm.", "en", "Hmm."),  # inferred standalone
        ("hmm so we should go", "en", "So we should go."),
        ("Ờ.", "vi", "Ờ."),
    ],
)
def test_a2_standalone_is_inferred(raw: str, language: str, expected: str) -> None:
    assert prepass(raw, language).clean_text == expected


def test_a2_turn_initial_after_question_is_kept() -> None:
    result = prepass("hmm I'm not sure", "en", prev_turn_is_question=True)
    assert result.clean_text == "Hmm I'm not sure."


def test_language_tags_and_auto_detection() -> None:
    assert prepass("um we ship", "en-US").clean_text == "We ship."
    assert prepass("ừm để em xem", "auto").clean_text == "Để em xem."
    assert prepass("um we ship", "auto").clean_text == "We ship."
    # An explicit unsupported language is left alone.
    unchanged = prepass("um 我们", "zh")
    assert unchanged.clean_text == "um 我们" and not unchanged.flags


def test_hyphenated_backchannels_are_one_token() -> None:
    assert tokenize("uh-oh, mm-hmm yes", "en") == ["uh-oh", "mm-hmm", "yes"]
    assert prepass("uh-oh, that broke", "en").clean_text == "Uh-oh, that broke."


def test_hallucinated_repeats_are_not_collapsed() -> None:
    raw = "Thank you. Thank you. Thank you."
    assert prepass(raw, "en").clean_text == raw


def test_acronym_is_not_a_filler() -> None:
    assert prepass("take him to the ER now", "en").clean_text == "Take him to the ER now."


def test_normalize_key() -> None:
    assert normalize_key("Ummm", "en") == "um"
    assert normalize_key("ừmmm", "vi") == "ừm"
    assert normalize_key("hòa", "vi") == normalize_key("hoà", "vi")
    assert normalize_key("エーーと", "ja") == "えーと"
    assert normalize_key("ｴｰﾄ", "ja") == "えーと"
    assert normalize_key("えー〜と", "ja") == "えーと"


@requires_morphology
def test_ja_tokenize_merges_split_fillers() -> None:
    assert tokenize("えーと、そのー、会議", "ja")[:2] == ["えーと", "そのー"]
    assert "ええと" in tokenize("ええと、ええ", "ja")


@requires_morphology
def test_ja_ascii_punctuation_and_halfwidth() -> None:
    assert prepass("ｴｰﾄ, 来週です", "ja").clean_text == "来週です。"


# --- question punctuation ---------------------------------------------------------------------

QUESTIONS: list[tuple[str, str, str]] = [
    # en
    ("Do you have the file", "en", "Do you have the file?"),
    ("Can we ship it on Friday", "en", "Can we ship it on Friday?"),
    ("Where are we on the budget", "en", "Where are we on the budget?"),
    ("We should ship it, right", "en", "We should ship it, right?"),
    ("It's ready, isn't it", "en", "It's ready, isn't it?"),
    ("So did Nhi send it", "en", "So did Nhi send it?"),
    ("Are we doing this or not", "en", "Are we doing this or not?"),
    ("What I mean is we're late", "en", "What I mean is we're late."),
    ("I know what you mean", "en", "I know what you mean."),
    ("I don't know what you mean", "en", "I don't know what you mean."),
    ("Do the dishes", "en", "Do the dishes."),
    ("Thank you. Is it ready.", "en", "Thank you. Is it ready?"),
    # vi
    ("Anh gửi báo cáo chưa", "vi", "Anh gửi báo cáo chưa?"),
    ("Mình có họp không", "vi", "Mình có họp không?"),
    ("Cái này bao nhiêu tiền", "vi", "Cái này bao nhiêu tiền?"),
    ("Anh đi đâu vậy", "vi", "Anh đi đâu vậy?"),
    ("Em không đi", "vi", "Em không đi."),
    ("Chị à, mai họp nhé", "vi", "Chị à, mai họp nhé."),
    ("Thế nào cũng được", "vi", "Thế nào cũng được."),
    ("Không sao", "vi", "Không sao."),
    ("Tôi cũng không", "vi", "Tôi cũng không."),
    # ja
    ("資料は届きましたか", "ja", "資料は届きましたか？"),
    ("明日来られませんか", "ja", "明日来られませんか？"),
    ("来るかどうか分からない", "ja", "来るかどうか分からない。"),
    ("雨が降るかもしれない", "ja", "雨が降るかもしれない。"),
    ("どこでしたっけ", "ja", "どこでしたっけ？"),
    ("行くの", "ja", "行くの"),
    ("本当ですか?", "ja", "本当ですか？"),
]


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        pytest.param(
            t, lang, e, id=f"{lang}-{i}", marks=[requires_morphology] if lang == "ja" else []
        )
        for i, (t, lang, e) in enumerate(QUESTIONS)
    ],
)
def test_terminal_punctuation(text: str, language: str, expected: str) -> None:
    assert normalize_terminal_punctuation(text, language) == expected


def test_stt_question_mark_is_never_removed() -> None:
    assert normalize_terminal_punctuation("I think so?", "en") == "I think so?"
    assert detect_question("I think so?", "en")
    assert prepass("um I think so?", "en").clean_text == "I think so?"


def test_unfinished_line_gets_no_period() -> None:
    assert normalize_terminal_punctuation("we need to,", "en") == "we need to,"


def test_detect_question_rejects_embedded_wh() -> None:
    assert not detect_question("I know what you mean", "en")
    assert detect_question("What time is it", "en")


# --- invariants -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean", "language", "violation"),
    [
        ("we do not ship", "we do ship", "en", "I2_negation_count"),
        ("we don't ship", "we ship", "en", "I2_negation_count"),
        ("em không đi", "em đi", "vi", "I2_negation_count"),
        ("we ship 3 units", "we ship units", "en", "I2_number_count"),
        ("we ship three units", "we ship units", "en", "I2_number_count"),
        ("we ship the units", "we ship units now", "en", "I1_not_subsequence"),
        ("we ship the units", "units we ship", "en", "I1_not_subsequence"),
        ("anh đi à", "anh đi", "vi", "I2_question_marker"),
        ("is it ready?", "is it ready.", "en", "I2_question_marker"),
    ],
)
def test_invariant_violations(raw: str, clean: str, language: str, violation: str) -> None:
    assert violation in check_invariants(raw, clean, language)


@requires_morphology
def test_ja_invariants() -> None:
    assert "I2_negation_count" in check_invariants("行かない。", "行く。", "ja")
    assert "I2_number_count" in check_invariants("3個です。", "個です。", "ja")
    assert not check_invariants("その、その件は", "その件は", "ja")
    assert "I1_not_subsequence" in check_invariants("その件は", "この件は", "ja")


def test_invariants_accept_punctuation_and_case_changes() -> None:
    assert not check_invariants("um so we go", "So we go.", "en")
    assert not check_invariants("mai anh đi Hà Nội à", "Mai anh đi Hà Nội à?", "vi")


def test_invariant_violation_returns_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    # `shared.disfluency.prepass` the attribute is the function; the module is in sys.modules.
    module = sys.modules["shared.disfluency.prepass"]
    monkeypatch.setattr(module, "check_invariants", lambda *_: ["I1_not_subsequence"])
    result = module.prepass("um we ship", "en")
    assert result.clean_text == "um we ship"
    assert result.flags == frozenset({FLAG_ESCALATE})
    assert "I1_not_subsequence" in result.escalate_reasons


def test_every_output_is_a_deletion_on_randomised_inputs() -> None:
    """No hypothesis: a seeded shuffle of fixture words plus fillers, checked against I1/I2."""
    rng = random.Random(716)
    pools = {
        "en": (
            "um uh hmm the the we we need to , very very no not 3 like I mean go "
            "con- contract pre- and post-launch yeah okay"
        ).split(),
        "vi": "ờ ừm thì cái tôi tôi là là nghĩ từ từ ba ba không đi à , chị mình".split(),
    }
    if ja_morphology_available():
        pools["ja"] = [
            "えーと",
            "、",
            "あのー",
            "その",
            "件",
            "は",
            "ない",
            "3",
            "私",
            "わ",
            "まだまだ",
        ]
    for language, pool in pools.items():
        sep = "" if language == "ja" else " "
        for _ in range(300):
            raw = sep.join(rng.choice(pool) for _ in range(rng.randint(1, 12)))
            result = prepass(raw, language)
            assert not check_invariants(raw, result.clean_text, language), (raw, result)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("えーと、あのー、来週の会議ですが", "来週の会議ですが"),
        ("ええと、ええ、そうです。", "ええ、そうです。"),
        # Too coarse to see the その、その repeat: only the delimited filler goes.
        ("その、その件は、えー、確認します。", "その、その件は、確認します。"),
        ("えーとですね", "えーとですね。"),  # not delimited → kept
    ],
)
def test_ja_fallback_tokenizer_only_deletes_delimited_fillers(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    """Without fugashi the prepass must still be safe, and must say it was guessing."""
    monkeypatch.setattr(sys.modules["shared.disfluency.tokenize"], "_tagger", lambda: None)
    result = prepass(raw, "ja")
    assert result.clean_text == expected
    assert FLAG_ESCALATE in result.flags
    assert "ja_fallback_tokenizer" in result.escalate_reasons
