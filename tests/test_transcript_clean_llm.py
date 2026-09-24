"""The LLM tier may only delete, and only deletions it can be held to.

Every test here is about the same question: what happens when the model is WRONG. It is shown
numbered tokens and answers with indices, so it cannot invent a word — but it can still name an
index that does not exist, delete a "not", delete half the sentence, or claim a self-repair that
is not one. Each of those has to end with revision 0 (the prepass line) standing.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from transcript_clean_worker.llm_cleaner import (
    REJECT_BAD_INDEX,
    REJECT_BAD_JSON,
    REJECT_NO_CHANGE,
    REJECT_RATIO,
    REJECT_TIMEOUT,
    LLMCleaner,
    apply_deletions,
    has_self_repair_marker,
    prepass_deletion_indices,
)


class FakeCompletions:
    def __init__(self, payloads: list[object], delay_s: float = 0.0) -> None:
        self.payloads = payloads
        self.delay_s = delay_s
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        payload = self.payloads.pop(0)
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeClient:
    def __init__(self, payloads: list[object], delay_s: float = 0.0) -> None:
        self.completions = FakeCompletions(payloads, delay_s)
        self.chat = SimpleNamespace(completions=self.completions)

    async def close(self) -> None:
        return None


def cleaner(payloads: list[object], *, delay_s: float = 0.0, **kwargs) -> LLMCleaner:
    instance = LLMCleaner(
        api_key="test-key",
        model=str(kwargs.pop("model", "gpt-4.1-mini")),
        timeout_s=float(kwargs.pop("timeout_s", 8.0)),
        max_delete_ratio=float(kwargs.pop("max_delete_ratio", 0.4)),
    )
    instance._client = FakeClient(payloads, delay_s)  # type: ignore[assignment]
    return instance


class TestAcceptedAnswers:
    async def test_a_valid_deletion_is_applied(self):
        subject = cleaner([{"delete": [0, 3], "self_repair": False}])
        result = await subject.clean(
            "um, so we we should ship it", "en", prepass_text="So we should ship it."
        )
        assert result is not None
        assert result.text == "So we should ship it."
        assert result.self_repair is False

    async def test_a_vietnamese_self_repair_keeps_the_corrected_half(self):
        subject = cleaner([{"delete": [1, 2, 3, 4], "self_repair": True}])
        result = await subject.clean("họp thứ hai, à không, thứ ba", "vi")
        assert result is not None
        assert result.text == "Họp thứ ba."
        assert result.self_repair is True

    async def test_a_question_mark_survives_the_clean(self):
        subject = cleaner([{"delete": [0], "self_repair": False}])
        result = await subject.clean("ờ anh gửi báo cáo chưa", "vi")
        assert result is not None
        assert result.text == "Anh gửi báo cáo chưa?"

    async def test_a_japanese_filler_inside_a_phrase_is_deletable(self):
        # Morphemes, not spaces: "明日えーとリリースします" has no word boundaries to split on.
        subject = cleaner([{"delete": [1], "self_repair": False}])
        result = await subject.clean("明日えーとリリースします", "ja")
        assert result is not None
        assert result.text == "明日リリースします。"


class TestRefusedAnswers:
    async def test_a_hallucinated_index_is_refused(self):
        subject = cleaner([{"delete": [0, 99], "self_repair": False}])
        assert await subject.clean("um, so we should ship it", "en") is None
        assert subject.rejections == {REJECT_BAD_INDEX: 1}

    async def test_deleting_a_negation_is_refused(self):
        # Index 2 is "not". The deletion is structurally legal and changes what was said, which
        # is what invariant I2_negation_count is for.
        subject = cleaner([{"delete": [2], "self_repair": False}])
        assert await subject.clean("we should not ship it today", "en") is None
        assert "I2_negation_count" in "".join(subject.rejections)

    async def test_deleting_a_number_is_refused(self):
        # Index 4 is "15th".
        subject = cleaner([{"delete": [4], "self_repair": False}])
        assert await subject.clean("we ship on the 15th of May", "en") is None
        assert "I2_number_count" in "".join(subject.rejections)

    async def test_deleting_more_than_the_cap_is_refused(self):
        # 4 of 7 tokens is 0.57, over the 0.4 cap: that is summarising, not cleaning.
        subject = cleaner([{"delete": [0, 1, 2, 3], "self_repair": False}])
        assert await subject.clean("um so we we should ship it", "en") is None
        assert subject.rejections == {REJECT_RATIO: 1}

    async def test_an_empty_deletion_list_publishes_nothing_new(self):
        subject = cleaner([{"delete": [], "self_repair": False}])
        assert await subject.clean("We should ship it today.", "en") is None
        assert subject.rejections == {REJECT_NO_CHANGE: 1}

    async def test_malformed_json_is_refused(self):
        subject = cleaner(["not json at all"])
        assert await subject.clean("um so we should ship it", "en") is None
        assert subject.rejections == {REJECT_BAD_JSON: 1}

    async def test_a_timeout_leaves_revision_zero_standing(self):
        subject = cleaner(
            [{"delete": [0], "self_repair": False}], delay_s=0.2, timeout_s=0.01
        )
        assert await subject.clean("um so we should ship it", "en") is None
        assert subject.rejections == {REJECT_TIMEOUT: 1}

    async def test_a_call_failure_is_swallowed(self):
        subject = cleaner([])  # popping from an empty list raises inside create()
        assert await subject.clean("um so we should ship it", "en") is None
        assert "call_failed" in subject.rejections

    async def test_without_a_client_nothing_is_attempted(self):
        subject = LLMCleaner(api_key="", model="gpt-4.1-mini")
        assert subject.is_available is False
        assert await subject.clean("um so we should ship it", "en") is None


class TestTheSelfRepairFlag:
    async def test_the_flag_is_dropped_when_no_marker_was_deleted(self):
        # The model may claim a repair on any deletion; the flag is only kept when what it
        # deleted actually contains a correction marker.
        subject = cleaner([{"delete": [0], "self_repair": True}])
        result = await subject.clean("um so we should ship it", "en")
        assert result is not None
        assert result.self_repair is False

    async def test_the_flag_needs_the_model_to_claim_it_too(self):
        # The deletion IS a repair and contains its marker; the model did not say so, so the
        # line is published without the flag rather than with one nobody claimed.
        subject = cleaner([{"delete": [4, 5, 6], "self_repair": False}])
        result = await subject.clean("We will ship on Monday, I mean Tuesday, next week", "en")
        assert result is not None
        assert result.text == "We will ship on Tuesday, next week."
        assert result.self_repair is False

    async def test_a_repair_may_delete_more_than_the_ordinary_cap(self):
        # 3 of 4 tokens — over max_delete_ratio, and exactly the shape of a self-repair.
        subject = cleaner([{"delete": [0, 1, 2], "self_repair": True}])
        result = await subject.clean("Monday, I mean Tuesday", "en")
        assert result is not None
        assert result.text == "Tuesday."
        assert result.self_repair is True

    def test_markers_are_recognised_in_all_three_languages(self):
        assert has_self_repair_marker("Monday, I mean Tuesday", "en", [0, 1, 2])
        assert has_self_repair_marker("họp thứ hai, à không, thứ ba", "vi", [1, 2, 3, 4])
        assert has_self_repair_marker("月曜、じゃなくて火曜です", "ja", [0, 1])
        assert not has_self_repair_marker("um so we should ship it", "en", [0])

    def test_a_contrast_is_not_a_repair(self):
        # "赤じゃなくて青がいい" — "not red, blue is better". Nothing is deleted, so there is
        # nothing for the flag to attach to either.
        assert not has_self_repair_marker("赤じゃなくて青がいい", "ja", [])


class TestTheRequest:
    async def test_the_prompt_asks_for_json_and_carries_the_prepass_suggestion(self):
        subject = cleaner([{"delete": [0], "self_repair": False}])
        await subject.clean(
            "um so we should ship it",
            "en",
            prepass_text="So we should ship it.",
            previous_line="When are we shipping?",
        )
        kwargs = subject._client.completions.calls[0]  # type: ignore[union-attr]
        assert kwargs["response_format"] == {"type": "json_object"}
        user_message = kwargs["messages"][1]["content"]
        assert "0:um" in user_message
        assert "[0]" in user_message  # the rule tier's suggestion
        assert "When are we shipping?" in user_message
        system_message = kwargs["messages"][0]["content"]
        assert "ONLY DELETE" in system_message
        # At most three few-shot examples: more of them measurably pushed the model into
        # deleting meaningful words.
        assert system_message.count("->") <= 3


class TestPureHelpers:
    def test_deletions_take_the_comma_that_belonged_to_them(self):
        assert apply_deletions("Monday, I mean Tuesday", "en", [0, 1, 2]) == "Tuesday."
        assert apply_deletions("um, so we should ship it", "en", [0]) == "So we should ship it."

    def test_a_comma_between_two_surviving_words_is_the_speakers(self):
        assert apply_deletions("um, we ship, then we test", "en", [0]) == (
            "We ship, then we test."
        )

    def test_japanese_rebuilds_without_spaces(self):
        assert apply_deletions("えーと、明日リリースします", "ja", [0]) == "明日リリースします。"

    def test_the_prepass_suggestion_is_recovered_by_alignment(self):
        assert prepass_deletion_indices(
            "um, so we we should ship it", "So we should ship it.", "en"
        ) == [0, 3]

    def test_an_unalignable_prepass_text_suggests_nothing(self):
        assert prepass_deletion_indices("we ship it", "something else entirely", "en") == []


@pytest.mark.parametrize(
    ("raw", "language", "indices", "expected"),
    [
        ("um so we we should ship it", "en", [0, 2], "So we should ship it."),
        ("thì mình deploy chiều nay", "vi", [0], "Mình deploy chiều nay."),
        ("えーとリリースは明日です", "ja", [0], "リリースは明日です。"),
    ],
)
def test_terminal_punctuation_is_applied_per_language(raw, language, indices, expected):
    assert apply_deletions(raw, language, indices) == expected
