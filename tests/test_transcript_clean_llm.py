"""The LLM tier may only delete, and only deletions it can be held to.

Every test here is about the same question: what happens when the model is WRONG. It is shown
numbered tokens and answers with indices, so it cannot invent a word — but it can still name an
index that does not exist, delete a "not", delete half the sentence, or claim a self-repair that
is not one. Each of those has to end with revision 0 (the prepass line) standing.

The self-repair cases have their own class, because that is the one path allowed to relax a
count invariant and therefore the only way this stage could ever publish a sentence that means
the opposite of what was said.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

from shared.config import WorkerSettings
from shared.schemas import STTResultMessage
from transcript_clean_worker.config import TranscriptCleanSettings
from transcript_clean_worker.llm_cleaner import (
    REJECT_BAD_INDEX,
    REJECT_BAD_JSON,
    REJECT_NEGATION_OUTSIDE_MARKER,
    REJECT_NO_CHANGE,
    REJECT_NUMBER_WITHOUT_REPLACEMENT,
    REJECT_RATIO,
    REJECT_TIMEOUT,
    LLMCleaner,
    apply_deletions,
    has_self_repair_marker,
    prepass_deletion_indices,
    self_repair_marker_span,
)
from transcript_clean_worker.worker import TranscriptCleanWorker


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
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


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
        self_repair_max_delete_ratio=float(kwargs.pop("self_repair_max_delete_ratio", 0.7)),
    )
    instance._client = FakeClient(payloads, delay_s)  # type: ignore[assignment]
    return instance


class FakeRedis:
    """Only what the worker touches: the pause-flag GET and the stream publishes."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.published: list[tuple[str, dict[str, str]]] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def publish(self, stream: str, data: dict[str, str]) -> bytes:
        self.published.append((stream, data))
        return b"1-0"


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
        subject = cleaner([{"delete": [0], "self_repair": False}], delay_s=0.2, timeout_s=0.01)
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
        # 3 of 7 tokens — over max_delete_ratio, and exactly the shape of a self-repair: the
        # weekday that is withdrawn is replaced by another weekday.
        subject = cleaner([{"delete": [3, 4, 5], "self_repair": True}])
        result = await subject.clean("We ship on Monday, I mean Tuesday", "en")
        assert result is not None
        assert result.text == "We ship on Tuesday."
        assert result.self_repair is True

    def test_markers_are_recognised_in_all_three_languages(self):
        assert has_self_repair_marker("Monday, I mean Tuesday", "en", [0, 1, 2])
        assert has_self_repair_marker("họp thứ hai, à không, thứ ba", "vi", [1, 2, 3, 4])
        assert has_self_repair_marker("月曜、じゃなくて火曜です", "ja", [0, 1])
        assert not has_self_repair_marker("um so we should ship it", "en", [0])

    def test_the_span_names_the_marker_tokens_and_not_the_reparandum(self):
        # Which tokens, not just whether: the span is what licenses a deleted negation, so
        # "không" at index 4 has to be inside it and "hai" at index 2 outside it.
        assert self_repair_marker_span("họp thứ hai, à không, thứ ba", "vi", [1, 2, 3, 4]) == (3, 4)
        assert self_repair_marker_span("Monday, I mean Tuesday", "en", [0, 1, 2]) == (1, 2)
        assert self_repair_marker_span("月曜、じゃなくて火曜です", "ja", [0, 1]) == (1,)
        assert self_repair_marker_span("um so we should ship it", "en", [0]) == ()

    def test_a_contrast_is_not_a_repair(self):
        # "赤じゃなくて青がいい" — "not red, blue is better". Nothing is deleted, so there is
        # nothing for the flag to attach to either.
        assert not has_self_repair_marker("赤じゃなくて青がいい", "ja", [])


class TestTheNarrowSelfRepairExceptions:
    """A verified repair may lose a negation or a number — but only in the two shapes below.

    The ruling on this ticket: faithfulness outranks tidiness, so a sentence that is wrongly
    classified as a self-repair must not be able to drop a "không"/"not"/"ない" or a figure and
    come out meaning the opposite. Everything here that is refused is refused in favour of the
    tier-1 wording, which the reader already has.
    """

    async def test_a_negation_inside_the_marker_is_allowed(self):
        # The Vietnamese correction marker IS "à không". Its negation goes because the marker
        # goes; that is the speaker cancelling a statement, not making one.
        subject = cleaner([{"delete": [1, 2, 3, 4], "self_repair": True}])
        result = await subject.clean("họp thứ hai, à không, thứ ba", "vi")
        assert result is not None
        assert result.text == "Họp thứ ba."
        assert result.self_repair is True

    async def test_a_negation_outside_the_marker_is_refused(self):
        # Two "không" in one line: the first belongs to the marker "à không", the second is the
        # sentence. Deleting both gives "Đồng ý." — the opposite of what was said — and the old
        # wholesale waiver let it through at a ratio (4 of 6) the cap never sees.
        subject = cleaner([{"delete": [0, 1, 2, 3], "self_repair": True}])
        assert await subject.clean("à không, mình không đồng ý", "vi") is None
        assert subject.rejections == {REJECT_NEGATION_OUTSIDE_MARKER: 1}

    async def test_an_english_negation_outside_the_marker_is_refused(self):
        subject = cleaner([{"delete": [2, 3, 4, 5], "self_repair": True}])
        assert await subject.clean("we should not ship, I mean, it today", "en") is None
        assert subject.rejections == {REJECT_NEGATION_OUTSIDE_MARKER: 1}

    async def test_a_number_replaced_by_another_number_is_allowed(self):
        subject = cleaner([{"delete": [2, 3, 4], "self_repair": True}])
        result = await subject.clean("we ship three, I mean four items", "en")
        assert result is not None
        assert result.text == "We ship four items."
        assert result.self_repair is True

    async def test_a_number_deleted_with_nothing_of_its_kind_left_is_refused(self):
        # A repair REPLACES. Nothing after the deleted span is a number, so this is not a
        # correction of a quantity, it is a quantity going missing.
        subject = cleaner([{"delete": [2, 3, 4], "self_repair": True}])
        assert await subject.clean("we ship three, I mean, soon", "en") is None
        assert subject.rejections == {REJECT_NUMBER_WITHOUT_REPLACEMENT: 1}

    async def test_a_weekday_deleted_with_no_weekday_left_is_refused(self):
        # shared.disfluency does not count weekdays as numbers, so I2 says nothing here: the
        # same-kind check in llm_cleaner is the only thing between the reader and a meeting that
        # quietly lost its day.
        subject = cleaner([{"delete": [3, 4, 5], "self_repair": True}])
        assert await subject.clean("We ship on Monday, I mean, soon", "en") is None
        assert subject.rejections == {REJECT_NUMBER_WITHOUT_REPLACEMENT: 1}

    async def test_a_japanese_contrast_is_left_exactly_as_it_was(self):
        # "赤じゃなくて青がいいです。" — "blue, not red" is a contrast, not a correction. The
        # model's right answer is to delete nothing, and nothing is what the reader keeps.
        subject = cleaner([{"delete": [], "self_repair": False}])
        assert await subject.clean("赤じゃなくて青がいいです。", "ja") is None
        assert subject.rejections == {REJECT_NO_CHANGE: 1}

    async def test_a_repair_over_the_repair_cap_is_refused(self):
        # "Monday, I mean Tuesday" is 3 of 4 tokens — 0.75: under the 0.8 this used to allow,
        # over the 0.7 it allows now. A whole sentence that is three quarters reparandum is as
        # easily a model summarising as a speaker correcting themselves, and on this ticket the
        # tie goes to the untidy-but-true tier-1 line.
        subject = cleaner([{"delete": [0, 1, 2], "self_repair": True}])
        assert await subject.clean("Monday, I mean Tuesday", "en") is None
        assert subject.rejections == {REJECT_RATIO: 1}

    async def test_the_repair_cap_can_be_moved_without_a_code_change(self):
        subject = cleaner(
            [{"delete": [0, 1, 2], "self_repair": True}], self_repair_max_delete_ratio=0.8
        )
        result = await subject.clean("Monday, I mean Tuesday", "en")
        assert result is not None
        assert result.text == "Tuesday."

    def test_the_repair_cap_is_a_setting_with_a_conservative_default(self):
        assert TranscriptCleanSettings().self_repair_max_delete_ratio == 0.7


class TestARejectionNeverCostsTheLine:
    async def test_a_refused_repair_leaves_revision_zero_exactly_as_published(self):
        """The whole point of refusing: the reader keeps the tier-1 sentence, negation and all.

        Driven through the worker rather than the cleaner alone, because "the line survives" is
        a claim about what reaches `transcript:clean`, not about a return value.
        """
        subject = cleaner([{"delete": [0, 1, 2, 3], "self_repair": True}])
        worker = TranscriptCleanWorker(
            clean_settings=TranscriptCleanSettings(),
            cleaner=subject,
            settings=WorkerSettings(),
        )
        redis = FakeRedis()
        worker.redis = redis  # type: ignore[assignment]
        payload = STTResultMessage(
            segment_id=str(uuid.uuid4()),
            meeting_id="11111111-1111-1111-1111-111111111111",
            speaker_id="22222222-2222-2222-2222-222222222222",
            text="à không, mình không đồng ý",
            language="vi",
            start_ms=0,
            end_ms=1000,
        ).to_redis()

        await worker.process(b"1-0", {k.encode(): v.encode() for k, v in payload.items()})
        await worker._flush_meeting("11111111-1111-1111-1111-111111111111", reason="meeting_end")
        for _ in range(5):  # let the detached LLM task run
            await asyncio.sleep(0)

        messages = [data for stream, data in redis.published if stream == "transcript:clean"]
        assert [message["revision"] for message in messages] == ["0"]
        assert messages[0]["source"] == "prepass"
        assert "mình không đồng ý" in messages[0]["clean_text"]
        assert subject.rejections == {REJECT_NEGATION_OUTSIDE_MARKER: 1}


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
        # At most four few-shot examples: more of them measurably pushed the model into
        # deleting meaningful words.
        assert 3 <= system_message.count("->") <= 4

    async def test_the_prompt_tells_the_model_to_delete_less_when_unsure(self):
        subject = cleaner([{"delete": [0], "self_repair": False}])
        await subject.clean("um so we should ship it", "en")
        system_message = subject._client.completions.calls[0]["messages"][0][  # type: ignore[union-attr]
            "content"
        ]
        assert "WHEN UNSURE, DELETE LESS" in system_message
        # An example whose right answer is an empty deletion, so the model has seen that
        # answering "nothing" is normal rather than a failure to do the job.
        assert '{"delete": [], "self_repair": false}' in system_message


class TestPureHelpers:
    def test_deletions_take_the_comma_that_belonged_to_them(self):
        assert apply_deletions("Monday, I mean Tuesday", "en", [0, 1, 2]) == "Tuesday."
        assert apply_deletions("um, so we should ship it", "en", [0]) == "So we should ship it."

    def test_a_comma_between_two_surviving_words_is_the_speakers(self):
        assert apply_deletions("um, we ship, then we test", "en", [0]) == ("We ship, then we test.")

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
