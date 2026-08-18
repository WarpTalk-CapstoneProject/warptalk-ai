"""Sentiment must never be able to damage the translation.

The translation is what the listener hears; the sentiment label is a nicety that unlocks an
emotion label the dub does not currently get at all. So the parse is deliberately lopsided, and
these tests weigh the safety properties far more heavily than the recognition.

The marker rides in free text because the primary translation path is OpenAI Realtime, which
returns free text — `response_format` and JSON mode are not available on it, and the Chat path
is only the fallback.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TranslationSettings
from shared.schemas import ProsodyEnvelope, STTResultMessage
from translation_worker.translator import OpenAITranslator
from translation_worker.valence import INSTRUCTION, MARKERS, split_valence
from translation_worker.worker import TranslationWorker


def _worker(mock_redis_client, worker_settings):
    """Same shape as test_translation_worker's factory, including the started-translation
    gate — process() drops a segment outright unless the room reports translation active, and
    these tests are about what happens once it is."""
    worker = TranslationWorker.__new__(TranslationWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.translation_settings = TranslationSettings()
    worker._paused_rooms = set()
    worker._route_states = {}
    worker._is_translation_active = lambda _room: True  # type: ignore[method-assign]
    worker._mt_glossaries = {}
    worker._recent_source_contexts = {}
    worker._speculative_translations = {}
    worker.worker_name = "translation"
    translator = MagicMock()
    translator.model = "gpt-4.1-mini"
    translator.translate_batch = AsyncMock(return_value=[])
    worker.translator = translator
    return worker


def _stt() -> STTResultMessage:
    return STTResultMessage(
        meeting_id="m1", speaker_id="s1", text="Great work", language="en", confidence=0.95
    )


def _published(mock_redis_client) -> list[dict]:
    return [
        call.args[1]
        for call in mock_redis_client._redis.xadd.call_args_list
        if "translate:results" in str(call.args[0])
    ]


@pytest.mark.parametrize(
    ("marker", "expected"),
    [("⟦+⟧", "positive"), ("⟦-⟧", "negative"), ("⟦=⟧", "neutral")],
)
def test_each_marker_is_recognised_and_removed(marker: str, expected: str) -> None:
    text, valence = split_valence(f"Chúng ta chốt phương án này. {marker}")

    assert text == "Chúng ta chốt phương án này."
    assert valence == expected


def test_a_reply_with_no_marker_is_returned_untouched() -> None:
    # The normal case for the batch path and for any older prompt. Untouched text, no valence —
    # exactly the behaviour before this existed.
    reply = "Chúng ta chốt phương án này."

    assert split_valence(reply) == (reply, None)


def test_an_invented_marker_is_removed_but_not_believed() -> None:
    """The failure that matters most.

    A model that writes `⟦positive⟧` instead of `⟦+⟧` must not have it spoken aloud in the dub,
    and must not have it believed either. Removed, and no valence.
    """
    text, valence = split_valence("Chúng ta chốt phương án này. ⟦positive⟧")

    assert "⟦" not in text
    assert text == "Chúng ta chốt phương án này."
    assert valence is None


def test_a_marker_in_the_middle_is_left_alone() -> None:
    # Only the very end is a marker position. Something bracket-shaped mid-sentence is content,
    # and cutting it would silently delete words the speaker said.
    reply = "Anh ấy nói ⟦+⟧ rồi bỏ đi."

    assert split_valence(reply) == (reply, None)


def test_a_long_bracketed_tail_is_not_swallowed() -> None:
    # The shape-cleanup is length-bounded so a real trailing clause can never be mistaken for a
    # malformed marker and deleted.
    reply = "Kết luận ⟦" + "x" * 80 + "⟧"

    text, valence = split_valence(reply)

    assert text == reply
    assert valence is None


def test_the_text_is_never_longer_than_what_arrived() -> None:
    # The invariant, stated as a property: this function only ever removes.
    for reply in [
        "Xin chào ⟦+⟧",
        "Xin chào",
        "⟦=⟧",
        "",
        "Xin chào ⟦không rõ⟧",
        "  Xin chào ⟦-⟧  ",
    ]:
        text, _ = split_valence(reply)
        assert len(text) <= len(reply)


def test_a_bare_marker_leaves_empty_text() -> None:
    # A reply that is only a marker means the model produced no translation. The caller's
    # existing empty-result path handles that; this must not manufacture text.
    text, valence = split_valence("⟦=⟧")

    assert text == ""
    assert valence == "neutral"


def test_trailing_whitespace_does_not_hide_a_marker() -> None:
    assert split_valence("Xong rồi ⟦+⟧   ")[1] == "positive"
    assert split_valence("Xong rồi ⟦+⟧\n")[1] == "positive"


def test_empty_input_is_safe() -> None:
    assert split_valence("") == ("", None)


def test_the_instruction_names_every_marker_it_expects_back() -> None:
    # A prompt that asks for a token the parser does not know yields a stripped marker and no
    # valence forever — silent, and only visible as "the dub never has emotion".
    for marker in MARKERS:
        assert marker in INSTRUCTION


def test_the_instruction_asks_about_the_speaker_not_the_translation() -> None:
    # Sentiment of what was SAID. A model rating its own output would label a well-translated
    # complaint positive.
    assert "SENTIMENT OF WHAT THE SPEAKER SAID" in INSTRUCTION


# ── End to end: the label has to actually reach the TTS worker ───────────────────────────────


class TestValenceReachesTheEnvelope:
    """A sentiment that is parsed and then dropped is the same as no sentiment at all.

    `to_generation_config` refuses to emit an emotion label without a valence, and today it
    never gets one — so this is the join that decides whether the whole feature does anything.
    """

    async def test_a_parsed_valence_rides_on_the_published_prosody(
        self, mock_redis_client, worker_settings
    ) -> None:
        worker = _worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        worker.translator.translate_with_valence = AsyncMock(return_value=("Tốt quá!", "positive"))
        stt = _stt().model_copy(
            update={"prosody": ProsodyEnvelope(pitch_lift=1.3, energy_ratio=1.4, arousal="high")}
        )

        await worker.process(b"m", stt.to_redis())

        published = _published(mock_redis_client)
        assert published, "nothing was published"
        envelope = ProsodyEnvelope.from_wire(published[0]["prosody"])
        assert envelope is not None
        assert envelope.valence == "positive"
        # The measured half must survive alongside it — valence is folded in, not substituted.
        assert envelope.arousal == "high"

    async def test_no_valence_leaves_the_envelope_exactly_as_measured(
        self, mock_redis_client, worker_settings
    ) -> None:
        worker = _worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        worker.translator.translate_with_valence = AsyncMock(return_value=("Xin chào", None))
        stt = _stt().model_copy(update={"prosody": ProsodyEnvelope(pitch_lift=1.3)})

        await worker.process(b"m", stt.to_redis())

        envelope = ProsodyEnvelope.from_wire(_published(mock_redis_client)[0]["prosody"])
        assert envelope is not None
        assert envelope.valence == "", "absent must stay absent, never collapse into 'neutral'"

    async def test_a_valence_without_a_measured_delivery_does_not_invent_one(
        self, mock_redis_client, worker_settings
    ) -> None:
        """An absent envelope means STT could not honestly measure this speaker yet.

        Manufacturing one just to carry a sentiment would tell the TTS worker a delivery was
        measured when none was — every ratio a default 1.0 presented as a reading.
        """
        worker = _worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {b"listener-1": b"vi"}
        worker.translator.translate_with_valence = AsyncMock(return_value=("Tệ quá", "negative"))

        await worker.process(b"m", _stt().to_redis())

        assert "prosody" not in _published(mock_redis_client)[0]


class TestTheTranslatorReallyAsksAndReallyStrips:
    """Drives the real translator, because the two failures that matter are invisible from the
    parser's own tests: a prompt that never asks for a marker, and a marker that is never
    stripped. The second one is spoken aloud in the dub."""

    def _translator(self, reply: str) -> OpenAITranslator:
        translator = OpenAITranslator.__new__(OpenAITranslator)
        translator.model = "gpt-4.1-mini"
        translator.max_tokens = 512
        translator.temperature = 0.1
        translator.realtime_model = ""  # force the Chat path
        response = MagicMock()
        response.choices[0].message.content = reply
        translator._client = MagicMock()
        translator._client.chat.completions.create = AsyncMock(return_value=response)
        return translator

    async def test_the_marker_never_reaches_the_caller(self) -> None:
        translator = self._translator("Xin chào ⟦+⟧")

        text, valence = await translator.translate_with_valence("Hello", "en", "vi")

        assert text == "Xin chào", "the marker leaked into the text that gets dubbed and stored"
        assert valence == "positive"

    async def test_the_plain_translate_helper_is_also_clean(self) -> None:
        # Every existing caller — tools, benchmarks, the batch fallback — goes through this one.
        translator = self._translator("Xin chào ⟦-⟧")

        assert await translator.translate("Hello", "en", "vi") == "Xin chào"

    async def test_the_prompt_actually_asks_for_a_marker(self) -> None:
        # Without this the parser is correct and permanently unused: no marker is ever sent, so
        # no valence is ever found, and the dub silently keeps having no emotion.
        translator = self._translator("Xin chào ⟦=⟧")

        await translator.translate_with_valence("Hello", "en", "vi")

        system_prompt = translator._client.chat.completions.create.call_args.kwargs["messages"][0][
            "content"
        ]
        assert INSTRUCTION.strip() in system_prompt
