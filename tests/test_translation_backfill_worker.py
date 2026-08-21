"""The after-the-fact half of translation: filling a saved transcript's gaps.

What is being protected here is not "does it translate" — translate_batch is already covered —
but the three ways a backfill can quietly do damage: sending a mixed-language batch to the model
as if it were one language, landing on the stream that drives text-to-speech, and letting the
live worker's out-of-scope suppression silently drop lines the reader asked to see.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from translation_worker.backfill_worker import (
    BACKFILL_RESULT_STREAM,
    TranslationBackfillWorker,
    _bare,
)

SEGMENT_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SEGMENT_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
SEGMENT_C = "cccccccc-cccc-cccc-cccc-cccccccccccc"
MEETING = "dddddddd-dddd-dddd-dddd-dddddddddddd"
STATUS_KEY = "transcript:backfill:t1:en"


def _worker(translations: dict[str, list[str]] | None = None) -> TranslationBackfillWorker:
    worker = TranslationBackfillWorker.__new__(TranslationBackfillWorker)
    worker.logger = MagicMock()
    worker.redis = MagicMock()
    worker.redis.get = AsyncMock(return_value=None)
    worker.redis.set_with_ttl = AsyncMock()

    translator = MagicMock()
    translator.model = "gpt-4.1"

    async def translate_batch(texts, source_lang, target_lang, **kwargs):
        if translations is not None and source_lang in translations:
            return translations[source_lang]
        return [f"[{target_lang}] {text}" for text in texts]

    translator.translate_batch = AsyncMock(side_effect=translate_batch)
    worker.translator = translator

    worker.published: list[tuple[str, str, dict]] = []

    async def publish(stream: str, meeting_id: str, data: dict) -> str:
        worker.published.append((stream, meeting_id, data))
        return "1-0"

    worker.publish = publish
    return worker


def _request(segments: list[dict[str, str]], target: str = "en") -> dict[bytes, bytes]:
    return {
        b"request_id": b"r1",
        b"transcript_id": b"t1",
        b"meeting_id": MEETING.encode(),
        b"workspace_id": b"w1",
        b"target_lang": target.encode(),
        b"status_key": STATUS_KEY.encode(),
        b"segments_json": json.dumps(segments).encode(),
    }


@pytest.mark.asyncio
async def test_splits_a_batch_by_the_language_each_line_was_spoken_in() -> None:
    # A meeting where a Vietnamese and a Japanese speaker alternate produces exactly this
    # request. Handing both to translate_batch under one source language tells the model the
    # Japanese sentences are Vietnamese.
    worker = _worker()

    await worker.process(
        b"1-0",
        _request(
            [
                {"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"},
                {"segment_id": SEGMENT_B, "text": "こんにちは", "source_lang": "ja"},
                {"segment_id": SEGMENT_C, "text": "cảm ơn", "source_lang": "vi"},
            ]
        ),
    )

    calls = worker.translator.translate_batch.await_args_list
    assert len(calls) == 2

    by_source = {call.args[1]: call.args[0] for call in calls}
    assert by_source["vi"] == ["xin chào", "cảm ơn"]
    assert by_source["ja"] == ["こんにちは"]
    assert {call.args[2] for call in calls} == {"en"}


@pytest.mark.asyncio
async def test_results_go_to_the_backfill_stream_and_never_to_the_one_tts_reads() -> None:
    # tts_worker consumes translate:results. A post-meeting backfill landing there would
    # synthesise and bill speech for every line of a meeting that already ended.
    worker = _worker()

    await worker.process(
        b"1-0",
        _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
    )

    streams = {stream for stream, _, _ in worker.published}
    assert streams == {BACKFILL_RESULT_STREAM}
    assert "translate:results" not in streams


@pytest.mark.asyncio
async def test_publishes_the_bare_segment_id_so_the_consumer_can_join_it_back() -> None:
    worker = _worker()

    await worker.process(
        b"1-0",
        _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
    )

    assert len(worker.published) == 1
    _, meeting_id, payload = worker.published[0]
    assert meeting_id == MEETING
    assert payload["segment_id"] == SEGMENT_A
    assert payload["target_lang"] == "en"
    assert payload["source_lang"] == "vi"
    assert payload["translated_text"] == "[en] xin chào"
    assert payload["translator_model"] == "gpt-4.1"
    # An absent latency means "not measured": one API call produced N sentences, so no single
    # sentence has a duration of its own.
    assert "latency_ms" not in payload


@pytest.mark.asyncio
async def test_never_passes_meeting_context_because_it_may_not_suppress_anything() -> None:
    # meeting_context turns on the live worker's out-of-scope suppression, which answers
    # OUT_OF_MEETING_SCOPE for speech it judges unrelated. A reader who asked to see the whole
    # transcript in one language must not have lines quietly dropped from it.
    worker = _worker()

    await worker.process(
        b"1-0",
        _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
    )

    kwargs = worker.translator.translate_batch.await_args_list[0].kwargs
    assert kwargs.get("meeting_context") is None


@pytest.mark.asyncio
async def test_skips_a_line_already_spoken_in_the_requested_language() -> None:
    worker = _worker()

    await worker.process(
        b"1-0",
        _request(
            [
                {"segment_id": SEGMENT_A, "text": "already english", "source_lang": "en"},
                {"segment_id": SEGMENT_B, "text": "xin chào", "source_lang": "vi"},
            ]
        ),
    )

    published_ids = [payload["segment_id"] for _, _, payload in worker.published]
    assert published_ids == [SEGMENT_B]


@pytest.mark.asyncio
async def test_a_locale_tag_still_counts_as_the_language_it_is() -> None:
    worker = _worker()

    await worker.process(
        b"1-0",
        _request(
            [{"segment_id": SEGMENT_A, "text": "already english", "source_lang": "en-US"}],
            target="en",
        ),
    )

    assert worker.published == []
    worker.translator.translate_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_marks_the_run_failed_and_re_raises_when_translation_breaks() -> None:
    worker = _worker()
    worker.translator.translate_batch = AsyncMock(side_effect=RuntimeError("openai down"))

    with pytest.raises(RuntimeError):
        await worker.process(
            b"1-0",
            _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
        )

    worker.redis.set_with_ttl.assert_awaited_once()
    key, value, _ = worker.redis.set_with_ttl.await_args.args
    assert key == STATUS_KEY
    assert value == "failed"


@pytest.mark.asyncio
async def test_unreadable_payload_is_marked_and_dropped_rather_than_retried_forever() -> None:
    worker = _worker()
    request = _request([])
    request[b"segments_json"] = b"{not json"

    await worker.process(b"1-0", request)

    worker.redis.set_with_ttl.assert_awaited_once()
    assert worker.published == []


@pytest.mark.asyncio
async def test_a_glossary_that_outlived_the_meeting_is_used_when_it_is_there() -> None:
    worker = _worker()
    worker.redis.get = AsyncMock(
        return_value=json.dumps([{"source": "pod", "target": "pod"}]).encode()
    )

    await worker.process(
        b"1-0",
        _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
    )

    kwargs = worker.translator.translate_batch.await_args_list[0].kwargs
    assert kwargs["glossary_terms"] == [{"source": "pod", "target": "pod"}]


@pytest.mark.asyncio
async def test_a_missing_glossary_is_not_an_error() -> None:
    # GlossaryStartedEventConsumer writes the key when a room starts and nothing renews it, so a
    # transcript read back days later almost always finds nothing there.
    worker = _worker()
    worker.redis.get = AsyncMock(side_effect=RuntimeError("redis gone"))

    await worker.process(
        b"1-0",
        _request([{"segment_id": SEGMENT_A, "text": "xin chào", "source_lang": "vi"}]),
    )

    assert len(worker.published) == 1


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("en-US", "en"), ("vi_VN", "vi"), ("JA", "ja"), ("  ko ", "ko"), ("", "")],
)
def test_bare_reduces_a_tag_to_the_code_segments_are_stored_with(raw: str, expected: str) -> None:
    assert _bare(raw) == expected
