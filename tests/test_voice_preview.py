"""A preview must be the SAME rendering the meeting would produce.

The play button exists to answer "is this me, and is this how I will sound?". A sample rendered
on a different code path answers a different question, so these tests pin the two things that
make it the same one: it goes through `synthesize` (which carries `speed="fast"`, a deliberate
choice a dub depends on), and it passes no generation_config — matching a real dub of an
utterance whose prosody could not be measured, which is exactly what a preview is.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import _PREVIEW_TEXT, TTSWorker


class _Redis:
    def __init__(self) -> None:
        self.written: dict[str, tuple[str, int]] = {}

    async def set_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None:
        self.written[key] = (value, ttl_seconds)


class _Cartesia:
    def __init__(self, audio: bytes = b"RIFFfake", raises: Exception | None = None) -> None:
        self._audio = audio
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    async def synthesize(
        self,
        text: str,
        language: str,
        voice_id: str | None = None,
        generation_config: dict[str, float | str] | None = None,
    ) -> tuple[bytes, int, str]:
        self.calls.append(
            {
                "text": text,
                "language": language,
                "voice_id": voice_id,
                "generation_config": generation_config,
            }
        )
        if self._raises is not None:
            raise self._raises
        return self._audio, 1500, voice_id or "default"


def _worker(cartesia: _Cartesia, redis: _Redis | None = None) -> tuple[TTSWorker, _Redis]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker._consumer_name = "tts-test"
    worker.worker_name = "tts"
    worker.cartesia = cartesia  # type: ignore[assignment]
    store = redis or _Redis()
    worker.redis = store  # type: ignore[assignment]
    return worker, store


def _request(voice_id: str, language: str) -> dict[bytes, bytes]:
    return {b"voice_id": voice_id.encode(), b"language": language.encode()}


def _answer(store: _Redis, key: str) -> dict[str, Any]:
    raw, _ttl = store.written[key]
    return json.loads(raw)


@pytest.mark.asyncio
async def test_renders_the_sample_and_stores_it_under_voice_and_language() -> None:
    cartesia = _Cartesia(audio=b"RIFFaudio")
    worker, store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-1", "vi"))

    answer = _answer(store, "voice:preview:voice-1:vi")
    assert base64.b64decode(answer["audio"]) == b"RIFFaudio"
    assert answer["error"] is None


@pytest.mark.asyncio
async def test_the_key_is_the_cache_so_a_second_play_can_skip_cartesia() -> None:
    """Keyed by (voice, language), never by request — that is what makes a repeat play free."""
    worker, store = _worker(_Cartesia())

    await worker._handle_preview_request(_request("voice-1", "en"))

    assert list(store.written) == ["voice:preview:voice-1:en"]
    _raw, ttl = store.written["voice:preview:voice-1:en"]
    assert ttl == 24 * 60 * 60


@pytest.mark.asyncio
async def test_sends_no_generation_config_matching_an_unmeasured_dub() -> None:
    """Prosody is measured from a speaker. A preview has none, and so does a real dub of a
    chunk that was mostly silence — passing None is what makes the two identical."""
    cartesia = _Cartesia()
    worker, _store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-1", "en"))

    assert cartesia.calls[0]["generation_config"] is None


@pytest.mark.asyncio
async def test_speaks_the_language_being_previewed() -> None:
    cartesia = _Cartesia()
    worker, _store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-1", "ja"))

    assert cartesia.calls[0]["language"] == "ja"
    assert cartesia.calls[0]["text"] == _PREVIEW_TEXT["ja"]


@pytest.mark.asyncio
async def test_a_locale_tag_is_reduced_to_the_language_cartesia_is_keyed_by() -> None:
    """ "vi-VN" compared verbatim matches nothing — the same bug that cloned a Vietnamese
    speaker as an English voice, and that starved the catalogue of every non-English language."""
    cartesia = _Cartesia()
    worker, store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-1", "vi-VN"))

    assert cartesia.calls[0]["language"] == "vi"
    assert cartesia.calls[0]["text"] == _PREVIEW_TEXT["vi"]
    assert "voice:preview:voice-1:vi" in store.written


@pytest.mark.asyncio
async def test_an_unknown_language_falls_back_to_english_rather_than_silence() -> None:
    cartesia = _Cartesia()
    worker, _store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-1", "sw"))

    assert cartesia.calls[0]["text"] == _PREVIEW_TEXT["en"]


@pytest.mark.asyncio
async def test_a_provider_failure_is_named_rather_than_left_pending() -> None:
    """A key that was never written and one still being written look identical to the waiting
    request, so silence renders as "still loading" until it times out — every retry, forever."""
    worker, store = _worker(_Cartesia(raises=RuntimeError("cartesia said no")))

    await worker._handle_preview_request(_request("voice-1", "en"))

    answer = _answer(store, "voice:preview:voice-1:en")
    assert answer["audio"] is None
    assert "cartesia said no" in answer["error"]


@pytest.mark.asyncio
async def test_an_empty_render_is_reported_as_a_failure_not_as_audio() -> None:
    worker, store = _worker(_Cartesia(audio=b""))

    await worker._handle_preview_request(_request("voice-1", "en"))

    answer = _answer(store, "voice:preview:voice-1:en")
    assert answer["audio"] is None
    assert answer["error"]


@pytest.mark.asyncio
async def test_a_provider_error_is_truncated_before_it_goes_to_a_person() -> None:
    worker, store = _worker(_Cartesia(raises=RuntimeError("x" * 900)))

    await worker._handle_preview_request(_request("voice-1", "en"))

    assert len(_answer(store, "voice:preview:voice-1:en")["error"]) <= 200


@pytest.mark.asyncio
async def test_a_request_with_no_voice_is_dropped_without_calling_the_provider() -> None:
    cartesia = _Cartesia()
    worker, store = _worker(cartesia)

    await worker._handle_preview_request(_request("", "en"))

    assert cartesia.calls == []
    assert store.written == {}
