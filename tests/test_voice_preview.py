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
from tts_worker.worker import _PREVIEW_TEXT, TTSWorker, _preview_failure


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


# ── WT-649: what a failed preview says ──────────────────────────────────────────
#
# A preview of a voice Cartesia does not have used to answer with the SDK's own exception,
# truncated to 200 characters:
#
#     Error code: 404 - {'error_code': 'voice_not_found', 'message': 'The requested voice was
#     not found.', 'title': 'Voice not found', 'request_id': 'e9d42fe9-…'}
#
# That went to somebody pressing a play button. The comment above the truncation had the
# diagnosis right — a stack trace is not a message for one — and truncating only made it a
# shorter stack trace. The classification happens here because this is the only side holding the
# Cartesia key: it is the only side that can tell a voice that does not exist from a key that has
# expired. AuthService picks the sentence from the code.


class _NotFoundError(Exception):
    pass


NotFoundError = _NotFoundError
NotFoundError.__name__ = "NotFoundError"


def _named(name: str) -> Exception:
    """An exception carrying the SDK's type NAME, which is what the mapping keys on."""
    return type(name, (Exception,), {})("provider said something internal")


@pytest.mark.parametrize(
    ("exception_name", "expected_code"),
    [
        ("NotFoundError", "VOICE_NOT_FOUND"),
        ("AuthenticationError", "PROVIDER_REJECTED"),
        ("PermissionDeniedError", "PROVIDER_REJECTED"),
        ("RateLimitError", "PROVIDER_BUSY"),
        ("APITimeoutError", "PROVIDER_UNREACHABLE"),
        ("APIConnectionError", "PROVIDER_UNREACHABLE"),
        ("InternalServerError", "PROVIDER_UNAVAILABLE"),
        ("BadRequestError", "VOICE_NOT_RENDERABLE"),
    ],
)
def test_each_provider_failure_gets_a_code_naming_what_went_wrong(
    exception_name: str, expected_code: str
) -> None:
    code, _message = _preview_failure(_named(exception_name))

    assert code == expected_code


def test_an_unknown_exception_says_so_rather_than_inventing_a_cause() -> None:
    code, message = _preview_failure(_named("SomethingNobodyHasSeen"))

    assert code == "UNKNOWN"
    # The message still carries something, but only for the log — AuthService shows the generic
    # line for an unrecognised code.
    assert "provider said something internal" in message


@pytest.mark.asyncio
async def test_a_failed_render_writes_a_code_beside_the_message() -> None:
    # The end the bug was reported at: the answer written to Redis, which is what AuthService
    # reads and what ultimately decides what a person is told.
    cartesia = _Cartesia(raises=_named("NotFoundError"))
    worker, store = _worker(cartesia)

    await worker._handle_preview_request(_request("voice-abc", "vi"))

    answer = _answer(store, "voice:preview:voice-abc:vi")
    assert answer["audio"] is None
    assert answer["error_code"] == "VOICE_NOT_FOUND"


@pytest.mark.asyncio
async def test_no_audio_is_a_named_outcome_too() -> None:
    worker, store = _worker(_Cartesia(audio=b""))

    await worker._handle_preview_request(_request("voice-abc", "vi"))

    answer = _answer(store, "voice:preview:voice-abc:vi")
    assert answer["error_code"] == "NO_AUDIO"
