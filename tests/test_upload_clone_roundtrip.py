"""WT-396 — turning a recording somebody uploaded of themselves into a usable voice.

`CreateProfileAsync` used to end at "bytes in a bucket, row marked active". Nothing anywhere
could make a voice out of them, so an uploaded profile was listed as ready in the UI and every
dub still came back in a stock catalogue voice.

Neither service can do this alone and that is deliberate: cloning needs the Cartesia key, which
only the AI side holds, and the recording lives in a bucket only AuthService has credentials for.
So the audio and the answer travel through Redis, the same way the voice catalogue already does
in the other direction.

These pin this half of it — that an answer is ALWAYS written, that the biometric bytes do not
outlive the work, and that one bad request cannot take the consumer down with it.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import (
    _CLONE_RESULT_PREFIX,
    _CLONE_RESULT_TTL_SECONDS,
    _CLONE_SAMPLE_PREFIX,
    TTSWorker,
)

PROFILE = "019fff06-2b98-7e1d-a923-1f53d10b455a"
SAMPLE = b"RIFF....fake wav bytes"


def _worker(sample: bytes | None = SAMPLE) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker.redis = AsyncMock()
    worker.redis.get = AsyncMock(return_value=sample)
    worker.redis.set_with_ttl = AsyncMock()
    worker.redis.delete = AsyncMock()
    synthesizer = MagicMock()
    synthesizer.clone_voice = AsyncMock(return_value="cartesia-voice-abc")
    worker.cartesia = synthesizer  # type: ignore[assignment]
    return worker


def _request(profile_id: str = PROFILE, language: str = "vi") -> dict[bytes, bytes]:
    return {
        b"profile_id": profile_id.encode(),
        b"user_id": b"019f0d00-0de0-7000-9000-000000000002",
        b"language": language.encode(),
    }


def _answer(worker: TTSWorker) -> dict:
    key, payload, ttl = worker.redis.set_with_ttl.await_args.args
    assert key == f"{_CLONE_RESULT_PREFIX}{PROFILE}"
    assert ttl == _CLONE_RESULT_TTL_SECONDS
    return json.loads(payload)


# ── the happy path, and what it hands back ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_uploaded_recording_becomes_a_provider_voice() -> None:
    worker = _worker()

    await worker._handle_upload_clone_request(_request())

    worker.cartesia.clone_voice.assert_awaited_once()
    assert worker.cartesia.clone_voice.await_args.args[0] == SAMPLE
    assert _answer(worker) == {
        "voiceId": "cartesia-voice-abc",
        "provider": "cartesia",
        "error": None,
    }


@pytest.mark.asyncio
async def test_a_language_cartesia_does_not_take_falls_back_rather_than_failing() -> None:
    # AudioChunkMessage.language can be "auto" and a profile can carry anything; Cartesia's
    # clone endpoint requires a real code. Refusing here would lose the recording over a hint.
    worker = _worker()

    await worker._handle_upload_clone_request(_request(language="auto"))

    assert worker.cartesia.clone_voice.await_args.kwargs["language"] == "en"
    assert _answer(worker)["voiceId"] == "cartesia-voice-abc"


# ── an answer is always written ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_sample_that_expired_is_reported_not_left_pending() -> None:
    """A missing answer and an unfinished one look identical to AuthService.

    It renders both as "not usable yet", forever. Saying the recording is gone is what lets the
    page tell somebody to upload again instead of waiting on nothing.
    """
    worker = _worker(sample=None)

    await worker._handle_upload_clone_request(_request())

    answer = _answer(worker)
    assert answer["voiceId"] is None
    assert "no longer available" in answer["error"]
    worker.cartesia.clone_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_provider_failure_is_reported_with_its_reason() -> None:
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(side_effect=RuntimeError("voice too short"))

    await worker._handle_upload_clone_request(_request())

    answer = _answer(worker)
    assert answer["voiceId"] is None
    assert "voice too short" in answer["error"]


@pytest.mark.asyncio
async def test_one_bad_request_does_not_raise_out_of_the_consumer() -> None:
    # This runs in a background task. An exception escaping would end the loop and every later
    # upload would sit unanswered — the silent failure this whole ticket is about.
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(side_effect=RuntimeError("boom"))

    await worker._handle_upload_clone_request(_request())  # must not raise


@pytest.mark.asyncio
async def test_a_request_with_no_profile_id_is_ignored() -> None:
    worker = _worker()

    await worker._handle_upload_clone_request({b"language": b"vi"})

    worker.redis.set_with_ttl.assert_not_awaited()
    worker.cartesia.clone_voice.assert_not_awaited()


# ── the bytes do not outlive the work ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_recording_is_deleted_once_it_has_been_cloned() -> None:
    worker = _worker()

    await worker._handle_upload_clone_request(_request())

    worker.redis.delete.assert_awaited_once_with(f"{_CLONE_SAMPLE_PREFIX}{PROFILE}")


@pytest.mark.asyncio
async def test_the_recording_is_deleted_even_when_cloning_failed() -> None:
    # Biometric audio. Keeping it after a failure buys nothing — the request is not retried from
    # this key — and the expiry is a backstop against a worker that never ran, not the plan.
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(side_effect=RuntimeError("boom"))

    await worker._handle_upload_clone_request(_request())

    worker.redis.delete.assert_awaited_once_with(f"{_CLONE_SAMPLE_PREFIX}{PROFILE}")
