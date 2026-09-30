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
    _clone_failure,
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


# ── the answer says WHY, in a form AuthService can store (2026-09-18) ────────────────────────


class APIStatusError(Exception):  # noqa: N818 — named as the Cartesia SDK names it
    """Stand-in for cartesia.APIStatusError: CI does not install the `tts` extra.

    402 has no SDK subclass, so this bare class is exactly what the SDK raises for it.
    """

    def __init__(self, status_code: int, body: object) -> None:
        super().__init__(f"Error code: {status_code} - {body}")
        self.status_code = status_code
        self.body = body


_PLAN_BODY = {
    "error_code": "plan_upgrade_required",
    "message": "This feature is not available on the free tier, please upgrade your subscription.",
    "title": "Feature not available",
    "request_id": "ba4473ac-d1ff-45b0-9c54-d8343dcd7110",
}


@pytest.mark.asyncio
async def test_a_free_plan_refusal_is_named_as_the_plan_not_as_a_provider_error() -> None:
    """The production failure of 2026-09-18, verbatim from the vendor.

    Filed under the generic "the voice provider returned an error" it told nobody that the fix
    was an account upgrade, and the only other record of it was a log line a deploy deleted.
    """
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(side_effect=APIStatusError(402, _PLAN_BODY))

    await worker._handle_upload_clone_request(_request())

    answer = _answer(worker)
    assert answer["voiceId"] is None
    assert answer["errorCode"] == "PROVIDER_PLAN_REQUIRED"
    assert "plan" in answer["error"]
    # The stored detail is for a person, not a log reader.
    assert "request_id" not in answer["error"]
    assert "ba4473ac" not in answer["error"]


@pytest.mark.asyncio
async def test_a_402_that_is_not_about_the_plan_is_read_as_credits() -> None:
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(
        side_effect=APIStatusError(402, {"error_code": "insufficient_credits", "message": "x"})
    )

    await worker._handle_upload_clone_request(_request())

    assert _answer(worker)["errorCode"] == "PROVIDER_QUOTA_EXCEEDED"


@pytest.mark.asyncio
async def test_a_refused_recording_carries_the_providers_one_line_reason() -> None:
    worker = _worker()
    worker.cartesia.clone_voice = AsyncMock(
        side_effect=APIStatusError(
            400, {"error_code": "invalid_clip", "message": "Clip is too short.", "request_id": "r"}
        )
    )

    await worker._handle_upload_clone_request(_request())

    answer = _answer(worker)
    assert answer["errorCode"] == "SAMPLE_REJECTED"
    assert answer["error"].endswith("Clip is too short.")
    assert "request_id" not in answer["error"]


@pytest.mark.asyncio
async def test_an_expired_sample_is_coded_so_the_page_can_ask_for_a_new_take() -> None:
    worker = _worker(sample=None)

    await worker._handle_upload_clone_request(_request())

    assert _answer(worker)["errorCode"] == "SAMPLE_EXPIRED"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (401, "PROVIDER_REJECTED"),
        (403, "PROVIDER_REJECTED"),
        (429, "PROVIDER_BUSY"),
        (503, "PROVIDER_UNAVAILABLE"),
        (422, "SAMPLE_REJECTED"),
    ],
)
def test_the_status_code_decides_the_clone_failure_code(status: int, code: str) -> None:
    assert _clone_failure(APIStatusError(status, {}))[0] == code


def test_an_unrecognised_exception_is_unknown_not_a_guessed_cause() -> None:
    assert _clone_failure(RuntimeError("boom")) == ("UNKNOWN", "boom")
