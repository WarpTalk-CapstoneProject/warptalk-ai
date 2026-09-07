"""A clone is made in the language the speaker is actually speaking, or not yet at all.

THE REPORT
    "voice clone tiếng việt hoạt động không đồng đều, lúc nghe tiếng việt lúc không."

WHAT IT WAS
    A Cartesia clone is keyed BY LANGUAGE. `_clone_language` mapped anything it did not
    recognise — "auto" included — to "en", and "auto" is exactly what livekit_ingress_worker
    sends until the speaker's language pick has landed in the room's `speak_languages` hash.

    So a Vietnamese speaker whose pick arrived before their first twenty seconds of audio was
    cloned as a Vietnamese voice and sounded right; the same person, in the same room, whose pick
    arrived a moment later was cloned as an ENGLISH voice and then asked to read Vietnamese for
    the rest of the meeting. Nothing was intermittent except the arrival time of a hint — and
    because the cached voice recorded no language, nothing could notice afterwards, and
    `_offer_carry_over` carried the mistake into the next meeting.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import AudioChunkMessage
from tests.test_clone_pitch_coverage import _varied
from tts_worker.worker import TTSWorker, _resolve_clone_language

SAMPLE_RATE = 16000


class _ScriptedRedis:
    """One batch of chunks, then the worker's outer loop stops.

    The two `sleep(0)`s per chunk are load-bearing for the same reason they are in
    test_clone_upgrade: the clone is dispatched with `create_task`, and without yielding, every
    chunk sees "not cloned yet" and the already-cloned branches are never reached.
    """

    def __init__(self, worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
        self._worker = worker
        self._chunks = chunks

    async def consume(self, **_kwargs: Any) -> Any:
        for index, chunk in enumerate(self._chunks):
            yield f"{index}-0".encode(), chunk.to_redis()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        self._worker._running = False


def _worker(**overrides: Any) -> tuple[TTSWorker, list[str]]:
    """A worker whose clones are recorded rather than sent, with the language they were made in.

    `_cloned_language` is stubbed to answer from that same record, which is what the real
    implementation does through Redis — the point of these tests is the DECISION, not the hash.
    """
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(**overrides)
    worker.logger = MagicMock()
    worker._route_states = {}
    worker._room_routes = {"m1": [{"SourceUserId": "s1", "VoiceCloneEnabled": True}]}
    worker._consumer_name = "test"
    worker.worker_name = "tts"
    worker._running = True

    languages: list[str] = []

    async def _clone_and_cache(
        _meeting: str,
        _speaker: str,
        _audio: bytes,
        language: str = "en",
        _sample_rate: int = 16000,
        _score: float | None = None,
    ) -> None:
        languages.append(language)

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return "voice-1" if languages else None

    async def _cloned_language(_meeting: str, _speaker: str) -> str | None:
        return languages[-1] if languages else None

    worker._clone_and_cache = _clone_and_cache  # type: ignore[method-assign]
    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    worker._cloned_language = _cloned_language  # type: ignore[method-assign]
    return worker, languages


def _chunks(language: str, count: int = 1, pcm: bytes | None = None) -> list[AudioChunkMessage]:
    audio = pcm if pcm is not None else _varied()
    return [
        AudioChunkMessage(
            meeting_id="m1",
            speaker_id="s1",
            chunk_index=index,
            audio_data=audio,
            language=language,
            sample_rate=SAMPLE_RATE,
        )
        for index in range(count)
    ]


async def _run(worker: TTSWorker, chunks: list[AudioChunkMessage]) -> None:
    worker.redis = _ScriptedRedis(worker, chunks)  # type: ignore[assignment]
    await worker._consume_audio_for_cloning()
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# ── the pure function ────────────────────────────────────────────────────────────


def test_an_unresolved_hint_is_not_a_language() -> None:
    # None, not "en". These are the strings the pipeline uses for "nobody has told me".
    assert _resolve_clone_language("auto") is None
    assert _resolve_clone_language("") is None
    assert _resolve_clone_language("unknown") is None


def test_a_locale_tag_is_still_its_language() -> None:
    assert _resolve_clone_language("vi-VN") == "vi"
    assert _resolve_clone_language("en-US") == "en"


def test_a_language_cartesia_cannot_serve_still_falls_back() -> None:
    # Waiting for an answer that is never coming would mean never cloning at all. "km" is a real
    # declaration, so it is answered — unlike "auto", which is the absence of one.
    assert _resolve_clone_language("km") == "en"


# ── the capture loop ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_does_not_clone_while_the_language_is_unresolved() -> None:
    worker, languages = _worker(voice_clone_min_seconds=10.0)
    await _run(worker, _chunks("auto", count=2))

    assert languages == [], (
        "an 'auto' hint must not be cloned as English — that is the whole defect"
    )


@pytest.mark.asyncio
async def test_clones_once_the_language_arrives() -> None:
    # The waiting is not a refusal: the buffer keeps sliding, and the moment the pick lands the
    # very next acceptable clip is cloned — in the right language.
    worker, languages = _worker(voice_clone_min_seconds=10.0)
    await _run(worker, _chunks("auto") + _chunks("vi"))

    assert languages == ["vi"]


@pytest.mark.asyncio
async def test_re_clones_when_the_speaker_turns_out_to_speak_something_else() -> None:
    # A clone made under one language and a speaker demonstrably on another is not a quality
    # question, so it must not be held back by the upgrade budget or the score margin.
    worker, languages = _worker(voice_clone_min_seconds=10.0, voice_clone_max_upgrades=0)
    await _run(worker, _chunks("en") + _chunks("vi"))

    assert languages == ["en", "vi"], (
        f"the voice should have been rebuilt in Vietnamese; got {languages}"
    )


@pytest.mark.asyncio
async def test_does_not_re_clone_while_the_language_is_unchanged() -> None:
    # The counterpart, and the one that stops this becoming a re-clone on every chunk: the same
    # language twice is not a mismatch, so the ordinary upgrade rules apply and nothing churns.
    worker, languages = _worker(voice_clone_min_seconds=10.0, voice_clone_max_upgrades=0)
    await _run(worker, _chunks("vi") + _chunks("vi"))

    assert languages == ["vi"]


@pytest.mark.asyncio
async def test_a_locale_tag_does_not_look_like_a_different_language() -> None:
    # "vi" then "vi-VN" is one language spelled two ways. Comparing verbatim would re-clone the
    # speaker's voice for nothing, on a paid API call, in the middle of the meeting.
    worker, languages = _worker(voice_clone_min_seconds=10.0, voice_clone_max_upgrades=0)
    await _run(worker, _chunks("vi") + _chunks("vi-VN"))

    assert languages == ["vi"]
