"""What reaches Cartesia's /voices/clone has to be an audio FILE, not a PCM stream.

THE OBSERVATION (production, all of it)

    transcript.audio_dubbings, every row ever written:

        voice_type | dubs | target_language
        -----------+------+----------------
        default    | 2153 | en 1715 / vi 422 / ja 16
        cloned     |    0 |
        profile    |    0 |

    Voice cloning had never produced a single dub, in any language. It was reported as
    "voice clone tiếng Việt không được" only because Vietnamese is what the team speaks.

    Everything upstream was working. `voice:clone:state` for meeting 01a00547 shows both
    speakers reaching the clone call with a PERFECT quality score:

        s..0001  capturing 2.8 → 23.1  →  cloning  score 1.0  ratio 0.782
        s..0004  capturing 2.2 → 20.5  →  cloning  score 1.0  ratio 0.850

    32 audio routes across 22 rooms had voice_clone_enabled = true, so consent was not it
    either. The clone was requested, correctly, over and over, and never once landed.

THE CAUSE

    `_consume_audio_for_cloning` accumulates `chunk.audio_data` — raw 16-bit mono PCM, which
    is exactly how `assess_clone_sample` reads it back (`np.frombuffer(..., dtype=np.int16)`).
    That buffer went straight to `voices.clone(clip=BytesIO(...))`, and Cartesia's clone
    endpoint takes an audio file with a container. Headerless PCM is not one, so every request
    was refused at the vendor.

    Two things in the same function disagreed about what the buffer was, and only one of them
    was right.

WHY IT SURVIVED

    `_clone_and_cache` caught the exception and only LOGGED it. Every other exit on this path
    publishes its reason to `voice:clone:state` (WT-420), so the stream read
    `capturing → cloning → <nothing>` — indistinguishable from a clone still in flight. The
    two tests below pin both halves: the bytes are decodable, and a refusal is never silent.
"""

from __future__ import annotations

import struct
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 24000
PCM = b"\x11\x22" * 1000


class _CapturingSynthesizer:
    """Stands in for Cartesia. Records exactly what it was asked to clone."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.clip: bytes | None = None
        self.language: str | None = None
        self._fail = fail

    async def clone_voice(self, audio_bytes: bytes, _label: str, language: str = "en") -> str:
        self.clip = audio_bytes
        self.language = language
        if self._fail:
            raise self._fail
        return "voice-abc"


class _StubRedis:
    async def hset(self, *_a: Any, **_k: Any) -> None: ...
    async def expire(self, *_a: Any, **_k: Any) -> None: ...
    async def publish_system_event(self, **_k: Any) -> None: ...


def _worker(fail: Exception | None = None) -> tuple[TTSWorker, _CapturingSynthesizer, list[str]]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker.redis = _StubRedis()  # type: ignore[assignment]

    synthesizer = _CapturingSynthesizer(fail)
    worker._require_cartesia = lambda: synthesizer  # type: ignore[method-assign,assignment]

    noted: list[str] = []

    async def _note(_key: tuple[str, str], reason: str, **_metrics: Any) -> None:
        noted.append(reason)

    worker._note_clone_state = _note  # type: ignore[method-assign]
    return worker, synthesizer, noted


@pytest.mark.asyncio
async def test_the_clip_sent_to_cartesia_is_a_riff_wav_not_bare_pcm() -> None:
    worker, synthesizer, _noted = _worker()

    await worker._clone_and_cache("m1", "s1", PCM, "vi", SAMPLE_RATE)

    clip = synthesizer.clip
    assert clip is not None, "nothing was sent to the provider at all"
    assert clip[:4] == b"RIFF" and clip[8:12] == b"WAVE", (
        "The clone buffer went to Cartesia as headerless PCM. That is the entire reason no "
        "cloned voice has ever been produced in production — the endpoint takes a file, and "
        f"this is not one: {clip[:12]!r}"
    )
    # The header has to describe THIS audio. A correct container carrying a wrong sample rate
    # is worse than no container: it is accepted, and comes back at the wrong pitch.
    (declared_rate,) = struct.unpack("<I", clip[24:28])
    (declared_data_len,) = struct.unpack("<I", clip[40:44])
    assert declared_rate == SAMPLE_RATE, (
        f"header claims {declared_rate}Hz, audio is {SAMPLE_RATE}Hz"
    )
    assert declared_data_len == len(PCM)
    assert clip[44:] == PCM, "the samples must be passed through untouched"


@pytest.mark.asyncio
async def test_a_provider_refusal_is_published_rather_than_only_logged() -> None:
    worker, _synthesizer, noted = _worker(fail=RuntimeError("unsupported audio format"))

    await worker._clone_and_cache("m1", "s1", PCM, "vi", SAMPLE_RATE)

    failures = [reason for reason in noted if reason.startswith("clone_failed:")]
    assert failures, (
        "A clone the provider refused reported nothing to voice:clone:state, so the stream "
        "stopped at 'cloning' forever — which is what made a dead feature look like a slow "
        f"one for its entire life. Saw: {noted}"
    )
    assert "unsupported audio format" in failures[0], "the verdict has to name what the vendor said"


@pytest.mark.asyncio
async def test_a_successful_clone_says_so() -> None:
    worker, _synthesizer, noted = _worker()

    await worker._clone_and_cache("m1", "s1", PCM, "vi", SAMPLE_RATE)

    assert "cloned" in noted, f"a clone that worked should close out its own state; saw {noted}"
