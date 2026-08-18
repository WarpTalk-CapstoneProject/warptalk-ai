"""The scaled energy floor has to be REACHED, not just written.

`_publish_speech_chunk` grew a `speech_samples` parameter so the floor could be weighed against
the speech in a chunk rather than against the VAD padding wrapped around it. The parameter
defaults to None, and every call site left it out — so the scaling branch never ran once, the
floor stayed at a flat 0.02, and the measurement that motivated the change described behaviour
the code did not have.

That is a whole class of defect this repo has hit repeatedly: the fix is written, reviewed and
correct, and connected to nothing. So there are two tests here and they fail for different
reasons. The first checks the arithmetic. The second checks that a caller actually asks for it —
which is the one that would have caught this.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest

from livekit_ingress_worker import worker as ingress_module
from livekit_ingress_worker.worker import LiveKitIngressWorker
from shared.config import LiveKitSettings, WorkerSettings

SAMPLE_RATE = 16000

# Loud enough to clear the floor once it is weighed against the speech, too quiet to clear the
# flat 0.02. Constant amplitude, so RMS is exactly this value: 491 / 32768 = 0.01498.
#
# Flat floor        0.02
# Scaled at 25%     0.02 * sqrt(0.25) = 0.01
_SAMPLE_VALUE = 491
_TOTAL_SAMPLES = SAMPLE_RATE  # 1s
_SPEECH_SAMPLES = SAMPLE_RATE // 4  # 25% of it was actually speech


def _worker() -> LiveKitIngressWorker:
    settings = WorkerSettings(
        livekit=LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")
    )
    return LiveKitIngressWorker(settings=settings)


def _chunk() -> bytearray:
    return bytearray(_SAMPLE_VALUE.to_bytes(2, "little", signed=True) * _TOTAL_SAMPLES)


async def _published(speech_samples: int | None) -> list[Any]:
    """Run one chunk through the gate; return whatever reached the stream."""
    worker = _worker()
    sent: list[Any] = []

    async def _capture(stream: str, room: str, payload: Any) -> None:
        sent.append((stream, room, payload))

    async def _language(_room: str, _speaker: str) -> str:
        return "vi"

    worker.publish = _capture  # type: ignore[method-assign]
    worker._speaker_language = _language  # type: ignore[method-assign]

    await worker._publish_speech_chunk(
        "room",
        "speaker",
        _chunk(),
        0,
        SAMPLE_RATE,
        speech_samples=speech_samples,
    )
    return sent


@pytest.mark.asyncio
async def test_flat_floor_still_rejects_this_chunk() -> None:
    """Without the speech share, the chunk is judged on padding-diluted RMS and dropped.

    This is the BEFORE behaviour, kept as a test so the one below is proving a difference rather
    than restating something that was always true.
    """
    assert await _published(None) == []


@pytest.mark.asyncio
async def test_speech_share_lets_a_short_utterance_through() -> None:
    """Same audio, same loudness — told how much of it was speech, the gate accepts it."""
    assert len(await _published(_SPEECH_SAMPLES)) == 1


def test_every_call_site_passes_speech_samples() -> None:
    """THE TEST THAT WOULD HAVE CAUGHT IT.

    The two above pass whether or not any caller supplies the argument, because they supply it
    themselves. Only the source can answer whether production does.
    """
    source = Path(inspect.getfile(ingress_module)).read_text(encoding="utf-8")

    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_publish_speech_chunk"
    ]

    assert calls, "no _publish_speech_chunk call sites found — did the method get renamed?"

    missing = [
        node.lineno
        for node in calls
        if not any(keyword.arg == "speech_samples" for keyword in node.keywords)
    ]
    assert not missing, (
        f"_publish_speech_chunk called without speech_samples at line(s) {missing}. "
        "The energy floor silently falls back to a flat 0.02 there, which is the whole "
        "behaviour the parameter exists to replace."
    )
