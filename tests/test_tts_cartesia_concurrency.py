"""One TTS process must never have more Cartesia generations in flight than the plan allows.

Production 2026-09-27..29: `429 concurrency_limited ... Current limit: 2`. The consume loop
dispatches up to 8 (speaker, language) keys at once and nothing bounded how many of them reached
Cartesia together, so a meeting with a few speakers and target languages overran the plan and
every excess sentence failed outright instead of waiting a fraction of a second for a slot.

The plan is Pro since 2026-10 (limit 3, measured with the production key); the default follows it.
"""

from __future__ import annotations

import asyncio

import pytest

from shared.config import TTSSettings
from tts_worker.synthesizer import CartesiaSynthesizer


def test_default_matches_the_cartesia_plan_limit() -> None:
    assert TTSSettings().cartesia_max_concurrency == 3


@pytest.mark.asyncio
async def test_generation_slot_caps_concurrent_generations() -> None:
    synth = CartesiaSynthesizer(api_key="test", max_concurrency=2)
    in_flight = 0
    peak = 0

    async def generate() -> None:
        nonlocal in_flight, peak
        async with synth.generation_slot():
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

    await asyncio.gather(*(generate() for _ in range(8)))

    assert peak == 2


@pytest.mark.asyncio
async def test_a_zero_limit_still_lets_one_generation_through() -> None:
    synth = CartesiaSynthesizer(api_key="test", max_concurrency=0)

    async with synth.generation_slot():
        pass
