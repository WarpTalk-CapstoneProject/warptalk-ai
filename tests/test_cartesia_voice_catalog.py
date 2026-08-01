"""CartesiaSynthesizer.list_voices — the per-language voice catalog scan.

Regression cover for the scan cap that made Vietnamese unreachable: Cartesia's public
library is ordered with English first, so a language whose voices sit deep in the list
is only found if the cap is above their position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator
from unittest.mock import MagicMock

import pytest

from tts_worker.synthesizer import CartesiaSynthesizer


@dataclass
class _FakeVoice:
    id: str
    name: str
    language: str
    gender: str | None = "feminine"


def _library(vi_positions: list[int], total: int = 900) -> list[_FakeVoice]:
    """A library of `total` voices, English everywhere except `vi_positions` (1-based)."""
    wanted = set(vi_positions)
    return [
        _FakeVoice(id=f"vi-{i}", name=f"Vietnamese {i}", language="vi")
        if i in wanted
        else _FakeVoice(id=f"en-{i}", name=f"English {i}", language="en")
        for i in range(1, total + 1)
    ]


def _synth_listing(library: list[_FakeVoice]) -> tuple[CartesiaSynthesizer, dict[str, int]]:
    """A synthesizer whose client streams `library`, plus a counter of items consumed."""
    consumed = {"count": 0}

    async def _iter(**_kwargs: Any) -> AsyncIterator[_FakeVoice]:
        for voice in library:
            consumed["count"] += 1
            yield voice

    client = MagicMock()
    client.voices.list = _iter

    synth = CartesiaSynthesizer(api_key="test-key")
    synth._client = client
    return synth, consumed


@pytest.mark.asyncio
async def test_finds_a_language_whose_voices_start_past_the_old_300_cap() -> None:
    # Production shape: the first `vi` voice sits at position 459 of ~843.
    synth, _ = _synth_listing(_library(vi_positions=[459, 512, 640, 799]))

    voices = await synth.list_voices("vi", limit=6)

    assert [v["id"] for v in voices] == ["vi-459", "vi-512", "vi-640", "vi-799"]
    assert voices[0]["name"] == "Vietnamese 459"
    assert voices[0]["gender"] == "feminine"


@pytest.mark.asyncio
async def test_stops_as_soon_as_limit_is_reached() -> None:
    # The scan must not walk the whole library once it has what it was asked for.
    synth, consumed = _synth_listing(_library(vi_positions=[10, 20, 30, 800]))

    voices = await synth.list_voices("vi", limit=3)

    assert [v["id"] for v in voices] == ["vi-10", "vi-20", "vi-30"]
    assert consumed["count"] == 30


@pytest.mark.asyncio
async def test_scan_cap_still_bounds_a_language_with_no_voices_at_all() -> None:
    synth, consumed = _synth_listing(_library(vi_positions=[], total=900))

    voices = await synth.list_voices("ja", limit=6, max_scanned=250)

    assert voices == []
    assert consumed["count"] == 250


@pytest.mark.asyncio
async def test_returns_empty_instead_of_raising_when_the_api_fails() -> None:
    # Callers treat [] as "fall back to _default_voice_id()" — never as a synthesis error.
    async def _boom(**_kwargs: Any) -> AsyncIterator[_FakeVoice]:
        raise RuntimeError("cartesia is down")
        yield  # pragma: no cover — makes this an async generator

    client = MagicMock()
    client.voices.list = _boom
    synth = CartesiaSynthesizer(api_key="test-key")
    synth._client = client

    assert await synth.list_voices("vi") == []


@pytest.mark.asyncio
async def test_missing_gender_is_normalised_to_empty_string() -> None:
    # The Gateway deserializes these into VoiceCatalogEntry(Id, Name, Gender) — a null
    # gender is tolerated there, but the catalog should be uniform at the source.
    synth, _ = _synth_listing([_FakeVoice(id="vi-1", name="Linh", language="vi", gender=None)])

    voices = await synth.list_voices("vi")

    assert voices == [{"id": "vi-1", "name": "Linh", "gender": ""}]
