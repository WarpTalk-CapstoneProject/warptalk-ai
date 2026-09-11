"""The voice library must exist before anybody has held a meeting in that language.

`voice_catalog:{lang}` was only ever written by the dubbing path, for the one language being
dubbed into, with a six-hour TTL. The Voice Profiles page reads those same keys — deliberately,
so it cannot drift from the in-meeting picker and so TTS_API_KEY stays confined to these
workers — and so it showed a language nothing at all until somebody had already been dubbed in
it, then lost it again six hours later.

These pin the warming pass that ends that, and the two properties whose failure would be
silent: that one walk serves every language, and that a failed refresh cannot take synthesis
down with it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tts_worker.synthesizer import CartesiaSynthesizer


class _Voice:
    def __init__(self, voice_id: str, name: str, language: str, gender: str = "") -> None:
        self.id = voice_id
        self.name = name
        self.language = language
        self.gender = gender


class _Voices:
    def __init__(self, voices: list[_Voice]) -> None:
        self._voices = voices
        self.calls = 0

    def list(self, **_: Any):  # noqa: ANN401 - mirrors the SDK's shape
        self.calls += 1

        async def _iter():
            for voice in self._voices:
                yield voice

        return _iter()


class _Client:
    def __init__(self, voices: list[_Voice]) -> None:
        self.voices = _Voices(voices)


def _synth(voices: list[_Voice]) -> CartesiaSynthesizer:
    synth = CartesiaSynthesizer(api_key="k", model="m", sample_rate=44100, speed=1.0)
    synth._client = _Client(voices)  # type: ignore[attr-defined]
    return synth


# The library really is ordered with English first; a Vietnamese voice near the end is the case
# that was unreachable, so the fixture reproduces that ordering rather than a tidy one.
LIBRARY = [
    _Voice("en-1", "Ana", "en", "female"),
    _Voice("en-2", "Ben", "en-US", "male"),
    _Voice("ja-1", "Hana", "ja"),
    _Voice("vi-1", "Lien", "vi-VN", "female"),
    _Voice("vi-2", "Minh", "vi", "male"),
]


@pytest.mark.asyncio
async def test_one_walk_answers_for_every_language() -> None:
    # The whole point of the method. Asking per language would walk the library once per
    # language — ~843 voices, forty times over, to learn what a single walk already knows.
    synth = _synth(LIBRARY)

    buckets = await synth.list_voices_by_language()

    assert synth._client.voices.calls == 1  # type: ignore[attr-defined]
    assert set(buckets) == {"en", "ja", "vi"}


@pytest.mark.asyncio
async def test_a_locale_tag_is_bucketed_under_its_primary_subtag() -> None:
    # Cartesia keys its library by primary subtag, and comparing full tags is what starved the
    # catalog before: `vi-VN` must land with `vi`, not beside it.
    buckets = await _synth(LIBRARY).list_voices_by_language()

    assert [voice["id"] for voice in buckets["vi"]] == ["vi-1", "vi-2"]
    assert [voice["id"] for voice in buckets["en"]] == ["en-1", "en-2"]


@pytest.mark.asyncio
async def test_every_voice_a_language_has_is_kept() -> None:
    # No per-language cap. Every voice Cartesia publishes for a language is one a person may
    # legitimately want to be dubbed in, and the page listing them has its own search.
    buckets = await _synth(LIBRARY).list_voices_by_language()

    assert len(buckets["vi"]) == 2
    assert len(buckets["en"]) == 2


@pytest.mark.asyncio
async def test_the_shape_matches_what_the_lazy_path_writes() -> None:
    # Both write the SAME key. A different shape here would be read by the gateway and the
    # Voice Profiles page as a corrupt catalog, silently, depending on which path wrote last.
    buckets = await _synth(LIBRARY).list_voices_by_language()

    for voices in buckets.values():
        for voice in voices:
            assert set(voice) == {"id", "name", "gender"}
            assert json.dumps(voice)  # serialisable exactly as the cache stores it


@pytest.mark.asyncio
async def test_a_voice_with_no_language_is_dropped_rather_than_bucketed_under_empty() -> None:
    # An empty key would be written to `voice_catalog:` — a key nothing reads, and one that
    # would make the warm log report a language that does not exist.
    buckets = await _synth([_Voice("x-1", "Nameless", "")]).list_voices_by_language()

    assert buckets == {}


@pytest.mark.asyncio
async def test_a_failing_library_returns_what_it_had_rather_than_raising() -> None:
    # Warming runs in a background task beside synthesis. If it could raise, a Cartesia outage
    # would stop being "the page is stale" and start being "the worker died".
    synth = CartesiaSynthesizer(api_key="k", model="m", sample_rate=44100, speed=1.0)

    class _Exploding:
        def list(self, **_: Any):  # noqa: ANN401
            raise RuntimeError("cartesia is down")

    class _BadClient:
        voices = _Exploding()

    synth._client = _BadClient()  # type: ignore[attr-defined]

    assert await synth.list_voices_by_language() == {}


@pytest.mark.asyncio
async def test_the_scan_guard_stops_a_runaway_without_losing_what_it_found() -> None:
    # The cap is a runaway guard, not a budget. Hitting it means the library outgrew it and some
    # language is being starved — the exact shape of the bug that made `vi` unreachable — so it
    # warns and keeps what it has rather than returning nothing.
    buckets = await _synth(LIBRARY).list_voices_by_language(max_scanned=3)

    assert set(buckets) == {"en", "ja"}
    assert "vi" not in buckets
