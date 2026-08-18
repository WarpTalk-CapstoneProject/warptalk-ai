"""Whose voice a dub is spoken in is the SPEAKER's decision; the listener chooses the LANGUAGE.

The same voice is rendered once per distinct target language, so A speaking Vietnamese with a
cloned voice is heard in English by an English listener — still in A's voice.

It did not work that way. `_get_explicit_voice_choices` was applied to every speaker, and the
client accepts only the preference track once a listener has one, so any listener who had ever
picked a voice silently stopped hearing every cloned speaker in their own voice — while the
speaker saw "My voice", watched the capture succeed, and had no way to learn it was discarded.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from tts_worker.worker import TTSWorker

MEETING = "m1"
SPEAKER = "s1"
LISTENER_VOICE = "listener-picked-voice"
OWN_PROFILE_VOICE = "speaker-profile-voice"
CLONED_VOICE = "speaker-cloned-voice"


def _worker(*, chosen: str | None, cloned: str | None, listener_choices: set[str]) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings()
    worker.logger = MagicMock()
    worker.worker_name = "tts"
    worker._consumer_name = "tts-test"

    worker.chosen_dub_voice = lambda _m, _s: chosen  # type: ignore[method-assign]

    async def _get_voice_id(_meeting: str, _speaker: str) -> str | None:
        return cloned

    async def _choices(_meeting: str, _lang: str) -> set[str]:
        return listener_choices

    async def _hashed(_lang: str, _speaker: str) -> str:
        return "hashed-catalog-voice"

    worker._get_voice_id = _get_voice_id  # type: ignore[method-assign]
    worker._get_explicit_voice_choices = _choices  # type: ignore[method-assign]
    worker._hashed_default_voice_id = _hashed  # type: ignore[method-assign]
    return worker


async def _variants(worker: TTSWorker, target_lang: str = "en") -> list[tuple[str, str, str]]:
    return await worker._resolve_voice_variants(MEETING, SPEAKER, target_lang)


@pytest.mark.asyncio
async def test_a_cloned_speaker_is_not_overridable_by_a_listener() -> None:
    worker = _worker(chosen=None, cloned=CLONED_VOICE, listener_choices={LISTENER_VOICE})

    variants = await _variants(worker)

    assert variants == [(CLONED_VOICE, "cloned", "")]


@pytest.mark.asyncio
async def test_a_speakers_own_profile_is_not_overridable_either() -> None:
    worker = _worker(chosen=OWN_PROFILE_VOICE, cloned=None, listener_choices={LISTENER_VOICE})

    variants = await _variants(worker)

    assert variants == [(OWN_PROFILE_VOICE, "profile", "")]


@pytest.mark.asyncio
async def test_a_listener_may_still_pick_a_stand_in_for_an_unvoiced_speaker() -> None:
    """A speaker on the hashed catalogue default has expressed no preference about how they
    sound, so there is nothing of theirs to override and the feature keeps working."""
    worker = _worker(chosen=None, cloned=None, listener_choices={LISTENER_VOICE})

    variants = await _variants(worker)

    assert ("hashed-catalog-voice", "default", "") in variants
    assert (LISTENER_VOICE, "preference", f"voice-{LISTENER_VOICE[:8]}") in variants


@pytest.mark.asyncio
async def test_the_same_voice_serves_every_target_language() -> None:
    """The "many language versions, one voice" half of the rule — A picks a voice once and is
    heard in it by listeners in every language, because the voice is resolved independently of
    target_lang while `process` is called per language."""
    worker = _worker(chosen=None, cloned=CLONED_VOICE, listener_choices=set())

    for language in ("en", "ja", "vi"):
        assert await _variants(worker, language) == [(CLONED_VOICE, "cloned", "")]


@pytest.mark.asyncio
async def test_an_unvoiced_speaker_with_no_listener_choices_is_still_rendered_once() -> None:
    worker = _worker(chosen=None, cloned=None, listener_choices=set())

    assert await _variants(worker) == [("hashed-catalog-voice", "default", "")]


@pytest.mark.asyncio
async def test_a_listener_choice_equal_to_the_default_is_not_rendered_twice() -> None:
    worker = _worker(chosen=None, cloned=None, listener_choices={"hashed-catalog-voice"})

    assert await _variants(worker) == [("hashed-catalog-voice", "default", "")]
