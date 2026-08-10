"""The default voice follows the language, not the exact spelling of its tag.

A Vietnamese-only meeting spoke English. Both language lookups in the synthesizer compared
tags verbatim against a table keyed by primary subtag, so "vi-VN" missed "vi" twice over:
_default_voice_id fell through to the English fallback, and list_voices matched nothing and
starved the catalog into that same fallback.

The fallback exists for a language nobody has a voice for. It must never be reached by a
different spelling of one that does.
"""

import pytest

from tts_worker.synthesizer import CartesiaSynthesizer

VIETNAMESE = "5619d38c-cf51-4d8e-9575-48f61a280413"
ENGLISH = "694f9389-aac1-45b6-b726-9d9369183238"


@pytest.mark.parametrize("tag", ["vi", "vi-VN", "VI", "  vi  ", "vi-vn"])
def test_every_spelling_of_vietnamese_gets_the_vietnamese_voice(tag: str) -> None:
    # "vi-VN" returning the English id is the whole bug, in one assertion.
    assert CartesiaSynthesizer._default_voice_id(tag) == VIETNAMESE


@pytest.mark.parametrize("tag", ["en", "en-US", "EN"])
def test_english_still_resolves_to_english(tag: str) -> None:
    assert CartesiaSynthesizer._default_voice_id(tag) == ENGLISH


def test_a_language_with_no_voice_still_falls_back() -> None:
    # The fallback is not removed — it is just no longer reachable by a spelling.
    assert CartesiaSynthesizer._default_voice_id("ja") == ENGLISH
    assert CartesiaSynthesizer._default_voice_id("") == ENGLISH
