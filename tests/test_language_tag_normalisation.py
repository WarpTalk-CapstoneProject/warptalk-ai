"""Language tags are compared by language, never by spelling.

The same mistake landed three times in one day, in three repositories' worth of code:

  meeting language picker   "vi-VN" not found in ["en","vi"], so it appended a duplicate
  TTS default voice         "vi-VN" missed the "vi" key, so a Vietnamese room spoke English
  translation passthrough   "vi" vs "vi-VN" read as different, so Vietnamese was "translated"
                            into Vietnamese

Every one of those sites had a normaliser available in a neighbouring module and did not use
it. These tests pin the rule where the helpers live, so the next comparison has something to
fail against.
"""

import pytest

from shared.lang import base_language, is_same_language

SPELLINGS_OF_VIETNAMESE = ["vi", "vi-VN", "VI", "vi-vn", "  vi  ", "vi-Latn-VN"]


@pytest.mark.parametrize("tag", SPELLINGS_OF_VIETNAMESE)
def test_every_spelling_folds_to_the_same_language(tag: str) -> None:
    assert base_language(tag) == "vi"


@pytest.mark.parametrize("tag", SPELLINGS_OF_VIETNAMESE)
def test_any_two_spellings_name_the_same_language(tag: str) -> None:
    # The comparison the translation worker makes to decide whether to translate at all.
    assert is_same_language("vi", tag)
    assert is_same_language(tag, "vi-VN")


def test_different_languages_stay_different() -> None:
    # Normalising must not collapse real distinctions — that would be a worse bug than the
    # one it fixes, because it would silently skip translation between two languages.
    assert not is_same_language("vi", "en")
    assert is_same_language("en-US", "en-GB") is not False  # same language, different region
    assert is_same_language("en-US", "en-GB")


def test_a_missing_tag_names_no_language() -> None:
    # Treating empty as a match would suppress real translations whenever STT failed to
    # report a language.
    assert not is_same_language("", "vi")
    assert not is_same_language("vi", "")
    assert not is_same_language("", "")
