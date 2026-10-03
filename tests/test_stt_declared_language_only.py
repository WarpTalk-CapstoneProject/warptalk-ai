"""A declared speaker's session expects their language and English — not the whole room.

Measured 4 Oct 2026: a Japanese-declared speaker in a vi/ja room was told `[ja, en, vi]` and 6 of 13
lines came back with Vietnamese inside the Japanese ("ấy ai" for "AI"). See _expected_languages.
"""

from stt_worker.model import OpenAISTT, _expected_languages


def test_a_declared_speaker_expects_their_language_and_english_only() -> None:
    assert _expected_languages("ja", {"vi", "ja"}) == ["ja", "en"]
    assert _expected_languages("vi", {"vi", "en", "ja"}) == ["vi", "en"]


def test_an_undeclared_speaker_still_expects_the_room() -> None:
    assert _expected_languages(None, {"vi", "ja"}) == ["ja", "vi", "en"]
    assert _expected_languages("auto", {"vi", "ja"}) == ["ja", "vi", "en"]


def test_the_session_payload_carries_it() -> None:
    stt = OpenAISTT.__new__(OpenAISTT)
    stt.model = "gpt-live-transcribe"
    stt.noise_reduction = "off"
    payload = stt._session_payload(language="ja", prompt=None, allowed_languages={"vi", "ja"})
    assert payload["audio"]["input"]["transcription"]["languages"] == ["ja", "en"]
