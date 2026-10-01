"""The rest of a turn survives when its early sentences differ from the final only in spacing.

Early sentences are re-joined with a space; the completed transcript has none between Japanese
sentences and none inside a number the splitter cut at each period. tools/meeting_sim caught
both on real gpt-live-transcribe, on items whose deltas and final agreed character for
character, and each time `stt_delta_final_mismatch` dropped the rest of the turn.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from stt_worker.model import TranscribedSegment, _after_flushed_prefix
from tests.test_stt_worker import _make_stt_with_conn


@pytest.mark.parametrize(
    ("final", "flushed", "rest"),
    [
        # Japanese: no space between sentences in the final.
        (
            "すみません。それは手順書に載っている作業ですか？それともミンさんがその場で判断しましたか？",
            "すみません。 それは手順書に載っている作業ですか？",
            "それともミンさんがその場で判断しましたか？",
        ),
        # A version number the splitter cut at each period.
        (
            "Cái PR đó có review, còn từ 0.2.27 tới 0.2.28 thì không.",
            "Cái PR đó có review, còn từ 0. 2. 27 tới 0.",
            "2.28 thì không.",
        ),
        # Identical text: unchanged behaviour.
        ("Hello there. How are you?", "Hello there.", "How are you?"),
        # Everything flushed.
        ("Hello there.", "Hello there.", ""),
    ],
)
def test_whitespace_alone_is_not_a_revision(final: str, flushed: str, rest: str) -> None:
    assert _after_flushed_prefix(final, flushed) == rest


@pytest.mark.parametrize(
    ("final", "flushed"),
    [
        # The model really did revise a flushed word: still a mismatch, nothing re-published.
        ("Okay, MD nhanh nha, một bốn mươi UTC.", "K, MD nhanh nha,"),
        ("Rule five X X, mẫu năm phần trăm.", "Rule five X X, một năm phần trăm."),
        # The final is shorter than what was flushed.
        ("Hello", "Hello there."),
    ],
)
def test_any_other_difference_is_still_a_revision(final: str, flushed: str) -> None:
    assert _after_flushed_prefix(final, flushed) is None


def _delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta=text)


def _completed(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="conversation.item.input_audio_transcription.completed", transcript=text
    )


async def test_the_rest_of_a_japanese_turn_is_published(sample_audio_bytes: bytes) -> None:
    events: list[Any] = [
        _delta("すみません。"),
        _delta("それは手順書に載っている作業ですか？"),
        _delta("それともミンさんがその場で判断しましたか？"),
        _completed(
            "すみません。それは手順書に載っている作業ですか？それともミンさんがその場で判断しましたか？"
        ),
    ]
    stt, _conn = _make_stt_with_conn(events)
    early: list[TranscribedSegment] = []

    async def on_early(seg: TranscribedSegment) -> None:
        early.append(seg)

    result = await stt.transcribe(
        sample_audio_bytes,
        language="ja",
        meeting_id="m1",
        speaker_id="s1",
        on_early_segment=on_early,
    )

    published = "".join(s.text for s in early) + "".join(s.text for s in result)
    # Every sentence is published once — the last one was dropped before this fix.
    assert published.count("それともミンさんがその場で判断しましたか？") == 1
    assert published.count("すみません。") == 1
