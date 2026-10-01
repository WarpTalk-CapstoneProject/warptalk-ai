"""A commit's early sentences come only from ITS OWN transcription item.

gpt-live-transcribe transcribes audio while it is appended, so a speaker's socket also carries
deltas for audio no commit of this call owns: a streamed turn that was abandoned and cleared
after the model had already spoken its opening words, or the opening of the next turn. Before
this fix every delta went into one buffer, and tools/meeting_sim reproduced what that does in a
three-person meeting:

    flushed "Ai Vậy mình cắt tính năng voice clone…"        ("Ai" is the NEXT turn's first word)
    flushed "…the onboarding flow Two more はい、でもスカ…"   (two other turns mixed in)

followed by `stt_delta_final_mismatch` — whose answer is to drop the rest of the turn — and
`filtered_repetition` throwing a whole real sentence away as a stutter.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from stt_worker.model import TranscribedSegment
from tests.test_stt_worker import _make_stt_with_conn


def _delta(text: str, item: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="conversation.item.input_audio_transcription.delta", delta=text, item_id=item
    )


def _committed(item: str) -> SimpleNamespace:
    return SimpleNamespace(type="input_audio_buffer.committed", item_id=item, previous_item_id=None)


def _completed(text: str, item: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="conversation.item.input_audio_transcription.completed", transcript=text, item_id=item
    )


async def _run(events: list[Any], sample_audio_bytes: bytes) -> tuple[list[str], list[str], Any]:
    stt, conn = _make_stt_with_conn(events)
    early: list[TranscribedSegment] = []

    async def on_early(seg: TranscribedSegment) -> None:
        early.append(seg)

    result = await stt.transcribe(
        sample_audio_bytes,
        language="vi",
        meeting_id="m1",
        speaker_id="s1",
        on_early_segment=on_early,
    )
    return [s.text for s in early], [s.text for s in result], stt


async def test_an_abandoned_turns_deltas_never_open_this_commit(sample_audio_bytes: bytes) -> None:
    events = [
        # The model already spoke the opening of a turn that was then cleared.
        _delta(" Ai", "item_orphan"),
        # This commit's own audio, streamed in while the speaker talked.
        _delta(" Vậy mình cắt tính năng voice clone ra khỏi bản đầu tiên.", "item_mine"),
        _committed("item_mine"),
        _delta(" Còn lại vẫn giữ ngày 15.", "item_mine"),
        _completed(
            "Vậy mình cắt tính năng voice clone ra khỏi bản đầu tiên. Còn lại vẫn giữ ngày 15.",
            "item_mine",
        ),
    ]

    early, final, _stt = await _run(events, sample_audio_bytes)

    assert early == [
        "Vậy mình cắt tính năng voice clone ra khỏi bản đầu tiên.",
        "Còn lại vẫn giữ ngày 15.",
    ]
    # Nothing was dropped as a delta/final mismatch: the whole turn is accounted for.
    assert final == []


async def test_the_rest_of_the_turn_is_not_dropped_when_another_item_interleaves(
    sample_audio_bytes: bytes,
) -> None:
    events = [
        _delta(" And the onboarding flow.", "item_mine"),
        _delta(" Two more", "item_next"),
        _committed("item_mine"),
        _delta(" Still needs work", "item_mine"),
        _completed("And the onboarding flow. Still needs work", "item_mine"),
    ]

    early, final, _stt = await _run(events, sample_audio_bytes)

    assert early == ["And the onboarding flow."]
    assert final == ["Still needs work"]


async def test_another_items_completion_does_not_end_this_commit(sample_audio_bytes: bytes) -> None:
    events = [
        _completed("", "item_old"),
        _committed("item_mine"),
        _delta(" Hai ngày nữa.", "item_mine"),
        _completed("Hai ngày nữa.", "item_mine"),
    ]

    early, final, _stt = await _run(events, sample_audio_bytes)

    assert early == ["Hai ngày nữa."]
    assert final == []


async def test_the_next_items_deltas_wait_on_the_session_for_its_own_commit(
    sample_audio_bytes: bytes,
) -> None:
    events = [
        _committed("item_mine"),
        _delta(" Đúng rồi.", "item_mine"),
        _delta(" Nhưng mà", "item_next"),
        _completed("Đúng rồi.", "item_mine"),
    ]

    early, final, stt = await _run(events, sample_audio_bytes)

    assert early == ["Đúng rồi."]
    session = next(iter(stt._sessions.values()))
    assert session["item_deltas"] == {"item_next": " Nhưng mà"}


async def test_events_without_item_ids_behave_as_before(sample_audio_bytes: bytes) -> None:
    events = [
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.delta", delta="Hello there."
        ),
        SimpleNamespace(type="conversation.item.input_audio_transcription.delta", delta=" How are"),
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed",
            transcript="Hello there. How are you today?",
        ),
    ]

    early, final, _stt = await _run(events, sample_audio_bytes)

    assert early == ["Hello there."]
    assert final == ["How are you today?"]
