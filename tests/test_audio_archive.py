"""Tests for the per-speaker meeting audio archive.

The property under test throughout is the TIMELINE. An archive whose samples do not line up
with the meeting clock is worse than no archive: a second pass would produce segments whose
timestamps look authoritative and point at the wrong moment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from livekit_ingress_worker.audio_archive import MeetingAudioArchive, describe

SR = 16000


def _tone(seconds: float, value: int = 5000) -> bytes:
    """`seconds` of a constant non-zero sample, so silence is distinguishable from speech."""
    return np.full(int(SR * seconds), value, dtype=np.int16).tobytes()


def _read(path: Path) -> np.ndarray:
    data, sample_rate = sf.read(str(path), dtype="int16")
    assert sample_rate == SR
    return data


def test_first_utterance_starts_the_clock_at_its_own_start(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    # Arrives at t=100.0 having lasted 2s, so it is the meeting's first speech and the
    # archive must begin with it rather than with two seconds of nothing.
    archive.append("room", "alice", _tone(2.0), SR, now=100.0)
    tracks = archive.close_meeting("room")

    assert len(tracks) == 1
    audio = _read(tracks[0].path)
    assert len(audio) == pytest.approx(SR * 2, abs=2)
    assert audio[0] != 0


def test_gap_between_utterances_becomes_silence_of_the_same_length(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(1.0), SR, now=100.0)  # speech 99.0 -> 100.0
    archive.append("room", "alice", _tone(1.0), SR, now=105.0)  # speech 104.0 -> 105.0

    tracks = archive.close_meeting("room")
    audio = _read(tracks[0].path)

    # 1s speech + 4s gap + 1s speech, measured from the first utterance's start.
    assert len(audio) == pytest.approx(SR * 6, abs=2)
    assert np.all(audio[:SR] != 0)
    assert np.all(audio[SR : SR * 5] == 0)
    assert np.all(audio[SR * 5 :] != 0)


def test_an_utterance_is_placed_so_that_it_ends_when_it_arrived(tmp_path: Path):
    """A chunk is published once its utterance is over, so arrival marks its END."""
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(0.5), SR, now=100.0)  # 99.5 -> 100.0
    archive.append("room", "alice", _tone(3.0), SR, now=110.0)  # 107.0 -> 110.0

    tracks = archive.close_meeting("room")
    audio = _read(tracks[0].path)

    # Clock starts at 99.5. Second utterance starts at 107.0, i.e. 7.5s in, and the track
    # runs to 110.0 — 10.5s total. Placing it by ARRIVAL instead would put it at 10.5s and
    # make the track 13.5s long.
    assert len(audio) == pytest.approx(SR * 10.5, abs=2)
    assert np.all(audio[int(SR * 7.5) + 1 :] != 0)


def test_speakers_are_separate_tracks_on_one_shared_clock(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(1.0), SR, now=100.0)
    archive.append("room", "bob", _tone(1.0), SR, now=103.0)  # speech 102.0 -> 103.0

    tracks = {track.speaker_id: track for track in archive.close_meeting("room")}
    assert set(tracks) == {"alice", "bob"}

    bob = _read(tracks["bob"].path)
    # Bob's file is not "one second of Bob". It is the meeting up to Bob, so that a sample
    # offset means the same thing in both files.
    assert len(bob) == pytest.approx(SR * 4, abs=2)
    assert np.all(bob[: int(SR * 2)] == 0)


def test_overlapping_utterances_from_one_speaker_never_rewind_the_head(tmp_path: Path):
    """One microphone cannot produce two overlapping utterances; a clock artefact can."""
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(5.0), SR, now=100.0)  # 95.0 -> 100.0
    archive.append("room", "alice", _tone(1.0), SR, now=99.0)  # claims 98.0 -> 99.0

    tracks = archive.close_meeting("room")
    audio = _read(tracks[0].path)

    # Appended after, not written over: 5s + 1s, with nothing lost.
    assert len(audio) == pytest.approx(SR * 6, abs=2)
    assert np.all(audio != 0)


def test_meetings_do_not_share_a_clock_or_a_directory(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room-a", "alice", _tone(1.0), SR, now=100.0)
    archive.append("room-b", "alice", _tone(1.0), SR, now=500.0)

    a = archive.close_meeting("room-a")
    b = archive.close_meeting("room-b")

    assert len(_read(a[0].path)) == pytest.approx(SR, abs=2)
    assert len(_read(b[0].path)) == pytest.approx(SR, abs=2)
    assert a[0].path.parent != b[0].path.parent


def test_a_room_where_nobody_spoke_closes_to_nothing_rather_than_failing(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)
    assert archive.close_meeting("silent-room") == []


def test_empty_and_malformed_chunks_are_ignored(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", b"", SR, now=100.0)
    archive.append("room", "alice", _tone(1.0), 0, now=100.0)

    assert archive.close_meeting("room") == []


def test_ids_that_look_like_paths_cannot_escape_the_archive_root(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("../../etc", "../passwd", _tone(0.2), SR, now=100.0)
    tracks = archive.close_meeting("../../etc")

    assert len(tracks) == 1
    assert tmp_path in tracks[0].path.parents


def test_closing_a_meeting_releases_its_tracks(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(0.2), SR, now=100.0)
    archive.append("room", "bob", _tone(0.2), SR, now=100.0)
    assert archive.open_track_count() == 2

    archive.close_meeting("room")
    assert archive.open_track_count() == 0


def test_describe_reports_what_was_written(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(1.0), SR, now=100.0)
    archive.append("room", "bob", _tone(1.0), SR, now=101.0)

    summary = describe(archive.close_meeting("room"))
    assert summary["tracks"] == 2
    assert sorted(summary["speakers"]) == ["alice", "bob"]
    assert summary["total_ms"] > 0


def test_the_span_index_says_where_the_speech_actually_is(tmp_path: Path):
    """The padding that buys a shared meeting clock also makes the file mostly silence, and a
    Whisper-family model fills silence rather than skipping it. A second pass needs to know which
    parts of an hour of audio anybody was actually speaking in."""
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(1.0), SR, now=100.0)  # 99.0 -> 100.0
    archive.append("room", "alice", _tone(1.0), SR, now=105.0)  # 104.0 -> 105.0
    track = archive.close_meeting("room")[0]

    index = json.loads(track.spans_path.read_text(encoding="utf-8"))
    spans = index["spans"]

    assert len(spans) == 2
    assert spans[0]["startMs"] == 0
    assert spans[0]["endMs"] == pytest.approx(1000, abs=2)
    # The five-second gap is silence, and the index says so by not covering it.
    assert spans[1]["startMs"] == pytest.approx(5000, abs=2)
    assert spans[1]["endMs"] == pytest.approx(6000, abs=2)


def test_a_clamped_overlap_is_indexed_where_the_audio_actually_landed(tmp_path: Path):
    """An utterance whose clock says it overlaps the previous one is appended after it, so the
    span has to describe where the samples are — not where they asked to be."""
    archive = MeetingAudioArchive(tmp_path)

    archive.append("room", "alice", _tone(5.0), SR, now=100.0)  # 95.0 -> 100.0
    archive.append("room", "alice", _tone(1.0), SR, now=99.0)  # claims 98.0 -> 99.0
    track = archive.close_meeting("room")[0]

    spans = json.loads(track.spans_path.read_text(encoding="utf-8"))["spans"]

    assert spans[1]["startMs"] == pytest.approx(5000, abs=2)
    assert spans[1]["endMs"] == pytest.approx(6000, abs=2)


def test_the_index_carries_what_a_reader_needs_to_use_it(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)
    archive.append("room", "alice", _tone(0.5), SR, now=100.0)
    track = archive.close_meeting("room")[0]

    index = json.loads(track.spans_path.read_text(encoding="utf-8"))

    assert index["meetingId"] == "room"
    assert index["speakerId"] == "alice"
    assert index["sampleRate"] == SR
    assert index["frames"] == track.frames


def test_a_meeting_with_no_speech_writes_no_index(tmp_path: Path):
    archive = MeetingAudioArchive(tmp_path)

    assert archive.close_meeting("silent-room") == []
    assert list(tmp_path.rglob("*.json")) == []
