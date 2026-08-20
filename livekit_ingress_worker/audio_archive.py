"""Keeping the speech this worker heard, so a second pass can hear it again.

WHAT THIS IS FOR
    The live transcript is produced under a two-second deadline, because the same text
    drives dubbing and somebody is waiting for it. A meeting *record* has no deadline and
    only has to be right. Those are two different jobs, and today one artifact does both —
    tuned for the one with the clock.

    Splitting them needs the audio to still exist after the meeting. It does not: the
    `audio:chunks` Redis stream is ephemeral, and the LiveKit room-composite recording is a
    mix of the room, which by then contains every `ai-interpreter` bot's dubbed voice
    layered over the speaker. There is nowhere to re-transcribe *from*.

    This module is that missing source. It costs no LiveKit egress minutes, because the
    audio is already here — the worker has it in hand to feed STT.

WHAT IT IS NOT
    Not the meeting recording. What lands here is what the VAD and the near-field gate let
    through — speech, with the silence and the rejected far-field audio absent by
    construction. Anyone wanting to *watch the meeting back* wants the composite recording,
    not this.

THE PROPERTY THAT MATTERS
    The archive is tapped at the same point the `audio:chunks` message is built, from the
    same bytes. So a second pass is handed exactly what the first pass was handed — same
    audio, different model, no other variable. That is what makes a WER comparison between
    the two passes mean anything at all; an archive recorded from a different tap would
    quietly turn the measurement into a comparison of two audio paths.

HOW THE TIMELINE IS REBUILT
    A chunk arrives once its utterance has *finished*, so its arrival time is the moment
    speech ended. Each utterance is therefore placed to END where it arrived, and the gap
    before it is filled with silence. That keeps a sample offset in the archive equal to a
    millisecond offset in the meeting, which is what lets a second-pass segment carry a
    timestamp the meeting page can still scroll to.

    Silence is why the files are FLAC rather than WAV. An hour of one person's meeting is
    mostly other people talking; as PCM that is ~115 MB of zeroes per speaker, and as FLAC
    it is almost nothing. The compression is lossless, so the speech itself is untouched.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from shared.logger import get_logger

logger = get_logger(__name__)

#: Silence longer than this between two utterances is still written in full; the constant
#: exists only to cap a single pad allocation so a pathological clock cannot ask for a
#: gigabyte of zeroes in one call. Anything longer is written in slices of this size.
_MAX_PAD_SECONDS = 60


@dataclass(frozen=True)
class SpeechSpan:
    """One utterance's place on the meeting clock, in milliseconds."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class ArchivedTrack:
    """One speaker's archived speech from one meeting."""

    meeting_id: str
    speaker_id: str
    path: Path
    sample_rate: int
    #: Samples actually written, silence included — the track's length on the meeting clock.
    frames: int
    #: Where the speech actually is. Written beside the audio as `{speaker}.json`.
    #:
    #: WHY THE SILENCE HAS TO BE DESCRIBED
    #:     The padding that makes an offset in this file mean an offset in the meeting also makes
    #:     the file mostly silence — an hour of one person's meeting is mostly other people
    #:     talking. Whisper-family models do not merely skip silence: they fill it, confidently,
    #:     with whatever the surrounding context suggests. `stt_worker/model.py` carries a whole
    #:     filter chain built from production hallucinations for exactly that reason.
    #:
    #:     A second pass therefore needs to know where nobody was speaking, so it can DROP a
    #:     segment the model placed there rather than trust it. Reconstructing that from the audio
    #:     means re-running VAD over a file whose VAD decisions are already known — this is those
    #:     decisions, recorded once, by the code that made them.
    spans: list[SpeechSpan] = field(default_factory=list)

    @property
    def spans_path(self) -> Path:
        return self.path.with_suffix(".json")

    @property
    def duration_ms(self) -> int:
        return int(self.frames * 1000 / self.sample_rate) if self.sample_rate else 0


class _SpeakerTrack:
    """The open FLAC file for one (meeting, speaker), and where its write head is."""

    def __init__(self, path: Path, sample_rate: int) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0
        self.spans: list[SpeechSpan] = []
        self._file = sf.SoundFile(
            str(path),
            mode="w",
            samplerate=sample_rate,
            channels=1,
            format="FLAC",
            subtype="PCM_16",
        )

    def pad_to(self, target_frame: int) -> None:
        """Write silence until the head sits at `target_frame`."""
        missing = target_frame - self.frames
        if missing <= 0:
            return
        block = self.sample_rate * _MAX_PAD_SECONDS
        while missing > 0:
            step = min(missing, block)
            self._file.write(np.zeros(step, dtype=np.int16))
            self.frames += step
            missing -= step

    def write(self, samples: npt.NDArray[np.int16]) -> None:
        # Recorded from the write head rather than from the requested start: an overlapping
        # utterance is clamped forward (see append), and the span has to describe where the audio
        # ACTUALLY is, not where it asked to be.
        start_ms = int(self.frames * 1000 / self.sample_rate)
        self._file.write(samples)
        self.frames += len(samples)
        self.spans.append(SpeechSpan(start_ms, int(self.frames * 1000 / self.sample_rate)))

    def close(self) -> None:
        self._file.close()


class MeetingAudioArchive:
    """Per-meeting, per-speaker FLAC of the speech this worker forwarded to STT.

    Deliberately synchronous and deliberately local. Writing a FLAC block is microseconds
    and happens on a path that is already doing far more expensive work; making it async
    would buy nothing and would let an archive failure interleave with the publish it is
    supposed to shadow. Uploading — which IS slow — is somebody else's job: this hands back
    finished files and says nothing about where they go.

    Every public method swallows its own failures. An archive is a second-order concern
    against carrying the meeting, so a full disk must degrade to "no archive for this
    meeting", never to "this meeting loses audio".
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._tracks: dict[tuple[str, str], _SpeakerTrack] = {}
        #: When each meeting's archive clock starts — set by its first chunk.
        self._started_at: dict[str, float] = {}
        self._failed: set[str] = set()

    # ------------------------------------------------------------------ writing

    def append(
        self,
        meeting_id: str,
        speaker_id: str,
        pcm: bytes,
        sample_rate: int,
        *,
        now: float | None = None,
    ) -> None:
        """Place one finished utterance on this speaker's timeline.

        `now` is the moment the utterance ENDED (its arrival), injectable so the placement
        rule can be tested without sleeping.
        """
        if not pcm or sample_rate <= 0 or meeting_id in self._failed:
            return

        try:
            arrived = time.monotonic() if now is None else now
            samples = np.frombuffer(pcm, dtype=np.int16)
            if samples.size == 0:
                return

            duration = samples.size / sample_rate
            started = self._started_at.setdefault(meeting_id, arrived - duration)
            track = self._track(meeting_id, speaker_id, sample_rate)

            # Place the utterance so it ends where it arrived. A start earlier than the write
            # head means two utterances from one speaker overlap — impossible for a single
            # microphone, so it is a clock artefact rather than the meeting, and clamping is
            # the honest repair: never move audio backwards over audio already written.
            start_frame = int(round((arrived - duration - started) * sample_rate))
            track.pad_to(max(start_frame, track.frames))
            track.write(samples)
        except Exception:
            # Named, then abandoned for this meeting: a track that stopped part-way through
            # would be a timeline with a hole in it, and a hole nobody can see is worse than
            # no archive at all.
            logger.warning(
                "audio_archive_append_failed",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                exc_info=True,
            )
            self._abandon(meeting_id)

    def _track(self, meeting_id: str, speaker_id: str, sample_rate: int) -> _SpeakerTrack:
        key = (meeting_id, speaker_id)
        track = self._tracks.get(key)
        if track is not None:
            return track

        directory = self.root / _safe(meeting_id)
        directory.mkdir(parents=True, exist_ok=True)
        track = _SpeakerTrack(directory / f"{_safe(speaker_id)}.flac", sample_rate)
        self._tracks[key] = track
        return track

    # ------------------------------------------------------------------ finishing

    def close_meeting(self, meeting_id: str) -> list[ArchivedTrack]:
        """Close every track for this meeting and describe what was written.

        Returns an empty list for a meeting that produced nothing — a room where nobody
        spoke is a real outcome, not a failure, and the caller should not have to tell the
        two apart.
        """
        finished: list[ArchivedTrack] = []

        for key in [key for key in self._tracks if key[0] == meeting_id]:
            track = self._tracks.pop(key)
            try:
                track.close()
                archived = ArchivedTrack(
                    meeting_id=meeting_id,
                    speaker_id=key[1],
                    path=track.path,
                    sample_rate=track.sample_rate,
                    frames=track.frames,
                    spans=list(track.spans),
                )
                _write_spans(archived)
                finished.append(archived)
            except Exception:
                logger.warning(
                    "audio_archive_close_failed",
                    meeting_id=meeting_id,
                    speaker_id=key[1],
                    exc_info=True,
                )

        self._started_at.pop(meeting_id, None)
        self._failed.discard(meeting_id)
        return finished

    def _abandon(self, meeting_id: str) -> None:
        """Give up on one meeting without touching any other."""
        self._failed.add(meeting_id)
        for key in [key for key in self._tracks if key[0] == meeting_id]:
            track = self._tracks.pop(key)
            try:
                track.close()
            except Exception:  # noqa: BLE001 — already on the failure path
                pass
        self._started_at.pop(meeting_id, None)

    # ------------------------------------------------------------------ introspection

    def open_track_count(self) -> int:
        """How many tracks are currently open, for the worker's own health reporting."""
        return len(self._tracks)


def _write_spans(track: ArchivedTrack) -> None:
    """The span index, beside the audio it describes.

    A separate file rather than a chunk of FLAC metadata: the audio is uploaded and read by
    whatever can open a FLAC, and a sidecar keeps that true. Failures are logged and swallowed —
    losing the index costs a second pass its silence filter, and losing the AUDIO costs it
    everything, so the audio must never fail because of this.
    """
    try:
        track.spans_path.write_text(
            json.dumps(
                {
                    "meetingId": track.meeting_id,
                    "speakerId": track.speaker_id,
                    "sampleRate": track.sample_rate,
                    "frames": track.frames,
                    "spans": [
                        {"startMs": span.start_ms, "endMs": span.end_ms} for span in track.spans
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        logger.warning(
            "audio_archive_spans_write_failed",
            meeting_id=track.meeting_id,
            speaker_id=track.speaker_id,
            exc_info=True,
        )


def _safe(value: str) -> str:
    """A path component that cannot escape the archive root.

    Meeting and speaker ids come from the meeting pipeline rather than from a user, but they
    are still strings arriving over a network and they are about to become file paths. The
    whitelist is narrow because the real ids are uuids and room names.
    """
    cleaned = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return cleaned[:128] or "unknown"


def describe(tracks: list[ArchivedTrack]) -> dict[str, Any]:
    """A log-friendly summary of one meeting's archive."""
    return {
        "tracks": len(tracks),
        "speakers": [track.speaker_id for track in tracks],
        "total_ms": sum(track.duration_ms for track in tracks),
    }
