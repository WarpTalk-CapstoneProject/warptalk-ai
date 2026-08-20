"""Turning per-speaker transcriptions back into one meeting.

WHY THIS IS EASY HERE AND HARD EVERYWHERE ELSE
    Speaker attribution is the expensive, error-prone half of meeting transcription — general
    notetakers cluster voices out of one mixed stream and guess. WarpTalk never has to: the archive
    holds ONE FILE PER SPEAKER, because LiveKit already knew who was talking. The speaker of a
    segment is the file it came out of, and that is not a judgement, it is a fact.

    And because every file is padded to the same meeting clock, an offset in one file means the
    same instant as the same offset in another. Merging is therefore a sort, not an alignment.

WHY SILENCE HAS TO BE FILTERED OUT AFTERWARDS
    The padding that buys that shared clock also makes each file mostly silence. Whisper-family
    models do not skip silence, they fill it — `stt_worker/model.py` carries a filter chain built
    entirely from production hallucinations. So the whole file is still transcribed, because
    whole-meeting context is the reason a second pass is worth running at all, and then every
    segment that landed where nobody was speaking is dropped.

    Dropped, not trimmed: a model that produced a sentence over silence did not mishear something,
    it invented it, and there is nothing in it worth keeping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechSpan:
    """Where a speaker actually spoke, from the archive's own VAD decisions."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SpeakerSegment:
    """One transcribed segment, before it is placed in the meeting."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


@dataclass(frozen=True)
class MergedSegment:
    """One line of the meeting, with the speaker the file already told us."""

    speaker_id: str
    start_ms: int
    end_ms: int
    text: str
    confidence: float | None = None


#: How much of a segment must sit inside known speech for it to be believed.
#:
#: Not "any overlap at all": a model that filled a silence often begins its invention just as the
#: previous utterance ends, so a single millisecond of contact would wave most of them through.
#: Not "entirely inside" either — a real utterance's boundaries are VAD's opinion, and speech
#: routinely starts a little before VAD notices. Half is the point where the segment is more inside
#: the speech than outside it.
MIN_SPEECH_OVERLAP = 0.5


def overlap_ms(start_ms: int, end_ms: int, spans: list[SpeechSpan]) -> int:
    """How much of a segment lies inside any known speech."""
    total = 0
    for span in spans:
        total += max(0, min(end_ms, span.end_ms) - max(start_ms, span.start_ms))
    return total


def is_within_speech(
    segment: SpeakerSegment, spans: list[SpeechSpan], min_overlap: float = MIN_SPEECH_OVERLAP
) -> bool:
    """Whether this segment sits where the speaker was actually speaking.

    With NO span index — an archive written before the sidecar, or one whose write failed — every
    segment is believed. The index makes the filter possible; its absence must not silently throw
    a meeting's transcript away.
    """
    if not spans:
        return True

    duration = segment.end_ms - segment.start_ms
    if duration <= 0:
        # A zero-length segment carries no evidence either way. Kept: it costs a line, and
        # dropping it would lose a real one-word utterance the model timed badly.
        return True

    return overlap_ms(segment.start_ms, segment.end_ms, spans) / duration >= min_overlap


def merge_speakers(
    transcribed: dict[str, list[SpeakerSegment]],
    spans: dict[str, list[SpeechSpan]] | None = None,
    min_overlap: float = MIN_SPEECH_OVERLAP,
) -> list[MergedSegment]:
    """One meeting timeline from per-speaker transcriptions.

    Ordered by when each line STARTED, then by speaker id — the tie-break is arbitrary but it has
    to be stable, because two runs of the same meeting producing different orders would show up as
    a diff in a transcript nobody changed.
    """
    spans = spans or {}
    merged: list[MergedSegment] = []

    for speaker_id, segments in transcribed.items():
        speaker_spans = spans.get(speaker_id, [])
        for segment in segments:
            if not segment.text.strip():
                continue
            if not is_within_speech(segment, speaker_spans, min_overlap):
                continue
            merged.append(
                MergedSegment(
                    speaker_id=speaker_id,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=segment.text.strip(),
                    confidence=segment.confidence,
                )
            )

    merged.sort(key=lambda segment: (segment.start_ms, segment.speaker_id))
    return merged


def load_spans(payload: dict[str, object] | None) -> list[SpeechSpan]:
    """Read a sidecar written by livekit_ingress_worker.audio_archive.

    Tolerant on purpose: a malformed index yields no spans, which the filter reads as "believe
    everything" rather than as "drop everything".
    """
    raw = (payload or {}).get("spans")
    if not isinstance(raw, list):
        return []

    spans: list[SpeechSpan] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        start, end = entry.get("startMs"), entry.get("endMs")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append(SpeechSpan(start, end))
    return spans
