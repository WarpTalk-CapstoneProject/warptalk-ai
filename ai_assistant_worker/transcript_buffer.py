"""The meeting transcript the summariser will read, kept somewhere a restart cannot erase.

WT-536. `AIAssistantWorker` accumulated a meeting's segments in a plain dict on the instance and
read it back when the end-of-meeting marker arrived. Anything that ended the process between the
first word and the last — a deploy, a crash, an OOM — took the whole meeting with it. The worker
then found nothing to summarise, published nothing, and `ArtifactsFinalizer` waited its 90
seconds and wrote the refusal users see:

    "The AI assistant could not generate a summary for this meeting
     (no transcript content was available or generation did not complete in time)."

Measured on production before the fix: **37 finished meetings carried that refusal while their
transcript was safely saved in the database** — 1789 segments in total, the worst single meeting
751 lines long. (A further 112 refusals were correct: those meetings genuinely had no speech.)

The transcript was never lost. Only the summariser's copy of it was.

WHY A REDIS LIST AND NOT THE SAVED TRANSCRIPT
    Re-reading the persisted transcript is what `SummaryTemplateWorker` does, and it is right for
    that path — a person asked for a rewrite, so their bearer token is in hand and the read is
    made AS them. The end-of-meeting path has no user and no token; giving this worker a
    privileged read of any meeting's transcript would be a new and much larger door than the
    problem needs.

    The list is written on exactly the path that already receives the segments, so it costs one
    pipelined append per segment and needs no new authority at all.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

#: One accumulated segment: (speaker, text, timestamp_ms).
TranscriptSegment = tuple[str, str, int]

#: Long enough to outlive any real meeting, short enough that an abandoned room's buffer expires
#: on its own. Matches the horizon the other per-meeting keys use.
BUFFER_TTL_S = 6 * 60 * 60

#: A hard ceiling on one meeting's buffer. An hour of steady conversation is well under a
#: thousand segments; this is a bound against a room left open, not a limit on real meetings.
MAX_BUFFERED_SEGMENTS = 5000


def buffer_key(meeting_id: str) -> str:
    return f"meeting:{meeting_id}:summary_segments"


def encode_segment(segment: TranscriptSegment) -> str:
    speaker, text, timestamp_ms = segment
    return json.dumps([speaker, text, timestamp_ms], ensure_ascii=False)


def decode_segments(raw: Iterable[bytes | str] | None) -> list[TranscriptSegment]:
    """The buffered segments, skipping anything unreadable.

    Never raises. A half-written entry costs that one line; refusing the whole meeting over it
    would reintroduce exactly the failure this module exists to remove.
    """
    segments: list[TranscriptSegment] = []
    for entry in raw or []:
        try:
            text = entry.decode("utf-8") if isinstance(entry, bytes) else entry
            speaker, spoken, timestamp_ms = json.loads(text)
        except Exception:
            continue
        if not isinstance(speaker, str) or not isinstance(spoken, str):
            continue
        if not isinstance(timestamp_ms, int):
            try:
                timestamp_ms = int(timestamp_ms)
            except Exception:
                continue
        segments.append((speaker, spoken, timestamp_ms))
    return segments


def choose_segments(
    in_memory: list[TranscriptSegment],
    buffered: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    """Whichever copy is more complete.

    Not "buffered if present". A restart mid-meeting leaves memory holding only the second half,
    and the buffer holding all of it — but a Redis outage leaves the buffer holding only what was
    written before it, and memory holding everything. Neither is reliably the longer one, so the
    rule is simply: summarise the fuller record.

    Ties go to memory, which is the copy that needed no network to be right.
    """
    return buffered if len(buffered) > len(in_memory) else in_memory
