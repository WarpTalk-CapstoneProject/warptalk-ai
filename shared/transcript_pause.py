"""Whether the host has paused TRANSCRIPT RECORDING for a room. WT-605.

READ THIS BEFORE REACHING FOR `BaseWorker._paused_rooms` — THEY ARE TWO DIFFERENT PAUSES
    `_paused_rooms` is *Pause Translation Room*: the whole translation session is suspended,
    the backend publishes `room_status = "PAUSED"` on `translationRoom:*:events`, and every
    worker downstream of it goes quiet on purpose — `livekit_ingress_worker` even flushes the
    speech buffer, so the room loses its dubbed audio entirely (see
    `tests/test_livekit_ingress_model.py`, which pins that behaviour as deliberate).

    THIS pause stops one thing and one thing only: writing the meeting down. Translation,
    dubbing, captions and LiveKit carry on untouched, and STT MUST keep running — it is the
    only producer of `stt:results`, which is `translation_worker`'s sole input, which is in
    turn `tts_worker`'s sole input. Gating audio or STT on this flag would silence the room
    for everyone while the host believed they had merely stopped the note-taking.

    So: never set `room_status` from this, never add a room to `_paused_rooms` because of it,
    and never gate a stage on it unless that stage's output ends up in the written record.

THE CROSS-REPO CONTRACT
    Redis key      `translationRoom:{roomId}:transcript_paused`
    Written by     warptalk-backend (translation-room), on TranscriptPaused / TranscriptResumed
    Read by        this repo, and by the backend's own
                   `TranscriptRedisConsumerService.IsRoomTranscriptPausedAsync`
    Value          any truthy string ("1", "true") while recording is paused. The key is
                   ABSENT — not "0" — while it is not. Both are honoured here so neither side
                   has to care which one the other writes.

    It is a DURABLE key rather than a pub/sub event, and that is the whole point. The backend
    also publishes `TranscriptPaused` on `warptalk:translation-room:commands`, but pub/sub has
    no replay: a worker restarted mid-meeting — which is what every deploy does to every
    worker — would never learn a pause that was announced while it was down, and would resume
    recording a meeting the host had muted. `shared/base_worker.py::_load_route_snapshot`
    documents that same failure for the route cache; this key is the version of it that cannot
    happen, because there is nothing to miss.

    Companion key, written by the backend and deliberately NOT read here:

        `translationRoom:{roomId}:transcript_paused_segments`   SET, TTL 30 minutes

    The backend records the segment ids it skipped while paused, so that a `translate:results`
    or `tts:results` message referring to one of them is dropped quietly instead of retried
    five times and dead-lettered. Nothing in this repo needs it — the AI side gates on the
    boolean above, before a segment is ever recorded — but it is written here so the next
    person to grep `transcript_paused` finds BOTH halves of the contract and does not
    re-derive the second one from an alert.

WHY THE FAILURE DIRECTION IS "NOT PAUSED"
    A Redis error, or a key nobody has written yet, means this side does not know. It answers
    "not paused" and keeps the segment, matching `IsRoomTranscriptPausedAsync` on the backend.
    Getting one sentence into a summary because Redis was down is a cheaper mistake than
    silently producing an empty record of a real meeting and presenting it as complete.
"""

from __future__ import annotations

from typing import Any, Protocol

#: The one true spelling of the key. Anything constructing it by hand will drift.
TRANSCRIPT_PAUSED_KEY = "translationRoom:{room_id}:transcript_paused"

#: The backend's skip list (see the module docstring). Named here so the contract is
#: discoverable from this repo; nothing here reads it.
TRANSCRIPT_PAUSED_SEGMENTS_KEY = "translationRoom:{room_id}:transcript_paused_segments"

#: Values that mean "the key exists but recording is running". The backend writes "1", but a
#: cleared flag left behind as "0"/"false" must not read as paused — that would stop recording
#: a meeting nobody paused, which is the expensive direction to be wrong in.
_NOT_PAUSED_VALUES = frozenset({"", "0", "false", "no", "off"})


def transcript_paused_key(room_id: str) -> str:
    return TRANSCRIPT_PAUSED_KEY.format(room_id=room_id)


def means_paused(raw: bytes | str | None) -> bool:
    """Read the flag's value the way both repositories agree to read it."""
    if raw is None:
        return False
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return text.strip().casefold() not in _NOT_PAUSED_VALUES


class _RedisGet(Protocol):
    async def get(self, key: str) -> bytes | str | None: ...


class _Logger(Protocol):
    def warning(self, event: str, **fields: Any) -> None: ...


async def is_transcript_paused(redis: _RedisGet, room_id: str, logger: _Logger) -> bool:
    """Whether this room's host has recording paused right now.

    Read straight from Redis on every call rather than cached: the flag's whole job is to take
    effect on the next sentence somebody says, and any cache window is a window of speech that
    reaches the record after the host asked for it not to. The read costs one GET on paths that
    are already making Redis calls per segment.

    Never raises. See the module docstring for why the unknown answer is "not paused".
    """
    try:
        raw = await redis.get(transcript_paused_key(room_id))
    except Exception as error:
        logger.warning("transcript_pause_flag_unreadable", room_id=room_id, error=str(error))
        return False
    return means_paused(raw)
