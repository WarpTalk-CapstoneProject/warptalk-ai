"""WT-473 — the transcript's timeline anchor reaching the consumer.

`start_ms` is a DURATION. Without the instant it is measured from, a transcript can be compared
between seats and lined up against nothing else — which is why "seek the recording to this line"
could not be built. The recording's origin is stored on its artifact; this is the other half, and
these tests pin that it actually travels.
"""

from __future__ import annotations

from shared.schemas import STTResultMessage


def _message(**overrides: object) -> STTResultMessage:
    defaults: dict[str, object] = {
        "meeting_id": "room-1",
        "speaker_id": "user-1",
        "text": "clearance granted",
        "language": "en",
        "start_ms": 4_000,
        "end_ms": 6_000,
    }
    defaults.update(overrides)
    return STTResultMessage(**defaults)  # type: ignore[arg-type]


def test_anchor_survives_a_redis_round_trip() -> None:
    original = _message(anchor_ms=1_786_962_240_000)

    restored = STTResultMessage.from_redis(original.to_redis())

    assert restored.anchor_ms == 1_786_962_240_000
    # The duration is unchanged — the anchor is additional context, not a replacement.
    assert restored.start_ms == 4_000


def test_a_message_without_an_anchor_reads_as_zero() -> None:
    """Wire compatibility.

    A worker that predates this field publishes no `anchor_ms`, and its messages must keep being
    consumable — 0 means "not stated", which a consumer treats as "no alignment available" rather
    than as the epoch.
    """
    wire = _message().to_redis()
    del wire["anchor_ms"]

    assert STTResultMessage.from_redis(wire).anchor_ms == 0


def test_the_anchor_is_absent_from_the_wire_only_when_never_set() -> None:
    """Default is 0 and it is still SENT, because a consumer keyed on presence would otherwise
    have to distinguish "old worker" from "new worker, no anchor" — and both mean the same thing."""
    assert _message().to_redis()["anchor_ms"] == "0"


def test_start_ms_plus_anchor_is_a_wall_clock_instant() -> None:
    """The property the whole feature rests on, stated as arithmetic.

    anchor + start_ms is the unix ms at which this segment began. That is what makes a transcript
    line comparable with a recording's own start, and it is why the anchor has to be a wall clock
    rather than another offset.
    """
    anchor = 1_786_962_240_000
    message = _message(anchor_ms=anchor, start_ms=4_000)

    assert message.anchor_ms + message.start_ms == 1_786_962_244_000
