"""Messages that travel down the pipeline without being anybody's speech.

THE ONE THAT EXISTS
    `MeetingRoomService.EndMeetingAsync` publishes a synthetic STT result to `stt:results` to
    tell `ai_assistant_worker` a meeting is over:

        segment_id = <new guid>   speaker_id = "system"
        language   = "system"     text       = "__MEETING_END__"

    That is a reasonable trigger — it reuses a stream both sides already speak — but
    `stt:results` FANS OUT, and only the assistant ever knew the sentinel was not speech.

WHAT THE OTHER CONSUMERS DID WITH IT
    - `translation_worker` translated it. Really translated it: the dead-lettered payloads in
      production contain `"Meeting end"`, `"End of meeting."` and `"Kết thúc cuộc họp"`, one
      paid LLM call per target language per meeting.
    - `tts_worker` then synthesized those, so every meeting ended with a paid Cartesia render
      of the words "Meeting end" — and published it on the interpreter track listeners are
      subscribed to.
    - `billing_worker` then tried to settle the charge with `user_id="system"`, which is not a
      UUID. It failed identically on all five deliveries and dead-lettered, twice per meeting,
      which is the `WarpTalkDeadLetterPresent` alert that has been arriving since.

    warptalk-web already had `isTranscriptControlMarker` to keep the sentinel out of the
    transcript panel — so the marker was known to leak all the way to the UI, and was filtered
    at the last possible moment instead of the first.

THE RULE
    A control marker is addressed to ONE worker. Every other consumer must recognise it and
    decline it — not translate it, not synthesize it, not bill it. Recognising it in one place
    is what stops the next consumer added to `stt:results` from repeating this.
"""

from __future__ import annotations

# The value MeetingRoomService.EndMeetingAsync publishes. A wire constant: changing it means
# changing that publisher in the same release.
MEETING_END_MARKER = "__MEETING_END__"

# The speaker id it carries. Not a user, and — the actual production failure — not a UUID.
SYSTEM_SPEAKER_ID = "system"

_MARKERS = frozenset({MEETING_END_MARKER})


def is_control_marker(text: str | None) -> bool:
    """True when this text is a pipeline control message rather than something somebody said.

    Case-insensitive and whitespace-tolerant, matching the comparison `ai_assistant_worker`
    has always used (`text.strip().upper()`), so the two cannot disagree about the same
    message.
    """
    if not text:
        return False
    return text.strip().upper() in _MARKERS


def is_system_speaker(speaker_id: str | None) -> bool:
    """True when this 'speaker' is the platform rather than a participant.

    Checked separately from the text because it is the field that actually broke billing, and
    because it stays meaningful for a control message whose text is not one of the markers
    above — a synthetic event nobody has thought of yet is still not a person to charge.
    """
    return (speaker_id or "").strip().casefold() == SYSTEM_SPEAKER_ID


# ---------------------------------------------------------------------------------------------
# WT-525 — the external-bridge stand-in
# ---------------------------------------------------------------------------------------------

#: LiveKit participant identity of the seat that represents everyone on the far side of an
#: external call (Google Meet, Zoom, Teams) in an EXTERNAL_BRIDGE room.
#:
#: Not a control marker in the sentinel sense — this IS a real speaker and must be transcribed,
#: translated and dubbed like any other. It lives here because it is the same KIND of fact as the
#: sentinel above: a string three repositories must agree on exactly, that nothing would fail to
#: compile over if one of them changed it.
#:
#: Written by warptalk-backend (translation-room seeds the seat, meeting mints its token — see
#: WarpTalk.Shared.ExternalBridgeConstants). Read here.
EXTERNAL_BRIDGE_SPEAKER_ID = "00000000-0000-0000-0000-00000000b21d"


def is_external_bridge_speaker(speaker_id: str | None) -> bool:
    """True for the stand-in seat carrying the far side of an external call.

    Callers use this to turn OFF processing that assumes a person at a microphone. The track is
    a line-level conference feed mixing several people in a different room — every near-mic
    assumption the pipeline makes about it is wrong.
    """
    if not speaker_id:
        return False
    return speaker_id.strip().lower() == EXTERNAL_BRIDGE_SPEAKER_ID
