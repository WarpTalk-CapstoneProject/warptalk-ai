"""The meeting-end sentinel must not be translated, dubbed, or billed.

WHAT PRODUCTION LOOKED LIKE

    translate:results:dead-letter   5 entries
    tts:results:dead-letter         5 entries

    Every one of them, on both streams:

        consumer_group    = billing-translation-workers | billing-tts-workers
        delivery_attempts = 5
        speaker_id        = "system"
        original_text     = "__MEETING_END__"
        translated_text   = "Meeting end" / "End of meeting." / "Kết thúc cuộc họp"

    One sentinel per meeting, five meetings, ten dead letters — and a
    `WarpTalkDeadLetterPresent` alert email that had been arriving for days.

THE CHAIN

    MeetingRoomService.EndMeetingAsync publishes __MEETING_END__ to `stt:results` so
    ai_assistant_worker knows to write the summary. But `stt:results` fans out, and the other
    two consumers had no idea the segment was synthetic:

        translation_worker  translated it — a real LLM call per target language. The
                            `translated_text` values above are the proof: those words exist
                            only because something paid to produce them.
        tts_worker          synthesized those words — a real Cartesia render, published onto
                            the interpreter track listeners subscribe to.
        billing_worker      tried to settle a charge with user_id="system". uuid.UUID("system")
                            raises, identically every time, so five deliveries and out.

    warptalk-web already filtered this marker out of the transcript panel
    (`isTranscriptControlMarker`), which means the leak was known at the LAST stage of the
    pipeline and never fixed at the first.

WHAT THESE TESTS PIN
    Each stage refuses it for its own reason, so no single revert can quietly re-open the chain.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from shared.control_markers import (
    MEETING_END_MARKER,
    is_control_marker,
    is_external_bridge_speaker,
    is_system_speaker,
)


class TestTheMarkerIsRecognised:
    def test_the_sentinel_the_backend_actually_sends(self) -> None:
        # Byte-for-byte what MeetingRoomService.EndMeetingAsync puts on the wire.
        assert is_control_marker(MEETING_END_MARKER)
        assert is_system_speaker("system")

    def test_matching_tolerates_what_the_assistant_always_tolerated(self) -> None:
        # ai_assistant_worker compared `text.strip().upper()`, so the shared predicate has to
        # accept the same inputs or the two would disagree about the same message.
        assert is_control_marker("  __meeting_end__  ")
        assert is_system_speaker("  SYSTEM ")

    def test_real_speech_is_not_a_marker(self) -> None:
        assert not is_control_marker("Kết thúc cuộc họp")
        assert not is_control_marker("meeting end")
        assert not is_control_marker("")
        assert not is_control_marker(None)
        assert not is_system_speaker(str(uuid.uuid4()))
        assert not is_system_speaker(None)


class TestBillingRefusesRatherThanDeadLetters:
    """The guard is a REFUSAL. A settlement that cannot name a user is not a charge to retry."""

    @staticmethod
    def _worker() -> object:
        from billing_worker.worker import BillingSettlementWorker

        worker = BillingSettlementWorker.__new__(BillingSettlementWorker)
        worker.logger = MagicMock()
        return worker

    def test_the_system_speaker_is_skipped(self) -> None:
        worker = self._worker()
        assert worker._is_unbillable("system", "room-1") is True  # type: ignore[attr-defined]

    def test_any_other_unparseable_speaker_is_skipped_and_warned_about(self) -> None:
        worker = self._worker()
        assert worker._is_unbillable("not-a-uuid", "room-1") is True  # type: ignore[attr-defined]
        # Louder than the "system" case on purpose: this one means something upstream is
        # emitting a shape nobody designed.
        worker.logger.warning.assert_called_once()  # type: ignore[attr-defined]

    def test_a_real_participant_is_still_billed(self) -> None:
        worker = self._worker()
        assert worker._is_unbillable(str(uuid.uuid4()), "room-1") is False  # type: ignore[attr-defined]

    def test_the_production_payload_would_no_longer_dead_letter(self) -> None:
        """The exact speaker id from every dead-lettered entry."""
        worker = self._worker()
        assert worker._is_unbillable("system", "01a00547-367f-7deb-88c0-c097396e3a62") is True  # type: ignore[attr-defined]
        # And prove why it used to: this is the call record_usage_and_charge makes.
        with pytest.raises(ValueError):
            uuid.UUID("system")


@pytest.mark.asyncio
async def test_translation_declines_the_marker_before_spending_anything() -> None:
    """The stage that stops the whole chain.

    Nothing downstream can charge for, synthesize, or dead-letter a translation that was never
    produced — so this one guard closes the LLM spend, the Cartesia spend, the interpreter-track
    leak and both dead letters at once.
    """
    from shared.schemas import STTResultMessage
    from translation_worker.worker import TranslationWorker

    worker = TranslationWorker.__new__(TranslationWorker)
    worker.logger = MagicMock()
    worker._paused_rooms = set()

    async def _never(_meeting_id: str) -> bool:
        raise AssertionError(
            "The control marker reached the translation-active gate, which means it also "
            "reached the translator. That gate is per-room; whether the platform's own "
            "sentinel gets translated is not a per-room setting."
        )

    worker._translation_active_for = _never  # type: ignore[method-assign]

    message = STTResultMessage(
        segment_id=str(uuid.uuid4()),
        meeting_id="01a00547-367f-7deb-88c0-c097396e3a62",
        speaker_id="system",
        text=MEETING_END_MARKER,
        language="system",
        confidence=1.0,
        start_ms=0,
        end_ms=0,
    )

    # Returns quietly rather than raising through _never.
    await worker.process(b"1-0", message.to_redis())  # type: ignore[arg-type]


# --- WT-525: the external-bridge stand-in ------------------------------------------------------
#
# The opposite question to the sentinel above. This identity IS speech and must go all the way
# through the pipeline; what it must NOT get is processing that assumes a person at a microphone.


def test_bridge_speaker_is_recognised_regardless_of_case():
    # Backend hands identities out as .NET Guid.ToString(), which is lowercase, but a token or a
    # config could carry the uppercase form. Failing to recognise it would not error — it would
    # silently re-enable near-field gating on a conference feed, whose symptom is "translation
    # works sometimes" with nothing in the logs.
    assert is_external_bridge_speaker("00000000-0000-0000-0000-00000000b21d")
    assert is_external_bridge_speaker("00000000-0000-0000-0000-00000000B21D")
    assert is_external_bridge_speaker("  00000000-0000-0000-0000-00000000b21d  ")


def test_a_real_participant_is_not_the_bridge_speaker():
    # The guard exists to turn OFF a quality filter. Matching too broadly would disable it for
    # real microphones, which is the hallucination risk the gate was built to prevent.
    assert not is_external_bridge_speaker("550e8400-e29b-41d4-a716-446655440000")
    assert not is_external_bridge_speaker("system")
    assert not is_external_bridge_speaker("")
    assert not is_external_bridge_speaker(None)
