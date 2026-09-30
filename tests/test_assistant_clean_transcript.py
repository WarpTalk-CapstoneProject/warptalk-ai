"""The summariser reads the CLEAN wording and cites the RAW moments. WT-716.

Two paths reach a model with a meeting's words, and both had to learn the same rule:

    live      `AIAssistantWorker.process` accumulates `stt:results` and summarises on the
              end-of-meeting marker. It now stores `STTResultMessage.display_text`.
    saved     `SummaryTemplateWorker._load_transcript` fetches the stored segments for a
              rewrite. It now prefers the row's `cleanText`.

WHAT THESE TESTS ARE REALLY GUARDING
    Not "the clean text is used" — that is one assertion and the easy half. The half that can
    break a shipped feature silently is the other one: a citation is an `atMs` offset measured
    from the meeting's first moment, `summary_grounding` checks membership in that set with
    zero tolerance (WT-663), and the meeting page resolves the number against the saved
    transcript, which still holds every raw segment. So the words may change and the numbers
    may not — including when the first thing anybody said was "Ummm" and the line the origin
    used to be taken from is now dropped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.summary_template_worker import SummaryTemplateWorker, _segment_text
from ai_assistant_worker.summary_templates import transcript_offsets
from ai_assistant_worker.transcript_buffer import buffer_key, decode_segments
from ai_assistant_worker.worker import AIAssistantWorker
from shared.config import WorkerSettings
from shared.control_markers import MEETING_END_MARKER
from shared.schemas import STTResultMessage, SummaryRequestMessage

MEETING = "019f6a39-a32c-7745-886e-1fe622c1f747"
ROOM = MEETING


class _Redis:
    """Only what `AIAssistantWorker` asks of Redis. None from `get` = recording is not paused."""

    def __init__(self) -> None:
        self.buffered: dict[str, list[str]] = {}
        self.deleted: list[str] = []
        self.stored: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return None

    async def rpush_capped(self, key: str, value: str, **_: Any) -> None:
        self.buffered.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int = 0, stop: int = -1) -> list[str]:
        return list(self.buffered.get(key, []))

    async def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.buffered.pop(key, None)

    async def hgetall(self, key: str) -> dict[bytes, bytes]:
        return {}

    async def hset(self, key: str, field: str, value: str) -> None:
        self.stored[f"{key}#{field}"] = value


def _worker() -> tuple[AIAssistantWorker, _Redis]:
    worker = AIAssistantWorker(settings=WorkerSettings())
    redis = _Redis()
    worker.redis = redis  # type: ignore[assignment]
    worker._generate_summary = AsyncMock()  # type: ignore[method-assign]
    return worker, redis


def _summarising_worker() -> tuple[AIAssistantWorker, list[str]]:
    """A worker that really runs `_generate_summary`, capturing the transcript it builds."""
    worker = AIAssistantWorker(settings=WorkerSettings())
    worker.redis = _Redis()  # type: ignore[assignment]
    worker.publish = AsyncMock()  # type: ignore[method-assign]
    seen: list[str] = []

    async def _summarize(transcript_text: str, **_: Any) -> str:
        seen.append(transcript_text)
        return "a summary"

    async def _structured(transcript_text: str, **_: Any) -> dict[str, Any]:
        return {"summary": "ok"}

    async def _actions(transcript_text: str, **_: Any) -> str:
        return "[]"

    worker._require_assistant = lambda: MagicMock(  # type: ignore[method-assign]
        summarize=_summarize,
        extract_action_items=_actions,
        generate_structured_summary=_structured,
    )
    return worker, seen


def _stt(
    text: str,
    *,
    clean_text: str | None = None,
    timestamp_ms: int = 1_000,
    segment_id: str = "segment-1",
) -> dict[bytes, bytes]:
    payload = STTResultMessage(
        segment_id=segment_id,
        meeting_id=MEETING,
        speaker_id="speaker-1",
        text=text,
        language="en",
        confidence=-0.1,
        timestamp_ms=timestamp_ms,
        clean_text=clean_text,
    ).to_redis()
    return {key.encode(): value.encode() for key, value in payload.items()}


# --- the live path ------------------------------------------------------------------------


class TestTheLiveStream:
    async def test_the_clean_wording_is_what_the_model_will_read(self) -> None:
        worker, redis = _worker()

        await worker.process(
            b"1-1",
            _stt("um so we uh we need to ship", clean_text="So we need to ship."),
        )

        assert worker._transcripts[MEETING] == [("speaker-1", "So we need to ship.", 1_000)]
        # And the copy a restart recovers from says the same thing, or the fix survives only as
        # long as the process does.
        assert decode_segments(redis.buffered[buffer_key(MEETING)])[0][1] == "So we need to ship."

    async def test_a_segment_with_no_clean_version_keeps_its_raw_words(self) -> None:
        """An older producer, or a language the prepass does not handle: nothing changes."""
        worker, _ = _worker()

        await worker.process(b"1-1", _stt("我们下周发布", clean_text=None))

        assert worker._transcripts[MEETING][0][1] == "我们下周发布"

    async def test_a_filler_only_line_never_reaches_the_summary(self) -> None:
        """`clean_text == ""` is not "no clean version": it is "there was nothing in it"."""
        worker, redis = _worker()

        await worker.process(b"1-1", _stt("Ummm", clean_text=""))

        assert MEETING not in worker._transcripts
        # Not buffered either: the buffer is capped, and a room full of "Ummm" would spend that
        # cap on lines the model will never be shown.
        assert redis.buffered == {}

    async def test_only_the_earliest_filler_only_moment_is_remembered(self) -> None:
        worker, _ = _worker()

        await worker.process(b"1-1", _stt("Ummm", clean_text="", timestamp_ms=5_000))
        await worker.process(b"1-2", _stt("Uh", clean_text="", timestamp_ms=2_000))

        assert worker._filler_only_ms[MEETING] == 2_000

    async def test_a_finished_meeting_forgets_its_filler_moments(self) -> None:
        worker, _ = _worker()
        await worker.process(b"1-1", _stt("Ummm", clean_text=""))

        await worker._forget_meeting(MEETING)

        assert MEETING not in worker._filler_only_ms

    async def test_the_end_of_meeting_marker_is_not_a_filler_only_line(self) -> None:
        """A marker whose clean text is empty must still end the meeting.

        The filler-only gate sits on the same stream `__MEETING_END__` travels down, and a gate
        in front of the marker means no meeting is ever summarised — the failure `process`
        already orders its pause gate to avoid.
        """
        worker, _ = _worker()

        await worker.process(b"1-1", _stt(MEETING_END_MARKER, clean_text=""))

        worker._generate_summary.assert_awaited_once_with(MEETING)  # type: ignore[attr-defined]


class TestTheCitedMoments:
    async def test_dropping_an_opening_filler_does_not_move_a_single_citation(self) -> None:
        """THE ANCHOR TEST, through the real `_generate_summary`.

        "Ummm" at 1s, a real line at 9s. The summary must offer t=8000 — the distance from when
        the meeting actually started — and not t=0, which would scroll every citation in the
        meeting back to the "Ummm" the saved transcript still holds.
        """
        worker, seen = _summarising_worker()
        await worker.process(b"1-1", _stt("Ummm", clean_text="", timestamp_ms=1_000))
        await worker.process(
            b"1-2",
            _stt("uh right, the budget", clean_text="Right, the budget.", timestamp_ms=9_000),
        )

        await worker._generate_summary(MEETING)

        assert seen == ["[t=8000] [Speaker 1] Right, the budget."]
        assert transcript_offsets(seen[0]) == {8_000}

    async def test_a_meeting_that_opens_with_speech_is_unchanged(self) -> None:
        """The ordinary case: no filler-only line, so the origin is the first line, as before."""
        worker, seen = _summarising_worker()
        await worker.process(b"1-1", _stt("we ship", clean_text="We ship.", timestamp_ms=4_000))
        await worker.process(b"1-2", _stt("on Friday", clean_text="On Friday.", timestamp_ms=9_000))

        await worker._generate_summary(MEETING)

        assert transcript_offsets(seen[0]) == {0, 5_000}

    async def test_a_meeting_of_nothing_but_fillers_is_not_summarised(self) -> None:
        worker, seen = _summarising_worker()
        await worker.process(b"1-1", _stt("Ummm", clean_text=""))

        await worker._generate_summary(MEETING)

        assert seen == []
        worker.publish.assert_not_awaited()  # type: ignore[attr-defined]


# --- the saved path -----------------------------------------------------------------------


class TestTheSavedTranscript:
    def test_a_cleaned_row_is_read_clean(self) -> None:
        row = {"originalText": "um so we ship", "cleanText": "So we ship."}
        assert _segment_text(row) == "So we ship."

    def test_an_uncleaned_row_is_read_raw(self) -> None:
        """NULL means NOT CLEANED — an older row, or one a human has corrected since."""
        assert _segment_text({"originalText": "so we ship", "cleanText": None}) == "so we ship"
        assert _segment_text({"originalText": "so we ship"}) == "so we ship"

    def test_a_filler_only_row_reads_as_nothing(self) -> None:
        assert _segment_text({"originalText": "Ummm", "cleanText": ""}) == ""

    async def test_the_fetched_transcript_is_clean_and_still_cites_stored_moments(self) -> None:
        worker = SummaryTemplateWorker(transcript_base_url="http://transcript")
        lookup = MagicMock(status_code=200, json=lambda: {"id": "tr-1"})
        segments = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [
                    {
                        "startTimeMs": 0,
                        "speakerName": "Tu",
                        "originalText": "um so we uh we need to ship",
                        "cleanText": "So we need to ship.",
                    },
                    # Filler-only: hidden from the Clean view, and from the model.
                    {
                        "startTimeMs": 4_000,
                        "speakerName": "Nhi",
                        "originalText": "Ummm",
                        "cleanText": "",
                    },
                    # Never cleaned: read exactly as it was before WT-716.
                    {
                        "startTimeMs": 90_210,
                        "speakerName": "Nhi",
                        "originalText": "cap it",
                        "cleanText": None,
                    },
                ]
            },
        )
        lookup.raise_for_status = MagicMock()
        segments.raise_for_status = MagicMock()
        client = MagicMock()
        client.get = AsyncMock(side_effect=[lookup, segments])
        worker._transcript_client = client

        transcript = await worker._load_transcript(
            SummaryRequestMessage(request_id="r", room_id=ROOM, workspace_id="w")
        )

        assert transcript == "[t=0] [Tu] So we need to ship.\n[t=90210] [Nhi] cap it"
        # The moments are the ones the rows were STORED with — read, never recomputed from the
        # text — so a summary of the clean wording still scrolls to the right segment. The
        # filler-only line's moment is gone from the set, which is correct: the model was never
        # shown that line, so it has nothing to cite there.
        assert transcript_offsets(transcript) == {0, 90_210}

    async def test_a_supplied_transcript_is_still_used_verbatim(self) -> None:
        """ArtifactsFinalizer sends raw, pre-rendered lines. There is no clean version to use.

        Pinned because the tempting "fix" — fetching the segments again for their clean column
        — would give a background finalization with no requester and no token a privileged read
        of any meeting's transcript.
        """
        worker = SummaryTemplateWorker(transcript_base_url="http://transcript")
        client = MagicMock()
        client.get = AsyncMock()
        worker._transcript_client = client

        transcript = await worker._load_transcript(
            SummaryRequestMessage(
                request_id="r",
                room_id=ROOM,
                workspace_id="w",
                transcript_text="[t=0] [Tu] um so we ship",
            )
        )

        assert transcript == "[t=0] [Tu] um so we ship"
        client.get.assert_not_awaited()
