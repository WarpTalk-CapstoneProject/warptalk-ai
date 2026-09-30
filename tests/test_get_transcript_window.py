"""WT-620: get_transcript reads the part of the meeting the question is about.

THE DEFECT THESE PIN
    get_transcript read `skip=0, take=200` and stopped. The endpoint is ordered ascending, so in
    any meeting past 200 segments the latest speech — exactly what somebody in a live call asks
    about ("what did they just propose?") — was never returned, and the model answered from the
    opening minutes as if they were the whole meeting.

THE FAKE
    A stand-in for TranscriptService that honours `skip`/`take` the way
    TranscriptQueryService.GetSegmentsAsync does: ascending by sequenceOrder, `totalCount` counted
    over the whole transcript. The citation behaviour (WT-647) is pinned separately in
    test_wt647_meeting_citations.py and only spot-checked here.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_tools import (
    EXTERNAL_BRIDGE_SPEAKER_LABEL,
    TOOLS,
    TRANSCRIPT_SEGMENT_LIMIT,
    ToolContext,
    _get_transcript,
)
from ai_assistant_worker.citations import SourceRegistry
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID

MEETING_ID = "019fd60a-e5f3-7342-804a-000000000002"
TRANSCRIPT_ID = "019fd60a-e5f3-7342-804a-0000000000aa"
LIMIT = TRANSCRIPT_SEGMENT_LIMIT


def _response(status_code: int, payload: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _segments(sequence_numbers: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "speakerName": "Mai",
            "originalLanguage": "en",
            "originalText": f"line {n}",
            "startTimeMs": n * 1000,
            "sequenceOrder": n,
        }
        for n in sequence_numbers
    ]


class FakeTranscriptService:
    """GET /api/v1/transcripts/by-room/{room} and /api/v1/transcripts/{id}/segments."""

    def __init__(
        self,
        segments: list[dict[str, Any]],
        *,
        lookup_status: int = 200,
        segments_status: int = 200,
    ) -> None:
        self.segments = sorted(segments, key=lambda s: int(s["sequenceOrder"]))
        self.lookup_status = lookup_status
        self.segments_status = segments_status
        self.reads: list[tuple[int, int]] = []

    async def get(self, url: str, **kwargs: Any) -> MagicMock:
        if url == f"/api/v1/transcripts/by-room/{MEETING_ID}":
            if self.lookup_status != 200:
                return _response(self.lookup_status, {"message": "not found"})
            return _response(200, {"id": TRANSCRIPT_ID, "status": "ACTIVE"})
        if url == f"/api/v1/transcripts/{TRANSCRIPT_ID}/segments":
            params = kwargs.get("params") or {}
            skip, take = int(params["skip"]), int(params["take"])
            self.reads.append((skip, take))
            if self.segments_status != 200:
                return _response(self.segments_status, {"message": "boom"})
            return _response(
                200,
                {"totalCount": len(self.segments), "items": self.segments[skip : skip + take]},
            )
        raise AssertionError(f"Unexpected request: {url}")


def _ctx(service: FakeTranscriptService, registry: SourceRegistry | None = None) -> ToolContext:
    room_client = AsyncMock()
    room_client.get.return_value = _response(200, {"id": MEETING_ID, "title": "Client call"})
    transcript_client = AsyncMock()
    transcript_client.get.side_effect = service.get
    return ToolContext(
        workspace_id="ws-1",
        user_id="user-1",
        bearer_token="Bearer test-token",
        workspace_client=AsyncMock(),
        transcript_client=transcript_client,
        translation_room_client=room_client,
        openai_client=None,
        model="gpt-4.1",
        redis=MagicMock(),
        citations=registry,
    )


async def _call(service: FakeTranscriptService, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        await _get_transcript(_ctx(service), {"meeting_id": MEETING_ID, **arguments})
    )
    return result


def _numbers(result: dict[str, Any]) -> list[int]:
    return [int(s["text"].removeprefix("line ")) for s in result["segments"]]


class TestShortMeeting:
    async def test_a_meeting_that_fits_in_one_window_is_read_whole_in_one_request(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 51))))

        result = await _call(service)

        assert _numbers(result) == list(range(1, 51))
        assert result["totalSegments"] == 50
        assert (result["returnedFrom"], result["returnedTo"]) == (1, 50)
        assert result["omittedEarlier"] is False
        assert result["omittedLater"] is False
        assert "note" not in result
        assert service.reads == [(0, LIMIT)]

    async def test_exactly_one_window_is_still_one_request(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, LIMIT + 1))))

        result = await _call(service)

        assert result["totalSegments"] == LIMIT
        assert result["omittedEarlier"] is False
        assert service.reads == [(0, LIMIT)]


class TestLongMeetingDefaultsToTheLatest:
    async def test_the_latest_window_is_returned_not_the_opening_one(self) -> None:
        """The bug: a live question about the last few minutes answered from the first 200."""
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service)

        assert _numbers(result) == list(range(301, 501))
        assert result["totalSegments"] == 500
        assert (result["returnedFrom"], result["returnedTo"]) == (301, 500)
        assert result["omittedEarlier"] is True
        assert result["omittedLater"] is False
        # The first page is the probe for totalCount; the second is the tail.
        assert service.reads == [(0, LIMIT), (300, LIMIT)]

    async def test_the_result_tells_the_model_how_to_read_further_back(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service)

        assert "before_sequence=301" in result["note"]

    async def test_explicit_latest_is_the_same_as_the_default(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        assert _numbers(await _call(service, range="latest")) == list(range(301, 501))


class TestRangeBeginning:
    async def test_beginning_reads_the_opening_window_in_one_request(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service, range="beginning")

        assert _numbers(result) == list(range(1, LIMIT + 1))
        assert (result["returnedFrom"], result["returnedTo"]) == (1, LIMIT)
        assert result["omittedEarlier"] is False
        assert result["omittedLater"] is True
        assert "most recent part" in result["note"]
        assert service.reads == [(0, LIMIT)]

    async def test_an_unknown_range_is_refused_before_any_request(self) -> None:
        service = FakeTranscriptService(_segments([1, 2, 3]))

        result = await _call(service, range="middle")

        assert "range must be one of" in result["error"]
        assert service.reads == []


class TestBeforeSequencePaging:
    async def test_pages_backwards_from_the_previous_window(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service, before_sequence=301)

        assert _numbers(result) == list(range(101, 301))
        assert (result["returnedFrom"], result["returnedTo"]) == (101, 300)
        assert result["omittedEarlier"] is True
        assert "before_sequence=101" in result["note"]
        # Dense numbering makes the skip arithmetic exact: one read, no probe.
        assert service.reads == [(100, LIMIT)]

    async def test_the_last_page_back_reaches_the_start(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service, before_sequence=101)

        assert _numbers(result) == list(range(1, 101))
        assert result["omittedEarlier"] is False
        assert "note" not in result

    async def test_nothing_precedes_the_first_segment(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service, before_sequence=1)

        assert result["segments"] == []
        assert result["returnedFrom"] is None
        assert result["omittedEarlier"] is False
        assert "marker" not in result

    async def test_holes_in_the_numbering_still_give_a_full_exact_window(self) -> None:
        """A failed save burns a sequence number, so index arithmetic overshoots; the window is
        re-aimed from what the overshoot reveals rather than coming back short or wrong."""
        numbers = [n for n in range(1, 541) if n % 10 != 0]  # every tenth number burned
        service = FakeTranscriptService(_segments(numbers))

        result = await _call(service, before_sequence=400)

        expected = [n for n in numbers if n < 400][-LIMIT:]
        assert _numbers(result) == expected
        assert result["returnedTo"] == 399
        assert result["omittedEarlier"] is True
        assert len(service.reads) == 2

    async def test_a_position_past_the_end_reads_the_tail(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 301))))

        result = await _call(service, before_sequence=10_000)

        assert _numbers(result) == list(range(101, 301))
        assert result["totalSegments"] == 300

    async def test_before_sequence_outranks_range(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = await _call(service, range="beginning", before_sequence=301)

        assert _numbers(result) == list(range(101, 301))

    async def test_a_numeric_string_is_accepted(self) -> None:
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        assert _numbers(await _call(service, before_sequence="301")) == list(range(101, 301))

    async def test_invalid_positions_are_refused_before_any_request(self) -> None:
        for bad in (0, -5, "soon", True, 1.5j):
            service = FakeTranscriptService(_segments([1, 2, 3]))

            result = await _call(service, before_sequence=bad)

            assert result["error"] == "before_sequence must be a positive integer.", bad
            assert service.reads == []


class TestMissingAndEmptyTranscripts:
    async def test_a_meeting_with_no_transcript_says_so(self) -> None:
        service = FakeTranscriptService([], lookup_status=404)

        result = await _call(service)

        assert result == {"segments": [], "note": "No transcript exists for this meeting yet."}
        assert service.reads == []

    async def test_an_empty_transcript_returns_an_empty_window_and_no_citation(self) -> None:
        registry = SourceRegistry()
        service = FakeTranscriptService([])
        ctx = _ctx(service, registry)

        result = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert result["segments"] == []
        assert result["totalSegments"] == 0
        assert (result["returnedFrom"], result["returnedTo"]) == (None, None)
        assert result["omittedEarlier"] is False
        assert "marker" not in result
        assert registry.registered() == []

    async def test_a_failing_segments_endpoint_is_reported_not_raised(self) -> None:
        service = FakeTranscriptService(_segments([1]), segments_status=500)

        result = await _call(service)

        assert result == {"error": "Could not look up the transcript segments right now."}


class TestCitationIsUnchanged:
    async def test_a_windowed_transcript_is_still_one_cited_source(self) -> None:
        registry = SourceRegistry()
        service = FakeTranscriptService(_segments(list(range(1, 501))))

        result = json.loads(
            await _get_transcript(_ctx(service, registry), {"meeting_id": MEETING_ID})
        )

        assert result["marker"] == "S1"
        source = registry.registered()[0]
        assert (source.kind, source.title, source.ref) == ("transcript", "Client call", MEETING_ID)


class TestBridgeStandInIsNamed:
    async def test_the_far_side_of_a_bridged_call_is_other_side_not_a_uuid(self) -> None:
        """The stand-in seat is not a user, so TranscriptService stores its uuid as the name."""
        far_side = {
            "speakerParticipantId": EXTERNAL_BRIDGE_SPEAKER_ID,
            "speakerName": EXTERNAL_BRIDGE_SPEAKER_ID,
            "originalLanguage": "en",
            "originalText": "line 2",
            "startTimeMs": 2000,
            "sequenceOrder": 2,
        }
        service = FakeTranscriptService([*_segments([1]), far_side])

        result = await _call(service)

        assert [s["speaker"] for s in result["segments"]] == [
            "Mai",
            EXTERNAL_BRIDGE_SPEAKER_LABEL,
        ]


class TestToolSchema:
    def test_the_schema_offers_range_and_before_sequence(self) -> None:
        tool = next(t for t in TOOLS if t.name == "get_transcript")
        properties = tool.parameters["properties"]

        assert properties["range"]["enum"] == ["latest", "beginning"]
        assert properties["before_sequence"]["type"] == "integer"
        assert tool.parameters["required"] == ["meeting_id"]
        assert "MOST RECENT" in tool.description
        assert "before_sequence" in tool.description
