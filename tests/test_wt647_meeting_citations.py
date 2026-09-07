"""WT-647: a meeting cites the same way whichever tool reached it.

THE DEFECT THESE PIN
    One meeting summary produced a CITED answer when the model reached it through
    semantic_search — which registers a source for every chunk, and maps an indexed
    `meeting_summary` onto the "meeting" chip — and an UNCITED one when it reached the same
    words through get_meeting_summary, which registered nothing. Which door the model walked
    through is invisible to the reader, so whether an answer carried provenance was effectively
    random. The transcript tool had the same hole.

WHAT IS NOT TESTED HERE
    Whether the caller may read the meeting at all. That is the S2 gate, covered by
    TestGetMeetingSummaryAuthorization in test_chat_tools.py. A citation is a label on content
    the caller was already handed; the only thing these tests say about authorization is that a
    refusal registers nothing, because a refusal returns nothing to label.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_tools import (
    ToolContext,
    _get_meeting_summary,
    _get_room_detail,
    _get_transcript,
    _list_recent_meetings,
    _source_kind,
)
from ai_assistant_worker.citations import SourceRegistry

MEETING_ID = "019fd60a-e5f3-7342-804a-000000000002"
TRANSCRIPT_ID = "019fd60a-e5f3-7342-804a-0000000000aa"
ROOM = {"id": MEETING_ID, "workspaceId": "ws-1", "title": "Sprint review"}

SEGMENTS = [{"speakerName": "Mai", "originalText": "Ship it Friday.", "sequenceOrder": 1}]


def _response(status_code: int, payload: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _routed(routes: dict[str, MagicMock]) -> AsyncMock:
    """A fake httpx.AsyncClient.get() dispatching on path prefix, longest route first.

    Longest first so `/translation-rooms/history` is not swallowed by the route registered for
    `/translation-rooms/`.
    """
    client = AsyncMock()

    async def _get(url: str, **_kwargs: Any) -> MagicMock:
        for path in sorted(routes, key=len, reverse=True):
            if url.startswith(path):
                return routes[path]
        raise AssertionError(f"Unexpected request: {url}")

    client.get.side_effect = _get
    return client


def _ctx(
    registry: SourceRegistry | None,
    *,
    room: Any = ROOM,
    room_status: int = 200,
    summary: dict[bytes, bytes] | None = None,
    transcript_routes: dict[str, MagicMock] | None = None,
    history: Any = None,
) -> ToolContext:
    redis = MagicMock()
    redis.hgetall = AsyncMock(
        return_value=summary if summary is not None else {b"content": b"We shipped the exporter."}
    )
    return ToolContext(
        workspace_id="ws-1",
        user_id="user-1",
        bearer_token="Bearer test-token",
        workspace_client=AsyncMock(),
        transcript_client=_routed(transcript_routes or {}),
        translation_room_client=_routed(
            {
                "/api/v1/translation-rooms/history": _response(200, {"rooms": history or []}),
                "/api/v1/translation-rooms/": _response(room_status, room),
            }
        ),
        openai_client=None,
        model="gpt-4.1",
        redis=redis,
        citations=registry,
    )


def _segment_routes(items: list[dict[str, Any]]) -> dict[str, MagicMock]:
    return {
        f"/api/v1/transcripts/by-room/{MEETING_ID}": _response(
            200, {"id": TRANSCRIPT_ID, "status": "FINALIZED"}
        ),
        f"/api/v1/transcripts/{TRANSCRIPT_ID}/segments": _response(200, {"items": items}),
    }


class TestMeetingSummaryIsCitable:
    async def test_the_summary_arrives_with_a_marker_the_model_can_cite(self) -> None:
        registry = SourceRegistry()

        result = json.loads(await _get_meeting_summary(_ctx(registry), {"meeting_id": MEETING_ID}))

        assert result["marker"] == "S1"
        assert result["summary"] == "We shipped the exporter."
        source = registry.registered()[0]
        assert (source.kind, source.title, source.ref) == ("meeting", "Sprint review", MEETING_ID)

    def test_both_doors_now_issue_the_same_kind(self) -> None:
        """The vector path already mapped an indexed summary onto "meeting". Agreeing with it is
        the whole of WT-647 — one meeting is one source however it was reached."""
        assert _source_kind("meeting_summary") == "meeting"

    async def test_a_meeting_with_no_name_gets_no_chip_rather_than_an_empty_one(self) -> None:
        registry = SourceRegistry()

        result = json.loads(
            await _get_meeting_summary(
                _ctx(registry, room={"id": MEETING_ID, "workspaceId": "ws-1", "title": ""}),
                {"meeting_id": MEETING_ID},
            )
        )

        # The answer is still worth having. What it does not get is a chip naming nothing.
        assert result["summary"] == "We shipped the exporter."
        assert "marker" not in result
        assert registry.registered() == []

    async def test_a_refused_meeting_registers_nothing(self) -> None:
        """A marker labels content the caller was allowed to receive. Where the S2 gate refuses,
        nothing is returned and nothing may be registered either."""
        registry = SourceRegistry()

        result = json.loads(
            await _get_meeting_summary(
                _ctx(registry, room={"id": MEETING_ID, "workspaceId": "ws-2", "title": "Theirs"}),
                {"meeting_id": MEETING_ID},
            )
        )

        assert result == {"error": "No meeting found with that id."}
        assert registry.registered() == []

    async def test_a_meeting_with_no_summary_yet_registers_nothing(self) -> None:
        registry = SourceRegistry()

        result = json.loads(
            await _get_meeting_summary(_ctx(registry, summary={}), {"meeting_id": MEETING_ID})
        )

        assert result["summary"] is None
        assert registry.registered() == []

    async def test_a_context_with_no_registry_still_answers(self) -> None:
        result = json.loads(await _get_meeting_summary(_ctx(None), {"meeting_id": MEETING_ID}))

        assert result["summary"] == "We shipped the exporter."
        assert "marker" not in result


class TestTranscriptIsCitable:
    async def test_the_transcript_arrives_with_a_marker(self) -> None:
        registry = SourceRegistry()
        ctx = _ctx(registry, transcript_routes=_segment_routes(SEGMENTS))

        result = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert result["marker"] == "S1"
        assert result["segments"][0]["text"] == "Ship it Friday."
        source = registry.registered()[0]
        assert (source.kind, source.title, source.ref) == (
            "transcript",
            "Sprint review",
            MEETING_ID,
        )

    async def test_an_empty_transcript_is_not_evidence_and_is_not_named(self) -> None:
        registry = SourceRegistry()
        ctx = _ctx(registry, transcript_routes=_segment_routes([]))

        result = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert result["segments"] == []
        assert "marker" not in result
        assert registry.registered() == []
        # And it did not spend a request learning a title for a chip nobody is offered.
        ctx.translation_room_client.get.assert_not_awaited()

    async def test_a_transcript_whose_meeting_cannot_be_named_still_answers(self) -> None:
        """Losing the transcript because its footnote could not be labelled would be trading the
        answer for the citation."""
        registry = SourceRegistry()
        ctx = _ctx(registry, room_status=503, transcript_routes=_segment_routes(SEGMENTS))

        result = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert result["segments"][0]["text"] == "Ship it Friday."
        assert "marker" not in result
        assert registry.registered() == []

    async def test_a_transcript_and_its_summary_are_two_different_sources(self) -> None:
        """Same meeting, different evidence: what was SAID and what was WRITTEN ABOUT what was
        said. A reader checking a quote needs to know which of the two the answer used."""
        registry = SourceRegistry()
        ctx = _ctx(registry, transcript_routes=_segment_routes(SEGMENTS))

        await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})
        await _get_transcript(ctx, {"meeting_id": MEETING_ID})

        assert [(s.kind, s.marker) for s in registry.registered()] == [
            ("meeting", "S1"),
            ("transcript", "S2"),
        ]


class TestRoomDetailIsCitable:
    async def test_room_detail_cites_the_meeting_it_describes(self) -> None:
        registry = SourceRegistry()

        result = json.loads(await _get_room_detail(_ctx(registry), {"room_id": MEETING_ID}))

        assert result["marker"] == "S1"
        source = registry.registered()[0]
        assert (source.kind, source.title, source.ref) == ("meeting", "Sprint review", MEETING_ID)

    async def test_one_meeting_read_through_two_tools_is_one_chip(self) -> None:
        """Identity is (kind, title, ref), so the same meeting reached twice dedupes. Two chips
        differing only in which tool fetched them is a distinction no reader can act on."""
        registry = SourceRegistry()
        ctx = _ctx(registry)

        summary = json.loads(await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID}))
        detail = json.loads(await _get_room_detail(ctx, {"room_id": MEETING_ID}))

        assert summary["marker"] == detail["marker"] == "S1"
        assert len(registry.registered()) == 1

    async def test_the_model_supplied_id_does_not_split_the_chip(self) -> None:
        """The ROOM's own id is registered, not the argument, so an uppercase UUID typed by a
        user does not produce a second source for the same meeting."""
        registry = SourceRegistry()
        ctx = _ctx(registry)

        await _get_room_detail(ctx, {"room_id": MEETING_ID.upper()})
        await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})

        assert len(registry.registered()) == 1


class TestListingMeetingsDoesNotCite:
    async def test_a_listing_of_candidates_registers_nothing(self) -> None:
        """DELIBERATE, and pinned so a later reader does not "fix" it by symmetry with the other
        meeting tools. A listing is what the model reads to pick an id, and the tool it then
        calls cites that meeting under the same identity — so the reader's chip is unchanged.
        Citing here would only change the answers that stop at the listing, letting "you had
        five meetings" claim five sources for a statement about the set rather than any one
        of them, and spending five of the eight slots an answer has for real evidence.
        """
        registry = SourceRegistry()
        history = [{"room": {"id": MEETING_ID, "title": "Sprint review", "status": "ENDED"}}]

        result = json.loads(await _list_recent_meetings(_ctx(registry, history=history), {}))

        assert result[0]["title"] == "Sprint review"
        assert registry.registered() == []
