"""WT-929 — the mention/tool resolvers read a meeting's record past its own access rules.

Four holes, each reproduced here against the tool or resolver that had it:

1. get_meeting_summary answered out of Redis after checking only room read + workspace, so a
   participant of a HOST_ONLY room (and a workspace Owner/Admin who hosts nothing) read through
   WarpBot a summary the web refuses them. It now reads through TranslationRoomService's own
   summary endpoints AS THE CALLER, where ArtifactAccessHelper decides.
2. `@minutes` is served by the summary, so the minutes' DRAFT gate was never asked. The worker
   now asks `GET /rooms/{id}/minutes` as the caller before pointing the model at anything.
3. get_room_detail and get_transcript compared no workspace.
4. ...without breaking a bridge room, whose participants come from several workspaces: taking
   part in the room is admitted whatever workspace it lives in.

(The `ai_retrieval` half of the ticket is enforced by WorkspaceService's extracted-text read and
tested there; get_document's side of it is pinned at the bottom.)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.chat_tools import (
    ToolContext,
    _get_document,
    _get_meeting_summary,
    _get_room_detail,
    _get_transcript,
)
from ai_assistant_worker.chat_worker import _format_mentions, _unreadable_minutes_mentions
from ai_assistant_worker.citations import SourceRegistry

MEETING_ID = "019fd60a-e5f3-7342-804a-000000000002"
TRANSCRIPT_ID = "019fd60a-e5f3-7342-804a-0000000000aa"
CALLER = "0190aaaa-0000-7000-8000-000000000001"
HOST = "0190aaaa-0000-7000-8000-0000000000ff"

SUMMARY_BASE = f"/api/v1/room-artifacts/rooms/{MEETING_ID}/summary"
ROOM_PATH = f"/api/v1/translation-rooms/{MEETING_ID}"
SECRET = "The acquisition closes on the 14th."


def _response(status_code: int, payload: Any = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def _routed(routes: dict[str, MagicMock]) -> AsyncMock:
    """A fake httpx get() dispatching on path prefix, longest route first."""
    client = AsyncMock()

    async def _get(url: str, **_kwargs: Any) -> MagicMock:
        for path in sorted(routes, key=len, reverse=True):
            if url.startswith(path):
                return routes[path]
        raise AssertionError(f"Unexpected request: {url}")

    client.get.side_effect = _get
    return client


def _room(**overrides: Any) -> dict[str, Any]:
    return {
        "id": MEETING_ID,
        "workspaceId": "ws-1",
        "hostId": HOST,
        "title": "Board sync",
        "status": "ENDED",
        **overrides,
    }


def _ctx(
    room_routes: dict[str, MagicMock],
    *,
    transcript_routes: dict[str, MagicMock] | None = None,
    workspace_client: AsyncMock | None = None,
    workspace_id: str = "ws-1",
) -> ToolContext:
    redis = MagicMock()
    # What the tool USED to answer from. Present on every context so each refusal can assert
    # that the leak's source was never even consulted.
    redis.hgetall = AsyncMock(return_value={b"content": SECRET.encode()})
    return ToolContext(
        workspace_id=workspace_id,
        user_id=CALLER,
        bearer_token="Bearer caller-token",
        workspace_client=workspace_client or AsyncMock(),
        transcript_client=_routed(transcript_routes or {}),
        translation_room_client=_routed(room_routes),
        openai_client=None,
        model="gpt-4.1",
        redis=redis,
        citations=SourceRegistry(),
    )


def _transcript_routes() -> dict[str, MagicMock]:
    return {
        f"/api/v1/transcripts/by-room/{MEETING_ID}": _response(
            200, {"id": TRANSCRIPT_ID, "status": "FINALIZED"}
        ),
        f"/api/v1/transcripts/{TRANSCRIPT_ID}/segments": _response(
            200, {"items": [{"speakerName": "Mai", "originalText": SECRET, "sequenceOrder": 1}]}
        ),
    }


def _paths(client: AsyncMock) -> list[str]:
    return [call.args[0] for call in client.get.await_args_list]


class TestSummaryFollowsArtifactAccess:
    """Hole 1. The backend's summary endpoints answer 403 when ArtifactAccessHelper refuses —
    a participant of a HOST_ONLY room, or an Owner/Admin who is not the host."""

    async def test_a_host_only_summary_is_refused_to_a_non_host(self) -> None:
        """THE BUG. Room read passes (they attended, or they administer the workspace), the
        workspace matches, and Redis holds the summary — the three things the old gate checked
        or read. The artifact gate says no, and that is the answer."""
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(403, {"errorCode": "UNAUTHORIZED"}),
            }
        )

        raw = await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})
        result = json.loads(raw)

        assert "has not been shared with you" in result["error"]
        assert SECRET not in raw
        ctx.redis.hgetall.assert_not_awaited()
        # Nothing was registered as a source either: a chip would name what was withheld.
        assert ctx.citations is not None and ctx.citations.registered() == []

    async def test_a_refusal_on_the_second_read_is_still_a_refusal(self) -> None:
        """The host un-shares between the two reads. Either door closing closes the tool."""
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(
                    200, [{"templateKey": "general", "language": "", "isCanonical": True}]
                ),
                SUMMARY_BASE: _response(403, {}),
            }
        )

        raw = await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})

        assert "has not been shared with you" in json.loads(raw)["error"]
        assert SECRET not in raw

    async def test_the_host_reads_the_published_summary(self) -> None:
        published = {"summary": SECRET, "decisions": [], "actionItems": []}
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(
                    200, [{"templateKey": "standup", "language": "ja", "isCanonical": True}]
                ),
                SUMMARY_BASE: _response(200, {"status": "ready", "content": json.dumps(published)}),
            }
        )

        result = json.loads(await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID}))

        assert result["summary"] == published
        assert (result["templateKey"], result["language"]) == ("standup", "ja")
        ctx.redis.hgetall.assert_not_awaited()

    async def test_it_asks_for_exactly_the_published_rendering(self) -> None:
        """`/summary` QUEUES a model call for a (shape, language) nobody has rendered. A read
        tool must never cause that, so it asks for the pair the renderings list calls canonical
        and no other."""
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(
                    200,
                    [
                        {"templateKey": "general", "language": "en", "isCanonical": False},
                        {"templateKey": "standup", "language": "ja", "isCanonical": True},
                    ],
                ),
                SUMMARY_BASE: _response(200, {"status": "ready", "content": "ok"}),
            }
        )

        await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})

        summary_call = ctx.translation_room_client.get.await_args_list[-1]
        assert summary_call.args[0] == SUMMARY_BASE
        assert summary_call.kwargs["params"] == {"template": "standup", "language": "ja"}
        assert summary_call.kwargs["headers"] == {"Authorization": "Bearer caller-token"}

    async def test_a_rendering_still_being_written_is_not_presented_as_the_summary(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(
                    200, [{"templateKey": "general", "language": "", "isCanonical": True}]
                ),
                SUMMARY_BASE: _response(202, {"status": "generating", "content": None}),
            }
        )

        result = json.loads(await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID}))

        assert result == {"error": "Could not look up the meeting summary right now."}

    async def test_a_meeting_still_running_has_no_summary_and_asks_nothing(self) -> None:
        ctx = _ctx({ROOM_PATH: _response(200, _room(status="IN_PROGRESS"))})

        result = json.loads(await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID}))

        assert result["summary"] is None
        assert _paths(ctx.translation_room_client) == [ROOM_PATH]
        ctx.redis.hgetall.assert_not_awaited()

    async def test_an_unreachable_summary_service_fails_closed(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room()),
                f"{SUMMARY_BASE}/renderings": _response(500, {}),
            }
        )

        raw = await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})

        assert json.loads(raw) == {"error": "Could not look up the meeting summary right now."}
        ctx.redis.hgetall.assert_not_awaited()


class TestMinutesMentionFollowsTheDraftGate:
    """Hole 2. `@minutes:` is answered from the summary, so MeetingMinutesService's DRAFT gate
    (host / secretary / Owner-Admin only until somebody signs) was never asked."""

    @staticmethod
    def _mentions() -> str:
        return json.dumps(
            [{"entityType": "minutes", "entityId": MEETING_ID, "label": "Board sync"}]
        )

    async def test_draft_minutes_are_refused_to_a_participant(self) -> None:
        """THE BUG. The backend answers a non-host 403/404 for a DRAFT; the mention must become
        a refusal, not an instruction to read the summary the draft was drawn from."""
        client = _routed({f"/api/v1/rooms/{MEETING_ID}/minutes": _response(403, {})})

        refused = await _unreadable_minutes_mentions(self._mentions(), client, "Bearer t")
        message = _format_mentions(self._mentions(), refused)

        assert refused == frozenset({MEETING_ID})
        assert message is not None
        assert "does NOT have access to these minutes" in message
        assert "MUST call get_meeting_summary" not in message
        assert "get_room_detail" not in message
        call = client.get.await_args_list[0]
        assert call.kwargs["headers"] == {"Authorization": "Bearer t"}

    async def test_signed_minutes_keep_the_existing_instruction(self) -> None:
        client = _routed(
            {f"/api/v1/rooms/{MEETING_ID}/minutes": _response(200, {"status": "SIGNED"})}
        )

        refused = await _unreadable_minutes_mentions(self._mentions(), client, "Bearer t")
        message = _format_mentions(self._mentions(), refused)

        assert refused == frozenset()
        assert message is not None
        assert f"MUST call get_meeting_summary with meeting_id={MEETING_ID}" in message

    async def test_every_way_of_not_knowing_is_a_refusal(self) -> None:
        mentions = self._mentions()
        broken = AsyncMock()
        broken.get.side_effect = TimeoutError("upstream")

        assert await _unreadable_minutes_mentions(mentions, broken, "Bearer t") == {MEETING_ID}
        assert await _unreadable_minutes_mentions(mentions, None, "Bearer t") == {MEETING_ID}
        no_token = _routed({f"/api/v1/rooms/{MEETING_ID}/minutes": _response(200, {})})
        assert await _unreadable_minutes_mentions(mentions, no_token, "") == {MEETING_ID}
        no_token.get.assert_not_awaited()

    async def test_an_id_that_is_not_an_id_never_reaches_the_path(self) -> None:
        client = AsyncMock()
        mentions = json.dumps([{"type": "minutes", "id": "../../admin", "display": "x"}])

        refused = await _unreadable_minutes_mentions(mentions, client, "Bearer t")

        assert refused == frozenset({"../../admin"})
        client.get.assert_not_awaited()

    async def test_other_mentions_cost_no_request(self) -> None:
        client = AsyncMock()
        mentions = json.dumps(
            [
                {"entityType": "summary", "entityId": MEETING_ID, "label": "Board sync"},
                {"id": "bot-warpbot", "display": "WarpBot", "type": "agent"},
            ]
        )

        assert await _unreadable_minutes_mentions(mentions, client, "Bearer t") == frozenset()
        client.get.assert_not_awaited()


class TestRoomDetailAndTranscriptCompareWorkspace:
    """Hole 3. Both forwarded the token and compared nothing, so the room read's own answer —
    which admits an invitee and an Owner/Admin of the ROOM's workspace — was the whole check."""

    async def test_another_workspaces_room_detail_is_refused(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(200, []),
            }
        )

        raw = await _get_room_detail(ctx, {"room_id": MEETING_ID})

        assert json.loads(raw) == {"error": "No meeting found with that id."}
        assert "Board sync" not in raw
        assert ctx.citations is not None and ctx.citations.registered() == []

    async def test_another_workspaces_transcript_is_refused_before_it_is_read(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(403, {}),
            },
            transcript_routes=_transcript_routes(),
        )

        raw = await _get_transcript(ctx, {"meeting_id": MEETING_ID})

        assert json.loads(raw) == {"error": "No meeting found with that id."}
        assert SECRET not in raw
        ctx.transcript_client.get.assert_not_awaited()

    async def test_a_room_the_service_refuses_reads_the_same_as_a_missing_one(self) -> None:
        ctx = _ctx({ROOM_PATH: _response(404, {})}, transcript_routes=_transcript_routes())

        detail = json.loads(await _get_room_detail(ctx, {"room_id": MEETING_ID}))
        transcript = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert detail == transcript == {"error": "No meeting found with that id."}
        ctx.transcript_client.get.assert_not_awaited()

    async def test_the_same_workspace_reads_as_before(self) -> None:
        ctx = _ctx({ROOM_PATH: _response(200, _room())}, transcript_routes=_transcript_routes())

        detail = json.loads(await _get_room_detail(ctx, {"room_id": MEETING_ID}))
        transcript = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert detail["title"] == "Board sync"
        assert transcript["segments"][0]["text"] == SECRET
        # No roster lookup when the workspace already matches.
        assert _paths(ctx.translation_room_client) == [ROOM_PATH, ROOM_PATH]


class TestBridgeParticipantsCrossWorkspaces:
    """Hole 3's guard rail. A bridge room takes every user in the same Google Meet, whichever
    workspace each works in — so for some of its participants the room is in "another"
    workspace, and they must still read the transcript of the meeting they sat in."""

    async def test_a_participant_reads_their_rooms_transcript_from_another_workspace(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(
                    200, [{"userId": HOST}, {"userId": CALLER.upper()}]
                ),
            },
            transcript_routes=_transcript_routes(),
        )

        transcript = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))
        detail = json.loads(await _get_room_detail(ctx, {"room_id": MEETING_ID}))

        assert transcript["segments"][0]["text"] == SECRET
        assert detail["title"] == "Board sync"

    async def test_the_host_is_admitted_without_a_roster_read(self) -> None:
        ctx = _ctx(
            {ROOM_PATH: _response(200, _room(workspaceId="ws-2", effectiveHostId=CALLER))},
            transcript_routes=_transcript_routes(),
        )

        transcript = json.loads(await _get_transcript(ctx, {"meeting_id": MEETING_ID}))

        assert transcript["segments"][0]["text"] == SECRET
        assert _paths(ctx.translation_room_client) == [ROOM_PATH]

    async def test_an_invitee_who_never_joined_is_not_a_participant(self) -> None:
        """The roster read ADMITS an invitee (they may see who is in the lobby). Being allowed
        to read the roster is not being on it."""
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(200, [{"userId": HOST}]),
            },
            transcript_routes=_transcript_routes(),
        )

        raw = await _get_transcript(ctx, {"meeting_id": MEETING_ID})

        assert json.loads(raw) == {"error": "No meeting found with that id."}
        ctx.transcript_client.get.assert_not_awaited()

    async def test_the_exemption_does_not_outrank_the_artifact_gate(self) -> None:
        """Crossing the workspace line is all participation buys. A bridge participant of a
        HOST_ONLY room is still refused its summary by the service that owns that rule."""
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(200, [{"userId": CALLER}]),
                f"{SUMMARY_BASE}/renderings": _response(403, {}),
            }
        )

        raw = await _get_meeting_summary(ctx, {"meeting_id": MEETING_ID})

        assert "has not been shared with you" in json.loads(raw)["error"]

    async def test_a_turn_with_no_workspace_gets_no_exemption(self) -> None:
        ctx = _ctx(
            {
                ROOM_PATH: _response(200, _room(workspaceId="ws-2")),
                f"{ROOM_PATH}/participants": _response(200, [{"userId": CALLER}]),
            },
            workspace_id="",
        )

        result = json.loads(await _get_room_detail(ctx, {"room_id": MEETING_ID}))

        assert result == {"error": "No meeting found with that id."}


class TestDocumentReadFollowsAiRetrieval:
    """Hole 2 of the ticket, from this side: WorkspaceService's extracted-text read is where
    `ai_retrieval` is enforced, and get_document must hand back nothing of the body when it
    refuses — not fall back to some other read."""

    async def test_a_document_the_assistant_may_not_use_comes_back_without_its_text(self) -> None:
        document_id = "0190bbbb-0000-7000-8000-000000000001"
        base = f"/api/v1/workspaces/ws-1/documents/{document_id}"
        workspace_client = _routed(
            {
                base: _response(200, {"id": document_id, "name": "Salaries.xlsx"}),
                f"{base}/extracted-text": _response(403, {"errorCode": "DOCUMENT_NOT_AI_ELIGIBLE"}),
            }
        )
        ctx = _ctx({}, workspace_client=workspace_client)

        result = json.loads(await _get_document(ctx, {"document_id": document_id}))

        assert result["excerpt"] is None
        assert result["name"] == "Salaries.xlsx"
        assert _paths(workspace_client) == [base, f"{base}/extracted-text"]
