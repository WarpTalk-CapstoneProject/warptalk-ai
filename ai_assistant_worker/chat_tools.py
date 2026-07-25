"""Tools the global chat assistant can call.

Each tool forwards the caller's own bearer token to a sibling .NET service's existing,
already-authorized REST endpoint — never a privileged "internal" bypass — so a user's
assistant can only ever see what that user could already see through the normal UI.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from shared.logger import get_logger
from shared.redis_client import RedisStreamClient

logger = get_logger(__name__)

SEMANTIC_SEARCH_TIMEOUT_SECONDS = 8.0
TRANSCRIPT_SEGMENT_LIMIT = 200
DOCUMENT_EXCERPT_CHAR_LIMIT = 4000


@dataclass
class ToolContext:
    workspace_id: str
    user_id: str
    bearer_token: str
    workspace_client: httpx.AsyncClient
    transcript_client: httpx.AsyncClient
    translation_room_client: httpx.AsyncClient
    openai_client: Any
    model: str
    redis: RedisStreamClient


@dataclass
class ChatTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Awaitable[str]]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _auth_headers(ctx: ToolContext) -> dict[str, str]:
    return {"Authorization": ctx.bearer_token} if ctx.bearer_token else {}


async def _search_workspace_members(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = (arguments or {}).get("query") or ""
    try:
        response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}/members",
            params={"search": query, "page": 1, "pageSize": 5},
            headers=_auth_headers(ctx),
        )
        if response.status_code != 200:
            logger.warning("search_workspace_members_failed", status=response.status_code)
            return json.dumps({"error": "Could not look up workspace members right now."})

        items = response.json().get("items", [])
        return json.dumps([
            {
                "name": m.get("fullName"),
                "email": m.get("email"),
                "role": m.get("roleName"),
                "status": m.get("status"),
            }
            for m in items
        ])
    except Exception:
        logger.exception("search_workspace_members_error")
        return json.dumps({"error": "Could not look up workspace members right now."})


async def _search_terminology(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = ((arguments or {}).get("query") or "").strip()
    if not query:
        return json.dumps({"error": "A search term is required."})

    query_lower = query.lower()
    matches: list[dict[str, Any]] = []

    try:
        glossaries_response = await ctx.transcript_client.get(
            f"/api/v1/glossaries/workspace/{ctx.workspace_id}",
            headers=_auth_headers(ctx),
        )
        if glossaries_response.status_code != 200:
            logger.warning("search_terminology_glossaries_failed", status=glossaries_response.status_code)
            return json.dumps({"error": "Could not look up terminology right now."})

        for glossary in glossaries_response.json():
            if not glossary.get("isActive", True):
                continue

            terms_response = await ctx.transcript_client.get(
                f"/api/v1/glossaries/{glossary['id']}/terms",
                headers=_auth_headers(ctx),
            )
            if terms_response.status_code != 200:
                continue

            for term in terms_response.json():
                if not term.get("isActive", True):
                    continue
                haystack = " ".join(
                    filter(None, [term.get("sourceTerm"), term.get("targetTerm"), term.get("context")])
                ).lower()
                if query_lower in haystack:
                    matches.append({
                        "source": "workspace",
                        "glossary": glossary.get("name"),
                        "term": term.get("sourceTerm"),
                        "translation": term.get("targetTerm"),
                        "context": term.get("context"),
                        "domain": term.get("domain"),
                    })
    except Exception:
        logger.exception("search_terminology_error")
        return json.dumps({"error": "Could not look up terminology right now."})

    # Fall back to the system-managed global glossary (docs/global-glossary-plan.md) only for
    # terms the workspace hasn't already defined itself — a workspace term always takes
    # precedence, same rule GlossaryStartedEventConsumer uses for the STT/MT prompt merge.
    # On-demand, tool-triggered lookup only (no unconditional system-prompt injection): the
    # assistant already burns a request for this the moment the user asks about a term, so
    # there's no baseline token cost paid on every turn regardless of whether it's needed —
    # the tradeoff workspace-memory-research.md §2.2 flags against context_snapshot.
    if len(matches) < 5:
        try:
            global_response = await ctx.transcript_client.get(
                "/api/v1/glossaries/global", headers=_auth_headers(ctx)
            )
            if global_response.status_code == 200:
                workspace_terms_lower = {m["term"].lower() for m in matches if m.get("term")}
                for term in global_response.json():
                    if len(matches) >= 8:
                        break
                    term_name = term.get("term") or ""
                    if term_name.lower() in workspace_terms_lower:
                        continue
                    haystack = " ".join(
                        filter(None, [term_name, term.get("preferredTranslation"), term.get("definition")])
                    ).lower()
                    if query_lower in haystack:
                        matches.append({
                            "source": "global",
                            "glossary": "System (Global Glossary)",
                            "term": term_name,
                            "translation": term.get("preferredTranslation"),
                            "context": term.get("definition"),
                            "domain": term.get("businessDomain"),
                        })
        except Exception:
            logger.warning("search_terminology_global_fallback_failed")

    return json.dumps(matches[:8])


async def _list_recent_meetings(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = (arguments or {}).get("query") or ""
    try:
        response = await ctx.translation_room_client.get(
            "/api/v1/translation-rooms/history",
            params={"search": query, "page": 1, "pageSize": 5},
            headers=_auth_headers(ctx),
        )
        if response.status_code != 200:
            logger.warning("list_recent_meetings_failed", status=response.status_code)
            return json.dumps({"error": "Could not look up recent meetings right now."})

        rooms = response.json().get("rooms", [])
        return json.dumps([
            {
                "id": r.get("room", {}).get("id"),
                "title": r.get("room", {}).get("title"),
                "code": r.get("room", {}).get("translationRoomCode"),
                "status": r.get("room", {}).get("status"),
                "endedAt": r.get("room", {}).get("endedAt"),
                "durationSeconds": r.get("room", {}).get("durationSeconds"),
            }
            for r in rooms
        ])
    except Exception:
        logger.exception("list_recent_meetings_error")
        return json.dumps({"error": "Could not look up recent meetings right now."})


async def _translate_text(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    text = ((arguments or {}).get("text") or "").strip()
    target_language = ((arguments or {}).get("target_language") or "").strip()
    if not text or not target_language:
        return json.dumps({"error": "Both 'text' and 'target_language' are required."})

    try:
        response = await ctx.openai_client.chat.completions.create(
            model=ctx.model,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Translate the user's text to {target_language}. "
                        "Reply with only the translation, no explanation or quotes."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        translation = (response.choices[0].message.content or "").strip()
        return json.dumps({"translation": translation, "target_language": target_language})
    except Exception:
        logger.exception("translate_text_error")
        return json.dumps({"error": "Could not translate that text right now."})


# The system-managed global glossary lives in its own Qdrant collection (published by
# GlobalGlossaryService.TryPublishEmbeddingIndexRequestAsync in the .NET TranscriptService) —
# not fanned out into every "workspace_{id}" collection, so a single publish stays O(1). Both
# constants must match that C# side exactly: EmbeddingSearchWorker.process (warptalk-ai) hard-
# filters vector search on payload["workspace_id"], so querying the global collection without
# this sentinel would silently return zero rows. See docs/global-glossary-plan.md §5.4.
GLOBAL_GLOSSARY_COLLECTION_ID = "global_glossary"
GLOBAL_GLOSSARY_WORKSPACE_SENTINEL = "global"


async def _run_embedding_search(
    ctx: ToolContext, *, collection_id: str, workspace_id: str, query: str, top_k: int
) -> list[dict[str, Any]]:
    """One request/reply round-trip against EmbeddingSearchWorker for a single collection.

    Returns [] on timeout or error rather than raising — callers merge results from multiple
    collections and a slow/absent one (e.g. no global terms published yet) must not sink the
    whole search.
    """
    job_id = str(uuid.uuid4())
    result_key = f"embedding:search_result:{job_id}"
    request = {
        "job_id": job_id,
        "workspace_id": workspace_id,
        "collection_id": collection_id,
        "query": query,
        "top_k": str(top_k),
        "timestamp_ms": str(int(time.time() * 1000)),
    }

    try:
        await ctx.redis.publish("embedding:search_requests", request)
        raw = await asyncio.wait_for(
            ctx.redis.redis.blpop([result_key], timeout=SEMANTIC_SEARCH_TIMEOUT_SECONDS),
            timeout=SEMANTIC_SEARCH_TIMEOUT_SECONDS + 1,
        )
        if raw is None:
            return []

        _key, payload = raw
        payload = payload.decode() if isinstance(payload, bytes) else payload
        result = json.loads(payload)
        return result.get("matches", [])
    except (asyncio.TimeoutError, TimeoutError):
        return []
    except Exception:
        logger.exception("semantic_search_collection_error", extra={"collection_id": collection_id})
        return []


async def _semantic_search(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    query = ((arguments or {}).get("query") or "").strip()
    if not query:
        return json.dumps({"error": "A search query is required."})

    top_k = 5

    # workspace_{id} is the collection-naming convention 3 producers already publish
    # into: TranscriptRedisConsumerService (per segment), DocumentSecurityGuardrail-
    # ConsumerService (per document), and GlossaryService (per term) — see
    # docs/workspace-memory-research.md for the full pipeline map. Global glossary terms live
    # in a separate collection (see constants above), so both are queried and merged — a
    # workspace's own knowledge should never be shadowed by the system baseline, but the
    # baseline is still useful context when the workspace has nothing on the topic itself.
    try:
        workspace_matches, global_matches = await asyncio.gather(
            _run_embedding_search(
                ctx,
                collection_id=f"workspace_{ctx.workspace_id}",
                workspace_id=ctx.workspace_id,
                query=query,
                top_k=top_k,
            ),
            _run_embedding_search(
                ctx,
                collection_id=GLOBAL_GLOSSARY_COLLECTION_ID,
                workspace_id=GLOBAL_GLOSSARY_WORKSPACE_SENTINEL,
                query=query,
                top_k=top_k,
            ),
        )
    except Exception:
        logger.exception("semantic_search_error")
        return json.dumps({"error": "Could not perform semantic search right now."})

    merged = sorted(workspace_matches + global_matches, key=lambda m: m.get("score", 0), reverse=True)[:top_k]
    if not merged:
        return json.dumps({"matches": [], "note": "No results available."})

    return json.dumps({"matches": merged})


async def _get_meeting_summary(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    meeting_id = ((arguments or {}).get("meeting_id") or "").strip()
    if not meeting_id:
        return json.dumps({
            "error": "A meeting_id is required — call list_recent_meetings first to find one."
        })

    try:
        summary_hash = await ctx.redis.hgetall(f"meeting:{meeting_id}:summary")
        if not summary_hash:
            return json.dumps({
                "summary": None,
                "note": (
                    "No summary has been generated for this meeting yet. Meeting summaries "
                    "are only produced automatically as a meeting's transcript pipeline "
                    "completes — there is currently no on-demand trigger."
                ),
            })

        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in summary_hash.items()
        }
        return json.dumps({
            "summary": decoded.get("content"),
            "action_items": decoded.get("action_items"),
        })
    except Exception:
        logger.exception("get_meeting_summary_error")
        return json.dumps({"error": "Could not look up the meeting summary right now."})


async def _get_room_detail(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    room_id = ((arguments or {}).get("room_id") or "").strip()
    if not room_id:
        return json.dumps({
            "error": "A room_id is required — call list_recent_meetings first to find one."
        })

    try:
        response = await ctx.translation_room_client.get(
            f"/api/v1/translation-rooms/{room_id}",
            headers=_auth_headers(ctx),
        )
        if response.status_code == 404:
            return json.dumps({"error": "No room found with that id."})
        if response.status_code != 200:
            logger.warning("get_room_detail_failed", status=response.status_code)
            return json.dumps({"error": "Could not look up that room right now."})

        room = response.json()
        return json.dumps({
            "id": room.get("id"),
            "title": room.get("title"),
            "code": room.get("translationRoomCode"),
            "status": room.get("status"),
            "sourceLanguage": room.get("sourceLanguage"),
            "targetLanguages": room.get("targetLanguages"),
            "hostId": room.get("hostId"),
            "scheduledAt": room.get("scheduledAt"),
            "startedAt": room.get("startedAt"),
            "endedAt": room.get("endedAt"),
        })
    except Exception:
        logger.exception("get_room_detail_error")
        return json.dumps({"error": "Could not look up that room right now."})


async def _get_transcript(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    meeting_id = ((arguments or {}).get("meeting_id") or "").strip()
    if not meeting_id:
        return json.dumps({
            "error": "A meeting_id is required — call list_recent_meetings first to find one."
        })

    try:
        transcript_response = await ctx.transcript_client.get(
            f"/api/v1/transcripts/by-room/{meeting_id}",
            headers=_auth_headers(ctx),
        )
        if transcript_response.status_code == 404:
            return json.dumps({"segments": [], "note": "No transcript exists for this meeting yet."})
        if transcript_response.status_code != 200:
            logger.warning("get_transcript_lookup_failed", status=transcript_response.status_code)
            return json.dumps({"error": "Could not look up the transcript right now."})

        transcript = transcript_response.json()
        transcript_id = transcript.get("id")
        if not transcript_id:
            return json.dumps({"segments": [], "note": "No transcript exists for this meeting yet."})

        segments_response = await ctx.transcript_client.get(
            f"/api/v1/transcripts/{transcript_id}/segments",
            params={"skip": 0, "take": TRANSCRIPT_SEGMENT_LIMIT},
            headers=_auth_headers(ctx),
        )
        if segments_response.status_code != 200:
            logger.warning("get_transcript_segments_failed", status=segments_response.status_code)
            return json.dumps({"error": "Could not look up the transcript segments right now."})

        items = segments_response.json().get("items", [])
        ordered = sorted(items, key=lambda s: s.get("sequenceOrder", 0))
        return json.dumps({
            "transcriptId": transcript_id,
            "status": transcript.get("status"),
            "segments": [
                {
                    "speaker": s.get("speakerName"),
                    "language": s.get("originalLanguage"),
                    "text": s.get("originalText"),
                    "startMs": s.get("startTimeMs"),
                }
                for s in ordered
            ],
        })
    except Exception:
        logger.exception("get_transcript_error")
        return json.dumps({"error": "Could not look up the transcript right now."})


async def _get_document(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    document_id = ((arguments or {}).get("document_id") or "").strip()
    if not document_id:
        return json.dumps({"error": "A document_id is required."})

    try:
        meta_response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}/documents/{document_id}",
            headers=_auth_headers(ctx),
        )
        if meta_response.status_code == 404:
            return json.dumps({"error": "No document found with that id."})
        if meta_response.status_code != 200:
            logger.warning("get_document_failed", status=meta_response.status_code)
            return json.dumps({"error": "Could not look up that document right now."})

        doc = meta_response.json()

        # Extracted text may not be ready yet (still ingesting) or the caller may lack
        # View permission on it even though they can see the document's metadata — either
        # way, degrade to metadata-only rather than failing the whole tool call.
        excerpt = None
        text_response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}/documents/{document_id}/extracted-text",
            headers=_auth_headers(ctx),
        )
        if text_response.status_code == 200:
            full_text = text_response.json().get("fullText") or ""
            excerpt = full_text[:DOCUMENT_EXCERPT_CHAR_LIMIT] or None

        return json.dumps({
            "id": doc.get("id"),
            "name": doc.get("name"),
            "status": doc.get("status"),
            "ingestionStatus": doc.get("ingestionStatus"),
            "isSensitive": doc.get("isSensitive"),
            "excerpt": excerpt,
        })
    except Exception:
        logger.exception("get_document_error")
        return json.dumps({"error": "Could not look up that document right now."})


TOOLS: list[ChatTool] = [
    ChatTool(
        name="search_workspace_members",
        description=(
            "Search for members of the current workspace by name or email. Use this when "
            "the user asks about a teammate, wants to know who is in the workspace, or "
            "needs someone's role."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name or email fragment to search for. Leave blank to list the first few members.",
                }
            },
            "required": [],
        },
        handler=_search_workspace_members,
    ),
    ChatTool(
        name="search_terminology",
        description=(
            "Search terminology for a term: first the workspace's own glossary, then falling "
            "back to the platform's system-managed global glossary of common IT/business "
            "terms. Use this when the user asks what a specific term means, how it should be "
            "translated, or what terminology exists."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Term or keyword to search for."},
            },
            "required": ["query"],
        },
        handler=_search_terminology,
    ),
    ChatTool(
        name="list_recent_meetings",
        description=(
            "List the user's recent translation room meetings (ended or cancelled), "
            "optionally filtered by a keyword in the title. Use this when the user asks "
            "about past or recent meetings."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional keyword to filter meeting titles by."},
            },
            "required": [],
        },
        handler=_list_recent_meetings,
    ),
    ChatTool(
        name="translate_text",
        description=(
            "Translate a piece of text into another language. Use this when the user "
            "asks you to translate something for them right now."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to translate."},
                "target_language": {
                    "type": "string",
                    "description": "The language to translate into, e.g. 'Vietnamese' or 'French'.",
                },
            },
            "required": ["text", "target_language"],
        },
        handler=_translate_text,
    ),
    ChatTool(
        name="semantic_search",
        description=(
            "Semantically search the workspace's indexed knowledge base (documents, "
            "transcripts, glossaries) for content related to a query. Use this for "
            "conceptual questions that a simple keyword search wouldn't answer well. "
            "May return no matches if nothing relevant has been indexed yet."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
            },
            "required": ["query"],
        },
        handler=_semantic_search,
    ),
    ChatTool(
        name="get_meeting_summary",
        description=(
            "Get the AI-generated summary and action items for a specific past meeting, "
            "if one has already been generated. Call list_recent_meetings first to find "
            "the meeting's id. Summaries are produced automatically as a meeting's "
            "transcript pipeline completes — there is no way to generate one on demand."
        ),
        parameters={
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The meeting's id, from a prior list_recent_meetings call.",
                },
            },
            "required": ["meeting_id"],
        },
        handler=_get_meeting_summary,
    ),
    ChatTool(
        name="get_room_detail",
        description=(
            "Get full details for a specific translation room/meeting — status, "
            "languages, host, and schedule. Call list_recent_meetings first to find the "
            "room's id if the user hasn't given you one directly (e.g. from page context)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "room_id": {
                    "type": "string",
                    "description": "The room's id, from page context or a prior list_recent_meetings call.",
                },
            },
            "required": ["room_id"],
        },
        handler=_get_room_detail,
    ),
    ChatTool(
        name="get_transcript",
        description=(
            "Get the transcribed segments (speaker, language, text) for a specific "
            "meeting's transcript. Use this when the user asks what was said, wants a "
            "quote, or wants something found within the meeting's transcript. Call "
            "list_recent_meetings first to find the meeting's id if you don't have one."
        ),
        parameters={
            "type": "object",
            "properties": {
                "meeting_id": {
                    "type": "string",
                    "description": "The meeting/room's id, from page context or a prior list_recent_meetings call.",
                },
            },
            "required": ["meeting_id"],
        },
        handler=_get_transcript,
    ),
    ChatTool(
        name="get_document",
        description=(
            "Get metadata and a text excerpt for a specific workspace document. Use this "
            "when the user asks about the content of a document, or references one by "
            "name/id (e.g. from page context or an @mention)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "document_id": {
                    "type": "string",
                    "description": "The document's id, from page context or an @mention.",
                },
            },
            "required": ["document_id"],
        },
        handler=_get_document,
    ),
]

TOOLS_BY_NAME: dict[str, ChatTool] = {t.name: t for t in TOOLS}
