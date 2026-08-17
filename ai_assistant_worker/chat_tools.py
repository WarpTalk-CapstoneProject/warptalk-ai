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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import httpx

from ai_assistant_worker.meeting_draft import (
    MEETING_TYPES,
    RECURRENCE_CHOICES,
    build_payload,
    draft_from_arguments,
    missing_fields,
    validate,
)
from shared.logger import get_logger
from shared.openai_options import completion_options
from shared.redis_client import RedisStreamClient

logger = get_logger(__name__)

SEMANTIC_SEARCH_TIMEOUT_SECONDS = 8.0
TRANSCRIPT_SEGMENT_LIMIT = 200
DOCUMENT_EXCERPT_CHAR_LIMIT = 4000
# search_documents returns names and ids for the model to choose from, not content, so a
# handful is enough to disambiguate "the onboarding spec" — and the cap keeps a workspace
# with hundreds of documents from spending the turn's tokens on a directory listing.
DOCUMENT_SEARCH_DEFAULT_LIMIT = 5
DOCUMENT_SEARCH_MAX_LIMIT = 20
# GlossaryTerm.Context and GlobalGlossaryTerm.Definition are unbounded TEXT columns in
# Postgres (unlike Term/PreferredTranslation, which are VARCHAR(255) with a DB-level cap) —
# without this, one admin- or workspace-authored term with a long free-text definition would
# make every _search_terminology call that surfaces it cost proportionally many tokens, with
# no ceiling. 300 chars (~75 tokens) is plenty for a term explanation; see
# docs/global-glossary-plan.md (token-cost follow-up on the search_terminology fallback).
TERMINOLOGY_CONTEXT_CHAR_LIMIT = 300


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
    #: Used to build document links in a created meeting's description. Last and defaulted so
    #: every existing construction site keeps working unchanged.
    workspace_slug: str | None = None
    #: Only get_platform_analytics reaches these, and only with the caller's own token. Optional
    #: so a worker that has not been given them simply cannot serve that one tool, rather than
    #: failing to start.
    billing_client: httpx.AsyncClient | None = None
    auth_client: httpx.AsyncClient | None = None


@dataclass
class ChatTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Awaitable[str]]

    def to_openai_schema(self) -> dict[str, Any]:
        """Tool declaration in the shape /v1/responses expects.

        FLAT, not the nested {"type": "function", "function": {...}} that chat
        completions takes. Verified against the live API: the flat form is accepted by
        both gpt-5.6-luna and gpt-4o-mini on this endpoint.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
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
        return json.dumps(
            [
                {
                    "name": m.get("fullName"),
                    "email": m.get("email"),
                    "role": m.get("roleName"),
                    "status": m.get("status"),
                }
                for m in items
            ]
        )
    except Exception:
        logger.exception("search_workspace_members_error")
        return json.dumps({"error": "Could not look up workspace members right now."})


def _truncate_terminology_context(text: str | None) -> str | None:
    """Bounds GlossaryTerm.Context / GlobalGlossaryTerm.Definition — both unbounded TEXT
    columns with no application-level length cap unlike Term/PreferredTranslation (VARCHAR
    255) — to TERMINOLOGY_CONTEXT_CHAR_LIMIT so one long-winded term definition can't blow up
    the token cost of every _search_terminology call that happens to surface it.
    """
    if not text:
        return text
    return text[:TERMINOLOGY_CONTEXT_CHAR_LIMIT]


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
            logger.warning(
                "search_terminology_glossaries_failed", status=glossaries_response.status_code
            )
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
                    filter(
                        None, [term.get("sourceTerm"), term.get("targetTerm"), term.get("context")]
                    )
                ).lower()
                if query_lower in haystack:
                    matches.append(
                        {
                            "source": "workspace",
                            "glossary": glossary.get("name"),
                            "term": term.get("sourceTerm"),
                            "translation": term.get("targetTerm"),
                            "context": _truncate_terminology_context(term.get("context")),
                            "domain": term.get("domain"),
                        }
                    )
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
                        filter(
                            None,
                            [term_name, term.get("preferredTranslation"), term.get("definition")],
                        )
                    ).lower()
                    if query_lower in haystack:
                        matches.append(
                            {
                                "source": "global",
                                "glossary": "System (Global Glossary)",
                                "term": term_name,
                                "translation": term.get("preferredTranslation"),
                                "context": _truncate_terminology_context(term.get("definition")),
                                "domain": term.get("businessDomain"),
                            }
                        )
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
            # The status travels with the message. It used to be logged and dropped, so the
            # answer a user could screenshot said only "could not look up recent meetings" —
            # which is why WT-373's report could not say whether this was auth, routing or an
            # outage, and the logs that knew had already been rotated away by a deploy.
            return json.dumps(
                {
                    "error": "Could not look up recent meetings right now.",
                    "status": response.status_code,
                }
            )

        rooms = response.json().get("rooms", [])
        return json.dumps(
            [
                {
                    "id": r.get("room", {}).get("id"),
                    "title": r.get("room", {}).get("title"),
                    "code": r.get("room", {}).get("translationRoomCode"),
                    "status": r.get("room", {}).get("status"),
                    "endedAt": r.get("room", {}).get("endedAt"),
                    "durationSeconds": r.get("room", {}).get("durationSeconds"),
                }
                for r in rooms
            ]
        )
    except Exception as error:
        logger.exception("list_recent_meetings_error")
        return json.dumps(
            {
                "error": "Could not look up recent meetings right now.",
                "reason": type(error).__name__,
            }
        )


async def _translate_text(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    text = ((arguments or {}).get("text") or "").strip()
    target_language = ((arguments or {}).get("target_language") or "").strip()
    if not text or not target_language:
        return json.dumps({"error": "Both 'text' and 'target_language' are required."})

    try:
        response = await ctx.openai_client.chat.completions.create(
            model=ctx.model,
            # Uncapped by design — a translation must not be truncated mid-sentence.
            **completion_options(ctx.model, temperature=0.0),
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


async def _caller_is_privileged(ctx: ToolContext) -> bool:
    """Whether this caller may reach document-sourced chunks through semantic search (WT-463).

    Resolved from the workspace read, with the CALLER'S OWN bearer token — so the answer is the
    role the backend attributes to them, not one this worker decided. `GET /workspaces/{id}`
    returns the requester's own `role` for exactly this reason.

    Fails CLOSED. A failed or unexpected response yields False, which costs an owner some
    document context in one answer; the opposite default would hand every member the contents of
    documents they cannot open, which is the bug being fixed.

    Phase 0 shape: coarse, and TRUSTED BY THE SEARCH WORKER because the flag travels in the
    request this worker writes. That is bounded on purpose — the threat model closed here is
    "a member uses WarpBot normally", not "an attacker controls the AI worker". Phase 3 moves the
    decision behind a signed or worker-resolved subject set so the flag stops being self-declared.
    """
    try:
        response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}",
            headers=_auth_headers(ctx),
        )
        if response.status_code != 200:
            logger.warning("caller_role_lookup_failed", status=response.status_code)
            return False
        role = str(response.json().get("role") or "").strip().lower()
        return role in {"owner", "admin"}
    except Exception:
        logger.exception("caller_role_lookup_error")
        return False


async def _run_embedding_search(
    ctx: ToolContext,
    *,
    collection_id: str,
    workspace_id: str,
    query: str,
    top_k: int,
    privileged: bool,
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
        # WT-463: the search worker treats an absent value as unprivileged, so this must be sent
        # explicitly on every request rather than only when true.
        "privileged": "true" if privileged else "false",
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
        return cast(list[dict[str, Any]], result.get("matches", []))
    except TimeoutError:
        return []
    except Exception:
        logger.exception("semantic_search_collection_error", extra={"collection_id": collection_id})
        return []


# The closed set the extractor writes — knowledge_facts.py FACT_CATEGORIES, mirrored by the web
# client's FACT_CATEGORIES. Closed on purpose: an open set produces a different label per chunk.
FACT_CATEGORIES = ("decision", "requirement", "definition", "commitment", "risk", "reference")


async def _search_facts(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    """The workspace's extracted knowledge facts, filtered and listed rather than embedded.

    `semantic_search` reaches the same facts through the vector index, and is the right tool for
    "what do we know about X". This one answers the question that index is bad at: "what was
    DECIDED", where the category is the query and completeness matters more than similarity. A
    vector search for "decision" returns whatever is nearest in embedding space; this returns the
    rows actually tagged as decisions, which is what the Knowledge page shows a human.
    """
    arguments = arguments or {}
    category = (arguments.get("fact_category") or "").strip().lower()
    query = (arguments.get("query") or "").strip().lower()

    if category and category not in FACT_CATEGORIES:
        return json.dumps(
            {"error": f"Unknown fact_category '{category}'.", "allowed": list(FACT_CATEGORIES)}
        )

    params: dict[str, Any] = {"pageSize": 100}
    if category:
        params["factCategory"] = category

    try:
        response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}/knowledge",
            params=params,
            headers=_auth_headers(ctx),
        )
        if response.status_code != 200:
            logger.warning("search_facts_failed", status=response.status_code)
            return json.dumps(
                {
                    "error": "Could not read the knowledge base right now.",
                    "status": response.status_code,
                }
            )

        items = response.json().get("items", []) or []
    except Exception as error:
        logger.exception("search_facts_error")
        return json.dumps(
            {
                "error": "Could not read the knowledge base right now.",
                "reason": type(error).__name__,
            }
        )

    matches = []
    for item in items:
        fact = item.get("fact")
        # A row with no extracted fact is an indexed chunk, not knowledge. Listing those would
        # bury six real decisions under a hundred transcript fragments.
        if not fact:
            continue
        if query and query not in f"{fact} {item.get('sourceTitle') or ''}".lower():
            continue
        matches.append(
            {
                "fact": fact,
                "category": item.get("factCategory"),
                "source": item.get("sourceTitle") or item.get("documentName"),
                "source_type": item.get("sourceType"),
            }
        )
        if len(matches) >= 25:
            break

    if not matches:
        return json.dumps(
            {
                "facts": [],
                "note": (
                    "No facts recorded for that filter. Facts are extracted after a meeting ends "
                    "or a document finishes indexing, so a very recent meeting may have none yet."
                ),
            }
        )
    return json.dumps({"facts": matches})


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
    # WT-463: resolved once per search, before either query, and applied to the workspace
    # collection only. The global glossary is published for every workspace by design, so it has
    # no per-subject dimension to honour — passing the caller's privilege there would imply a
    # restriction that does not exist.
    privileged = await _caller_is_privileged(ctx)

    try:
        workspace_matches, global_matches = await asyncio.gather(
            _run_embedding_search(
                ctx,
                collection_id=f"workspace_{ctx.workspace_id}",
                workspace_id=ctx.workspace_id,
                query=query,
                top_k=top_k,
                privileged=privileged,
            ),
            _run_embedding_search(
                ctx,
                collection_id=GLOBAL_GLOSSARY_COLLECTION_ID,
                workspace_id=GLOBAL_GLOSSARY_WORKSPACE_SENTINEL,
                query=query,
                top_k=top_k,
                privileged=True,
            ),
        )
    except Exception:
        logger.exception("semantic_search_error")
        return json.dumps({"error": "Could not perform semantic search right now."})

    merged = sorted(
        workspace_matches + global_matches, key=lambda m: m.get("score", 0), reverse=True
    )[:top_k]
    if not merged:
        return json.dumps({"matches": [], "note": "No results available."})

    return json.dumps({"matches": merged})


async def _authorize_meeting_access(ctx: ToolContext, meeting_id: str) -> str | None:
    """None if this caller may read this meeting's derived data, else a tool-visible error.

    S2. `meeting_id` is a MODEL-SUPPLIED tool argument — the assistant will pass whatever id
    appears in the conversation, including one a user simply typed. Tools that answer out of
    a sibling service inherit that service's authorization for free by forwarding the
    caller's own bearer token (see this module's docstring); tools that answer straight out
    of Redis inherit nothing, because Redis has no notion of who is asking. Every such tool
    has to re-establish the check that the HTTP call would have made.

    Two things are checked, and both are needed:

    - The room is fetched with the caller's own bearer token. That is what proves the token
      is presently valid: this worker performs no signature verification and no expiry check
      of its own, so an unauthenticated (or expired) request must be refused by the .NET
      service, not by us.
    - The room's workspace must be the workspace this chat turn is scoped to.
      GET /api/v1/translation-rooms/{id} is [Authorize] but performs no workspace or
      participant check of its own, so a 200 alone only proves the room EXISTS — any
      authenticated user in any workspace gets one. Without the workspace comparison this
      gate would still hand a user another workspace's meeting summary, which is the bug.
    """
    try:
        uuid.UUID(meeting_id)
    except ValueError:
        # Not merely tidiness: the id is interpolated into a Redis key, so an unvalidated
        # value lets a crafted argument name a key that is not a meeting summary at all.
        return json.dumps({"error": "That does not look like a valid meeting id."})

    if not ctx.bearer_token:
        logger.warning("meeting_summary_denied_no_token", meeting_id=meeting_id)
        return json.dumps({"error": "You are not signed in to view that meeting."})

    try:
        response = await ctx.translation_room_client.get(
            f"/api/v1/translation-rooms/{meeting_id}",
            headers=_auth_headers(ctx),
        )
    except Exception:
        logger.exception("meeting_summary_authorization_error", meeting_id=meeting_id)
        return json.dumps({"error": "Could not look up the meeting summary right now."})

    if response.status_code in (401, 403, 404):
        # One indistinguishable answer for "no such meeting" and "not yours" — telling them
        # apart turns this tool into an oracle for which meeting ids exist elsewhere.
        logger.warning(
            "meeting_summary_denied",
            meeting_id=meeting_id,
            user_id=ctx.user_id,
            status=response.status_code,
        )
        return json.dumps({"error": "No meeting found with that id."})

    if response.status_code != 200:
        logger.warning("meeting_summary_authorization_failed", status=response.status_code)
        return json.dumps({"error": "Could not look up the meeting summary right now."})

    try:
        room_workspace_id = str(response.json().get("workspaceId") or "")
    except ValueError:
        logger.warning("meeting_summary_authorization_unreadable", meeting_id=meeting_id)
        return json.dumps({"error": "Could not look up the meeting summary right now."})

    # Fail closed: a room whose workspace we cannot read is a room we cannot clear.
    if not room_workspace_id or room_workspace_id.lower() != (ctx.workspace_id or "").lower():
        logger.warning(
            "meeting_summary_denied_cross_workspace",
            meeting_id=meeting_id,
            user_id=ctx.user_id,
        )
        return json.dumps({"error": "No meeting found with that id."})

    return None


async def _get_meeting_summary(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    meeting_id = ((arguments or {}).get("meeting_id") or "").strip()
    if not meeting_id:
        return json.dumps(
            {"error": "A meeting_id is required — call list_recent_meetings first to find one."}
        )

    denial = await _authorize_meeting_access(ctx, meeting_id)
    if denial is not None:
        return denial

    try:
        summary_hash = await ctx.redis.hgetall(f"meeting:{meeting_id}:summary")
        if not summary_hash:
            return json.dumps(
                {
                    "summary": None,
                    "note": (
                        "No summary has been generated for this meeting yet. Meeting summaries "
                        "are only produced automatically as a meeting's transcript pipeline "
                        "completes — there is currently no on-demand trigger."
                    ),
                }
            )

        decoded = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in summary_hash.items()
        }
        return json.dumps(
            {
                "summary": decoded.get("content"),
                "action_items": decoded.get("action_items"),
            }
        )
    except Exception:
        logger.exception("get_meeting_summary_error")
        return json.dumps({"error": "Could not look up the meeting summary right now."})


async def _get_room_detail(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    room_id = ((arguments or {}).get("room_id") or "").strip()
    if not room_id:
        return json.dumps(
            {"error": "A room_id is required — call list_recent_meetings first to find one."}
        )

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
        return json.dumps(
            {
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
            }
        )
    except Exception:
        logger.exception("get_room_detail_error")
        return json.dumps({"error": "Could not look up that room right now."})


async def _get_transcript(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    meeting_id = ((arguments or {}).get("meeting_id") or "").strip()
    if not meeting_id:
        return json.dumps(
            {"error": "A meeting_id is required — call list_recent_meetings first to find one."}
        )

    try:
        transcript_response = await ctx.transcript_client.get(
            f"/api/v1/transcripts/by-room/{meeting_id}",
            headers=_auth_headers(ctx),
        )
        if transcript_response.status_code == 404:
            return json.dumps(
                {"segments": [], "note": "No transcript exists for this meeting yet."}
            )
        if transcript_response.status_code != 200:
            logger.warning("get_transcript_lookup_failed", status=transcript_response.status_code)
            return json.dumps({"error": "Could not look up the transcript right now."})

        transcript = transcript_response.json()
        transcript_id = transcript.get("id")
        if not transcript_id:
            return json.dumps(
                {"segments": [], "note": "No transcript exists for this meeting yet."}
            )

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
        return json.dumps(
            {
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
            }
        )
    except Exception:
        logger.exception("get_transcript_error")
        return json.dumps({"error": "Could not look up the transcript right now."})


async def _search_documents(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    """Find workspace documents by name.

    get_document needs an id, and the only ids the model could previously obtain came from an
    @mention or from page context — so a user who simply named a document ("what does the
    onboarding spec say?") could not be answered at all. This is the missing name→id step.

    An empty query is deliberately allowed: "what documents do we have?" is a real question,
    and the endpoint treats a missing search as "list the first page".
    """
    query = ((arguments or {}).get("query") or "").strip()
    limit = (arguments or {}).get("limit")
    try:
        page_size = int(limit) if limit is not None else DOCUMENT_SEARCH_DEFAULT_LIMIT
    except (TypeError, ValueError):
        page_size = DOCUMENT_SEARCH_DEFAULT_LIMIT
    page_size = max(1, min(page_size, DOCUMENT_SEARCH_MAX_LIMIT))

    params: dict[str, Any] = {"page": 1, "pageSize": page_size}
    if query:
        params["search"] = query

    try:
        response = await ctx.workspace_client.get(
            f"/api/v1/workspaces/{ctx.workspace_id}/documents",
            params=params,
            headers=_auth_headers(ctx),
        )
        if response.status_code != 200:
            logger.warning("search_documents_failed", status=response.status_code)
            return json.dumps({"error": "Could not look up workspace documents right now."})

        items = response.json().get("items") or []
        return json.dumps(
            [
                {
                    "id": doc.get("id"),
                    "name": doc.get("name"),
                    "status": doc.get("status"),
                    "ingestionStatus": doc.get("ingestionStatus"),
                    # Whether the assistant is allowed to read this document's contents at
                    # all. A false here means get_document will come back metadata-only, so
                    # the model can say why rather than reporting an empty document.
                    "isAiAllowed": doc.get("isAiAllowed"),
                    "confidentialityLevel": doc.get("confidentialityLevel"),
                }
                for doc in items
            ]
        )
    except Exception:
        logger.exception("search_documents_error")
        return json.dumps({"error": "Could not look up workspace documents right now."})


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

        return json.dumps(
            {
                "id": doc.get("id"),
                "name": doc.get("name"),
                "status": doc.get("status"),
                "ingestionStatus": doc.get("ingestionStatus"),
                "isSensitive": doc.get("isSensitive"),
                "excerpt": excerpt,
            }
        )
    except Exception:
        logger.exception("get_document_error")
        return json.dumps({"error": "Could not look up that document right now."})


async def _ask_user(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    """Put a question card in front of the user and STOP.

    This tool does not answer anything. It exists so the assistant can decline to guess.

    The worker turns the arguments into a `question` event on the result stream, the web client
    renders the choices as a card, and the user's pick comes back as an ordinary chat message on
    the NEXT turn. Nothing here blocks: pausing a Redis request/response loop mid-flight to wait
    for a human would hold a worker slot open for as long as somebody takes to read, and a
    reconnect would strand the turn forever.

    The string returned is for the MODEL, not the user — it is the instruction to stop talking
    now rather than filling the silence with an assumption, which is what a model does when a
    tool returns nothing useful.
    """
    questions = (arguments or {}).get("questions") or []
    if not isinstance(questions, list) or not questions:
        return json.dumps({"error": "ask_user needs at least one question."})

    return json.dumps(
        {
            "status": "asked",
            "question_count": len(questions),
            "instruction": (
                "The question card is now on the user's screen. End your turn WITHOUT calling "
                "another tool and WITHOUT guessing an answer. Say one short sentence telling "
                "them you need these details, then stop. Their reply arrives as a normal "
                "message on your next turn."
            ),
        }
    )


async def _create_meeting(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    """Create a translation room. The assistant's first tool that writes anything.

    Gated twice before the network is touched — once on what is missing, once on what would come
    back as a 400 — because the failure modes here are not "a wrong answer on screen" but a real
    room, real invitation emails, and for a recurring booking a whole series of them.
    """
    draft = draft_from_arguments(arguments)
    # The slug is the worker's to supply, not the model's: it comes from the request context, and
    # a model-invented slug would produce links that look clickable and go nowhere.
    draft.workspace_slug = getattr(ctx, "workspace_slug", None) or None

    absent = missing_fields(draft)
    if absent:
        # Handed back as a list the model can pass straight to ask_user, rather than as prose it
        # has to parse out of an error string.
        return json.dumps(
            {
                "status": "needs_more_information",
                "missing": absent,
                "instruction": (
                    "Do NOT create the meeting. Call ask_user with one question per missing "
                    "field, then stop."
                ),
            }
        )

    problems = validate(draft)
    if problems:
        return json.dumps({"status": "invalid", "problems": problems})

    payload = build_payload(draft, ctx.workspace_id)

    try:
        response = await ctx.translation_room_client.post(
            "/api/v1/translation-rooms",
            json=payload,
            headers=_auth_headers(ctx),
        )
    except Exception:
        logger.exception("create_meeting_request_error")
        return json.dumps({"error": "Could not reach the meeting service."})

    if response.status_code not in (200, 201):
        # The server's own words, not a generic failure: it is the only thing that knows why, and
        # "you are not allowed to create meetings in this workspace" is something the user can act
        # on where "the tool failed" is not.
        detail = ""
        try:
            body = response.json()
            detail = body.get("error") or body.get("message") or ""
        except Exception:
            detail = ""
        logger.warning("create_meeting_failed", status=response.status_code)
        return json.dumps(
            {
                "status": "failed",
                "http_status": response.status_code,
                "reason": detail or "The meeting service refused the request.",
            }
        )

    try:
        created = response.json()
    except Exception:
        created = {}

    # A recurring booking answers with {series, firstOccurrence}; a single meeting answers with
    # the room itself. Both are reported, so the model can say "every weekday from Monday" rather
    # than "created" and leave the user to go and check.
    room = created.get("firstOccurrence") or created
    return json.dumps(
        {
            "status": "created",
            "id": room.get("id"),
            "title": room.get("title"),
            "room_code": room.get("translationRoomCode"),
            "scheduled_at": room.get("scheduledAt"),
            "recurring": bool(created.get("series")),
            "invited_count": len(draft.invited_emails),
        }
    )


#: Reports get_platform_analytics can fetch, and which client serves each. Kept beside the tool
#: declaration's enum so the two cannot drift into offering a report nothing can answer.
PLATFORM_ANALYTICS_REPORTS = ("overview", "revenue", "feedback", "users", "health")

#: The window a feedback report may cover. Matches the server's own cap — asking for more comes
#: back a validation error, and there is no reason to spend a round trip discovering that.
PLATFORM_ANALYTICS_MAX_DAYS = 366
PLATFORM_ANALYTICS_DEFAULT_DAYS = 30


def _not_authorized(report: str) -> dict[str, Any]:
    return {
        "error": "not_authorized",
        "report": report,
        "message": (
            "Platform analytics require the platform administrator role. Say so plainly; do not "
            "estimate the figure."
        ),
    }


async def _admin_get(
    client: httpx.AsyncClient | None,
    path: str,
    ctx: ToolContext,
    report: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One admin GET, with the caller's own token and every failure named rather than empty.

    The token is the caller's, never a service credential — the platform admin policy is checked
    server-side on every one of these paths, so an ordinary user's assistant sees exactly the 403
    that user would see in a browser. There is no elevation here to get wrong.
    """
    if client is None:
        return {
            "error": "unavailable",
            "report": report,
            "message": "This deployment has no client configured for that report.",
        }

    try:
        response = await client.get(path, params=params, headers=_auth_headers(ctx))
    except Exception:
        logger.exception("platform_analytics_request_failed", report=report)
        return {"error": "unavailable", "report": report, "message": "The service did not answer."}

    # 401 as well as 403: an expired token and a missing role are both "you cannot have this",
    # and neither is a number to guess at.
    if response.status_code in (401, 403):
        return _not_authorized(report)

    if response.status_code != 200:
        logger.warning("platform_analytics_bad_status", report=report, status=response.status_code)
        return {
            "error": "unavailable",
            "report": report,
            "message": f"The service answered {response.status_code}.",
        }

    try:
        return cast("dict[str, Any]", response.json())
    except ValueError:
        logger.warning("platform_analytics_bad_body", report=report)
        return {"error": "unavailable", "report": report, "message": "Unreadable response."}


async def _analytics_overview(ctx: ToolContext, _days: int) -> dict[str, Any]:
    workspaces, meetings = await asyncio.gather(
        # pageSize 1: the envelope's `total` is the whole answer, and a page of rows would be
        # tokens spent on data nobody asked for.
        _admin_get(
            ctx.workspace_client,
            "/api/v1/admin/workspaces",
            ctx,
            "overview",
            {"page": 1, "pageSize": 1},
        ),
        _admin_get(ctx.translation_room_client, "/api/v1/admin/meetings/counts", ctx, "overview"),
    )

    if workspaces.get("error"):
        return workspaces

    result: dict[str, Any] = {"workspaces_total": workspaces.get("total")}
    if meetings.get("error"):
        result["meetings_note"] = meetings.get("message")
    else:
        result["meetings_live_now"] = meetings.get("liveNow")
        result["meetings_started_today"] = meetings.get("startedToday")
    return result


async def _analytics_revenue(ctx: ToolContext, _days: int) -> dict[str, Any]:
    summary = await _admin_get(
        ctx.billing_client, "/api/v1/admin/subscriptions/summary", ctx, "revenue"
    )
    if summary.get("error"):
        return summary

    # monthlyRecurring is a LIST, one entry per currency, and is never summed across them.
    # Flattening it to a single number here would invent an exchange rate nobody chose.
    return {
        "monthly_recurring": summary.get("monthlyRecurring"),
        "active": summary.get("activeCount"),
        "trialing": summary.get("trialingCount"),
        "cancelling": summary.get("cancellingCount"),
        "note": (
            "monthly_recurring is one figure per currency and must be reported per currency. "
            "Do not add them together."
        ),
    }


async def _analytics_feedback(ctx: ToolContext, days: int) -> dict[str, Any]:
    from_at = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")

    summary = await _admin_get(
        ctx.translation_room_client,
        "/api/v1/admin/feedback/summary",
        ctx,
        "feedback",
        {"from": from_at},
    )
    if summary.get("error"):
        return summary

    return {
        "window_days": days,
        "responses": summary.get("responseCount"),
        "meetings_rated": summary.get("ratedMeetings"),
        "meetings_ended": summary.get("endedMeetings"),
        "response_rate": summary.get("responseRate"),
        # responseCount travels with every dimension deliberately: four of the five are optional,
        # so an average of 4.8 from three people must not be quoted as if it were from three
        # hundred.
        "dimensions": summary.get("dimensions"),
        "note": (
            "Each dimension's averageRating is over its own responseCount, not over all "
            "responses. A null averageRating means nobody rated it — not a score of zero. A null "
            "response_rate means no meeting ended in the window."
        ),
    }


async def _analytics_users(ctx: ToolContext, _days: int) -> dict[str, Any]:
    directory = await _admin_get(
        ctx.auth_client, "/api/v1/admin/users", ctx, "users", {"page": 1, "pageSize": 1}
    )
    if directory.get("error"):
        return directory
    return {"users_total": directory.get("total")}


async def _analytics_health(ctx: ToolContext, _days: int) -> dict[str, Any]:
    health = await _admin_get(ctx.workspace_client, "/api/v1/admin/platform-health", ctx, "health")
    if health.get("error"):
        return health

    # An unreachable metrics store is NOT an outage, and this is the one place the distinction
    # could be lost — a model handed empty lists would reasonably report that everything is down.
    if not health.get("monitoringAvailable"):
        return {
            "monitoring_available": False,
            "reason": health.get("monitoringUnavailableReason"),
            "note": (
                "Monitoring could not be read. This says NOTHING about whether the platform is "
                "healthy — report it as 'I cannot see the metrics', not as an outage."
            ),
        }

    targets = health.get("targets") or []
    workers = health.get("workers") or []
    dead_letters = health.get("deadLetters") or []

    return {
        "monitoring_available": True,
        "targets_down": [t.get("job") for t in targets if not t.get("isUp")],
        "targets_total": len(targets),
        "workers_at_zero": [w.get("worker") for w in workers if not w.get("replicas")],
        "firing_alerts": [
            {"name": a.get("name"), "severity": a.get("severity"), "summary": a.get("summary")}
            for a in (health.get("alerts") or [])
        ],
        "dead_letter_streams": [
            {"stream": d.get("stream"), "length": d.get("length")}
            for d in dead_letters
            if d.get("length")
        ],
        "stage_latency_p95_ms": health.get("stageLatencies"),
    }


_ANALYTICS_HANDLERS: dict[str, Callable[[ToolContext, int], Awaitable[dict[str, Any]]]] = {
    "overview": _analytics_overview,
    "revenue": _analytics_revenue,
    "feedback": _analytics_feedback,
    "users": _analytics_users,
    "health": _analytics_health,
}


async def _get_platform_analytics(ctx: ToolContext, arguments: dict[str, Any]) -> str:
    raw_reports = (arguments or {}).get("reports") or []
    if isinstance(raw_reports, str):
        raw_reports = [raw_reports]

    # Deduplicated, order preserved. A model that asks for "overview, overview, revenue" should
    # not pay for the first report twice.
    requested: list[str] = []
    unknown: list[str] = []
    for report in raw_reports:
        name = str(report).strip().lower()
        if name not in PLATFORM_ANALYTICS_REPORTS:
            if name and name not in unknown:
                unknown.append(name)
            continue
        if name not in requested:
            requested.append(name)

    if not requested:
        return json.dumps(
            {
                "error": "no_report_requested",
                "message": (f"Name at least one report: {', '.join(PLATFORM_ANALYTICS_REPORTS)}."),
                "unknown_reports": unknown,
            }
        )

    days = (arguments or {}).get("days")
    try:
        window = int(days) if days is not None else PLATFORM_ANALYTICS_DEFAULT_DAYS
    except (TypeError, ValueError):
        window = PLATFORM_ANALYTICS_DEFAULT_DAYS
    # Clamped rather than rejected: an out-of-range window is a model slip, not a reason to
    # answer nothing, and the response states which window was actually used.
    window = max(1, min(window, PLATFORM_ANALYTICS_MAX_DAYS))

    results = await asyncio.gather(
        *(_ANALYTICS_HANDLERS[name](ctx, window) for name in requested),
        return_exceptions=True,
    )

    payload: dict[str, Any] = {}
    for name, result in zip(requested, results, strict=True):
        if isinstance(result, BaseException):
            logger.exception("platform_analytics_report_failed", report=name, exc_info=result)
            payload[name] = {
                "error": "unavailable",
                "report": name,
                "message": "That report could not be produced.",
            }
        else:
            payload[name] = result

    if unknown:
        payload["unknown_reports"] = unknown

    return json.dumps(payload)


TOOLS: list[ChatTool] = [
    ChatTool(
        name="ask_user",
        description=(
            "Ask the user one or more multiple-choice questions and STOP. Use this the moment "
            "you need a detail you do not have — never guess a meeting's title, languages, type "
            "or time. The questions appear as a card the user picks from; their answer arrives "
            "as a normal message on your next turn. Ask everything you need in ONE call: three "
            "questions in one card is a form, three cards in a row is an interrogation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "Between one and four questions, asked together.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question, ending in a question mark.",
                            },
                            "header": {
                                "type": "string",
                                "description": "A 1-2 word chip label, e.g. 'Language' or 'Type'.",
                            },
                            "options": {
                                "type": "array",
                                "description": (
                                    "Two to four concrete choices. The user can always type "
                                    "something else, so do not add an 'Other' option."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {"type": "string"},
                                        "description": {
                                            "type": "string",
                                            "description": "One line on what this choice means.",
                                        },
                                    },
                                    "required": ["label"],
                                },
                            },
                            "multi_select": {
                                "type": "boolean",
                                "description": (
                                    "True when several answers may be picked at once, e.g. "
                                    "target languages."
                                ),
                            },
                        },
                        "required": ["question", "header", "options"],
                    },
                }
            },
            "required": ["questions"],
        },
        handler=_ask_user,
    ),
    ChatTool(
        name="create_meeting",
        description=(
            "Create a translation room in the current workspace. Call this ONLY once you know "
            "the title, meeting type, source language and target languages — if any of those is "
            "missing, call ask_user first. Supports a one-off time (scheduled_at) OR a repeating "
            "rule (recurrence_*), never both. Invited people receive an email, so only pass "
            "addresses the user actually gave you."
        ),
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "What the meeting is called."},
                "description": {
                    "type": "string",
                    "description": "Free-text purpose of the meeting.",
                },
                "agenda": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Agenda items in order. Appended to the description under an 'Agenda' "
                        "heading — the API has no separate agenda field."
                    ),
                },
                "translation_room_type": {
                    "type": "string",
                    "enum": list(MEETING_TYPES),
                    "description": "CHANNEL_MEETING suits most internal team meetings.",
                },
                "source_language": {
                    "type": "string",
                    "description": "Language the speakers will use, e.g. 'vi'.",
                },
                "target_languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Languages to translate into, e.g. ['en'].",
                },
                "scheduled_at": {
                    "type": "string",
                    "description": (
                        "ISO-8601 UTC start for a ONE-OFF meeting. Leave empty for a repeating one."
                    ),
                },
                "invited_emails": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email addresses to invite. Each one receives a real email.",
                },
                "recurrence_type": {
                    "type": "string",
                    "enum": list(RECURRENCE_CHOICES),
                    "description": (
                        "NONE for a meeting that happens once — which is most of them. Only "
                        "DAILY/WEEKLY/MONTHLY when the user asked for something repeating."
                    ),
                },
                "recurrence_start_time_local": {
                    "type": "string",
                    "description": "24-hour HH:mm in the user's own zone, e.g. '09:00'.",
                },
                "recurrence_time_zone": {
                    "type": "string",
                    "description": "IANA zone, e.g. 'Asia/Ho_Chi_Minh'.",
                },
                "recurrence_start_date_local": {
                    "type": "string",
                    "description": "yyyy-MM-dd for the first occurrence.",
                },
                "recurrence_end_date_local": {
                    "type": "string",
                    "description": "yyyy-MM-dd, inclusive. Omit for the server's default span.",
                },
                "max_participants": {
                    "type": "integer",
                    "description": "Seat cap. Omit to let the meeting type decide.",
                },
                "documents": {
                    "type": "array",
                    "description": (
                        "Workspace documents to put in front of attendees. Find them with "
                        "search_documents first and pass back its exact title and id — they are "
                        "rendered as markdown links in the meeting description, since a meeting "
                        "has no attachment field. Never invent an id."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "id": {"type": "string"},
                        },
                        "required": ["title", "id"],
                    },
                },
            },
            "required": ["title", "translation_room_type", "source_language", "target_languages"],
        },
        handler=_create_meeting,
    ),
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
                    "description": (
                        "Name or email fragment to search for. Leave blank to "
                        "list the first few members."
                    ),
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
                "query": {
                    "type": "string",
                    "description": "Optional keyword to filter meeting titles by.",
                },
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
        name="search_facts",
        description=(
            "List the knowledge facts recorded for this workspace — decisions, requirements, "
            "definitions, commitments, risks and references extracted from meetings and "
            "documents. Prefer this over semantic_search when the question names a category "
            '("what was decided", "what did we commit to", "what risks were raised"), '
            "because it returns everything tagged that way rather than whatever is nearest in "
            "embedding space."
        ),
        parameters={
            "type": "object",
            "properties": {
                "fact_category": {
                    "type": "string",
                    "enum": list(FACT_CATEGORIES),
                    "description": "Restrict to one category. Omit for all of them.",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional keyword to narrow by, matched against the fact "
                        "and its source title."
                    ),
                },
            },
            "required": [],
        },
        handler=_search_facts,
    ),
    ChatTool(
        name="semantic_search",
        description=(
            "Semantically search everything indexed for this workspace: documents, meeting "
            "transcripts, glossaries, AND the knowledge facts extracted from meetings — "
            "decisions, requirements, definitions, commitments, risks and references. Use this "
            "for conceptual questions, and whenever asked what was decided, agreed, required or "
            "flagged. May return no matches if nothing relevant has been indexed yet."
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
                    "description": (
                        "The room's id, from page context or a prior list_recent_meetings call."
                    ),
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
                    "description": (
                        "The meeting/room's id, from page context or a prior "
                        "list_recent_meetings call."
                    ),
                },
            },
            "required": ["meeting_id"],
        },
        handler=_get_transcript,
    ),
    ChatTool(
        name="search_documents",
        description=(
            "Find workspace documents by name, or list them when the user asks what "
            "documents exist. Use this whenever the user refers to a document by name "
            "rather than by id — it returns ids to pass to get_document. Matches on the "
            "document's name, not on its contents; use semantic_search to find a passage."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Part of the document's name. Omit or leave empty to list the "
                        "workspace's most recent documents."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "How many documents to return (default 5, max 20).",
                },
            },
            "required": [],
        },
        handler=_search_documents,
    ),
    ChatTool(
        name="get_document",
        description=(
            "Get metadata and a text excerpt for a specific workspace document. Use this "
            "when the user asks about the content of a document, or references one by "
            "name/id (e.g. from page context or an @mention). Call search_documents first "
            "if you only know the document's name."
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
    ChatTool(
        name="get_platform_analytics",
        description=(
            "Platform-wide figures for a WarpTalk PLATFORM ADMINISTRATOR: workspace and meeting "
            "counts, recurring revenue, product feedback ratings, sign-ups, and infrastructure "
            "health. Ask for every report you need in one call — they are fetched together. "
            "\n\n"
            "This reads the same admin API the /admin console does, using the asking user's own "
            "credentials, so it returns data ONLY for someone who holds the platform admin role. "
            "Anyone else gets not_authorized back, and you must tell them plainly that this is "
            "restricted — never estimate, extrapolate or invent a platform figure. A report that "
            "comes back with an error is a report you do not have."
            "\n\n"
            "These are PLATFORM totals across every tenant. For one workspace's own meetings use "
            "list_recent_meetings instead."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reports": {
                    "type": "array",
                    "description": (
                        "Which figures to fetch. Ask for everything the question needs at once."
                    ),
                    "items": {
                        "type": "string",
                        "enum": [
                            "overview",
                            "revenue",
                            "feedback",
                            "users",
                            "health",
                        ],
                    },
                    "minItems": 1,
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "Window in days for the feedback report, 1-366. Defaults to 30. Ignored "
                        "by the others, which are point-in-time."
                    ),
                },
            },
            "required": ["reports"],
        },
        handler=_get_platform_analytics,
    ),
]

TOOLS_BY_NAME: dict[str, ChatTool] = {t.name: t for t in TOOLS}
