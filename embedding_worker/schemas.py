"""Schemas for text/RAG embedding indexing jobs."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class EmbeddingChunk(BaseModel):
    """One text chunk to embed and store in the vector database."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingIndexRequest(BaseModel):
    """Backend → embedding worker request.

    Policy flags are evaluated before any provider call so privacy-sensitive
    workspaces never leak content to external embedding APIs.
    """

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str
    collection_id: str
    source_type: str
    source_id: str
    chunks: list[EmbeddingChunk]
    external_llm_allowed: bool = True
    ai_retrieval_allowed: bool = True
    retention_state: str = "active"
    deletion_state: str = "active"
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "chunks_json": json.dumps([chunk.model_dump() for chunk in self.chunks]),
            "external_llm_allowed": _bool_to_redis(self.external_llm_allowed),
            "ai_retrieval_allowed": _bool_to_redis(self.ai_retrieval_allowed),
            "retention_state": self.retention_state,
            "deletion_state": self.deletion_state,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> EmbeddingIndexRequest:
        d = _decode_dict(data)
        chunks = json.loads(d.get("chunks_json", "[]"))
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            workspace_id=d["workspace_id"],
            collection_id=d["collection_id"],
            source_type=d["source_type"],
            source_id=d["source_id"],
            chunks=[EmbeddingChunk(**chunk) for chunk in chunks],
            external_llm_allowed=_redis_to_bool(d.get("external_llm_allowed", "true")),
            ai_retrieval_allowed=_redis_to_bool(d.get("ai_retrieval_allowed", "true")),
            retention_state=d.get("retention_state", "active"),
            deletion_state=d.get("deletion_state", "active"),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class EmbeddingIndexResult(BaseModel):
    """Embedding worker → backend indexing result."""

    job_id: str
    workspace_id: str
    collection_id: str
    source_type: str
    source_id: str
    status: str  # indexed | blocked | failed | deleted
    chunks_indexed: int = 0
    provider: str = ""
    model: str = ""
    dimensions: int = 0
    reason: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "status": self.status,
            "chunks_indexed": str(self.chunks_indexed),
            "provider": self.provider,
            "model": self.model,
            "dimensions": str(self.dimensions),
            "reason": self.reason,
            "timestamp_ms": str(self.timestamp_ms),
        }


class EmbeddingSearchRequest(BaseModel):
    """Chat assistant → embedding worker semantic-search request.

    Delivered over the `embedding:search_requests` stream; the reply is NOT a stream
    (point-to-point, not broadcast) — it's a single JSON blob RPUSHed to a per-job
    `embedding:search_result:{job_id}` list key, which the requester BLPOPs with a
    timeout. That fits an RPC-shaped call better than a second consumer-group stream.
    """

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str
    collection_id: str
    query: str
    top_k: int = 5
    #: JSON array of meeting ids this caller may open, or "" for "do not scope".
    #:
    #: A STRING because Redis stream fields are strings; the worker parses it. Empty string and
    #: an empty array mean different things and both are real: "" is a privileged caller who is
    #: not scoped at all, [] is a member who can open no meetings and must therefore match none.
    allowed_room_ids_json: str = ""
    #: JSON array of document ids this caller may have the assistant answer from, or "" for
    #: "do not scope".
    #:
    #: Produced by GET /workspaces/{id}/documents/ai-retrievable, read AS THE CALLER, which
    #: evaluates the `ai_retrieval` permission through DocumentAccessEvaluator. The ACL therefore
    #: stays in the service that owns it and is never re-derived here — this end only filters on
    #: ids it was handed.
    #:
    #: Same three states as the room list, and the same reason: "" is a request that carried no
    #: allowlist, [] is a caller entitled to no documents and must match none, unreadable is [].
    allowed_document_ids_json: str = ""
    # WT-463 phase 0. Who is asking — until now, nobody.
    #
    # The search filtered on workspace_id and ai_retrieval alone. `ai_retrieval` is a real gate
    # but a GLOBAL one: it says whether the AI may use a resource at all, for everyone. It has no
    # per-subject dimension, so a document that is AI-retrievable was retrievable by every member
    # of the workspace regardless of who was allowed to open it — and documents carry a genuine
    # per-subject ACL (WorkspaceDocumentAccessPolicy) that the REST path enforces and this path
    # did not. The result: ask WarpBot, receive passages from a document you cannot open.
    #
    # `privileged` is deliberately coarse for phase 0. The honest fix is the resource's own ACL
    # travelling in the vector payload (phase 2) so the filter is per-subject; that needs a
    # re-index of everything already stored. This closes the bypass in the meantime, and the
    # field it introduces is the one phase 3 replaces with a resolved subject set.
    #
    # DEFAULT FALSE. An older or hand-built request that omits it is treated as unprivileged, so
    # the failure mode of a missing field is "sees less" rather than "sees everything".
    privileged: bool = False
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def allowed_room_ids(self) -> list[str] | None:
        """The meeting allowlist, or None when this request is not room-scoped.

        The two empty cases are NOT the same and the distinction is the whole gate: `""` is a
        privileged caller who should see every meeting, `"[]"` is a member who can open none and
        must therefore match none. Collapsing them either way is a silent failure — one way opens
        the leak this closes, the other hides every meeting from everybody — so the parsing lives
        here rather than being re-derived at each call site.
        """
        return _parse_id_allowlist(self.allowed_room_ids_json)

    def allowed_document_ids(self) -> list[str] | None:
        """The document allowlist, or None when this request carried none.

        Reports what was SENT. Whether a caller may go UNSCOPED is a policy question, and it is
        answered in EmbeddingSearchWorker.process — so that a request which simply omits the field
        cannot quietly come to mean "every document in the workspace".
        """
        return _parse_id_allowlist(self.allowed_document_ids_json)

    def to_redis(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "query": self.query,
            "top_k": str(self.top_k),
            "allowed_room_ids_json": self.allowed_room_ids_json,
            "allowed_document_ids_json": self.allowed_document_ids_json,
            "privileged": _bool_to_redis(self.privileged),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> EmbeddingSearchRequest:
        d = _decode_dict(data)
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            workspace_id=d["workspace_id"],
            collection_id=d["collection_id"],
            query=d.get("query", ""),
            top_k=int(d.get("top_k", "5")),
            # Absent means "not scoped", which is the shape every request had before this field
            # existed — an old producer keeps exactly the behaviour it had.
            allowed_room_ids_json=d.get("allowed_room_ids_json", ""),
            allowed_document_ids_json=d.get("allowed_document_ids_json", ""),
            # "false" on absence, matching the field's default: unknown is not privileged.
            privileged=_redis_to_bool(d.get("privileged", "false")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


def _parse_id_allowlist(raw_json: str) -> list[str] | None:
    """Parse one allowlist field: None for "carried no allowlist", a list otherwise.

    The two empty cases are NOT the same and the distinction is the whole gate: `""` means the
    request carried nothing, `"[]"` means it carried an empty allowlist — a caller entitled to
    nothing, which must therefore match nothing. Collapsing them either way is a silent failure,
    so the parsing lives in one place rather than being written out once per field.
    """
    raw = raw_json.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Unreadable is not "unrestricted". A malformed allowlist means we cannot establish what
        # this caller may see, and the safe reading of that is "nothing".
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _decode_dict(data: Mapping[Any, Any]) -> dict[str, str]:
    return {
        k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else str(v))
        for k, v in data.items()
    }


def _bool_to_redis(value: bool) -> str:
    return "true" if value else "false"


def _redis_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}
