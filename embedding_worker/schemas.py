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

    def to_redis(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "query": self.query,
            "top_k": str(self.top_k),
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
            # "false" on absence, matching the field's default: unknown is not privileged.
            privileged=_redis_to_bool(d.get("privileged", "false")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


def _decode_dict(data: Mapping[Any, Any]) -> dict[str, str]:
    return {
        k.decode() if isinstance(k, bytes) else k: (v.decode() if isinstance(v, bytes) else str(v))
        for k, v in data.items()
    }


def _bool_to_redis(value: bool) -> str:
    return "true" if value else "false"


def _redis_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}
