"""Schemas for text/RAG embedding indexing jobs."""

from __future__ import annotations

import json
import time
import uuid
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
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> EmbeddingIndexRequest:
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
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "job_id": self.job_id,
            "workspace_id": self.workspace_id,
            "collection_id": self.collection_id,
            "query": self.query,
            "top_k": str(self.top_k),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> EmbeddingSearchRequest:
        d = _decode_dict(data)
        return cls(
            job_id=d.get("job_id", str(uuid.uuid4())),
            workspace_id=d["workspace_id"],
            collection_id=d["collection_id"],
            query=d.get("query", ""),
            top_k=int(d.get("top_k", "5")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


def _decode_dict(data: dict[bytes | str, bytes | str]) -> dict[str, str]:
    return {
        k.decode() if isinstance(k, bytes) else k: (
            v.decode() if isinstance(v, bytes) else str(v)
        )
        for k, v in data.items()
    }


def _bool_to_redis(value: bool) -> str:
    return "true" if value else "false"


def _redis_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}
