"""Embedding Search Worker — semantic-search RPC for the global chat assistant.

Pipeline:
    Redis Stream (embedding:search_requests) — own consumer group, independent of
    EmbeddingWorker's indexing pipeline (embedding:index_requests)
    → embed the query text with the same provider used for indexing
    → VectorStore.search()
    → reply is NOT published to a stream: it's RPUSHed as one JSON blob to a
      per-job `embedding:search_result:{job_id}` list key (with a short TTL), which
      the chat_worker BLPOPs with a timeout. A point-to-point RPC fits this call
      shape better than a second broadcast stream + consumer group.
"""

from __future__ import annotations

import json
from typing import Any

from embedding_worker.providers import EmbeddingProvider, create_embedding_provider
from embedding_worker.schemas import EmbeddingSearchRequest
from embedding_worker.vector_store import VectorStore, create_vector_store
from shared.base_worker import BaseWorker
from shared.config import EmbeddingSettings, VectorDbSettings

RESULT_TTL_SECONDS = 30

# WT-463 phase 0. What an unprivileged caller may not reach through semantic search.
#
# THE HOLE THIS PLUGS
#   The filter here was `workspace_id` + `ai_retrieval` and nothing else. `ai_retrieval` is a real
#   gate but a GLOBAL one — it decides whether the AI may use a resource at all, for everyone, and
#   has no per-subject dimension. Documents DO have a per-subject ACL
#   (WorkspaceDocumentAccessPolicy, enforced by DocumentAccessEvaluator on every REST read), and
#   this path never consulted it. So a member could ask WarpBot a question and receive passages
#   from a document they are not allowed to open.
#
# WHY BY SOURCE TYPE, WHICH IS BLUNT
#   Filtering by the document's actual ACL requires that ACL to be IN the vector payload, and
#   nothing indexed so far carries it — a per-subject filter would need a re-index of everything
#   already stored (phase 2). Excluding the class outright is lossy for members, and it is correct
#   in the only direction that matters here: it can hide something a member was entitled to, never
#   reveal something they were not.
#
#   `meeting_summary` is deliberately NOT in this list, though its artifacts have their own access
#   levels (HOST_ONLY / ALL_PARTICIPANTS). That is a second, real instance of the same bug; it is
#   recorded in WT-463 rather than fixed by widening a blunt instrument here.
#
# WHAT STAYS REACHABLE
#   transcript, glossary_term, global_glossary_term, meeting_summary — everything a workspace
#   member can already read through the product's own surfaces.
UNPRIVILEGED_EXCLUDED_SOURCES: dict[str, list[str]] = {"source_type": ["document"]}


class EmbeddingSearchWorker(BaseWorker):
    """Consumes semantic-search requests and replies via a per-job Redis list key."""

    worker_name = "embedding-search"
    input_stream = "embedding:search_requests"
    consumer_group = "embedding-search-workers"

    def __init__(
        self,
        embedding_settings: EmbeddingSettings | None = None,
        vector_settings: VectorDbSettings | None = None,
        provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_settings = embedding_settings or EmbeddingSettings()
        self.vector_settings = vector_settings or VectorDbSettings()
        self.provider = provider
        self.vector_store = vector_store

    async def load_model(self) -> None:
        if self.provider is None:
            self.provider = create_embedding_provider(self.embedding_settings)
        if self.vector_store is None:
            self.vector_store = create_vector_store(self.vector_settings)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = EmbeddingSearchRequest.from_redis(data)
        result_key = f"embedding:search_result:{request.job_id}"

        try:
            if not request.query.strip():
                await self._reply(result_key, {"matches": []})
                return

            provider = self._require_provider()
            vector_store = self._require_vector_store()
            vectors = await provider.embed_texts([request.query])
            if not vectors:
                await self._reply(result_key, {"matches": []})
                return

            matches = await vector_store.search(
                collection=request.collection_id,
                vector=vectors[0],
                top_k=request.top_k,
                filters={"workspace_id": request.workspace_id, "ai_retrieval": True},
                exclude=None if request.privileged else UNPRIVILEGED_EXCLUDED_SOURCES,
            )
            await self._reply(result_key, {"matches": matches})
        except Exception as exc:
            self.logger.exception("semantic_search_failed", job_id=request.job_id)
            await self._reply(result_key, {"matches": [], "error": str(exc)})

    async def _reply(self, result_key: str, payload: dict[str, Any]) -> None:
        await self.redis.redis.rpush(result_key, json.dumps(payload))
        await self.redis.expire(result_key, RESULT_TTL_SECONDS)

    def _require_provider(self) -> EmbeddingProvider:
        if self.provider is None:
            raise RuntimeError("Embedding provider is not loaded")
        return self.provider

    def _require_vector_store(self) -> VectorStore:
        if self.vector_store is None:
            raise RuntimeError("Vector store is not loaded")
        return self.vector_store
