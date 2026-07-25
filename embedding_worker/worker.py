"""Embedding worker for WarpBot text/RAG indexing."""

from __future__ import annotations

from itertools import islice

from shared.base_worker import BaseWorker
from shared.config import EmbeddingSettings, VectorDbSettings

from embedding_worker.providers import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from embedding_worker.schemas import EmbeddingIndexRequest, EmbeddingIndexResult
from embedding_worker.vector_store import VectorStore, create_vector_store


class EmbeddingWorker(BaseWorker):
    """Consumes document/transcript/glossary chunks and stores text vectors."""

    worker_name = "embedding"
    input_stream = "embedding:index_requests"
    consumer_group = "embedding-workers"

    def __init__(
        self,
        embedding_settings: EmbeddingSettings | None = None,
        vector_settings: VectorDbSettings | None = None,
        provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.embedding_settings = embedding_settings or EmbeddingSettings()
        self.vector_settings = vector_settings or VectorDbSettings()
        self.provider = provider
        self.vector_store = vector_store
        # Per-instance override of BaseWorker's class-level default (1) — see
        # EmbeddingSettings.concurrency for why this worker specifically is safe to
        # parallelize. process() has no shared mutable state across calls (each call's
        # request/chunks/vectors/payloads are locals); self.provider/self.vector_store are
        # async clients designed for concurrent use.
        self.concurrency = self.embedding_settings.concurrency

    async def load_model(self) -> None:
        if self.provider is None:
            self.provider = create_embedding_provider(self.embedding_settings)
        if self.vector_store is None:
            self.vector_store = create_vector_store(self.vector_settings)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = EmbeddingIndexRequest.from_redis(data)
        block_reason = self._block_reason(request)
        if block_reason:
            await self._publish_result(request, status="blocked", reason=block_reason)
            return

        try:
            chunks = [chunk for chunk in request.chunks if chunk.text.strip()]
            if not chunks:
                await self._publish_result(request, status="indexed", chunks_indexed=0)
                return

            vectors: list[list[float]] = []
            for batch in _batched(chunks, self.embedding_settings.batch_size):
                vectors.extend(await self.provider.embed_texts([chunk.text for chunk in batch]))

            self._validate_dimensions(vectors)
            payloads = [
                {
                    **chunk.metadata,
                    "workspace_id": request.workspace_id,
                    "source_type": request.source_type,
                    "source_id": request.source_id,
                    "chunk_id": chunk.id,
                    "ai_retrieval": request.ai_retrieval_allowed,
                    "retention_state": request.retention_state,
                    "deletion_state": request.deletion_state,
                }
                for chunk in chunks
            ]
            await self.vector_store.upsert(
                collection=request.collection_id,
                ids=[chunk.id for chunk in chunks],
                vectors=vectors,
                payloads=payloads,
                dimensions=self.embedding_settings.dimensions,
            )
            await self._publish_result(
                request,
                status="indexed",
                chunks_indexed=len(chunks),
            )
        except Exception as exc:
            self.logger.exception("embedding_index_failed", job_id=request.job_id)
            await self._publish_result(request, status="failed", reason=str(exc))

    def _block_reason(self, request: EmbeddingIndexRequest) -> str:
        if not request.ai_retrieval_allowed:
            return "ai_retrieval_not_allowed"
        if request.deletion_state != "active":
            return "source_deleted"
        if request.retention_state != "active":
            return "source_not_active"
        if (
            not request.external_llm_allowed
            and isinstance(self.provider, OpenAIEmbeddingProvider)
        ):
            return "external_llm_disabled_without_local_embedding_provider"
        return ""

    def _validate_dimensions(self, vectors: list[list[float]]) -> None:
        for index, vector in enumerate(vectors):
            if len(vector) != self.embedding_settings.dimensions:
                raise ValueError(
                    "Embedding dimension mismatch: "
                    f"expected {self.embedding_settings.dimensions}, "
                    f"got {len(vector)} at index {index}"
                )

    async def _publish_result(
        self,
        request: EmbeddingIndexRequest,
        status: str,
        chunks_indexed: int = 0,
        reason: str = "",
    ) -> None:
        result = EmbeddingIndexResult(
            job_id=request.job_id,
            workspace_id=request.workspace_id,
            collection_id=request.collection_id,
            source_type=request.source_type,
            source_id=request.source_id,
            status=status,
            chunks_indexed=chunks_indexed,
            provider=self.embedding_settings.provider,
            model=self.embedding_settings.model,
            dimensions=self.embedding_settings.dimensions,
            reason=reason,
        )
        await self.publish("embedding:index_results", request.workspace_id, result.to_redis())


def _batched(items: list, batch_size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, max(1, batch_size))):
        yield batch
