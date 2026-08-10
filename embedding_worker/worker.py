"""Embedding worker for WarpBot text/RAG indexing."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from itertools import islice
from typing import Any, TypeVar, cast

from embedding_worker.facts import (
    build_extraction_prompt,
    fact_payload_fields,
    parse_fact_response,
)
from embedding_worker.providers import (
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
)
from embedding_worker.schemas import EmbeddingIndexRequest, EmbeddingIndexResult
from embedding_worker.vector_store import VectorStore, create_vector_store
from shared.base_worker import BaseWorker
from shared.config import EmbeddingSettings, VectorDbSettings, resolve_openai_api_key

T = TypeVar("T")


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
        **kwargs: Any,
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
        request = EmbeddingIndexRequest.from_redis(cast(Any, data))
        assert self.provider is not None
        assert self.vector_store is not None

        # An explicit deletion (a term/document/segment archived or removed at the source —
        # see GlobalGlossaryService.ArchiveTermAsync/DeleteTermAsync and GlossaryService.
        # DeleteTermAsync) must actually remove the previously-indexed vector, not just get
        # silently blocked from re-indexing: deletion_state used to only ever gate new
        # upserts (via _block_reason below), so an archived/deleted term's vector stayed in
        # Qdrant forever with nothing left to ever clean it up. No embedding call needed here
        # — deleting is pure vector-store bookkeeping, keyed by chunk id.
        if request.deletion_state == "deleted":
            chunk_ids = [chunk.id for chunk in request.chunks]
            if chunk_ids:
                await self.vector_store.delete(collection=request.collection_id, ids=chunk_ids)
            await self._publish_result(request, status="deleted", chunks_indexed=len(chunk_ids))
            return

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
            facts = await self._extract_facts(chunks)

            payloads = [
                {
                    **chunk.metadata,
                    # The text itself. It was embedded and then thrown away, which left the
                    # vector store holding 1536 floats and no way for a person — or a
                    # review UI — to see what had been indexed about their own workspace.
                    "text": chunk.text,
                    **fact_payload_fields(facts[index]),
                    "workspace_id": request.workspace_id,
                    "source_type": request.source_type,
                    "source_id": request.source_id,
                    "chunk_id": chunk.id,
                    "ai_retrieval": request.ai_retrieval_allowed,
                    "retention_state": request.retention_state,
                    "deletion_state": request.deletion_state,
                }
                for index, chunk in enumerate(chunks)
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
        # deletion_state == "deleted" is handled explicitly in process() before this is ever
        # called (it deletes the vector rather than merely blocking a re-index) — by the time
        # execution reaches here, deletion_state is always "active".
        if not request.ai_retrieval_allowed:
            return "ai_retrieval_not_allowed"
        if request.retention_state != "active":
            return "source_not_active"
        if not request.external_llm_allowed and isinstance(self.provider, OpenAIEmbeddingProvider):
            return "external_llm_disabled_without_local_embedding_provider"
        return ""

    async def _extract_facts(self, chunks: list[Any]) -> list[dict[str, str] | None]:
        """One readable fact per chunk, or None where the chunk carries none.

        Failure here must never fail the index. Retrieval is the feature people depend on;
        the fact is a convenience laid on top of it, and an OpenAI outage that stopped
        documents being searchable in order to protect a table column would be the wrong
        trade every time. Every path returns a list of the right length, so the payload
        builder can index into it without checking.
        """
        if not self.embedding_settings.facts_enabled or not chunks:
            return [None] * len(chunks)

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=resolve_openai_api_key(self.embedding_settings.api_key))
        except Exception:
            self.logger.warning("fact_extraction_unavailable", chunks=len(chunks))
            return [None] * len(chunks)

        prompt = build_extraction_prompt()
        limit = self.embedding_settings.facts_max_input_chars

        async def extract_one(text: str) -> dict[str, str] | None:
            try:
                response = await client.chat.completions.create(
                    model=self.embedding_settings.facts_model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text[:limit]},
                    ],
                    temperature=0,
                )
                return parse_fact_response(response.choices[0].message.content)
            except Exception:
                # Per chunk, so one bad response does not cost the rest their facts.
                self.logger.warning("fact_extraction_failed", exc_info=True)
                return None

        return await asyncio.gather(*(extract_one(chunk.text) for chunk in chunks))

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


def _batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    iterator = iter(items)
    while batch := list(islice(iterator, max(1, batch_size))):
        yield batch
