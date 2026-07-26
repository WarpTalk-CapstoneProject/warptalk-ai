"""Tests for embedding indexing worker policy and vector validation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from embedding_worker.providers import EmbeddingProvider, OpenAIEmbeddingProvider
from embedding_worker.schemas import EmbeddingChunk, EmbeddingIndexRequest
from embedding_worker.vector_store import VectorStore
from embedding_worker.worker import EmbeddingWorker
from shared.config import EmbeddingSettings


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vectors: list[list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self.vectors[: len(texts)]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.upsert_mock = AsyncMock()
        self.search_mock = AsyncMock(return_value=[])
        self.delete_mock = AsyncMock()

    async def upsert(self, *args, **kwargs) -> None:
        await self.upsert_mock(*args, **kwargs)

    async def search(self, *args, **kwargs):
        return await self.search_mock(*args, **kwargs)

    async def delete(self, *args, **kwargs) -> None:
        await self.delete_mock(*args, **kwargs)


def _request(**overrides) -> EmbeddingIndexRequest:
    defaults = {
        "job_id": "job-1",
        "workspace_id": "workspace-1",
        "collection_id": "workspace-1-rag",
        "source_type": "transcript",
        "source_id": "meeting-1",
        "chunks": [
            EmbeddingChunk(id="chunk-1", text="hello world", metadata={"order": 1}),
        ],
    }
    defaults.update(overrides)
    return EmbeddingIndexRequest(**defaults)


class TestEmbeddingWorker:
    async def test_blocks_openai_when_external_llm_disabled(self) -> None:
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(
            provider="openai",
            api_key="test-key",
            dimensions=2,
        )
        worker.provider = OpenAIEmbeddingProvider(
            settings=worker.embedding_settings,
            client=object(),
        )
        worker.vector_store = FakeVectorStore()
        worker.publish = AsyncMock()

        await worker.process(
            b"msg-1",
            _request(external_llm_allowed=False).to_redis(),
        )

        worker.vector_store.upsert_mock.assert_not_awaited()
        result = worker.publish.call_args.args[2]
        assert result["status"] == "blocked"
        assert result["reason"] == "external_llm_disabled_without_local_embedding_provider"

    async def test_rejects_dimension_mismatch_before_upsert(self) -> None:
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(
            provider="openai",
            api_key="test-key",
            dimensions=3,
        )
        worker.provider = FakeEmbeddingProvider(vectors=[[0.1, 0.2]])
        worker.vector_store = FakeVectorStore()
        worker.logger = MagicMock()
        worker.publish = AsyncMock()

        await worker.process(b"msg-1", _request().to_redis())

        worker.vector_store.upsert_mock.assert_not_awaited()
        result = worker.publish.call_args.args[2]
        assert result["status"] == "failed"
        assert "Embedding dimension mismatch" in result["reason"]

    async def test_upserts_vectors_with_policy_payload(self) -> None:
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(
            provider="openai",
            api_key="test-key",
            dimensions=2,
            batch_size=8,
        )
        worker.provider = FakeEmbeddingProvider(vectors=[[0.1, 0.2]])
        worker.vector_store = FakeVectorStore()
        worker.publish = AsyncMock()

        await worker.process(b"msg-1", _request().to_redis())

        worker.vector_store.upsert_mock.assert_awaited_once()
        kwargs = worker.vector_store.upsert_mock.await_args.kwargs
        assert kwargs["collection"] == "workspace-1-rag"
        assert kwargs["ids"] == ["chunk-1"]
        assert kwargs["vectors"] == [[0.1, 0.2]]
        assert kwargs["payloads"][0]["workspace_id"] == "workspace-1"
        assert kwargs["payloads"][0]["source_type"] == "transcript"
        assert kwargs["payloads"][0]["ai_retrieval"] is True
        result = worker.publish.call_args.args[2]
        assert result["status"] == "indexed"
        assert result["chunks_indexed"] == "1"

    async def test_deletion_state_deletes_vector_instead_of_indexing(self) -> None:
        """A deletion_state="deleted" request (e.g. GlobalGlossaryService.ArchiveTermAsync/
        DeleteTermAsync) must actually remove the previously-indexed vector — not just get
        silently blocked, which used to leave it in Qdrant forever with no other path to
        ever clean it up.
        """
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(provider="openai", api_key="test-key", dimensions=2)
        worker.provider = FakeEmbeddingProvider(vectors=[[0.1, 0.2]])
        worker.vector_store = FakeVectorStore()
        worker.publish = AsyncMock()

        await worker.process(b"msg-1", _request(deletion_state="deleted").to_redis())

        worker.vector_store.delete_mock.assert_awaited_once_with(collection="workspace-1-rag", ids=["chunk-1"])
        worker.vector_store.upsert_mock.assert_not_awaited()
        worker.provider.calls == []
        result = worker.publish.call_args.args[2]
        assert result["status"] == "deleted"
        assert result["chunks_indexed"] == "1"

    async def test_deletion_state_skips_delete_call_when_no_chunk_ids(self) -> None:
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(provider="openai", api_key="test-key", dimensions=2)
        worker.provider = FakeEmbeddingProvider(vectors=[])
        worker.vector_store = FakeVectorStore()
        worker.publish = AsyncMock()

        await worker.process(b"msg-1", _request(deletion_state="deleted", chunks=[]).to_redis())

        worker.vector_store.delete_mock.assert_not_awaited()
        result = worker.publish.call_args.args[2]
        assert result["status"] == "deleted"
        assert result["chunks_indexed"] == "0"

    async def test_deletion_state_takes_priority_over_other_block_reasons(self) -> None:
        """A deletion request must delete even when ai_retrieval_allowed/external_llm_allowed
        would otherwise have blocked an index request — those policy flags govern whether
        NEW content may be embedded, not whether stale vectors may be cleaned up.
        """
        worker = EmbeddingWorker.__new__(EmbeddingWorker)
        worker.embedding_settings = EmbeddingSettings(provider="openai", api_key="test-key", dimensions=2)
        worker.provider = OpenAIEmbeddingProvider(settings=worker.embedding_settings, client=object())
        worker.vector_store = FakeVectorStore()
        worker.publish = AsyncMock()

        await worker.process(
            b"msg-1",
            _request(deletion_state="deleted", ai_retrieval_allowed=False, external_llm_allowed=False).to_redis(),
        )

        worker.vector_store.delete_mock.assert_awaited_once()
        result = worker.publish.call_args.args[2]
        assert result["status"] == "deleted"
