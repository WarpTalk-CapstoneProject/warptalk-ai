"""WT-463: semantic search must not hand a member documents they cannot open.

THE BUG THESE PIN
    EmbeddingSearchWorker filtered on `workspace_id` and `ai_retrieval` alone. `ai_retrieval` is
    a real gate but a GLOBAL one — it says whether the AI may use a resource at all, for
    everyone, and carries no per-subject dimension. Documents DO have a per-subject ACL
    (WorkspaceDocumentAccessPolicy, enforced by DocumentAccessEvaluator on every REST read), and
    this path never consulted it. Ask WarpBot, receive a passage from a document you are not
    allowed to open.

    Every assertion below fails against that version: with no `privileged` concept, `exclude` was
    never passed at all.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from embedding_worker.schemas import EmbeddingSearchRequest
from embedding_worker.search_worker import UNPRIVILEGED_EXCLUDED_SOURCES, EmbeddingSearchWorker
from embedding_worker.vector_store import VectorStore


class FakeProvider:
    async def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]


class RecordingVectorStore(VectorStore):
    def __init__(self) -> None:
        self.search_mock = AsyncMock(return_value=[])
        self.upsert_mock = AsyncMock()
        self.delete_mock = AsyncMock()

    async def upsert(self, *args, **kwargs) -> None:
        await self.upsert_mock(*args, **kwargs)

    async def search(self, *args, **kwargs):
        return await self.search_mock(*args, **kwargs)

    async def delete(self, *args, **kwargs) -> None:
        await self.delete_mock(*args, **kwargs)


def _worker() -> tuple[EmbeddingSearchWorker, RecordingVectorStore]:
    worker = EmbeddingSearchWorker.__new__(EmbeddingSearchWorker)
    store = RecordingVectorStore()
    worker.provider = FakeProvider()
    worker.vector_store = store
    worker.redis = AsyncMock()
    worker.logger = AsyncMock()
    worker._reply = AsyncMock()
    return worker, store


def _request(**overrides) -> dict:
    defaults = {
        "job_id": "job-1",
        "workspace_id": "workspace-1",
        "collection_id": "workspace_workspace-1",
        "query": "what did we agree about pricing",
        "top_k": 5,
    }
    defaults.update(overrides)
    return EmbeddingSearchRequest(**defaults).to_redis()


@pytest.mark.asyncio
async def test_a_member_cannot_reach_document_chunks() -> None:
    worker, store = _worker()

    await worker.process(b"msg-1", _request(privileged=False))

    kwargs = store.search_mock.await_args.kwargs
    assert kwargs["exclude"] == UNPRIVILEGED_EXCLUDED_SOURCES
    assert "document" in kwargs["exclude"]["source_type"]


@pytest.mark.asyncio
async def test_an_owner_or_admin_still_reaches_everything() -> None:
    worker, store = _worker()

    await worker.process(b"msg-1", _request(privileged=True))

    assert store.search_mock.await_args.kwargs["exclude"] is None


@pytest.mark.asyncio
async def test_a_request_that_omits_privilege_is_treated_as_a_member() -> None:
    """Fail closed. An older producer, a replayed message or a hand-built request carries no
    `privileged` field, and the safe reading of "unknown" is the least privilege — not the most.
    """
    raw = _request(privileged=True)
    del raw["privileged"]

    worker, store = _worker()
    await worker.process(b"msg-1", raw)

    assert store.search_mock.await_args.kwargs["exclude"] == UNPRIVILEGED_EXCLUDED_SOURCES


@pytest.mark.asyncio
async def test_the_workspace_and_retrieval_gates_are_still_applied() -> None:
    """The new exclusion is added to the existing filter, not swapped in for it."""
    worker, store = _worker()

    await worker.process(b"msg-1", _request(privileged=True))

    filters = store.search_mock.await_args.kwargs["filters"]
    assert filters["workspace_id"] == "workspace-1"
    assert filters["ai_retrieval"] is True


def test_privilege_survives_the_redis_round_trip() -> None:
    # The flag crosses a Redis stream as a string; a bool that serialises to "True" and parses
    # back as False would disable the whole gate silently.
    for value in (True, False):
        restored = EmbeddingSearchRequest.from_redis(_request(privileged=value))
        assert restored.privileged is value
