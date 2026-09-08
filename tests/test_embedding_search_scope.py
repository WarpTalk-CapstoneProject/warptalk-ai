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


# ── The per-MEETING half: transcripts and summaries ────────────────────────────────────────
#
# Excluding documents left the bigger door open. A transcript is the verbatim conversation and a
# meeting summary defaults to HOST_ONLY, and BOTH were returned to every member of the workspace:
# they were not in UNPRIVILEGED_EXCLUDED_SOURCES, and the search worker's own comment called that
# out as "a second, real instance of the same bug".
#
# They are narrowed rather than dropped because, unlike documents, their meeting id IS in the
# payload already — so no re-index was needed to start filtering on it.


@pytest.mark.asyncio
async def test_a_member_is_scoped_to_the_meetings_they_can_open() -> None:
    worker, store = _worker()

    await worker.process(
        b"msg-1",
        _request(privileged=False, allowed_room_ids_json='["room-a", "room-b"]'),
    )

    assert store.search_mock.await_args.kwargs["allowed_room_ids"] == ["room-a", "room-b"]


@pytest.mark.asyncio
async def test_an_owner_or_admin_is_not_scoped_to_any_meeting_list() -> None:
    """None, not []. An admin can open every meeting through the product; narrowing them here
    would hide from the assistant what the meetings list already hands them."""
    worker, store = _worker()

    await worker.process(b"msg-1", _request(privileged=True))

    assert store.search_mock.await_args.kwargs["allowed_room_ids"] is None


@pytest.mark.asyncio
async def test_a_member_who_can_open_nothing_matches_nothing() -> None:
    """[] and "" are different answers and the gate lives in the difference.

    An empty ARRAY is a real answer — a member with no meetings — and must match none. Reading it
    as "unrestricted" would hand that member every transcript in the workspace, which is the leak
    inverted rather than closed.
    """
    worker, store = _worker()

    await worker.process(b"msg-1", _request(privileged=False, allowed_room_ids_json="[]"))

    assert store.search_mock.await_args.kwargs["allowed_room_ids"] == []


def test_an_unreadable_allowlist_is_read_as_no_meetings() -> None:
    """Malformed is not unrestricted. If we cannot establish what the caller may open, the safe
    reading is nothing — it costs one search, where the other way costs a confidentiality
    boundary."""
    request = EmbeddingSearchRequest(
        job_id="j",
        workspace_id="w",
        collection_id="c",
        query="q",
        allowed_room_ids_json="{not json",
    )

    assert request.allowed_room_ids() == []


def test_the_allowlist_survives_the_redis_round_trip() -> None:
    # It crosses a Redis stream as a string. An array that serialises and parses back as None
    # would silently reopen the leak for every member.
    raw = _request(privileged=False, allowed_room_ids_json='["room-a"]')
    assert EmbeddingSearchRequest.from_redis(raw).allowed_room_ids() == ["room-a"]

    absent = _request(privileged=False)
    del absent["allowed_room_ids_json"]
    # Absent means "not scoped" — the shape every request had before this field existed. The
    # document exclusion still applies to those callers, so an old producer loses nothing and
    # gains nothing.
    assert EmbeddingSearchRequest.from_redis(absent).allowed_room_ids() is None
