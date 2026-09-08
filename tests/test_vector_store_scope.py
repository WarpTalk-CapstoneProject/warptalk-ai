"""The Qdrant filter that keeps one meeting's words, and one document's, where they belong.

WHY THIS IS ITS OWN FILE
    The search worker's tests prove the allowlist REACHES the vector store. They cannot prove the
    filter built from it is the right shape, and that is where this fails silently: a filter that
    is too narrow returns nothing and reads as "the assistant found nothing", while one that is
    too wide returns somebody else's transcript and reads as success. Neither raises.

WHY THE TWO SOURCE TYPES READ DIFFERENT KEYS
    Two producers chose differently and both are already in the index.
    TranscriptRedisConsumerService sets `source_id` to the TRANSCRIPT id and carries the room id
    in the chunk metadata as `translation_room_id`; a meeting summary is published with
    `source_id` set to the room id itself. Filtering the wrong key for either type returns zero
    rows for it — which looks like an empty knowledge base, not a bug.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from embedding_worker.vector_store import (
    DOCUMENT_SCOPED_SOURCES,
    ROOM_SCOPED_SOURCES,
    QdrantVectorStore,
)


def _store_with_captured_filter() -> tuple[QdrantVectorStore, dict[str, Any]]:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    captured: dict[str, Any] = {}

    client = MagicMock()
    client.get_collection = AsyncMock(return_value=object())

    async def query_points(**kwargs):
        captured["query_filter"] = kwargs.get("query_filter")
        return MagicMock(points=[])

    client.query_points = query_points
    store._get_client = AsyncMock(return_value=client)  # type: ignore[method-assign]
    return store, captured


def _flatten(condition: Any) -> list[Any]:
    """Every FieldCondition anywhere in a nested Filter, so a test can ask what it constrains."""
    found: list[Any] = []
    for attribute in ("must", "should", "must_not"):
        for item in getattr(condition, attribute, None) or []:
            if hasattr(item, "key"):
                found.append(item)
            else:
                found.extend(_flatten(item))
    return found


@pytest.mark.asyncio
async def test_both_meeting_scoped_types_are_constrained_by_their_own_key() -> None:
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1", "ai_retrieval": True},
        allowed_room_ids=["room-a"],
    )

    keys = {condition.key for condition in _flatten(captured["query_filter"])}
    # The pairing is the assertion. Reading `source_id` for a transcript would match nothing,
    # because a transcript's source_id is its transcript row id.
    assert ROOM_SCOPED_SOURCES == {
        "transcript": "translation_room_id",
        "meeting_summary": "source_id",
    }
    assert "translation_room_id" in keys
    assert "source_id" in keys


@pytest.mark.asyncio
async def test_a_privileged_search_builds_no_room_constraint() -> None:
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_room_ids=None,
    )

    keys = {condition.key for condition in _flatten(captured["query_filter"])}
    assert "translation_room_id" not in keys
    assert keys == {"workspace_id"}


@pytest.mark.asyncio
async def test_an_empty_allowlist_still_constrains() -> None:
    """[] is a member who can open no meetings, and it must not fall through to unrestricted.

    The bug this guards is a falsy check: `if allowed_room_ids:` would treat the empty list as
    "not scoped" and hand that member every transcript in the workspace.
    """
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_room_ids=[],
    )

    keys = {condition.key for condition in _flatten(captured["query_filter"])}
    assert "translation_room_id" in keys


@pytest.mark.asyncio
async def test_sources_that_belong_to_the_workspace_are_not_narrowed() -> None:
    """A glossary term belongs to the workspace, not to a meeting, and must survive scoping.

    Expressed as a `should` branch that matches anything which is NOT one of the meeting-scoped
    types — so a source type added later is reachable by default rather than silently hidden.
    """
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_room_ids=["room-a"],
    )

    scope = [item for item in (captured["query_filter"].must or []) if not hasattr(item, "key")]
    assert len(scope) == 1, "the room scope is one nested Filter on the must list"
    branches = scope[0].should or []
    # One escape hatch for non-meeting sources, plus one branch per meeting-scoped type.
    assert len(branches) == 1 + len(ROOM_SCOPED_SOURCES)
    assert any(getattr(branch, "must_not", None) for branch in branches)


# ── The per-DOCUMENT dimension ─────────────────────────────────────────────────────────────
#
# Documents used to be dropped wholesale for an unprivileged caller, because their per-subject
# ACL could not be consulted from the worker. They are narrowed now, from an allowlist the
# workspace service resolves through the `ai_retrieval` permission — and no re-index was needed,
# because `source_id` has carried the document id since the first document was indexed.


@pytest.mark.asyncio
async def test_documents_are_constrained_by_their_own_key() -> None:
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1", "ai_retrieval": True},
        allowed_document_ids=["doc-a"],
    )

    keys = {condition.key for condition in _flatten(captured["query_filter"])}
    assert DOCUMENT_SCOPED_SOURCES == {"document": "source_id"}
    assert "source_id" in keys


@pytest.mark.asyncio
async def test_an_empty_document_allowlist_still_constrains() -> None:
    """[] is a caller entitled to no documents, and must not fall through to unrestricted.

    The bug this guards is the same falsy check the room list has: `if allowed_document_ids:`
    would read the empty list as "not scoped" and hand that caller every document in the
    workspace — the leak inverted rather than closed.
    """
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_document_ids=[],
    )

    keys = {condition.key for condition in _flatten(captured["query_filter"])}
    assert "source_id" in keys


@pytest.mark.asyncio
async def test_scoping_documents_does_not_narrow_transcripts() -> None:
    """The two dimensions are independent, and the escape hatch has to know which.

    The "not scoped" branch lists the types scoped by THIS request. If it listed every type that
    could ever be scoped, then scoping documents alone would also hide transcripts from a caller
    nobody meant to restrict — a silent, total loss of meeting answers.
    """
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_document_ids=["doc-a"],
        allowed_room_ids=None,
    )

    scope = [item for item in (captured["query_filter"].must or []) if not hasattr(item, "key")]
    assert len(scope) == 1
    branches = scope[0].should or []
    assert len(branches) == 1 + len(DOCUMENT_SCOPED_SOURCES)

    escape_hatch = branches[0].must_not[0]
    assert escape_hatch.match.any == ["document"]


@pytest.mark.asyncio
async def test_both_dimensions_scope_side_by_side() -> None:
    """One branch per scoped type, plus one escape hatch — for the ordinary member's search."""
    store, captured = _store_with_captured_filter()

    await store.search(
        collection="workspace_1",
        vector=[0.1, 0.2],
        top_k=5,
        filters={"workspace_id": "w1"},
        allowed_room_ids=["room-a"],
        allowed_document_ids=["doc-a"],
    )

    scope = [item for item in (captured["query_filter"].must or []) if not hasattr(item, "key")]
    branches = scope[0].should or []
    assert len(branches) == 1 + len(ROOM_SCOPED_SOURCES) + len(DOCUMENT_SCOPED_SOURCES)

    escape_hatch = branches[0].must_not[0]
    assert set(escape_hatch.match.any) == set(ROOM_SCOPED_SOURCES) | set(DOCUMENT_SCOPED_SOURCES)
