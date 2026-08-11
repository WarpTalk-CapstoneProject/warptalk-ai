"""Tests for the agent that turns workspace content into indexed facts.

The contract worth protecting here is the seam: this worker decides WHAT a fact is, and
EmbeddingWorker stores whatever `metadata` it is handed. If the fact stops travelling in
metadata, nothing crashes — the Knowledge page's Fact column just goes quietly empty.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from ai_assistant_worker.knowledge_fact_worker import KnowledgeFactWorker
from ai_assistant_worker.knowledge_facts import _clean
from shared.schemas import KnowledgeFactRequestMessage


class FakeExtractor:
    def __init__(self, facts: list[dict[str, str]] | None = None) -> None:
        self.facts = facts or []
        self.calls: list[tuple[str, str]] = []

    async def extract(self, title: str, text: str) -> list[dict[str, str]]:
        self.calls.append((title, text))
        return self.facts


def _worker(extractor: FakeExtractor) -> KnowledgeFactWorker:
    worker = KnowledgeFactWorker.__new__(KnowledgeFactWorker)
    worker.extractor = extractor
    worker.redis = AsyncMock()
    worker.logger = MagicMock()
    return worker


def _request(**overrides) -> KnowledgeFactRequestMessage:
    defaults = {
        "request_id": "req-1",
        "workspace_id": "ws-1",
        "source_type": "meeting_summary",
        "source_id": "room-1",
        "title": "Sprint planning",
        "text": "We decided to ship on the 14th.",
        "index_source_text": True,
    }
    defaults.update(overrides)
    return KnowledgeFactRequestMessage(**defaults)


def _published(worker: KnowledgeFactWorker) -> dict:
    stream, data = worker.redis.publish.await_args.args
    assert stream == "embedding:index_requests"
    return data


class TestKnowledgeFactWorker:
    async def test_a_fact_reaches_the_index_as_chunk_metadata(self) -> None:
        extractor = FakeExtractor(
            [
                {
                    "fact": "Release is on the 14th.",
                    "category": "decision",
                    "quote": "ship on the 14th",
                }
            ]
        )
        worker = _worker(extractor)

        await worker.process(b"msg-1", _request().to_redis())

        chunks = json.loads(_published(worker)["chunks_json"])
        fact_chunk = next(c for c in chunks if "fact" in c["metadata"])
        assert fact_chunk["metadata"]["fact"] == "Release is on the 14th."
        assert fact_chunk["metadata"]["fact_category"] == "decision"
        # The quote is what gets embedded — the words the meeting actually used.
        assert fact_chunk["text"] == "ship on the 14th"

    async def test_a_summary_is_indexed_alongside_its_facts(self) -> None:
        """A meeting summary has never been indexed by anyone else, so this is the only
        chance it gets. Without it the Knowledge page would list facts about a meeting whose
        summary is absent from the index WarpBot searches."""
        worker = _worker(FakeExtractor([]))

        await worker.process(b"msg-1", _request(index_source_text=True).to_redis())

        chunks = json.loads(_published(worker)["chunks_json"])
        assert [c["text"] for c in chunks] == ["We decided to ship on the 14th."]

    async def test_a_document_is_not_indexed_twice(self) -> None:
        """RedisEmbeddingIndexPublisher already chunked and indexed the document before this
        request was made. Re-indexing the whole text here would duplicate every chunk under a
        second set of ids, and both copies would come back from one search."""
        extractor = FakeExtractor(
            [{"fact": "Retention is 30 days.", "category": "requirement", "quote": "30 days"}]
        )
        worker = _worker(extractor)

        await worker.process(
            b"msg-1",
            _request(source_type="document", source_id="doc-1", index_source_text=False).to_redis(),
        )

        chunks = json.loads(_published(worker)["chunks_json"])
        assert len(chunks) == 1
        assert chunks[0]["metadata"]["fact_category"] == "requirement"

    async def test_external_llm_disabled_extracts_nothing(self) -> None:
        """The workspace's privacy flag is not advisory. Extraction is the step that would
        send content to OpenAI, so it must not run at all — not run-and-discard."""
        extractor = FakeExtractor(
            [{"fact": "Should never appear.", "category": "risk", "quote": "x"}]
        )
        worker = _worker(extractor)

        await worker.process(b"msg-1", _request(external_llm_allowed=False).to_redis())

        assert extractor.calls == []
        chunks = json.loads(_published(worker)["chunks_json"])
        assert all("fact" not in c["metadata"] for c in chunks)

    async def test_chunk_ids_are_stable_so_a_rerun_upserts(self) -> None:
        extractor = FakeExtractor([{"fact": "A.", "category": "decision", "quote": "a"}])

        first = _worker(extractor)
        await first.process(b"msg-1", _request().to_redis())
        second = _worker(extractor)
        await second.process(b"msg-2", _request().to_redis())

        ids_first = [c["id"] for c in json.loads(_published(first)["chunks_json"])]
        ids_second = [c["id"] for c in json.loads(_published(second)["chunks_json"])]
        assert ids_first == ids_second

    async def test_empty_content_publishes_nothing(self) -> None:
        worker = _worker(FakeExtractor([]))

        await worker.process(b"msg-1", _request(text="   ").to_redis())

        worker.redis.publish.assert_not_awaited()

    async def test_a_source_with_no_facts_and_nothing_to_index_publishes_nothing(self) -> None:
        """An index request carrying zero chunks is pure noise on the stream."""
        worker = _worker(FakeExtractor([]))

        await worker.process(
            b"msg-1", _request(source_type="document", index_source_text=False).to_redis()
        )

        worker.redis.publish.assert_not_awaited()

    async def test_the_collection_is_the_workspaces_own(self) -> None:
        """Facts must land in the same collection QdrantKnowledgeChunkReader scrolls —
        `workspace_{id}` — or they are indexed somewhere nobody reads."""
        worker = _worker(FakeExtractor([]))

        await worker.process(b"msg-1", _request().to_redis())

        assert _published(worker)["collection_id"] == "workspace_ws-1"


class TestFactCleaning:
    def test_an_invented_category_is_dropped(self) -> None:
        """A seventh category would be indexed under a label no Knowledge tab can match —
        present in the store, invisible in the UI, and impossible to explain."""
        assert _clean([{"fact": "A.", "category": "vibes", "quote": "a"}]) == []

    def test_duplicate_facts_collapse(self) -> None:
        cleaned = _clean(
            [
                {"fact": "Ship on the 14th.", "category": "decision", "quote": "a"},
                {"fact": "ship on the 14th.", "category": "decision", "quote": "b"},
            ]
        )
        assert len(cleaned) == 1

    def test_a_missing_quote_falls_back_to_the_fact(self) -> None:
        cleaned = _clean([{"fact": "Ship on the 14th.", "category": "decision"}])
        assert cleaned[0]["quote"] == "Ship on the 14th."

    def test_a_malformed_response_yields_no_facts(self) -> None:
        assert _clean(None) == []
        assert _clean("facts") == []
        assert _clean([{"category": "decision"}]) == []
