"""Extract facts from workspace content and hand the result to the index.

WHERE THIS SITS
    Producers (translation-room when a meeting summary is written, workspace when a
    document finishes indexing) publish `knowledge:fact_requests`. This worker reads the
    content, asks the model for durable facts, and publishes ordinary
    `embedding:index_requests` — the same stream RedisEmbeddingIndexPublisher writes to.

WHY IT PUBLISHES INDEX REQUESTS RATHER THAN "FACT RESULTS"
    A fact only matters once it is retrievable, and the embedding worker is already the one
    thing that talks to Qdrant. Inventing a second write path would mean two places that can
    disagree about collection naming, policy flags, and deletion. So facts ride the existing
    contract: they are chunks whose `metadata` happens to carry `fact` and `fact_category`,
    which EmbeddingWorker spreads into the payload without knowing what they are.

WHY IT WRITES THE STREAM DIRECTLY
    BaseWorker.publish() fans out to `{prefix}:{id}` *and* `{prefix}`, which is right for
    per-meeting result streams and wrong here — it would leave an `embedding:index_requests:
    {source_id}` stream per source that no consumer group ever drains.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from ai_assistant_worker.knowledge_facts import KnowledgeFactExtractor
from shared.base_worker import BaseWorker
from shared.config import AssistantSettings, resolve_openai_api_key
from shared.schemas import KnowledgeFactRequestMessage

INDEX_REQUESTS_STREAM = "embedding:index_requests"

# The same namespace the whole chunk-id scheme lives in, so a fact's id can never collide
# with a document chunk's.
_CHUNK_NAMESPACE = uuid.NAMESPACE_URL


class KnowledgeFactWorker(BaseWorker):
    """Consumes `knowledge:fact_requests`, publishes `embedding:index_requests`."""

    worker_name = "knowledge-fact"
    input_stream = "knowledge:fact_requests"
    consumer_group = "knowledge-fact-workers"

    def __init__(
        self,
        assistant_settings: AssistantSettings | None = None,
        extractor: KnowledgeFactExtractor | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.assistant_settings = assistant_settings or AssistantSettings()
        self.extractor = extractor

    async def load_model(self) -> None:
        if self.extractor is None:
            self.extractor = KnowledgeFactExtractor(
                api_key=resolve_openai_api_key(self.assistant_settings.api_key),
                model=self.assistant_settings.model,
                max_tokens=self.assistant_settings.max_tokens,
                temperature=self.assistant_settings.temperature,
            )
            await self.extractor.load()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        request = KnowledgeFactRequestMessage.from_redis(data)

        if not request.text.strip():
            return

        facts: list[dict[str, str]] = []
        if request.external_llm_allowed:
            assert self.extractor is not None, "load_model() must run before process()"
            facts = await self.extractor.extract(request.title, request.text)
        else:
            # The workspace forbids sending its content to an external model. The content
            # itself may still be indexed below (embedding can run on a local provider);
            # what is skipped is only the part that would have left the deployment.
            self.logger.info(
                "knowledge_facts_skipped_external_llm_disabled",
                workspace_id=request.workspace_id,
                source_id=request.source_id,
            )

        chunks = self._build_chunks(request, facts)
        if not chunks:
            return

        await self.redis.publish(
            INDEX_REQUESTS_STREAM,
            {
                "job_id": f"facts:{request.source_type}:{request.source_id}",
                "workspace_id": request.workspace_id,
                "collection_id": f"workspace_{request.workspace_id}",
                "source_type": request.source_type,
                "source_id": request.source_id,
                "chunks_json": json.dumps(chunks, ensure_ascii=False),
                "external_llm_allowed": "true" if request.external_llm_allowed else "false",
                "ai_retrieval_allowed": "true",
                "retention_state": request.retention_state,
                "deletion_state": request.deletion_state,
                "timestamp_ms": str(request.timestamp_ms),
            },
        )

        self.logger.info(
            "knowledge_facts_indexed",
            workspace_id=request.workspace_id,
            source_type=request.source_type,
            source_id=request.source_id,
            facts=len(facts),
            chunks=len(chunks),
        )

    def _build_chunks(
        self, request: KnowledgeFactRequestMessage, facts: list[dict[str, str]]
    ) -> list[dict[str, Any]]:
        """One chunk for the content (when it is not already indexed), one per fact.

        Chunk ids are deterministic in the source and the position, so re-running extraction
        on the same source upserts the same Qdrant points instead of duplicating them.

        KNOWN LIMIT: if a later run yields FEWER facts than an earlier one, the surplus
        points from the earlier run stay behind — nothing here can address a point it no
        longer knows the id of. In practice a source is extracted once; a re-extraction that
        shrinks is the case to watch if fact counts ever look too high.
        """
        # `source_title` rather than `document_name`: a meeting is not a document, and a row
        # that borrows the document field would render as one on the Knowledge page. The
        # reader maps this to WorkspaceKnowledgeChunkDto.SourceTitle.
        provenance = {"source_title": request.title}

        chunks: list[dict[str, Any]] = []

        if request.index_source_text:
            chunks.append(
                {
                    "id": self._chunk_id(request, "content", 0),
                    "text": request.text,
                    "metadata": {**provenance, "chunk_index": 0},
                }
            )

        for index, fact in enumerate(facts):
            chunks.append(
                {
                    # The quote is what gets embedded: it is the language the content
                    # actually used, so it matches a user's phrasing better than the model's
                    # tidied-up restatement would.
                    "id": self._chunk_id(request, "fact", index),
                    "text": fact["quote"],
                    "metadata": {
                        **provenance,
                        "fact": fact["fact"],
                        "fact_category": fact["category"],
                    },
                }
            )

        return chunks

    @staticmethod
    def _chunk_id(request: KnowledgeFactRequestMessage, kind: str, index: int) -> str:
        return str(
            uuid.uuid5(
                _CHUNK_NAMESPACE,
                f"warptalk:{request.source_type}:{request.source_id}:{kind}:{index}",
            )
        )
