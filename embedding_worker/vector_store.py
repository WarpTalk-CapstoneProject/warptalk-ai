"""Vector store adapters for text/RAG embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shared.config import VectorDbSettings

#: Source types that belong to ONE MEETING, and the payload key holding that meeting's id.
#:
#: They differ because two producers chose differently and both are already in the index:
#: TranscriptRedisConsumerService sets `source_id` to the TRANSCRIPT id and puts the room id in
#: the chunk metadata as `translation_room_id`, while a meeting summary is published with
#: `source_id` set to the room id itself. Filtering has to know which key to read for which type;
#: guessing one would silently return nothing for the other, which is a failure that looks like
#: "the assistant found nothing" rather than like a bug.
ROOM_SCOPED_SOURCES: dict[str, str] = {
    "transcript": "translation_room_id",
    "meeting_summary": "source_id",
}

ROOM_SCOPED_SOURCE_KEYS: tuple[str, ...] = tuple(ROOM_SCOPED_SOURCES)


class VectorStore(ABC):
    """Interface for storing text embeddings."""

    @abstractmethod
    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        dimensions: int,
    ) -> None:
        """Create/update vectors and payloads in the configured store."""

    @abstractmethod
    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, list[str]] | None = None,
        allowed_room_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top_k nearest payloads (with score) for `vector`.

        `filters` are ANDed exact matches. `exclude` maps a payload key to values that must NOT
        appear — WT-463 needs it to keep document-sourced chunks out of an unprivileged caller's
        results, which `filters` alone cannot express.

        `allowed_room_ids` narrows the two MEETING-scoped source types (see
        `ROOM_SCOPED_SOURCES`) to meetings this caller can actually open. `None` means do not
        scope, which is what a privileged caller sends; an empty list is a real answer — a member
        who can open no meetings — and matches none of them.

        The exclusion belongs in the QUERY, not in a pass over the results: post-filtering still
        spends the top_k budget on points the caller may not see, so a workspace whose best
        matches are all restricted returns fewer rows the more restricted content it has — which
        is itself a signal about content the caller was not allowed to learn about.

        Must return an empty list — never raise — when the collection doesn't exist yet
        (nothing has been indexed into it), since callers treat "no results" as a normal,
        honest answer rather than a failure.
        """

    @abstractmethod
    async def delete(self, collection: str, ids: list[str]) -> None:
        """Remove points by id from `collection`.

        Must be a no-op — never raise — when the collection doesn't exist, or when an id
        isn't present in it: callers (EmbeddingWorker.process, on a deletion_state="deleted"
        request) fire this on every archive/delete of a source row, including ones that were
        never actually indexed (e.g. a draft term archived without ever being published).
        """


class QdrantVectorStore(VectorStore):
    """Qdrant vector store used by production WarpBot RAG."""

    def __init__(self, settings: VectorDbSettings | None = None, client: Any | None = None):
        self.settings = settings or VectorDbSettings()
        self._client = client

    async def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import AsyncQdrantClient

            self._client = AsyncQdrantClient(
                url=self.settings.url,
                api_key=self.settings.api_key or None,
            )
        return self._client

    async def upsert(
        self,
        collection: str,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
        dimensions: int,
    ) -> None:
        client = await self._get_client()
        await self._ensure_collection(client, collection, dimensions)

        from qdrant_client import models

        points = [
            models.PointStruct(id=ids[index], vector=vector, payload=payloads[index])
            for index, vector in enumerate(vectors)
        ]
        await client.upsert(collection_name=collection, points=points)

    async def search(
        self,
        collection: str,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        exclude: dict[str, list[str]] | None = None,
        allowed_room_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        client = await self._get_client()

        try:
            await client.get_collection(collection)
        except Exception:
            # Nothing has been indexed into this collection yet — an empty result is the
            # honest answer, not an error.
            return []

        from qdrant_client import models

        query_filter = None
        # list[Any] rather than list[FieldCondition]: Qdrant's Filter takes a union of condition
        # types and Python lists are invariant, so a precisely-typed list of one member of that
        # union is not assignable to it.
        must: list[Any] = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, value in (filters or {}).items()
        ]
        # WT-463: `must_not` rather than a filter on the results. Qdrant applies it while
        # searching, so the top_k the caller asked for is filled entirely with points they are
        # allowed to see.
        must_not: list[Any] = [
            models.FieldCondition(key=key, match=models.MatchValue(value=value))
            for key, values in (exclude or {}).items()
            for value in values
        ]
        # WT-463: the per-MEETING gate, for the two source types that belong to one meeting
        # rather than to the whole workspace.
        #
        # A transcript and a meeting summary are readable through the product only by people who
        # can open that meeting — `BuildListableRoomsQueryAsync` decides it, and it is a
        # per-participant question. `ai_retrieval` cannot express that: it is one global flag
        # meaning "the AI may use this at all", with no subject in it. So a workspace member could
        # ask WarpBot a question and receive the verbatim transcript of a meeting they were never
        # in, and the summary of one whose record was never shared with them.
        #
        # Filterable WITHOUT a re-index, which is what makes fixing it possible now rather than
        # after a migration: transcript points already carry `translation_room_id`
        # (TranscriptRedisConsumerService writes it into the chunk metadata and the embedding
        # worker spreads metadata into the payload), and meeting summaries are indexed with
        # `source_id` set to the room id.
        #
        # `None` means "do not scope" and is what a privileged caller sends. An EMPTY list is a
        # real answer — a member who can open no meetings — and correctly matches nothing.
        if allowed_room_ids is not None:
            must.append(
                models.Filter(
                    should=[
                        # Everything not tied to one meeting — glossary terms, workspace context —
                        # is unaffected. Written as "not one of the room-scoped types" so a new
                        # source type is reachable by default; the alternative silently hides any
                        # future type until somebody notices.
                        models.Filter(
                            must_not=[
                                models.FieldCondition(
                                    key="source_type",
                                    match=models.MatchAny(any=list(ROOM_SCOPED_SOURCE_KEYS)),
                                )
                            ]
                        ),
                        *[
                            models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="source_type",
                                        match=models.MatchValue(value=source_type),
                                    ),
                                    models.FieldCondition(
                                        key=room_key,
                                        match=models.MatchAny(any=allowed_room_ids),
                                    ),
                                ]
                            )
                            for source_type, room_key in ROOM_SCOPED_SOURCES.items()
                        ],
                    ]
                )
            )

        if must or must_not:
            query_filter = models.Filter(must=must or None, must_not=must_not or None)

        result = await client.query_points(
            collection_name=collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return [
            {"id": str(point.id), "score": point.score, "payload": point.payload or {}}
            for point in result.points
        ]

    async def delete(self, collection: str, ids: list[str]) -> None:
        if not ids:
            return

        client = await self._get_client()

        try:
            await client.get_collection(collection)
        except Exception:
            # Nothing has ever been indexed into this collection — deleting from it is
            # already a no-op, same honest-empty reasoning as search() above.
            return

        from qdrant_client import models

        await client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list[int | str](ids)),
        )

    async def _ensure_collection(self, client: Any, collection: str, dimensions: int) -> None:
        from qdrant_client import models

        try:
            await client.get_collection(collection)
            return
        except Exception:
            distance_name = self.settings.distance_metric.upper()
            distance = getattr(models.Distance, distance_name, models.Distance.COSINE)
            await client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=dimensions, distance=distance),
            )


def create_vector_store(settings: VectorDbSettings | None = None) -> VectorStore:
    settings = settings or VectorDbSettings()
    if settings.provider == "qdrant":
        return QdrantVectorStore(settings)
    raise ValueError(f"Unsupported vector DB provider: {settings.provider}")
