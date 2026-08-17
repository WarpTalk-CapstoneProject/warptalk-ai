"""Vector store adapters for text/RAG embeddings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shared.config import VectorDbSettings


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
    ) -> list[dict[str, Any]]:
        """Return the top_k nearest payloads (with score) for `vector`.

        `filters` are ANDed exact matches. `exclude` maps a payload key to values that must NOT
        appear — WT-463 needs it to keep document-sourced chunks out of an unprivileged caller's
        results, which `filters` alone cannot express.

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
