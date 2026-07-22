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
    ) -> list[dict[str, Any]]:
        """Return the top_k nearest payloads (with score) for `vector`.

        Must return an empty list — never raise — when the collection doesn't exist yet
        (nothing has been indexed into it), since callers treat "no results" as a normal,
        honest answer rather than a failure.
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
        if filters:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )

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
