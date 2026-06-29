"""Embedding providers for WarpBot knowledge retrieval.

Voice-clone embeddings are intentionally separate and remain in tts_worker.
This module only handles text/RAG embeddings for Qdrant-backed retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from shared.config import EmbeddingSettings, resolve_openai_api_key


class EmbeddingProvider(ABC):
    """Interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text embedding provider used by default for WarpBot RAG."""

    def __init__(
        self,
        settings: EmbeddingSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or EmbeddingSettings()
        self._client = client

    async def _get_client(self) -> Any:
        api_key = resolve_openai_api_key(self.settings.api_key)
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings")

        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=api_key)
        return self._client

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text for text in texts if text and text.strip()]
        if not clean_texts:
            return []

        client = await self._get_client()
        response = await client.embeddings.create(
            model=self.settings.model,
            input=clean_texts,
            dimensions=self.settings.dimensions,
            encoding_format="float",
        )
        return [list(item.embedding) for item in response.data]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Placeholder for future privacy-preserving local embeddings."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Local embedding provider is not configured")


def create_embedding_provider(settings: EmbeddingSettings | None = None) -> EmbeddingProvider:
    settings = settings or EmbeddingSettings()
    if settings.provider == "openai":
        return OpenAIEmbeddingProvider(settings)
    if settings.provider == "local":
        return LocalEmbeddingProvider()
    raise ValueError(f"Unsupported embedding provider: {settings.provider}")
