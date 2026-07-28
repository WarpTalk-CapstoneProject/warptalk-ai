"""Tests for knowledge embedding providers used by WarpBot RAG."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from embedding_worker.providers import OpenAIEmbeddingProvider, create_embedding_provider
from shared.config import EmbeddingSettings


class TestOpenAIEmbeddingProvider:
    async def test_sends_model_and_dimensions_to_openai(self) -> None:
        settings = EmbeddingSettings(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )
        client = SimpleNamespace(
            embeddings=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        data=[
                            SimpleNamespace(embedding=[0.1, 0.2]),
                            SimpleNamespace(embedding=[0.3, 0.4]),
                        ]
                    )
                )
            )
        )
        provider = OpenAIEmbeddingProvider(settings=settings, client=client)

        vectors = await provider.embed_texts(["hello", "world"])

        assert vectors == [[0.1, 0.2], [0.3, 0.4]]
        client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-small",
            input=["hello", "world"],
            dimensions=1536,
            encoding_format="float",
        )

    async def test_requires_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = EmbeddingSettings(provider="openai", api_key="")
        provider = OpenAIEmbeddingProvider(settings=settings, client=None)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            await provider.embed_texts(["hello"])


def test_rejects_unimplemented_local_provider_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider: local"):
        create_embedding_provider(EmbeddingSettings(provider="local"))
