"""Shared configuration loader using pydantic-settings.

Each worker has its own settings class loaded from environment variables
with a unique prefix (STT_, TRANSLATION_, TTS_, ASSISTANT_, EMBEDDING_).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


def resolve_openai_api_key(stage_api_key: str = "") -> str:
    """Prefer a stage-specific key, then fall back to shared OPENAI_API_KEY."""
    return stage_api_key or os.getenv("OPENAI_API_KEY", "")


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = {"env_prefix": "REDIS_"}

    url: str = "redis://localhost:6379"
    password: str = ""
    # livekit_ingress alone holds 1 connection per concurrent room (pubsub listener)
    # plus one per in-flight XADD from VAD-triggered chunk publishes — 10 was only
    # enough for a single active room at a time; opening a second meeting while one
    # is still live exhausted the pool ("MaxConnectionsError: Too many connections"),
    # which read as the whole AI pipeline going unresponsive.
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    stream_maxlen: int = 1000  # MAXLEN ~ for XADD trimming
    retry_max_attempts: int = 5
    retry_base_delay: float = 0.5


class LiveKitSettings(BaseSettings):
    """LiveKit Server connection settings."""

    model_config = {"env_prefix": "LIVEKIT_"}

    url: str = "ws://localhost:7880"
    api_key: str = "YOUR_LIVEKIT_API_KEY"
    api_secret: str = "YOUR_LIVEKIT_API_SECRET"


class WorkerSettings(BaseSettings):
    """Base worker settings shared by all workers."""

    model_config = {"env_prefix": ""}

    log_level: str = "INFO"
    chunk_duration_ms: int = 3000  # 3s chunk — balance between latency and STT context
    redis: RedisSettings = RedisSettings()
    livekit: LiveKitSettings = LiveKitSettings()

    # VAD gating settings (used by ingress worker)
    vad_threshold: float = 0.3        # Speech detection threshold
    vad_pre_speech_ms: int = 300      # Pre-speech buffer to capture word onsets
    vad_silence_hangover_ms: int = 600  # 600ms hangover — faster turnaround
    vad_min_speech_ms: int = 500      # Minimum speech length to publish


class STTSettings(BaseSettings):
    """Speech-to-Text worker settings."""

    model_config = {"env_prefix": "STT_"}

    provider: str = "openai"
    api_key: str = ""
    # gpt-4o-transcribe (not gpt-realtime-whisper): the realtime-whisper model is
    # dialogue-optimized (forced min temperature 0.6) and supports neither a `prompt`
    # field nor confidence signals, so it hallucinates more and can't be steered.
    # gpt-4o-transcribe runs over the same realtime transcription session but accepts a
    # free-text `prompt` for contextual biasing (glossary/key terms) — the research-
    # backed way to cut hallucination (arXiv 2410.18363). See model.py.
    model: str = "gpt-4o-transcribe"
    language: str = "auto"  # Auto-detect for code-switching (Vi + En)


class TranslationSettings(BaseSettings):
    """Translation worker settings."""

    model_config = {"env_prefix": "TRANSLATION_"}

    provider: str = "openai"  # 'openai' only — no fallback
    api_key: str = ""
    model: str = "gpt-4.1-mini"  # Best cost/quality for short-text translation
    max_tokens: int = 512
    # Fully deterministic, not just "near" — measured via the real pipeline benchmark
    # that 0.1 let identical repeated sentences translate to different (equally valid)
    # phrasings across separate calls, breaking tts_worker's text-based synthesis
    # cache: a real repeated meeting phrase missed the cache and paid a full ~1s
    # Cartesia call instead of a ~2ms cache hit. See translation_worker/translator.py.
    temperature: float = 0.0


class TTSSettings(BaseSettings):
    """Text-to-Speech worker settings."""

    model_config = {"env_prefix": "TTS_"}

    provider: str = "cartesia"
    api_key: str = ""
    # sonic-turbo does not support Vietnamese (confirmed via a live 400 "language_not_supported"
    # response) — sonic-3.5 is Cartesia's current model with Vietnamese in its language table.
    # This product's whole premise is cross-language dubbing, so the default must support the
    # target languages it actually needs, not just English.
    model: str = "sonic-3.5"
    sample_rate: int = 44100
    voice_clone_min_seconds: float = 10.0  # Buffer threshold before calling /voices/clone
    min_clone_chars: int = 8
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    voice_clone_key_ttl_seconds: int = 43200  # 12h — unbounded before this; matches AudioRouteCacheService's own Redis TTL


class AssistantSettings(BaseSettings):
    """AI Assistant worker settings."""

    model_config = {"env_prefix": "ASSISTANT_"}

    api_key: str = ""
    model: str = "gpt-4.1"
    max_tokens: int = 2048
    temperature: float = 0.3


class ChatAssistantSettings(BaseSettings):
    """Global AI assistant (chat-with-tools) worker settings.

    Distinct from AssistantSettings (per-meeting summarization) — this worker answers
    free-form questions in the global "Ask WarpTalk" widget and can call tools that read
    real workspace data from sibling .NET services.
    """

    model_config = {"env_prefix": "ASSISTANT_CHAT_"}

    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.4
    max_tool_iterations: int = 5
    # Flush a streamed chunk to Redis every N characters rather than per-token — keeps
    # Redis Stream / SignalR traffic bounded, matching the rest of the pipeline's coarse
    # buffered-unit convention (STT/TTS/AI-assistant results are never per-token either).
    chunk_flush_chars: int = 40
    workspace_service_url: str = "http://localhost:5106"
    transcript_service_url: str = "http://localhost:5103"
    translation_room_service_url: str = "http://localhost:5102"


class EmbeddingSettings(BaseSettings):
    """Knowledge embedding settings for WarpBot RAG."""

    model_config = {"env_prefix": "EMBEDDING_"}

    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    timeout_ms: int = 30000


class DatabaseSettings(BaseSettings):
    """Postgres connection settings for the billing settlement worker.

    No AI worker wrote to Postgres before billing_worker — everything else in
    warptalk-ai is Redis-only. Keep this settings class scoped to billing_worker
    only; do not reach for it from stt/translation/tts workers, which must stay
    on the real-time Redis Streams path.
    """

    model_config = {"env_prefix": "BILLING_DB_"}

    dsn: str = "postgresql://postgres:postgres@localhost:5432/warptalk"
    min_pool_size: int = 1
    max_pool_size: int = 5


class BillingSettings(BaseSettings):
    """Billing settlement worker settings."""

    model_config = {"env_prefix": "BILLING_"}

    database: DatabaseSettings = DatabaseSettings()
    # Subscription lookups (translation_room_id -> subscription_id) are cached for the
    # room's lifetime — refresh periodically in case a workspace's active subscription
    # changes mid-room (plan upgrade/downgrade).
    subscription_cache_ttl_seconds: int = 300


class VectorDbSettings(BaseSettings):
    """Vector database settings for text/RAG embeddings."""

    model_config = {"env_prefix": "VECTOR_DB_"}

    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str = ""
    distance_metric: str = "cosine"
