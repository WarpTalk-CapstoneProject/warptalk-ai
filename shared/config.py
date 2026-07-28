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
    # XREADGROUP uses a blocking read (currently 2s for AI workers). Keep a
    # generous socket margin for Docker Desktop/Redis scheduling jitter so a
    # normal long-poll does not become a retry storm under load.
    socket_timeout: float = 15.0
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

    # VAD gating settings (used by ingress worker). 0.5 matches Silero/OpenAI's own
    # documented default and guidance ("a higher threshold ... might perform better in
    # noisy environments") — raised from an earlier, more permissive 0.3 that let
    # distant/muffled speech and ambiguous noise trip VAD too easily, reaching the STT
    # model (which has no confidence signal of its own to reject them — see
    # STTSettings.model) and getting hallucinated into a full sentence.
    vad_threshold: float = 0.5        # Speech detection threshold
    vad_pre_speech_ms: int = 300      # Pre-speech buffer to capture word onsets
    vad_silence_hangover_ms: int = 600  # 600ms hangover — faster turnaround
    vad_min_speech_ms: int = 500      # Minimum speech length to publish

    # Near-field energy gate (ingress worker only, see livekit_ingress_worker/near_field_gate.py)
    # — rejects a speech chunk whose peak amplitude is much quieter than this SAME
    # track's own established near-field baseline. VAD alone can't tell "close and
    # clear" from "far away and muffled" — both are speech-shaped. Pure energy/peak
    # math, no ML dependency, negligible latency.
    near_field_gate_enabled: bool = True
    # chunk peak must be >= 35% of this track's own established near-field peak
    near_field_gate_relative_floor: float = 0.35
    near_field_gate_min_baseline_chunks: int = 2
    near_field_gate_baseline_ema_alpha: float = 0.3


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
    # OpenAI Realtime's own input-side denoiser, applied before VAD/the model ever see
    # the audio — "far_field" is tuned for laptop/room mics (WarpTalk's actual usage),
    # "near_field" for headset mics, "off" disables it. Distinct from and upstream of
    # livekit_ingress_worker's Silero VAD / NearFieldGate, which run on raw PCM before
    # this session ever receives it.
    noise_reduction: str = "far_field"


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
    # 12h — unbounded before this; matches AudioRouteCacheService's own Redis TTL
    voice_clone_key_ttl_seconds: int = 43200
    # How many public Cartesia voices to offer per language — both for the per-speaker
    # hashed default (auto-diversity when nobody has cloned/chosen a voice) and for the
    # control-bar voice picker's option list.
    voice_catalog_size: int = 6
    # Cartesia's public library doesn't churn often — cache the per-language catalog
    # in Redis this long before re-fetching, to avoid a /voices call on every miss.
    voice_catalog_cache_ttl_seconds: int = 21600  # 6h


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
    # How many index-request messages (each = one document/transcript/glossary source, itself
    # made of one or more chunks) EmbeddingWorker processes at once. Unlike stt/tts/translation,
    # these jobs are independent of each other and I/O-bound (an OpenAI embed call + a Qdrant
    # upsert), so this is a real throughput win rather than a correctness risk — see
    # Keep embedding pressure bounded while sharing Redis with real-time audio.
    # Increase deliberately after measuring Redis timeout/error rates.
    concurrency: int = 2


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
    accumulator_flush_interval_seconds: int = 10
    max_credits_per_flush: int = 2000
    billing_event_dedupe_ttl_seconds: int = 86400


class VectorDbSettings(BaseSettings):
    """Vector database settings for text/RAG embeddings."""

    model_config = {"env_prefix": "VECTOR_DB_"}

    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str = ""
    distance_metric: str = "cosine"


class SecuritySettings(BaseSettings):
    """Security scanning worker settings."""

    model_config = {"env_prefix": "SECURITY_"}

    api_key: str = ""
    model: str = "gpt-4o-mini"
    max_tokens: int = 2000
    temperature: float = 0.0
    max_analyze_length: int = 20000
    result_ttl_seconds: int = 300



