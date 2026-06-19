"""Shared configuration loader using pydantic-settings.

Each worker has its own settings class loaded from environment variables
with a unique prefix (STT_, TRANSLATION_, TTS_, OPENAI_).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    model_config = {"env_prefix": "REDIS_"}

    url: str = "redis://localhost:6379"
    password: str = ""
    max_connections: int = 10
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
    chunk_duration_ms: int = 3000  # 3s chunk — mlx-whisper has ~4s fixed overhead, longer=better RTF
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

    model: str = "large-v3-turbo"  # MLX: best Vietnamese accuracy
    device: str = "cpu"  # Ignored by MLX — auto-selects Apple GPU
    compute_type: str = "int8"  # Ignored by MLX — uses own quantization
    language: str = "auto"  # Auto-detect for code-switching (Vi + En)
    beam_size: int = 1  # Greedy for lowest latency
    vad_filter: bool = False  # VAD handled by ingress worker, not Whisper


class TranslationSettings(BaseSettings):
    """Translation worker settings."""

    model_config = {"env_prefix": "TRANSLATION_"}

    model: str = "facebook/nllb-200-distilled-600M"
    device: str = "cuda"
    fallback_provider: str = "google"  # 'google' or 'none'
    max_length: int = 512


class TTSSettings(BaseSettings):
    """Text-to-Speech worker settings."""

    model_config = {"env_prefix": "TTS_"}

    anchor_provider: str = "edge"
    clone_provider: str = "xtts"
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cuda"
    embedding_min_seconds: float = 5.0  # Min audio for first voice embedding
    embedding_refine_seconds: float = 15.0  # Audio threshold for refined embedding
    default_voice: str = "en-US-AriaNeural"  # Edge-TTS default voice
    sample_rate: int = 24000  # XTTS v2 output sample rate
    blend_enabled: bool = True
    min_clone_chars: int = 8
    default_clone_strength: float = 0.6
    cache_enabled: bool = True
    cache_ttl_seconds: int = 900
    max_synthesis_ms: int = 6000


class AssistantSettings(BaseSettings):
    """AI Assistant worker settings."""

    model_config = {"env_prefix": "OPENAI_"}

    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.3


class EmbeddingSettings(BaseSettings):
    """Knowledge embedding settings for WarpBot RAG."""

    model_config = {"env_prefix": "EMBEDDING_"}

    provider: str = "openai"
    api_key: str = ""
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    timeout_ms: int = 30000


class VectorDbSettings(BaseSettings):
    """Vector database settings for text/RAG embeddings."""

    model_config = {"env_prefix": "VECTOR_DB_"}

    provider: str = "qdrant"
    url: str = "http://localhost:6333"
    api_key: str = ""
    distance_metric: str = "cosine"
