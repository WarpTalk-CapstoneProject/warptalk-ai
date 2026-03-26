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


class WorkerSettings(BaseSettings):
    """Base worker settings shared by all workers."""

    model_config = {"env_prefix": ""}

    log_level: str = "INFO"
    chunk_duration_ms: int = 1000  # 1s chunks for sub-2s latency
    redis: RedisSettings = RedisSettings()


class STTSettings(BaseSettings):
    """Speech-to-Text worker settings."""

    model_config = {"env_prefix": "STT_"}

    model: str = "medium"  # Whisper model size (medium for speed/accuracy balance)
    device: str = "cuda"
    compute_type: str = "int8"  # INT8 quantization for ~150ms inference
    language: str = "auto"  # Auto-detect or specify (e.g. 'en', 'vi')
    beam_size: int = 1  # Greedy for lowest latency, increase for accuracy
    vad_filter: bool = True  # Voice Activity Detection filter


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

    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "cuda"
    embedding_min_seconds: float = 5.0  # Min audio for first voice embedding
    embedding_refine_seconds: float = 15.0  # Audio threshold for refined embedding
    default_voice: str = "en-US-AriaNeural"  # Edge-TTS default voice
    sample_rate: int = 24000  # XTTS v2 output sample rate


class AssistantSettings(BaseSettings):
    """AI Assistant worker settings."""

    model_config = {"env_prefix": "OPENAI_"}

    api_key: str = ""
    model: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.3
