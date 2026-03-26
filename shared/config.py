"""Shared configuration loader using pydantic-settings."""

from pydantic_settings import BaseSettings


class RedisSettings(BaseSettings):
    """Redis connection settings."""

    url: str = "redis://localhost:6379"
    password: str = ""

    class Config:
        env_prefix = "REDIS_"


class WorkerSettings(BaseSettings):
    """Base worker settings."""

    log_level: str = "INFO"
    chunk_duration_ms: int = 2000
    redis: RedisSettings = RedisSettings()

    class Config:
        env_prefix = ""


settings = WorkerSettings()
