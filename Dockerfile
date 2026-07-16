# =============================================================
# WarpTalk AI Workers — Multi-stage Dockerfile
# =============================================================
# Base image: Python 3.11 runtime for API-backed AI workers
# Stages: base → builder → stt | translation | tts | assistant | embedding
#
# Build:
#   docker build --target stt -t warptalk-ai/stt .
#   docker build --target tts -t warptalk-ai/tts .
#
# Run:
#   docker run --env-file .env warptalk-ai/stt
# =============================================================

# ---- Base: Python + system deps ----
FROM python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Builder: install deps ----
FROM base AS builder

COPY pyproject.toml ./
COPY shared/ shared/
COPY stt_worker/ stt_worker/
COPY translation_worker/ translation_worker/
COPY tts_worker/ tts_worker/
COPY ai_assistant_worker/ ai_assistant_worker/
COPY embedding_worker/ embedding_worker/
COPY billing_worker/ billing_worker/
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install OpenAI-backed workers plus Cartesia TTS, Qdrant embedding, and asyncpg billing extras.
RUN python -m pip install --no-cache-dir -e ".[tts,embeddings,billing]"

# ---- STT Worker ----
FROM builder AS stt

# Non-root user for security
RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "stt_worker"]

# ---- Translation Worker ----
FROM builder AS translation

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "translation_worker"]

# ---- TTS Worker ----
FROM builder AS tts

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "tts_worker"]

# ---- AI Assistant Worker ----
FROM builder AS assistant

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "ai_assistant_worker"]

# ---- Embedding Worker ----
FROM builder AS embedding

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "embedding_worker"]

# ---- Billing Settlement Worker ----
FROM builder AS billing

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "billing_worker"]
