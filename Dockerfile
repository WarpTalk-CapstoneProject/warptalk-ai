# =============================================================
# WarpTalk AI Workers — Multi-stage Dockerfile
# =============================================================
# Base image: NVIDIA CUDA 12.1 + Python 3.11 (GPU inference)
# Stages: base → builder → stt | translation | tts | assistant
#
# Build:
#   docker build --target stt -t warptalk-ai/stt .
#   docker build --target tts -t warptalk-ai/tts .
#
# Run:
#   docker run --gpus all --env-file .env warptalk-ai/stt
# =============================================================

# ---- Base: CUDA + Python + system deps ----
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ---- Builder: install deps ----
FROM base AS builder

COPY pyproject.toml ./
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install core dependencies
RUN python -m pip install --no-cache-dir -e "."

# Install all optional deps into a venv-like layer
RUN python -m pip install --no-cache-dir \
    "faster-whisper>=1.0" \
    "sentencepiece>=0.1.99" \
    "deep-translator>=1.11.4" \
    "TTS>=0.22" \
    "edge-tts>=6.1" \
    "openai>=1.6"

# Copy source
COPY shared/ shared/
COPY stt_worker/ stt_worker/
COPY translation_worker/ translation_worker/
COPY tts_worker/ tts_worker/
COPY ai_assistant_worker/ ai_assistant_worker/

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
