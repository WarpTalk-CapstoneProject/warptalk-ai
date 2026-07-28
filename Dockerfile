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
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1 AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Builder: install deps ----
FROM base AS builder

COPY pyproject.toml uv.lock ./
COPY shared/ shared/
COPY stt_worker/ stt_worker/
COPY translation_worker/ translation_worker/
COPY tts_worker/ tts_worker/
COPY ai_assistant_worker/ ai_assistant_worker/
COPY embedding_worker/ embedding_worker/
COPY billing_worker/ billing_worker/
COPY livekit_ingress_worker/ livekit_ingress_worker/
COPY security_worker/ security_worker/
COPY metrics_exporter/ metrics_exporter/
# Install exactly the dependency graph recorded in uv.lock.
RUN uv sync --frozen --no-dev --extra tts --extra embeddings --extra billing

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

# ---- LiveKit Ingress Worker (VAD-gated audio capture) ----
FROM base AS livekit-ingress

COPY pyproject.toml uv.lock ./
COPY shared/ shared/
COPY livekit_ingress_worker/ livekit_ingress_worker/
RUN uv sync --frozen --no-dev --extra ingress

RUN groupadd -r worker && useradd -r -g worker -d /app worker

# Pin and fetch the model during the image build. A production restart must not
# depend on GitHub or silently receive a different model from a mutable branch.
ENV TORCH_HOME=/app/.cache/torch
RUN mkdir -p /app/.cache \
    && uv run python -m livekit_ingress_worker.prefetch_model \
    && chown -R worker:worker /app
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "livekit_ingress_worker"]

# ---- Security Worker ----
FROM builder AS security

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "security_worker"]

# ---- Redis Streams / Worker Metrics Exporter ----
FROM base AS metrics

COPY pyproject.toml uv.lock ./
COPY shared/ shared/
COPY metrics_exporter/ metrics_exporter/
RUN uv sync --frozen --no-dev

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app \
    METRICS_PORT=9108
EXPOSE 9108
CMD ["python", "-m", "metrics_exporter"]
