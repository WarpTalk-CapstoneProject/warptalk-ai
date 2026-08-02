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
# Pinned by digest so a rebuild is reproducible. Bumping it is a security action, not
# housekeeping: the previous digest carried openssl 3.0.x, and the CVEs published against
# it (CVE-2026-45447 and the CVE-2026-283xx set) began failing the release Trivy gate the
# moment its vulnerability database picked them up — with the image itself unchanged.
FROM python:3.11.13-slim-bookworm@sha256:b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba AS base
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/app/.venv/bin:$PATH \
    UV_HTTP_TIMEOUT=120

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# setuptools ships its own vendored copies of jaraco.context and wheel, and the versions
# baked into the base image are the two remaining HIGH findings after the digest bump.
# Neither is used at runtime — the workers run from .venv — but Trivy scans the whole
# filesystem, so upgrading setuptools (which replaces what it vendors) is what actually
# clears them.
RUN pip install --no-cache-dir --upgrade \
        "setuptools>=80.9.0" \
        "wheel>=0.46.2" \
    && rm -rf /root/.cache/pip

WORKDIR /app

# ---- Builder: install deps ----
FROM base AS builder

COPY pyproject.toml uv.lock ./
COPY shared/ shared/
COPY stt_worker/ stt_worker/
COPY translation_worker/ translation_worker/
COPY tts_worker/ tts_worker/
COPY ai_assistant_worker/ ai_assistant_worker/
COPY suggestion_worker/ suggestion_worker/
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

# ---- Inline Transcript Suggestion Worker ----
FROM builder AS suggestion

RUN groupadd -r worker && useradd -r -g worker -d /app worker
USER worker

ENV PYTHONPATH=/app
CMD ["python", "-m", "suggestion_worker"]

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
