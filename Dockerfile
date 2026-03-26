# ====================================================================
# WarpTalk AI Worker — Multi-stage Dockerfile
# Build: docker build --build-arg WORKER=stt_worker -t warptalk-stt .
# ====================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY shared/ shared/
COPY stt_worker/ stt_worker/
COPY translation_worker/ translation_worker/
COPY tts_worker/ tts_worker/
COPY ai_assistant_worker/ ai_assistant_worker/

# Worker to run (override via build arg or env)
ARG WORKER=stt_worker
ENV WORKER_MODULE=${WORKER}

CMD ["sh", "-c", "python -m ${WORKER_MODULE}"]
