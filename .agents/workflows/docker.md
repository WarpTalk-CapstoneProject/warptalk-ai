---
description: How to build and run AI workers with Docker
---

## Build

1. Build the Docker image:
```bash
cd /Users/danchoingoinhinmuaroi/Projects/WarpTalk\ -\ Capstone\ Project/warptalk-ai
docker build -t warptalk-ai .
```

## Run Individual Workers

2. Run a specific worker with GPU support:
```bash
# STT Worker
docker run --gpus '"device=0"' --env-file .env -e WORKER_TYPE=stt warptalk-ai

# Translation Worker
docker run --env-file .env -e WORKER_TYPE=translation warptalk-ai

# TTS Worker (needs more GPU memory)
docker run --gpus '"device=1"' --env-file .env -e WORKER_TYPE=tts warptalk-ai

# AI Assistant Worker
docker run --env-file .env -e WORKER_TYPE=ai_assistant warptalk-ai
```

## Run via Infrastructure Compose

3. Use the full stack compose from warptalk-infrastructure:
```bash
cd ../warptalk-infrastructure
docker compose up stt-worker translation-worker tts-worker ai-assistant-worker
```

## Health Check

4. Verify workers are running:
```bash
docker ps --filter "name=warptalk" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```
