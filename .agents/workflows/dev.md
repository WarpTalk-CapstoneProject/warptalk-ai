---
description: How to run AI workers locally for development
---

## Prerequisites

- Python 3.11+
- Redis running (via docker-compose in warptalk-infrastructure)
- CUDA toolkit (for GPU workers) or CPU fallback

## Steps

1. Create and activate virtual environment:
```bash
cd /Users/danchoingoinhinmuaroi/Projects/WarpTalk\ -\ Capstone\ Project/warptalk-ai
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"
```

3. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your local Redis/Qdrant connection details
```

4. Start Redis (if not already running):
```bash
cd ../warptalk-infrastructure && docker compose up redis -d
```

5. Run a specific worker:
```bash
# STT Worker
python -m stt_worker

# Translation Worker
python -m translation_worker

# TTS Worker
python -m tts_worker

# AI Assistant Worker
python -m ai_assistant_worker
```

6. Run tests:
// turbo
```bash
pytest tests/ -v
```

7. Lint & format:
// turbo
```bash
ruff check . && ruff format .
```
