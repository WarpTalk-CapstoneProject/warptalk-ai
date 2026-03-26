# WarpTalk AI Workers

Python-based AI processing workers for the WarpTalk real-time translation platform.

## Architecture

```
Audio Stream → STT Worker → Translation Worker → TTS Worker → Audio Output
                                    ↓
                            AI Assistant Worker
```

All workers communicate via **Redis Streams** with 2-second overlapping audio chunks for low-latency streaming.

## Workers

| Worker | Purpose | Model |
|--------|---------|-------|
| `stt-worker` | Speech-to-Text | Whisper (OpenAI) |
| `translation-worker` | Real-time translation | NLLB / Helsinki-NLP |
| `tts-worker` | Text-to-Speech + Voice Cloning | Coqui TTS / XTTS |
| `ai-assistant-worker` | Meeting summarization & Q&A | LLaMA / GPT API |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Copy environment config
cp .env.example .env

# 4. Run a worker
python -m stt_worker
python -m translation_worker
python -m tts_worker
python -m ai_assistant_worker

# 5. Run tests
pytest
```

## Project Structure

```
warptalk-ai/
├── shared/                     # Shared utilities
│   ├── redis_client.py         # Redis Streams consumer/producer
│   ├── audio_utils.py          # Audio processing helpers
│   ├── config.py               # Environment config loader
│   └── logger.py               # Structured logging
├── stt_worker/                 # Speech-to-Text
│   ├── __main__.py
│   ├── worker.py
│   └── models.py
├── translation_worker/         # Translation
│   ├── __main__.py
│   ├── worker.py
│   └── models.py
├── tts_worker/                 # Text-to-Speech
│   ├── __main__.py
│   ├── worker.py
│   └── models.py
├── ai_assistant_worker/        # AI Summarization
│   ├── __main__.py
│   ├── worker.py
│   └── models.py
├── tests/
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## GPU Requirements

| Worker | VRAM | Recommended GPU |
|--------|------|-----------------|
| STT | 4 GB | RTX 3060+ |
| TTS/Voice Clone | 8 GB | RTX 3080+ |
| Translation | 2 GB | Any CUDA GPU |
| AI Assistant | 8 GB+ | RTX 4080+ (or API) |
