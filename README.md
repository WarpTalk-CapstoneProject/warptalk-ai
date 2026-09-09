# WarpTalk AI Workers

Python-based AI processing workers for the WarpTalk real-time translation platform.

## Architecture

```
Audio Stream → STT Worker → Translation Worker → TTS Worker → Audio Output
                                    ↓
                            AI Assistant Worker
                                    ↓
                            Embedding Worker
```

All workers communicate via **Redis Streams** with overlapping audio chunks for low-latency streaming.

## Workers

| Worker | Purpose | Model |
|--------|---------|-------|
| `stt-worker` | Speech-to-Text | OpenAI `gpt-transcribe` |
| `translation-worker` | Real-time translation | OpenAI `gpt-5.4-nano` + Realtime fallback |
| `tts-worker` | Text-to-Speech + Voice Cloning | Cartesia `sonic-3.5` |
| `ai-assistant-worker` | Meeting summarization & Q&A | OpenAI `gpt-4.1` |
| `embedding-worker` | WarpBot RAG indexing | OpenAI `text-embedding-3-small` + Qdrant |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -e ".[dev,tts,embeddings]"

# 3. Copy environment config
cp .env.example .env
# Fill OPENAI_API_KEY and TTS_API_KEY.

# 4. Run a worker
python -m stt_worker
python -m translation_worker
python -m tts_worker
python -m ai_assistant_worker
python -m embedding_worker

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
├── embedding_worker/           # Text/RAG embedding indexing
│   ├── __main__.py
│   ├── worker.py
│   ├── providers.py
│   └── vector_store.py
├── tests/
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## Metrics exporter

`metrics_exporter` answers `/metrics` for Prometheus from the Redis Streams the pipeline runs
on. Consumer groups are **discovered with `XINFO GROUPS` on every scrape**: a new group shows up
in the next scrape without a code change, and a destroyed group disappears from it. What the
configuration decides is which streams are asked:

| Variable | Default | Meaning |
|----------|---------|---------|
| `METRICS_PORT` | `9108` | Listen port. |
| `METRICS_GLOBAL_STREAMS` | every global stream the platform publishes to (`shared.config.DEFAULT_GLOBAL_STREAMS`) | Comma-separated. Reported on every scrape whether or not the key exists — an absent one gets `redis_stream_groups{stream="..."} 0` rather than no series. Any further non-room stream found in Redis is reported as well. |
| `METRICS_EXPORT_PER_ROOM_STREAMS` | `false` | Also report the `<stream>:<roomId>` copies. Leave off in production: there is one set per meeting ever held, so this grows the series count without bound. Their groups are the same groups already counted on the global stream. |

## Runtime Requirements

The current architecture is API-backed and does not require a local GPU for STT,
translation, TTS, assistant, or embedding workers. Redis is required for streams,
and Qdrant is required for embedding storage.
