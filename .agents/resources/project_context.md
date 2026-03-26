# WarpTalk AI Workers — Project Context

> **Repo**: `warptalk-ai` — Python GPU AI Workers for WarpTalk Capstone Project
> **Parent Project**: WarpTalk — AI Speech Translation Platform for Real-Time Multilingual Communication with Voice Cloning

---

## Overview

This repo contains the Python-based AI worker services that power WarpTalk's real-time speech translation pipeline. Each worker consumes/produces messages via **Redis Streams** and processes audio/text with GPU-accelerated models.

## Workers

| Worker | Purpose | Model | Input Stream | Output Stream |
|--------|---------|-------|-------------|--------------|
| `stt_worker` | Speech-to-Text | Fast-Whisper | `audio:chunks:{meetingId}` | `stt:results:{meetingId}` |
| `translation_worker` | Machine Translation | LLM | `stt:results:{meetingId}` | `translate:results:{meetingId}` |
| `tts_worker` | Text-to-Speech + Voice Cloning | XTTS v2 / Cartesia / ElevenLabs | `translate:results:{meetingId}` | `tts:results:{meetingId}` |
| `ai_assistant_worker` | Meeting Summarization & AI Assistant | LLM | Meeting events | Summaries, action items |

## Pipeline Architecture

```
Audio (2s chunks, overlapped)
  → STT Worker (0.5s) → Redis Stream
  → Translation Worker (0.4s) → Redis Stream
  → TTS/Voice Clone Worker (0.6s) → Client
  = ~1.5s first output latency (vs 8s batch)
```

## Tech Stack

- **Language**: Python 3.11+
- **Message Broker**: Redis Streams + Consumer Groups
- **GPU**: CUDA (NVIDIA), resource-isolated per worker
- **Package Manager**: pyproject.toml (PEP 621)
- **Vector DB**: Qdrant (voice embeddings, glossary, transcript context)
- **Communication with Backend**: Redis Streams (async), gRPC Protobuf (future)

## Key Design Decisions

1. **2-second streaming chunks** — overlapped processing for <2s latency
2. **GPU isolation** — each worker gets dedicated CUDA device
3. **Consumer Groups** — Redis Streams for backpressure and horizontal scaling
4. **Schema-per-service** — AI workers interact with `transcript` schema tables via gRPC to backend

## Related Resources

- [Database Schema](file:///Users/danchoingoinhinmuaroi/Projects/WarpTalk%20-%20Capstone%20Project/.agents/resources/database_schema.md) — Full 33-table schema
- [Implementation Plan](file:///Users/danchoingoinhinmuaroi/Projects/WarpTalk%20-%20Capstone%20Project/.agents/resources/implementation_plan.md) — Architecture & infrastructure
- [Capstone Register](file:///Users/danchoingoinhinmuaroi/Projects/WarpTalk%20-%20Capstone%20Project/.agents/resources/SP26SE016_WARPTALK_CAPSTONE_REGISTER.md) — Requirements & scope

## Folder Structure

```
warptalk-ai/
├── shared/              # Redis client, audio utils, config, Protobuf
├── stt_worker/          # Speech-to-Text (Fast-Whisper)
├── translation_worker/  # Machine Translation (LLM)
├── tts_worker/          # TTS + Voice Cloning
├── ai_assistant_worker/ # Meeting AI Assistant
├── tests/               # Pytest test suite
├── pyproject.toml       # Dependencies & project config
├── Dockerfile           # Multi-stage GPU build
└── .env.example         # Environment variables template
```
