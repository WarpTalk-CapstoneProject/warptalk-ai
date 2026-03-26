# WarpTalk AI Workers — Implementation Plan

> **Target**: Sub-2s latency (~1.5s). **Strategy**: Overlapping pipeline + progressive voice cloning.

See full plan: [implementation_plan.md](file:///Users/danchoingoinhinmuaroi/Projects/WarpTalk%20-%20Capstone%20Project/warptalk-ai/.agents/resources/implementation_plan.md)

## Key Architecture Decisions

1. **1s chunks + overlapping pipeline** — process chunk N while recording N+1
2. **Progressive voice cloning** — Edge-TTS (0-5s) → XTTS v2 clone (5s+) → refined (15s+)
3. **XTTS v2 streaming mode** — `inference_stream()` for lower TTFB
4. **Whisper medium INT8** — ~150ms/chunk (trade accuracy 98%→95% for speed)
5. **NLLB distilled** — ~50ms/chunk translation

## Latency Budget

| Step | Duration |
|------|----------|
| Chunk buffer | 1000ms |
| STT | ~150ms |
| Translation | ~50ms |
| TTS (XTTS streaming) | ~300ms |
| Network overhead | ~15ms |
| **Total** | **~1.5s** |

## Phases

| Phase | Files | Status |
|-------|-------|--------|
| 1. Shared Infra | `schemas.py`, `base_worker.py`, `config.py`, `redis_client.py` | Pending |
| 2. STT | `stt_worker/worker.py`, `model.py` | Pending |
| 3. Translation | `translation_worker/worker.py`, `translator.py` | Pending |
| 4. TTS | `tts_worker/worker.py`, `synthesizer.py`, `embedding_extractor.py` | Pending |
| 5. AI Assistant | `ai_assistant_worker/worker.py`, `assistant.py` | Pending |
| 6. Docker/Tests | `Dockerfile`, `pyproject.toml`, `tests/` | Pending |
