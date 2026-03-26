# AI Pipeline Architecture

## Streaming Pipeline (2s Chunks)

```
BEFORE (batch):  Audio 10s → STT 3s → Translate 2s → TTS 3s = 8s latency

AFTER (streaming, 2s chunks):
  Chunk 1 → STT 0.5s → Translate 0.4s → TTS 0.6s = 1.5s first output
  Chunk 2 → overlapped with Chunk 1 processing
  Chunk 3 → ...
```

## Redis Streams

```
Stream Keys:
├── audio:chunks:{meetingId}         # Raw audio → STT
├── stt:results:{meetingId}          # STT → Translator
├── translate:results:{meetingId}    # Translator → TTS
├── tts:results:{meetingId}          # TTS → Client
├── events:notification              # → NotificationService
└── events:subscription              # → SubscriptionService
```

## GPU Resource Isolation (Docker)

```yaml
stt-worker:
  deploy:
    resources:
      reservations:
        devices: [{ driver: nvidia, count: 1, capabilities: [gpu] }]
      limits: { memory: 4G }
  environment:
    CUDA_VISIBLE_DEVICES: "0"

tts-worker:
  deploy:
    resources:
      limits: { memory: 8G }
  environment:
    CUDA_VISIBLE_DEVICES: "1"
```

## Auto-Scaling Triggers

| Service | Trigger | Action |
|---------|---------|--------|
| STT Worker | Stream lag > 100 msgs | +1 worker |
| TTS Worker | Stream lag > 50 msgs | +1 worker |

## Qdrant Vector DB Collections

| Collection | Dims | Metadata | Use Case |
|---|---|---|---|
| `voice_embeddings` | 256-512 | `user_id`, `language`, `duration`, `quality_score` | Voice cloning match |
| `glossary_embeddings` | 768 | `glossary_id`, `source_term`, `target_term`, `domain` | Semantic glossary search |
| `transcript_embeddings` | 768 | `meeting_id`, `segment_id`, `speaker_id`, `language` | AI Assistant context |

## Relevant Database Schemas

AI workers primarily interact with:

### `transcript` Schema (via gRPC to backend)
- `transcripts` — Meeting transcript metadata
- `transcript_segments` — Individual speech segments (partitioned by `created_at`)
- `transcript_translations` — Translated text per segment
- `transcript_corrections` — User corrections that trigger re-translation
- `glossaries` / `glossary_terms` — Enterprise terminology

### `meeting` Schema (read-only via gRPC)
- `meetings` — Meeting metadata, languages
- `meeting_participants` — Speaker info, language preferences
- `meeting_audio_routes` — Audio routing config
- `meeting_summaries` — AI-generated summaries
