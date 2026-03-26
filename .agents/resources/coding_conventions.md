# WarpTalk AI — Coding Conventions

## Python Standards

- **Version**: Python 3.11+
- **Style**: PEP 8, enforced via `ruff`
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public classes/functions
- **Async**: Use `asyncio` for I/O-bound operations (Redis, HTTP)

## Project Structure

```
each_worker/
├── __init__.py          # Worker package init
├── __main__.py          # Entry point (python -m worker_name)
├── consumer.py          # Redis Stream consumer logic
├── processor.py         # Core ML/AI processing
├── models.py            # Pydantic models for messages
└── config.py            # Worker-specific config
```

## Shared Module

```
shared/
├── config.py            # Global config (env vars, Redis, Qdrant)
├── redis_client.py      # Redis Streams consumer/producer helpers
├── audio_utils.py       # Audio format conversion, chunking
├── models.py            # Shared Pydantic models
└── logging.py           # Structured logging setup
```

## Dependencies

- Use `pyproject.toml` (PEP 621) for dependency management
- Pin major versions, allow patch updates
- GPU dependencies (torch, faster-whisper) in optional groups

## Testing

- **Framework**: pytest + pytest-asyncio
- **Coverage**: Target ≥80% for shared module
- **Naming**: `test_<module>_<function>_<scenario>.py`

## Environment Variables

- All config via environment variables (12-factor)
- `.env.example` as template — never commit `.env`
- Use `pydantic-settings` for typed config

## Error Handling

- Use structured logging (JSON format) for all workers
- Implement retry with exponential backoff for transient failures
- Dead-letter streams for messages that fail processing after max retries
- Health check endpoint per worker

## Message Format (Redis Streams)

All messages use JSON serialization with these standard fields:
```json
{
  "meeting_id": "uuid",
  "participant_id": "uuid",
  "timestamp": "ISO-8601",
  "sequence": 0,
  "data": { ... }
}
```
