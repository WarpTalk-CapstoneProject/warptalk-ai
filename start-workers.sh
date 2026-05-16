#!/bin/bash
export REDIS_URL=redis://localhost:6379
.venv/bin/python3 -m stt_worker &
.venv/bin/python3 -m translation_worker &
.venv/bin/python3 -m tts_worker &
wait
wait
