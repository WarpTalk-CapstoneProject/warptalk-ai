#!/bin/bash
# Export all variables from .env
set -a
source .env
set +a
.venv/bin/python3 -m stt_worker &
.venv/bin/python3 -m translation_worker &
.venv/bin/python3 -m tts_worker &
.venv/bin/python3 -m livekit_ingress_worker &
wait
wait
