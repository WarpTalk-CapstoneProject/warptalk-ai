"""Billing settlement worker.

The only AI worker that talks to Postgres. Every other worker (stt/translation/tts)
stays purely on the real-time Redis Streams path. This one runs out-of-band,
consuming *:results streams after the fact and turning them into
subscription.usage_records + subscription.credit_transactions rows.

Before this existed, nothing in the codebase (Python or C#) wrote to those tables for
STT/TRANSLATION/AUDIO_DUBBING usage — see migration
017-15-07-2026-translation-cluster-finalize.sql for the schema this settles into.
"""
