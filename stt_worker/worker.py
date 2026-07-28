"""STT Worker — Consumes audio chunks, produces text segments.

Pipeline:
    Redis Stream (audio:chunks:{meetingId})
    → OpenAI gpt-4o-mini-transcribe (async API call)
    → Redis Stream (stt:results:{meetingId})
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping
from typing import Any

from shared.base_worker import BaseWorker
from shared.config import STTSettings, resolve_openai_api_key
from shared.schemas import AudioChunkMessage, STTResultMessage
from stt_worker.model import OpenAISTT, TranscribedSegment, _normalize_language


def _decode_field(data: Mapping[Any, Any], key: str) -> str:
    raw = data.get(key)
    if raw is None:
        raw = data.get(key.encode())
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


def _extract_speaker_key(
    data: Mapping[Any, Any],
) -> tuple[str, str]:
    """Cheap (meeting_id, speaker_id) extraction from a raw audio:chunks Redis entry.

    Deliberately avoids AudioChunkMessage.from_redis(), which base64-decodes the (often
    large) audio_data field — that decode would otherwise happen twice per chunk once
    _consume_loop needs this key up front (to pick which speaker's lock to acquire)
    ahead of process()'s own full parse.
    """
    meeting_id = _decode_field(data, "meeting_id") or _decode_field(data, "translation_room_id")
    speaker_id = _decode_field(data, "speaker_id")
    return meeting_id, speaker_id


# Generic, domain-neutral biasing prompt fed to every STT session even when the room has
# no glossary configured. transcription.prompt (gpt-4o-transcribe) conditions the model
# on representative in-domain text, steering it toward normal meeting speech and away
# from the caption-style hallucinations Whisper-family models emit on silence/noise
# ("thanks for watching", "subscribe", etc. — see model._HALLUCINATIONS). It is
# deliberately generic so it never fabricates specifics: no names, no product terms, just
# the register of a real work meeting. Any room glossary is appended AFTER this base.
#
# The code-switching sentence exists because session.audio.input.transcription.language is
# pinned to the speaker's declared language (see _session_payload in model.py) — a real,
# necessary fix for a different bug (cross-script hallucination), but it means the model
# decodes the WHOLE utterance in that one language's phoneme space. An English word dropped
# into a Vietnamese sentence (e.g. "architect") then gets forced into the nearest
# Vietnamese-sounding syllables instead of transcribed as English (e.g. "hai kiên thách") —
# see docs/code-switching-research.md for the full diagnosis. This sentence doesn't fix
# recognition on its own (that's what the glossary prompt from GlossaryStartedEventConsumer
# is for — see _get_stt_prompt below); it only stops a correctly-recognized English word
# from being re-spelled phonetically once the model does catch it.
_GENERIC_STT_BASE_PROMPT = (
    "This is a live professional work meeting. Participants discuss projects, tasks, "
    "schedules, decisions, requirements, features, and technical details in clear, "
    "conversational speech. Transcribe exactly what is said, verbatim. Do not add "
    "greetings, sign-offs, video captions, channel/subscribe phrases, or any words that "
    "were not spoken. Speakers often mix in English words or phrases (e.g. product, "
    "technical, or business terms) mid-sentence even when speaking another language — "
    "write those words in their original English spelling, never phonetically "
    "transliterated into another script or language."
)

# The room language set is derived from participants' declared speak-languages, which
# change as people join/leave. Cache it briefly rather than per room-lifetime (unlike the
# prompt) so a newly joined speaker's language is picked up within a few seconds.
_ROOM_LANGUAGES_TTL_S = 15.0


def _build_segment_id(
    meeting_id: str,
    speaker_id: str,
    source_message_id: bytes,
    start_ms: int,
    end_ms: int,
    text: str,
) -> str:
    """Create an idempotent segment id from the immutable Redis source event."""
    source_id = source_message_id.decode("utf-8", errors="replace")
    material = f"warptalk:stt:{meeting_id}:{speaker_id}:{source_id}:{start_ms}:{end_ms}:{text}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material))


class STTWorker(BaseWorker):
    """Speech-to-Text worker using OpenAI gpt-4o-mini-transcribe."""

    worker_name = "stt"
    input_stream = "audio:chunks"
    consumer_group = "stt-workers"

    def __init__(
        self,
        stt_settings: STTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.stt_settings = stt_settings or STTSettings()
        self.model: OpenAISTT | None = None
        # meeting_id -> contextual-biasing prompt (glossary/key terms), published to
        # Redis by the backend when the room starts. Cached for the room's lifetime so
        # we don't hit Redis on every chunk; the STT session is created once per speaker
        # and reuses the prompt fetched here, so a mid-room glossary change won't apply
        # until that speaker's session is swept for idleness — acceptable, glossaries
        # are configured before a meeting, not during.
        self._stt_prompts: dict[str, str] = {}
        # meeting_id -> (set of declared language codes, monotonic timestamp fetched).
        # Refreshed every _ROOM_LANGUAGES_TTL_S so late joiners' languages are picked up.
        self._room_languages: dict[str, tuple[set[str], float]] = {}
        # (meeting_id, speaker_id) -> lock serializing THAT speaker's own chunks — see
        # _consume_loop for why.
        self._speaker_locks: dict[tuple[str, str], asyncio.Lock] = {}

    async def load_model(self) -> None:
        self.model = OpenAISTT(
            api_key=resolve_openai_api_key(self.stt_settings.api_key),
            model=self.stt_settings.model,
            noise_reduction=self.stt_settings.noise_reduction,
        )
        await self.model.load()

    async def _consume_loop(self) -> None:
        """Dispatch process() concurrently across DIFFERENT speakers, while keeping
        each speaker's OWN chunks strictly ordered.

        BaseWorker's default _consume_loop awaits process() for one message before
        even reading the next — so before this override, if speaker A and speaker B
        spoke around the same time, B's chunk sat idle behind A's entire transcribe()
        call (a real OpenAI Realtime round-trip; model.py's own benchmark comment
        puts commit -> completed at ~1.8s for a ~4s utterance) even though they use
        completely independent Realtime sessions (OpenAISTT._sessions is already keyed
        per (meeting_id, speaker_id)). Dispatching concurrently here is what actually
        lets concurrent speakers be transcribed in parallel end-to-end — having
        separate sessions was necessary but not sufficient while the consume loop
        itself was single-file.

        The per-speaker lock is still required: two chunks from the SAME speaker
        committing audio into / reading completions from the SAME reused WebSocket
        session concurrently would interleave the transcription stream. Locking keeps
        that path exactly as ordered as before; only cross-speaker work is now parallel.

        RedisStreamClient.consume_concurrent ties XACK to successful handler
        completion. Failed work remains pending for BaseWorker's reclaim/DLQ path.
        """
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )

        async def _run(message_id: bytes, data: dict[bytes, bytes]) -> None:
            key = _extract_speaker_key(data)
            lock = self._speaker_locks.setdefault(key, asyncio.Lock())
            async with lock:
                await self._process_and_log_errors(message_id, data)

        while not self._shutdown_event.is_set():
            try:
                await self._recover_stale_messages()
                await self.redis.consume_concurrent(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    handler=_run,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=8,
                    concurrency=8,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                await asyncio.sleep(1.0)

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        # These three dicts are otherwise unbounded for the life of the process — one
        # entry per meeting (+ per speaker for locks) ever seen. Room-ended is the one
        # reliable signal we get that a meeting is truly done.
        self._stt_prompts.pop(room_id, None)
        self._room_languages.pop(room_id, None)
        stale_speakers = [key for key in self._speaker_locks if key[0] == room_id]
        for key in stale_speakers:
            self._speaker_locks.pop(key, None)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Process one audio chunk: transcribe and publish results."""
        chunk = AudioChunkMessage.from_redis(data)

        if chunk.meeting_id in self._paused_rooms:
            self.logger.debug("skipping_paused_room", meeting_id=chunk.meeting_id)
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - chunk.timestamp_ms
        await self.redis.publish_telemetry(chunk.meeting_id, self.worker_name, e2e_latency_ms)

        self.logger.debug(
            "processing_chunk",
            meeting_id=chunk.meeting_id,
            speaker_id=chunk.speaker_id,
            chunk_index=chunk.chunk_index,
            is_final=chunk.is_final_chunk,
        )

        chunk_offset_ms = chunk.chunk_index * self.settings.chunk_duration_ms
        language_hint = chunk.language if chunk.language != "auto" else None
        prompt = await self._get_stt_prompt(chunk.meeting_id)
        allowed_languages = await self._get_room_languages(chunk.meeting_id)

        async def publish_early(segment: TranscribedSegment) -> None:
            # Never final — the chunk's real is_final_chunk flag belongs to whichever
            # trailing segment(s) come back from transcribe() below, once the whole
            # chunk is done. This is published the moment a complete sentence shows up
            # in the Realtime API's incremental delta stream, so translation/TTS can
            # start on it without waiting for the rest of the chunk to transcribe.
            result = STTResultMessage(
                segment_id=_build_segment_id(
                    chunk.meeting_id,
                    chunk.speaker_id,
                    message_id,
                    segment.start_ms,
                    segment.end_ms,
                    segment.text,
                ),
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                text=segment.text,
                language=segment.language,
                confidence=segment.confidence,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                chunk_index=chunk.chunk_index,
                is_final_chunk=False,
                timestamp_ms=chunk.timestamp_ms,
            )
            await self.publish("stt:results", chunk.meeting_id, result.to_redis())
            self.logger.info(
                "segment_transcribed_early",
                meeting_id=chunk.meeting_id,
                text=segment.text[:80],
                language=segment.language,
            )

        t0 = time.monotonic()
        try:
            model = self._require_model()
            segments = await model.transcribe(
                chunk.audio_data,
                sample_rate=chunk.sample_rate,
                language=language_hint,
                chunk_offset_ms=chunk_offset_ms,
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                prompt=prompt,
                allowed_languages=allowed_languages,
                on_early_segment=publish_early,
            )
        except Exception as exc:
            await self.redis.publish_system_event(
                room_id=chunk.meeting_id,
                event_type="stt_unavailable",
                payload={
                    "speakerId": chunk.speaker_id,
                    "chunkIndex": chunk.chunk_index,
                    "model": self.stt_settings.model,
                    "error": str(exc),
                },
            )
            self.logger.error(
                "stt_unavailable",
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                chunk_index=chunk.chunk_index,
                error=str(exc),
            )
            segments = []
        inference_ms = int((time.monotonic() - t0) * 1000)

        self.logger.info(
            "inference_complete",
            inference_ms=inference_ms,
            segments=len(segments),
            chunk_index=chunk.chunk_index,
        )

        for segment in segments:
            result = STTResultMessage(
                segment_id=_build_segment_id(
                    chunk.meeting_id,
                    chunk.speaker_id,
                    message_id,
                    segment.start_ms,
                    segment.end_ms,
                    segment.text,
                ),
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                text=segment.text,
                language=segment.language,
                confidence=segment.confidence,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                chunk_index=chunk.chunk_index,
                is_final_chunk=chunk.is_final_chunk,
                timestamp_ms=chunk.timestamp_ms,
            )

            await self.publish("stt:results", chunk.meeting_id, result.to_redis())

            self.logger.info(
                "segment_transcribed",
                meeting_id=chunk.meeting_id,
                text=segment.text[:80],
                language=segment.language,
                confidence=segment.confidence,
                inference_ms=inference_ms,
            )

        if not segments and chunk.is_final_chunk:
            result = STTResultMessage(
                segment_id=_build_segment_id(
                    chunk.meeting_id,
                    chunk.speaker_id,
                    message_id,
                    0,
                    0,
                    "",
                ),
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                text="",
                language=chunk.language,
                is_final_chunk=True,
                timestamp_ms=chunk.timestamp_ms,
            )
            await self.publish("stt:results", chunk.meeting_id, result.to_redis())

    def _require_model(self) -> OpenAISTT:
        if self.model is None:
            raise RuntimeError("STT model is not loaded")
        return self.model

    async def _get_stt_prompt(self, meeting_id: str) -> str | None:
        """Contextual-biasing prompt for this room: a generic anti-hallucination base
        (always present) plus this room's glossary/key terms if the backend has published
        any to `translationRoom:{meeting_id}:stt_prompt`.

        The room glossary is cached for the room's lifetime — the value is stable per
        meeting and the STT session (created once per speaker) reuses whatever it was
        first given. Even when no glossary is configured, the generic base is returned so
        the session still opens with hallucination-reducing context, not just a language
        hint.
        """
        cached = self._stt_prompts.get(meeting_id)
        if cached is None:
            raw = await self.redis.get(f"translationRoom:{meeting_id}:stt_prompt")
            glossary = ""
            if raw:
                glossary = (raw.decode() if isinstance(raw, bytes) else raw).strip()
            self._stt_prompts[meeting_id] = glossary
            if glossary:
                self.logger.info("stt_prompt_loaded", meeting_id=meeting_id, chars=len(glossary))
        else:
            glossary = cached

        if glossary:
            return f"{_GENERIC_STT_BASE_PROMPT}\n\n{glossary}"
        return _GENERIC_STT_BASE_PROMPT

    async def _get_room_languages(self, meeting_id: str) -> set[str]:
        """The set of languages this meeting is allowed to produce — the distinct
        profile speak-languages of its currently-joined participants.

        Read from the Redis hash `translationRoom:{meeting_id}:speak_languages`
        (userId -> normalized speak language), which TranslationRoomHub.JoinTranslationRoom
        populates and OnDisconnected/Leave clears. This is the "languages present in the
        meeting" set the room declares implicitly by who is in it — no separate config
        needed. Empty set ⇒ nothing declared yet; _filter_segments falls back to its
        default and always keeps the speaker's own hint language, so transcript is never
        dropped just because this hash hasn't populated.
        """
        now = time.monotonic()
        cached = self._room_languages.get(meeting_id)
        if cached is not None and now - cached[1] < _ROOM_LANGUAGES_TTL_S:
            return cached[0]

        raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:speak_languages")
        langs: set[str] = set()
        for value in (raw or {}).values():
            code = value.decode() if isinstance(value, bytes) else value
            code = _normalize_language(code.strip()) if code else ""
            if code and code != "auto":
                langs.add(code)
        self._room_languages[meeting_id] = (langs, now)
        return langs
