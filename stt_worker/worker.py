"""STT Worker — Consumes audio chunks, produces text segments.

Pipeline:
    Redis Stream (audio:chunks:{meetingId})
    → OpenAI gpt-transcribe (persistent Realtime transcription session)
    → Redis Stream (stt:results:{meetingId})
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any

from shared.base_worker import BaseWorker
from shared.config import STTSettings, resolve_openai_api_key
from shared.prosody import (
    SpeakerBaseline,
    measure,
    pcm16_to_float,
    to_delivery,
    update_baseline,
)
from shared.schemas import AudioChunkMessage, ProsodyEnvelope, STTResultMessage
from shared.text_utils import split_into_sentences
from stt_worker.model import OpenAISTT, _normalize_language


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


# The room language set is derived from participants' declared speak-languages, which
# change as people join/leave. Cache it briefly rather than per room-lifetime (unlike the
# prompt) so a newly joined speaker's language is picked up within a few seconds.
_ROOM_LANGUAGES_TTL_S = 15.0

_RECENT_CONTEXT_SEGMENTS = 4
_MAX_STT_PROMPT_CHARS = 600
_MAX_GLOSSARY_CHARS = 240
_MAX_STT_KEYWORDS = 100
_CONTEXT_MIN_CONFIDENCE = -0.35
_ACTIVE_TRANSLATION_STATES = {"IN_PROGRESS", "AUDIO_ROUTING_ACTIVE"}


def _language_hint_for_stt(language: str) -> str | None:
    normalized = language.strip().lower().split("-", 1)[0]
    if not normalized or normalized == "auto":
        return None
    return normalized


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
    """Speech-to-Text worker using OpenAI gpt-transcribe."""

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
        # Retained only for backward-compatible prompt parsing/filter tests. Production
        # sessions do NOT receive this prose: gpt-transcribe can recite or translate the
        # prompt itself on marginal audio. Structured `_stt_keywords` below is the safe
        # provider-side vocabulary bias; translation owns the full meeting context.
        self._stt_prompts: dict[str, str] = {}
        # Structured source/target terms from the existing MT glossary Redis contract.
        # Unlike prose prompts, provider keyword hints cannot be echoed as instructions.
        self._stt_keywords: dict[str, list[str]] = {}
        # Retain accepted completed transcripts only for defensive echo detection.
        # Never feed them back into the transcription prompt: marginal audio can cause
        # Realtime STT to copy prior transcript verbatim instead of transcribing speech.
        self._recent_transcripts: dict[str, deque[str]] = {}
        # meeting_id -> (set of declared language codes, monotonic timestamp fetched).
        # Refreshed every _ROOM_LANGUAGES_TTL_S so late joiners' languages are picked up.
        self._room_languages: dict[str, tuple[set[str], float]] = {}
        # (meeting_id, speaker_id) -> lock serializing THAT speaker's own chunks — see
        # _consume_loop for why.
        self._speaker_locks: dict[tuple[str, str], asyncio.Lock] = {}
        # (meeting_id, speaker_id) -> how that speaker normally sounds in THIS room. Held in
        # memory and mirrored to Redis so a restart or a second replica does not start every
        # speaker from scratch — see _speaker_baseline.
        self._prosody_baselines: dict[tuple[str, str], SpeakerBaseline] = {}
        self._prewarm_listener_task: asyncio.Task[None] | None = None

    async def load_model(self) -> None:
        self.model = OpenAISTT(
            api_key=resolve_openai_api_key(self.stt_settings.api_key),
            model=self.stt_settings.model,
            noise_reduction=self.stt_settings.noise_reduction,
            min_avg_logprob=self.stt_settings.min_avg_logprob,
            min_avg_logprob_by_language=self.stt_settings.min_avg_logprob_by_language,
        )
        await self.model.load()
        await self.model.warm_up(pool_size=self.stt_settings.realtime_pool_size)
        self._prewarm_listener_task = asyncio.create_task(self._listen_for_track_prewarm())

    async def _listen_for_track_prewarm(self) -> None:
        """Prepare the speaker's Realtime socket during room join, before first speech."""
        pubsub = self.redis.redis.pubsub()
        try:
            await pubsub.subscribe("meeting.track_published")
            while not self._shutdown_event.is_set():
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message:
                    await self._prewarm_from_track_event(message["data"])
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("stt_prewarm_listener_failed")
        finally:
            try:
                await pubsub.close()
            except Exception:
                self.logger.warning("stt_prewarm_listener_close_failed")

    async def _prewarm_from_track_event(self, serialized_event: bytes | str) -> None:
        try:
            envelope = json.loads(serialized_event)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return
        if (
            envelope.get("event_type") != "meeting.track_published"
            or envelope.get("schema_version") != 1
            or envelope.get("producer") != "meeting-service"
        ):
            return
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            return
        meeting_id = payload.get("room_name")
        speaker_id = payload.get("participant_identity")
        if not isinstance(meeting_id, str) or not meeting_id:
            return
        if not isinstance(speaker_id, str) or not speaker_id:
            return

        # Gateway normally persists the selected speak language immediately after join.
        # Give that write a brief chance to land so the prepared session is language-pinned
        # and will not be discarded/reopened on the first audio chunk.
        raw_language: bytes | str | None = None
        for attempt in range(3):
            raw_language = await self.redis.hget(
                f"translationRoom:{meeting_id}:speak_languages",
                speaker_id,
            )
            if raw_language:
                break
            if attempt < 2:
                await asyncio.sleep(0.25)
        # Prewarm ANYWAY when the language write has not landed. This used to return, so a
        # speaker whose gateway write was even half a second late got no prepared socket at
        # all and paid the full Realtime handshake on their first sentence — exactly the
        # "I joined, I spoke, and the transcript lagged" complaint. Claiming the socket
        # unpinned is still most of the win: the handshake is the expensive part, and the
        # language is applied by an ordinary session.update on the first chunk.
        declared_language = (
            raw_language.decode() if isinstance(raw_language, bytes) else raw_language
        ) or None
        if declared_language is None:
            self.logger.info(
                "stt_prewarm_without_language",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
            )
        allowed_languages = await self._get_room_languages(meeting_id)
        keywords = await self._get_stt_keywords(meeting_id)
        await self._require_model().prepare_session(
            meeting_id,
            speaker_id,
            language=_language_hint_for_stt(declared_language) if declared_language else None,
            prompt=None,
            allowed_languages=allowed_languages,
            keywords=keywords,
        )
        self.logger.info(
            "stt_session_prewarmed",
            meeting_id=meeting_id,
            speaker_id=speaker_id,
            language=declared_language,
        )

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
        getattr(self, "_stt_keywords", {}).pop(room_id, None)
        getattr(self, "_recent_transcripts", {}).pop(room_id, None)
        self._room_languages.pop(room_id, None)
        stale_speakers = [key for key in self._speaker_locks if key[0] == room_id]
        for key in stale_speakers:
            self._speaker_locks.pop(key, None)
        # Baselines are per (meeting, speaker) and describe a microphone in a room that no
        # longer exists. The Redis copy expires on its own TTL; this is the in-process one.
        baselines = self._baselines()
        for key in [key for key in baselines if key[0] == room_id]:
            baselines.pop(key, None)

    async def _cleanup(self) -> None:
        task = getattr(self, "_prewarm_listener_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.model is not None:
            await self.model.close()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Process one audio chunk: transcribe and publish results."""
        chunk = AudioChunkMessage.from_redis(data)

        if not await self._translation_state_allows_stt(chunk.meeting_id):
            self.logger.info(
                "skipping_inactive_room",
                meeting_id=chunk.meeting_id,
                route_state=getattr(self, "_route_states", {}).get(chunk.meeting_id),
            )
            return

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
        language_hint = _language_hint_for_stt(chunk.language)
        allowed_languages = await self._get_room_languages(chunk.meeting_id)
        keywords = await self._get_stt_keywords(chunk.meeting_id)

        # Measured CONCURRENTLY with recognition, not before it. Transcription is a
        # network round trip of hundreds of milliseconds; the measurement is single-digit
        # milliseconds of CPU in a thread. Started here and collected after, it costs the
        # pipeline no wall-clock at all — which is the only way a delivery feature earns its
        # place in front of a live meeting.
        prosody_task: asyncio.Task[ProsodyEnvelope | None] | None = None
        if self.stt_settings.prosody_enabled:
            prosody_task = asyncio.create_task(self._measure_prosody(chunk))

        t0 = time.monotonic()
        try:
            model = self._require_model()

            async def publish_speculative(segment: Any) -> None:
                """Warm translation on a complete delta sentence without exposing it.

                This Pub/Sub hint is deliberately ephemeral. Translation may cache the
                result, but only the later completed STT segment (with real logprobs)
                can cause a durable translate:results/UI publish.
                """
                text = " ".join(segment.text.split())
                if not text:
                    return
                await self.redis.redis.publish(
                    "stt:speculative",
                    json.dumps(
                        {
                            "meeting_id": chunk.meeting_id,
                            "speaker_id": chunk.speaker_id,
                            "text": text,
                            "language": segment.language or chunk.language,
                            "chunk_index": chunk.chunk_index,
                            "timestamp_ms": chunk.timestamp_ms,
                        },
                        ensure_ascii=False,
                    ),
                )
                self.logger.info(
                    "stt_speculative_sentence",
                    meeting_id=chunk.meeting_id,
                    chunk_index=chunk.chunk_index,
                    chars=len(text),
                    inference_offset_ms=int((time.monotonic() - t0) * 1000),
                )

            segments = await model.transcribe(
                chunk.audio_data,
                sample_rate=chunk.sample_rate,
                language=language_hint,
                chunk_offset_ms=chunk_offset_ms,
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                # Never send title/description/instruction prose to STT. A production
                # failure transcribed and translated that prose verbatim. Keywords retain
                # vocabulary bias without giving the model a sentence it can recite.
                prompt=None,
                allowed_languages=allowed_languages,
                keywords=keywords,
                # Realtime delta events carry neither confidence nor no-speech
                # probability. Publishing them allowed hallucinated partials to reach
                # translation/UI before the completed event could be validated. They
                # may only warm the private speculative translation cache below.
                on_early_segment=None,
                on_speculative_segment=publish_speculative,
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

        # Awaited even when recognition failed, so the task is never left orphaned — and its
        # baseline update is still worth keeping: the speaker did speak, whatever the model
        # made of it.
        prosody = await prosody_task if prosody_task is not None else None

        self.logger.info(
            "inference_complete",
            inference_ms=inference_ms,
            segments=len(segments),
            chunk_index=chunk.chunk_index,
        )

        for segment in segments:
            if segment.confidence >= _CONTEXT_MIN_CONFIDENCE:
                self._remember_transcript(chunk.meeting_id, segment.text)
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
                # Every segment recognised in this chunk shares the chunk's delivery. The
                # measurement's granularity is the audio, and splitting it per segment would
                # be inventing precision that was never measured.
                prosody=prosody,
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

    async def _measure_prosody(self, chunk: AudioChunkMessage) -> ProsodyEnvelope | None:
        """How this chunk was said, relative to how this speaker normally says things.

        Returns None whenever there is nothing honest to report — unusable audio, or a speaker
        the room has not heard enough of yet. None is not a failure and not "neutral": it means
        the field is omitted, and the TTS worker then synthesizes exactly as it did before this
        feature existed.

        Never raises. A meeting must not lose its transcript because a measurement of tone went
        wrong, so every failure here degrades to None and is logged.
        """
        try:
            pcm = pcm16_to_float(chunk.audio_data)
            # measure() is a per-frame Python loop over numpy — ~14ms for a full 6s chunk. Off
            # the event loop so it cannot stall the other speakers being handled concurrently.
            features = await asyncio.to_thread(measure, pcm, chunk.sample_rate)
            if not features.is_usable:
                return None

            key = (chunk.meeting_id, chunk.speaker_id)
            baseline = await self._speaker_baseline(key)

            # Compare FIRST, then fold in. The other order would let a shout become part of the
            # normal it is being measured against, and every strong utterance would partly
            # cancel its own detection.
            delivery = to_delivery(features, baseline)
            await self._store_baseline(key, update_baseline(baseline, features))

            if not delivery.is_measured:
                return None

            return ProsodyEnvelope(
                pitch_lift=delivery.pitch_lift,
                pitch_variation=delivery.pitch_variation,
                energy_ratio=delivery.energy_ratio,
                rate_ratio=delivery.rate_ratio,
                arousal=delivery.arousal,
                # Left empty on purpose: valence is a judgement about the words, and this worker
                # has only heard the sound. See ProsodyEnvelope.valence.
                valence="",
            )
        except Exception:
            self.logger.warning(
                "prosody_measurement_failed",
                meeting_id=chunk.meeting_id,
                speaker_id=chunk.speaker_id,
                chunk_index=chunk.chunk_index,
                exc_info=True,
            )
            return None

    def _baselines(self) -> dict[tuple[str, str], SpeakerBaseline]:
        """The in-process baseline table, created on demand — same defensive shape as
        `_route_states` below, and for the same reason: workers built without __init__."""
        baselines: dict[tuple[str, str], SpeakerBaseline] | None = getattr(
            self, "_prosody_baselines", None
        )
        if baselines is None:
            baselines = {}
            self._prosody_baselines = baselines
        return baselines

    def _baseline_key(self, key: tuple[str, str]) -> str:
        meeting_id, speaker_id = key
        return f"prosody:baseline:{meeting_id}:{speaker_id}"

    async def _speaker_baseline(self, key: tuple[str, str]) -> SpeakerBaseline:
        """This speaker's rolling normal, from memory or (once) from Redis.

        Redis is what makes the baseline survive a worker restart and lets a second replica
        pick up a speaker mid-meeting without starting their normal over. Two replicas holding
        the same speaker can lose an update to each other; that is accepted rather than locked
        against, because the value is an exponential moving average whose whole purpose is to
        be insensitive to any single sample.
        """
        baselines = self._baselines()
        cached = baselines.get(key)
        if cached is not None:
            return cached

        baseline = SpeakerBaseline()
        try:
            raw = await self.redis.get(self._baseline_key(key))
            if raw:
                d = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                baseline = SpeakerBaseline(
                    pitch_median_hz=float(d["pitch_median_hz"]),
                    pitch_iqr_hz=float(d["pitch_iqr_hz"]),
                    rms=float(d["rms"]),
                    speech_rate=float(d["speech_rate"]),
                    sample_count=int(d["sample_count"]),
                )
        except Exception:
            # A corrupt or unreachable baseline means this speaker starts over, which costs
            # them MIN_BASELINE_SAMPLES utterances of plain delivery. Nothing else.
            self.logger.warning("prosody_baseline_read_failed", exc_info=True)

        baselines[key] = baseline
        return baseline

    async def _store_baseline(self, key: tuple[str, str], baseline: SpeakerBaseline) -> None:
        self._baselines()[key] = baseline
        try:
            await self.redis.set_with_ttl(
                self._baseline_key(key),
                json.dumps(
                    {
                        "pitch_median_hz": round(baseline.pitch_median_hz, 3),
                        "pitch_iqr_hz": round(baseline.pitch_iqr_hz, 3),
                        "rms": round(baseline.rms, 6),
                        "speech_rate": round(baseline.speech_rate, 4),
                        "sample_count": baseline.sample_count,
                    },
                    separators=(",", ":"),
                ),
                self.stt_settings.prosody_baseline_ttl_seconds,
            )
        except Exception:
            # In-memory copy is already updated, so this worker keeps working; only the
            # hand-over to a restart or a sibling replica is lost.
            self.logger.warning("prosody_baseline_write_failed", exc_info=True)

    async def _translation_state_allows_stt(self, meeting_id: str) -> bool:
        """Reject queued audio when the authoritative room state is known inactive.

        LiveKit ingress is the primary capture gate. This second gate prevents a stale
        Redis chunk (or a producer regression) from creating transcript before Start.
        Unknown legacy state remains fail-open so an unavailable cache cannot erase valid
        live speech; current rooms persist ``audio_routes`` before publishing audio.
        """
        route_states = getattr(self, "_route_states", None)
        if route_states is None:
            route_states = {}
            self._route_states = route_states
        state = route_states.get(meeting_id)
        if state is None:
            try:
                raw = await self.redis.get(f"translationRoom:{meeting_id}:audio_routes")
                if raw:
                    serialized = raw.decode() if isinstance(raw, bytes) else raw
                    payload = json.loads(serialized)
                    persisted_state = payload.get("room_status")
                    if isinstance(persisted_state, str) and persisted_state:
                        state = persisted_state
                        route_states[meeting_id] = state
            except Exception:
                self.logger.warning(
                    "stt_room_status_hydration_failed",
                    meeting_id=meeting_id,
                    exc_info=True,
                )

        return state is None or state in _ACTIVE_TRANSLATION_STATES

    def _require_model(self) -> OpenAISTT:
        if self.model is None:
            raise RuntimeError("STT model is not loaded")
        return self.model

    async def _get_stt_prompt(self, meeting_id: str) -> str | None:
        """Return the bounded room glossary used for contextual vocabulary biasing.

        The room glossary is cached for the room's lifetime — the value is stable per
        meeting and the STT session (created once per speaker) reuses whatever it was
        first given. Generic instruction prose is deliberately omitted because Realtime
        transcription can echo prompt text into output on silence or marginal audio.
        Accepted transcript is deliberately excluded for the same reason.
        """
        cached = self._stt_prompts.get(meeting_id)
        if cached is None:
            raw = await self.redis.get(f"translationRoom:{meeting_id}:stt_prompt")
            glossary = ""
            if raw:
                glossary = (raw.decode() if isinstance(raw, bytes) else raw).strip()
            if glossary:
                # Track publication/prewarm can race the backend's meeting.started
                # consumer. Cache only a real value; caching "" permanently removed
                # context from every later utterance in that room.
                self._stt_prompts[meeting_id] = glossary
                self.logger.info("stt_prompt_loaded", meeting_id=meeting_id, chars=len(glossary))
        else:
            glossary = cached

        return glossary[:_MAX_GLOSSARY_CHARS] or None

    async def _get_stt_keywords(self, meeting_id: str) -> list[str]:
        """Return structured glossary terms for the provider's keyword-bias field."""
        caches: dict[str, list[str]] | None = getattr(self, "_stt_keywords", None)
        if caches is None:
            caches = {}
            self._stt_keywords = caches
        cached = caches.get(meeting_id)
        if cached is not None:
            return cached

        raw = await self.redis.get(f"translationRoom:{meeting_id}:stt_keywords")
        if not raw:
            return []
        serialized = raw.decode() if isinstance(raw, bytes) else raw
        try:
            entries = json.loads(serialized)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            self.logger.warning("stt_keywords_invalid_json", meeting_id=meeting_id)
            return []
        if not isinstance(entries, list):
            return []

        keywords: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            values: list[object] = [entry] if isinstance(entry, str) else []
            # Be tolerant of an object payload during rolling upgrades, while the
            # dedicated key's canonical contract remains a compact string array.
            if isinstance(entry, dict):
                values = [entry.get("source"), entry.get("target")]
            for value in values:
                if not isinstance(value, str):
                    continue
                cleaned = " ".join(value.split())[:100]
                normalized = cleaned.casefold()
                if not cleaned or normalized in seen:
                    continue
                seen.add(normalized)
                keywords.append(cleaned)
                if len(keywords) >= _MAX_STT_KEYWORDS:
                    break
            if len(keywords) >= _MAX_STT_KEYWORDS:
                break

        # As with the prompt, avoid permanently caching a race-time empty value.
        if keywords:
            caches[meeting_id] = keywords
            self.logger.info(
                "stt_keywords_loaded",
                meeting_id=meeting_id,
                count=len(keywords),
            )
        return keywords

    def _remember_transcript(self, meeting_id: str, text: str) -> None:
        contexts = getattr(self, "_recent_transcripts", None)
        if contexts is None:
            contexts = {}
            self._recent_transcripts = contexts
        window = contexts.setdefault(meeting_id, deque(maxlen=_RECENT_CONTEXT_SEGMENTS))
        # Realtime completed events return one flat chunk. Store its sentences
        # independently so the next chunk's prompt-echo filter can recognize one
        # repeated background lyric instead of seeing an opaque multi-sentence blob.
        for sentence in split_into_sentences(text):
            cleaned = " ".join(sentence.split())
            if cleaned:
                window.append(cleaned)

    async def _get_room_languages(self, meeting_id: str) -> set[str]:
        """Every language this meeting may contain.

        TWO SOURCES, AND THE SECOND ONE IS THE ANSWER TO A REAL BUG
            `speak_languages` is what the people currently in the room are SPEAKING —
            written by TranslationRoomHub.JoinTranslationRoom, cleared on leave.

            `room_languages` is what the room was CONFIGURED for — its SourceLanguage plus
            TargetLanguages, published on the audio_routes payload by
            AudioRouteCacheService.

            They are not the same set, and using only the first is what produced the
            report. A room configured Vietnamese + Japanese, where both participants SPEAK
            Vietnamese and one LISTENS in Japanese, has speak_languages = {vi}. Japanese —
            which the host explicitly configured, and which the room exists to produce —
            was not in the set, so STT treated Japanese characters as foreign to the room
            and deleted every one of them. "Đọc Kanji thì không bắt transcript."

            The union is the honest set: a language is possible here if somebody is
            speaking it OR the room was set up for it.

        Empty set ⇒ nothing declared yet, and _filter_segments then filters nothing rather
        than filtering on an assumption.
        """
        now = time.monotonic()
        cached = self._room_languages.get(meeting_id)
        if cached is not None and now - cached[1] < _ROOM_LANGUAGES_TTL_S:
            return cached[0]

        langs: set[str] = set()

        raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:speak_languages")
        for value in (raw or {}).values():
            code = value.decode() if isinstance(value, bytes) else value
            code = _normalize_language(code.strip()) if code else ""
            if code and code != "auto":
                langs.add(code)

        langs |= await self._get_configured_room_languages(meeting_id)

        self._room_languages[meeting_id] = (langs, now)
        return langs

    async def _get_configured_room_languages(self, meeting_id: str) -> set[str]:
        """The room's own language configuration, from the audio_routes payload.

        Same key `_translation_state_allows_stt` already reads, so this adds no round trip
        beyond the one this worker was making anyway. Best-effort: an unreadable or
        older-format payload (one published before `room_languages` existed) yields an
        empty set and the speak-languages half stands alone, which is exactly the previous
        behaviour.
        """
        try:
            raw = await self.redis.get(f"translationRoom:{meeting_id}:audio_routes")
            if not raw:
                return set()
            payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            configured = payload.get("room_languages") or payload.get("roomLanguages") or []
            langs = set()
            for value in configured:
                if not isinstance(value, str):
                    continue
                code = _normalize_language(value.strip())
                if code and code != "auto":
                    langs.add(code)
            return langs
        except Exception:
            self.logger.warning(
                "room_configured_languages_unreadable", meeting_id=meeting_id, exc_info=True
            )
            return set()
