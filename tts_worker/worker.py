"""TTS Worker — Consumes translated text, produces synthesized audio.

Pipeline:
    Redis Stream (translate:results:{meetingId})
    → Cartesia Sonic Turbo (default voice until clone ready, then cloned voice)
    → Redis Stream (tts:results:{meetingId})

Voice cloning:
    Background task buffers audio:chunks per speaker.
    Once voice_clone_min_seconds of audio collected → POST /voices/clone.
    voice_id cached in Redis: voice:{meeting_id}:{speaker_id}
    All synthesis calls after that use the cloned voice.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage
from tts_worker.livekit_publisher import LiveKitTTSPublisher
from tts_worker.synthesizer import CartesiaSynthesizer

# Standard WAV header size for the pcm_s16le format CartesiaSynthesizer requests —
# used to strip the header before feeding audio into the LiveKit track (which wants
# raw PCM frames, not a WAV container).
_WAV_HEADER_BYTES = 44

# Cartesia's voices.clone() requires a concrete `language`, but AudioChunkMessage.language
# defaults to "auto" (STT does language auto-detection, not the audio-chunk producer) — so
# fall back to "en" for anything Cartesia's SDK wouldn't accept as a real language code.
_CARTESIA_SUPPORTED_LANGUAGES = {
    "en", "fr", "de", "es", "pt", "zh", "ja", "hi", "it", "ko", "nl", "pl", "ru", "sv",
    "tr", "tl", "bg", "ro", "ar", "cs", "el", "fi", "hr", "ms", "sk", "da", "ta", "uk",
    "hu", "no", "vi", "bn", "th", "he", "ka", "id", "te", "gu", "kn", "ml", "mr", "pa",
}


def _clone_language(hint: str) -> str:
    return hint if hint in _CARTESIA_SUPPORTED_LANGUAGES else "en"


def _decode_field(data: dict, key: str) -> str:
    raw = data.get(key)
    if raw is None:
        raw = data.get(key.encode())
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


def _extract_tts_key(data: dict) -> tuple[str, str, str]:
    """Cheap (meeting_id, speaker_id, target_lang) extraction from a raw translate:results
    Redis entry, ahead of process()'s own full TranslationResultMessage parse — this is
    the unit of ordering _consume_loop must preserve (one LiveKit track per this triple)."""
    return (
        _decode_field(data, "meeting_id"),
        _decode_field(data, "speaker_id"),
        _decode_field(data, "target_lang"),
    )


class TTSWorker(BaseWorker):
    """Text-to-Speech worker using Cartesia Sonic Turbo."""

    worker_name = "tts"
    input_stream = "translate:results"
    consumer_group = "tts-workers"
    _audio_consumer_group = "tts-audio-workers"
    _running = True

    def __init__(
        self,
        tts_settings: TTSSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_settings = tts_settings or TTSSettings()
        self.cartesia: CartesiaSynthesizer | None = None
        self.livekit_publisher: LiveKitTTSPublisher | None = None
        # (meeting_id, speaker_id, target_lang) -> lock serializing that key's own
        # messages — see _consume_loop for why.
        self._key_locks: dict[tuple[str, str, str], asyncio.Lock] = {}

    async def load_model(self) -> None:
        self.cartesia = CartesiaSynthesizer(
            api_key=self.tts_settings.api_key,
            model=self.tts_settings.model,
            sample_rate=self.tts_settings.sample_rate,
        )
        await self.cartesia.load()
        self.livekit_publisher = LiveKitTTSPublisher(self.settings.livekit)
        asyncio.create_task(self._consume_audio_for_cloning())
        self.logger.info("tts_worker_ready", model=self.tts_settings.model)

    async def _consume_loop(self) -> None:
        """Dispatch process() concurrently across DIFFERENT (speaker, target_lang)
        keys, while keeping each key's OWN messages strictly ordered.

        BaseWorker's default _consume_loop awaits process() for one message before
        even reading the next, so — before this override — synthesizing speaker A's
        sentence into English fully blocked speaker B's sentence (a different speaker,
        a different LiveKit track, no shared state at all) from even STARTING its own
        Cartesia call. Dispatching concurrently is what lets concurrent speakers (and a
        single speaker's multiple target languages) be synthesized and dubbed in true
        parallel, matching how livekit_publisher already gives each (speaker,
        target_lang) its own independent WebRTC track.

        The per-key lock keeps messages for the SAME key (e.g. sentence 1 then sentence
        2 of one utterance, same speaker, same target_lang) processed one at a time, in
        order — Cartesia's per-call synthesis latency varies, so without this a later
        sentence could finish synthesizing first and get published to the shared track
        before an earlier one, playing the dub back in the wrong order. This trades a
        little same-key pipelining (sentence 2 can't start synthesizing until sentence
        1's audio has fully been pushed to the track) for guaranteed in-order playback —
        the right trade-off, since real speech itself paces how fast new same-key
        sentences even arrive.

        Same acking trade-off as translation_worker's override: RedisStreamClient.
        consume() acks a message right after yielding it, not after process() returns,
        so a crash mid-flight loses whatever's currently dispatched instead of it being
        redelivered.
        """
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )

        async def _run(message_id: bytes, data: dict[bytes, bytes]) -> None:
            key = _extract_tts_key(data)
            lock = self._key_locks.setdefault(key, asyncio.Lock())
            async with lock:
                try:
                    await self.process(message_id, data)
                except Exception:
                    self.logger.exception(
                        "process_error", message_id=message_id, stream=self.input_stream
                    )

        while not self._shutdown_event.is_set():
            try:
                async for message_id, data in self.redis.consume(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                ):
                    if self._shutdown_event.is_set():
                        break
                    asyncio.create_task(_run(message_id, data))
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                await asyncio.sleep(1.0)

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        stale_keys = [key for key in self._key_locks if key[0] == room_id]
        for key in stale_keys:
            self._key_locks.pop(key, None)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Synthesize one translated text segment — into every DISTINCT voice this
        (speaker, target_lang) needs (see _resolve_voice_variants): the shared
        default/cloned track everyone hears absent a preference, plus one extra
        track per distinct voice a listener explicitly picked via SetVoicePreference.
        """
        translation = TranslationResultMessage.from_redis(data)
        text = translation.translated_text

        route_status = self._route_states.get(translation.meeting_id, "AUDIO_ROUTING_ACTIVE")
        if route_status == "PAUSED":
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - translation.timestamp_ms
        await self.redis.publish_telemetry(translation.meeting_id, self.worker_name, e2e_latency_ms)

        if route_status == "TEXT_ONLY_MODE" or not text.strip():
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

        variants = await self._resolve_voice_variants(
            translation.meeting_id, translation.speaker_id, translation.target_lang
        )

        for voice_id, voice_type, voice_key in variants:
            await self._synthesize_and_publish(translation, text, voice_id, voice_type, voice_key)

        # Exactly once per message regardless of how many voice variants rendered —
        # billing_worker/TranscriptRedisConsumerService key off this event, not off
        # per-variant synthesis.
        if translation.is_final_chunk:
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="final_chunk_processed",
                payload={"segmentId": translation.segment_id},
            )

    async def _resolve_voice_variants(
        self, meeting_id: str, speaker_id: str, target_lang: str
    ) -> list[tuple[str, str, str]]:
        """Every distinct (voice_id, voice_type, voice_key) this (speaker, target_lang)
        must be rendered into.

        Always includes exactly one "default" entry (voice_key="") — the speaker's own
        cloned voice if available, else a voice deterministically hashed from
        speaker_id out of that language's Cartesia catalog (so two un-cloned speakers
        dubbed into the same language sound different from each other by default,
        instead of both using Cartesia's single hardcoded fallback voice — the "A and B
        sound identical when they talk over each other" problem). This is the ONLY
        variant billed (see _synthesize_and_publish) and the only one with a backward-
        compatible LiveKit identity (ai-interpreter-{lang}-{speakerId}, unchanged).

        Plus one extra entry per DISTINCT voice a listener explicitly chose via
        SetVoicePreference for this language (deduped — two listeners picking the same
        voice share one synthesis+track, same principle as _get_target_languages
        deduping identical listen-languages). Skipped entirely if it happens to equal
        the default voice already being rendered.
        """
        cloned_voice_id = await self._get_voice_id(meeting_id, speaker_id)
        if cloned_voice_id:
            default_voice_id, default_voice_type = cloned_voice_id, "cloned"
        else:
            default_voice_id = await self._hashed_default_voice_id(target_lang, speaker_id)
            default_voice_type = "default"

        variants: list[tuple[str, str, str]] = [(default_voice_id, default_voice_type, "")]

        explicit_choices = await self._get_explicit_voice_choices(meeting_id, target_lang)
        for voice_id in explicit_choices:
            if voice_id == default_voice_id:
                continue
            variants.append((voice_id, "preference", f"voice-{voice_id[:8]}"))

        return variants

    async def _get_voice_catalog(self, language: str) -> list[dict]:
        """Redis-cached (TTL) list of public Cartesia voices for a language.

        Falls back to [] on any cache/fetch problem — callers must fall back to
        CartesiaSynthesizer._default_voice_id() rather than fail synthesis.
        """
        cache_key = f"voice_catalog:{language}"
        cached = await self.redis.get(cache_key)
        if cached:
            try:
                raw = cached.decode() if isinstance(cached, bytes) else cached
                return json.loads(raw)
            except Exception:
                self.logger.warning("voice_catalog_cache_corrupt", language=language)

        voices = await self.cartesia.list_voices(
            language,
            limit=self.tts_settings.voice_catalog_size,
        )
        if voices:
            await self.redis.set_with_ttl(
                cache_key, json.dumps(voices), self.tts_settings.voice_catalog_cache_ttl_seconds
            )
        return voices

    async def _hashed_default_voice_id(self, language: str, speaker_id: str) -> str:
        """Deterministic per-speaker pick from this language's voice catalog."""
        catalog = await self._get_voice_catalog(language)
        if not catalog:
            return CartesiaSynthesizer._default_voice_id(language)
        index = int(hashlib.sha256(speaker_id.encode()).hexdigest(), 16) % len(catalog)
        return catalog[index]["id"]

    async def _get_explicit_voice_choices(self, meeting_id: str, target_lang: str) -> set[str]:
        """Distinct voice_ids explicitly chosen (via TranslationRoomHub.
        SetVoicePreference) by listeners currently tuned to target_lang — cross-
        references the languages hash (who's listening in target_lang right now)
        against the voice_preferences hash (their chosen voice, if any). A listener
        who changes target_lang stops being counted here on their very next
        utterance, same as _get_target_languages already behaves for language itself.
        """
        languages_raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:languages")
        listeners_in_lang = {
            (uid.decode() if isinstance(uid, bytes) else uid)
            for uid, lang in (languages_raw or {}).items()
            if (lang.decode() if isinstance(lang, bytes) else lang) == target_lang
        }
        if not listeners_in_lang:
            return set()

        prefs_raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:voice_preferences")
        choices: set[str] = set()
        for uid, voice_id in (prefs_raw or {}).items():
            user_id = uid.decode() if isinstance(uid, bytes) else uid
            if user_id not in listeners_in_lang:
                continue
            value = voice_id.decode() if isinstance(voice_id, bytes) else voice_id
            if value:
                choices.add(value)
        return choices

    async def _synthesize_and_publish(
        self,
        translation: TranslationResultMessage,
        text: str,
        voice_id: str,
        voice_type: str,
        voice_key: str,
    ) -> None:
        cache_key = self._cache_key(
            speaker_id=translation.speaker_id,
            target_lang=translation.target_lang,
            # Two different voice_ids must never share a cache entry even when
            # voice_type matches (e.g. two distinct "preference" picks) — the concrete
            # voice_id, not just the type, is part of what was actually rendered.
            text=text,
            voice_mode=f"{voice_type}:{voice_id}",
        )

        if self.tts_settings.cache_enabled:
            cached_audio = await self.redis.get(cache_key)
            if cached_audio:
                if voice_key:
                    # Extra voice variant — LiveKit only, never a second billing event
                    # for content already billed via the default variant's publish.
                    await self._publish_livekit_only(translation, cached_audio, voice_key)
                else:
                    await self._publish_result(
                        translation=translation,
                        audio_bytes=cached_audio,
                        duration_ms=0,
                        voice_type=voice_type,
                        voice_key=voice_key,
                        provider_voice_id=voice_id,
                        cache_key=cache_key,
                        cache_hit=True,
                        synthesis_latency_ms=0,
                    )
                return

        t0 = time.monotonic()
        try:
            audio_bytes, duration_ms, resolved_voice_id = await self.cartesia.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
            )
        except Exception as e:
            self.logger.error("cartesia_synthesis_failed", error=str(e), voice_type=voice_type)
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="tts_unavailable",
                payload={"error": str(e)},
            )
            return

        synthesis_latency_ms = int((time.monotonic() - t0) * 1000)

        if audio_bytes:
            if voice_key:
                await self._publish_livekit_only(translation, audio_bytes, voice_key)
            else:
                await self._publish_result(
                    translation=translation,
                    audio_bytes=audio_bytes,
                    duration_ms=duration_ms,
                    voice_type=voice_type,
                    voice_key=voice_key,
                    provider_voice_id=resolved_voice_id,
                    cache_key=cache_key,
                    cache_hit=False,
                    synthesis_latency_ms=synthesis_latency_ms,
                )
            if self.tts_settings.cache_enabled:
                await self.redis.set_with_ttl(
                    cache_key, audio_bytes, self.tts_settings.cache_ttl_seconds
                )

        self.logger.info(
            "audio_synthesized",
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            voice_type=voice_type,
            voice_key=voice_key,
            duration_ms=duration_ms,
            synthesis_latency_ms=synthesis_latency_ms,
            text=text[:60],
            is_final=translation.is_final_chunk,
        )

    async def _publish_result(
        self,
        translation: TranslationResultMessage,
        audio_bytes: bytes,
        duration_ms: int,
        voice_type: str,
        voice_key: str,
        provider_voice_id: str,
        cache_key: str,
        cache_hit: bool,
        synthesis_latency_ms: int,
    ) -> None:
        """Full publish: tts:results (billing_worker/TranscriptRedisConsumerService
        depend on this) + LiveKit track.

        ONLY called for the default/cloned variant (voice_key=""). An explicit-
        preference variant is a re-render of content already billed via this call for
        the same utterance — it must reach LiveKit (see _publish_livekit_only) but
        must NOT publish a second tts:results event, or billing_worker (which charges
        per tts:results message, keyed by segment_id+target_lang) would double-charge
        the workspace for one translated utterance just because a listener picked an
        alternate voice.
        """
        result = TTSResultMessage(
            segment_id=translation.segment_id,
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            audio_data=audio_bytes,
            duration_ms=duration_ms,
            char_count=len(translation.translated_text),
            voice_type=voice_type,
            voice_mode=voice_type,
            clone_strength=1.0 if voice_type == "cloned" else 0.0,
            anchor_provider="cartesia",
            clone_provider="cartesia" if voice_type == "cloned" else "",
            provider_voice_id=provider_voice_id,
            render_location="server",
            cache_key=cache_key,
            cache_hit=cache_hit,
            synthesis_latency_ms=synthesis_latency_ms,
            fallback_reason="" if voice_type == "cloned" else "voice_profile_not_ready",
            target_lang=translation.target_lang,
            is_final_chunk=translation.is_final_chunk,
            timestamp_ms=translation.timestamp_ms,
        )
        await self.publish("tts:results", translation.meeting_id, result.to_redis())
        await self._publish_livekit_only(translation, audio_bytes, voice_key)

    async def _publish_livekit_only(
        self, translation: TranslationResultMessage, audio_bytes: bytes, voice_key: str
    ) -> None:
        """Push to this variant's LiveKit track only — no tts:results.

        Awaited (not fire-and-forget) so this key's _consume_loop lock genuinely
        covers the full publish, not just synthesis — otherwise a later sentence for
        the same key could start capturing to the track before an earlier one
        finishes, playing the dub back out of order. Safe to await unconditionally:
        publish_pcm() catches every internal failure itself and never raises.
        """
        pcm = audio_bytes[_WAV_HEADER_BYTES:] if len(audio_bytes) > _WAV_HEADER_BYTES else b""
        if pcm and self.livekit_publisher is not None:
            await self.livekit_publisher.publish_pcm(
                translation.meeting_id,
                translation.speaker_id,
                translation.target_lang,
                pcm,
                self.tts_settings.sample_rate,
                voice_key=voice_key,
            )

    async def _get_voice_id(self, meeting_id: str, speaker_id: str) -> str | None:
        """Return cached Cartesia voice_id for this speaker, or None.

        Re-checks consent on every call (not just before cloning) — if the speaker
        revokes voice clone consent mid-session, synthesis must fall back to the
        default voice immediately, even though a voice_id is still cached.
        """
        if not self.is_voice_clone_consented(meeting_id, speaker_id):
            return None
        cached = await self.redis.hget(f"voice:{meeting_id}:{speaker_id}", "voice_id")
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached
        return None

    async def _consume_audio_for_cloning(self) -> None:
        """Buffer raw audio per speaker; clone voice once enough is collected."""
        # {(meeting_id, speaker_id): accumulated_audio_bytes}
        buffers: dict[tuple[str, str], bytearray] = {}
        buffer_seconds: dict[tuple[str, str], float] = {}
        buffer_lang: dict[tuple[str, str], str] = {}

        while self._running:
            try:
                async for _msg_id, data in self.redis.consume(
                    stream="audio:chunks",
                    group=self._audio_consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=5,
                ):
                    try:
                        chunk = AudioChunkMessage.from_redis(data)
                        key = (chunk.meeting_id, chunk.speaker_id)

                        # Consent gate: never buffer/clone a speaker's voice (biometric
                        # data) unless they have at least one current outgoing route with
                        # VoiceCloneEnabled = true. See base_worker.is_voice_clone_consented.
                        if not self.is_voice_clone_consented(chunk.meeting_id, chunk.speaker_id):
                            buffers.pop(key, None)
                            buffer_seconds.pop(key, None)
                            buffer_lang.pop(key, None)
                            continue

                        # Skip if voice already cloned for this speaker
                        if await self._get_voice_id(chunk.meeting_id, chunk.speaker_id):
                            continue

                        buffers.setdefault(key, bytearray()).extend(chunk.audio_data)
                        # PCM 16-bit mono: 2 bytes per sample
                        duration_s = len(chunk.audio_data) / 2 / max(chunk.sample_rate, 1)
                        buffer_seconds[key] = buffer_seconds.get(key, 0.0) + duration_s
                        buffer_lang[key] = chunk.language

                        if buffer_seconds[key] >= self.tts_settings.voice_clone_min_seconds:
                            audio_snapshot = bytes(buffers.pop(key))
                            del buffer_seconds[key]
                            clone_lang = _clone_language(buffer_lang.pop(key, "en"))
                            asyncio.create_task(
                                self._clone_and_cache(
                                    chunk.meeting_id, chunk.speaker_id, audio_snapshot, clone_lang
                                )
                            )
                    except Exception:
                        self.logger.exception("audio_chunk_processing_error")
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("audio_consumer_error")
                await asyncio.sleep(2)

    async def _clone_and_cache(
        self, meeting_id: str, speaker_id: str, audio_bytes: bytes, language: str = "en"
    ) -> None:
        """Clone voice via Cartesia and cache voice_id in Redis."""
        label = f"speaker-{speaker_id[:8]}-{meeting_id[:8]}"
        try:
            voice_id = await self.cartesia.clone_voice(audio_bytes, label, language)
            cache_key = f"voice:{meeting_id}:{speaker_id}"
            await self.redis.hset(cache_key, "voice_id", voice_id)
            # hset has no TTL of its own — without this the key lives in Redis forever.
            await self.redis.expire(cache_key, self.tts_settings.voice_clone_key_ttl_seconds)
            self.logger.info(
                "voice_cached",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                voice_id=voice_id,
            )
            await self.redis.publish_system_event(
                room_id=meeting_id,
                event_type="voice_clone_ready",
                payload={"speakerId": speaker_id, "voiceId": voice_id},
            )
        except Exception as e:
            self.logger.error(
                "voice_clone_failed",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                error=str(e),
            )

    @staticmethod
    def _cache_key(speaker_id: str, target_lang: str, text: str, voice_mode: str) -> str:
        normalized = " ".join(text.casefold().split())
        digest = hashlib.sha256(
            f"{speaker_id}|{target_lang}|{normalized}|{voice_mode}".encode()
        ).hexdigest()
        return f"tts:cache:{digest}"
