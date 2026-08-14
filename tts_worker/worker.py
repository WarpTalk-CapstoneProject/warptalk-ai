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
from collections.abc import Mapping
from typing import Any, cast

from shared import isochrony
from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.lang import is_same_language
from shared.prosody import Arousal, Delivery, Valence, to_generation_config
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage
from tts_worker.clone_sample_quality import assess_clone_sample
from tts_worker.livekit_publisher import LiveKitTTSPublisher
from tts_worker.prosody_context import ProsodyContext
from tts_worker.synthesizer import CartesiaSynthesizer

# Standard WAV header size for the pcm_s16le format CartesiaSynthesizer requests —
# used to strip the header before feeding audio into the LiveKit track (which wants
# raw PCM frames, not a WAV container).
_WAV_HEADER_BYTES = 44

# Cartesia's voices.clone() requires a concrete `language`, but AudioChunkMessage.language
# defaults to "auto" (STT does language auto-detection, not the audio-chunk producer) — so
# fall back to "en" for anything Cartesia's SDK wouldn't accept as a real language code.
_CARTESIA_SUPPORTED_LANGUAGES = {
    "en",
    "fr",
    "de",
    "es",
    "pt",
    "zh",
    "ja",
    "hi",
    "it",
    "ko",
    "nl",
    "pl",
    "ru",
    "sv",
    "tr",
    "tl",
    "bg",
    "ro",
    "ar",
    "cs",
    "el",
    "fi",
    "hr",
    "ms",
    "sk",
    "da",
    "ta",
    "uk",
    "hu",
    "no",
    "vi",
    "bn",
    "th",
    "he",
    "ka",
    "id",
    "te",
    "gu",
    "kn",
    "ml",
    "mr",
    "pa",
}


def _clone_language(hint: str) -> str:
    return hint if hint in _CARTESIA_SUPPORTED_LANGUAGES else "en"


def _decode_field(data: Mapping[Any, Any], key: str) -> str:
    raw = data.get(key)
    if raw is None:
        raw = data.get(key.encode())
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


def _extract_tts_key(
    data: Mapping[Any, Any],
) -> tuple[str, str, str]:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_settings = tts_settings or TTSSettings()
        self.cartesia: CartesiaSynthesizer | None = None
        self.livekit_publisher: LiveKitTTSPublisher | None = None
        # (meeting_id, speaker_id, target_lang) -> lock serializing that key's own
        # messages — see _consume_loop for why.
        self._key_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        # One in-flight spoken turn per (meeting, speaker, language, voice). The per-key lock
        # above is what makes a plain dict safe here: a key's sentences are processed one at a
        # time, so a turn can never be pushed into concurrently.
        self._turns: dict[tuple[str, ...], ProsodyContext] = {}
        self._turn_connections: dict[tuple[str, ...], Any] = {}
        # Isochrony state, per (meeting, speaker, target language): how this speaker's dubs have
        # been running against the clock, and the turn currently being accumulated.
        self._dub_fits: dict[tuple[str, str, str], isochrony.DubFit] = {}
        self._turn_dub_ms: dict[tuple[str, str, str], int] = {}

    async def load_model(self) -> None:
        self.cartesia = CartesiaSynthesizer(
            api_key=self.tts_settings.api_key,
            model=self.tts_settings.model,
            sample_rate=self.tts_settings.sample_rate,
            speed=self.tts_settings.speed,
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
            key = _extract_tts_key(data)
            lock = self._key_locks.setdefault(key, asyncio.Lock())
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
        stale_keys = [key for key in self._key_locks if key[0] == room_id]
        for key in stale_keys:
            self._key_locks.pop(key, None)
        # getattr, because the tests build workers with __new__ and never run __init__ — the
        # same guard the rest of this codebase uses for that pattern. A worker with no turns
        # dict has no turns to abandon.
        turns: dict[tuple[str, ...], ProsodyContext] = getattr(self, "_turns", {})
        for turn_key in [k for k in turns if k[0] == room_id]:
            turn = turns.pop(turn_key, None)
            if turn is not None:
                # Fire-and-forget: _cleanup_room is sync (it is called from the route-state
                # broadcast handler), and a room that has ended is not waiting on a socket.
                asyncio.create_task(turn.abandon())

    async def _synthesize_sentence(
        self,
        *,
        translation: TranslationResultMessage,
        text: str,
        voice_id: str | None,
        voice_key: str,
        generation_config: dict[str, float | str] | None,
    ) -> tuple[bytes, int, str]:
        """One sentence of a turn, spoken in prosodic continuity with the ones before it.

        WT-371 follow-up / Level 4. A spoken turn is routinely split into several sentences
        (chunk_index > 0), and each used to be an independent one-shot generation with no memory
        of the one before it — so the model opened every sentence at its own default baseline and
        the dub came back as a list of separately-read sentences. Cartesia's contexts exist for
        exactly this; see tts_worker/prosody_context.py.

        Falls back to the proven one-shot path on ANY failure, and when the feature is off. That
        is not defensive padding: this WebSocket path has never run against the real API from
        this codebase, and a dub that fails is silence in a live meeting.
        """
        synthesizer = self._require_cartesia()
        resolved_voice_id = voice_id or CartesiaSynthesizer._default_voice_id(
            translation.target_lang
        )

        if not self.tts_settings.prosody_continuity:
            return await synthesizer.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
                generation_config=generation_config,
            )

        # Keyed by voice as well as by speaker and language: a clone upgrade replaces the voice
        # mid-meeting (voice_clone_max_upgrades), and continuing a turn into a different voice
        # would be worse than the seam this removes.
        key = (
            translation.meeting_id,
            translation.speaker_id,
            translation.target_lang,
            voice_key,
            resolved_voice_id,
        )

        try:
            turn = self._turns.get(key)
            if turn is None:
                turn, connection = await synthesizer.open_prosody_context(
                    context_id=f"{translation.speaker_id}:{translation.target_lang}:{voice_key}",
                    language=translation.target_lang,
                    voice_id=voice_id,
                )
                self._turns[key] = turn
                self._turn_connections[key] = connection

            audio_bytes, duration_ms = await turn.speak(text, generation_config)
            return audio_bytes, duration_ms, resolved_voice_id
        except Exception:
            self.logger.warning(
                "prosody_context_failed_falling_back",
                meeting_id=translation.meeting_id,
                exc_info=True,
            )
            await self._end_turn(key)
            return await synthesizer.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
                generation_config=generation_config,
            )
        finally:
            # The turn ends where the SPEAKER stopped, not where a chunk boundary fell —
            # is_final_chunk is the only signal that carries that.
            if translation.is_final_chunk:
                await self._end_turn(key)

    async def _end_turn(self, key: tuple[str, ...]) -> None:
        turn = self._turns.pop(key, None)
        connection = self._turn_connections.pop(key, None)
        if turn is not None:
            await turn.aclose()
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                self.logger.debug("prosody_connection_close_failed", exc_info=True)

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

        # S6. Never dub a listener back into the language the speaker is already speaking.
        # The listener is subscribed to BOTH the ai-interpreter track this would publish and
        # that speaker's raw mic, so synthesizing here plays the real voice and a synthetic
        # echo of the same sentence over each other.
        #
        # The producing side (translation_worker._get_target_languages) no longer builds
        # these messages, so in a healthy pipeline this never fires. It stays because this
        # worker is where the LiveKit track is actually published, and that is the last
        # place the echo can still be stopped: translate:results is a Redis stream that
        # outlives a deploy, so messages built by the previous revision are replayed into
        # this one, and any future producer inherits the guard for free rather than having
        # to remember it. Placed above the empty-text check so the final-chunk bookkeeping
        # below still runs — billing_worker and TranscriptRedisConsumerService key off
        # final_chunk_processed, and swallowing it would stall them on a silent segment.
        if is_same_language(translation.source_lang, translation.target_lang):
            self.logger.info(
                "same_language_synthesis_skipped",
                meeting_id=translation.meeting_id,
                speaker_id=translation.speaker_id,
                segment_id=translation.segment_id,
                lang=translation.target_lang,
            )
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

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

    async def _get_voice_catalog(self, language: str) -> list[dict[str, Any]]:
        """Redis-cached (TTL) list of public Cartesia voices for a language.

        Falls back to [] on any cache/fetch problem — callers must fall back to
        CartesiaSynthesizer._default_voice_id() rather than fail synthesis.
        """
        cache_key = f"voice_catalog:{language}"
        cached = await self.redis.get(cache_key)
        if cached:
            try:
                raw = cached.decode() if isinstance(cached, bytes) else cached
                return cast(list[dict[str, Any]], json.loads(raw))
            except Exception:
                self.logger.warning("voice_catalog_cache_corrupt", language=language)

        voices = await self._require_cartesia().list_voices(
            language, limit=self.tts_settings.voice_catalog_size
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
        return str(catalog[index]["id"])

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
            # Listeners store whatever tag their picker gave them, so an exact match dropped
            # anyone whose choice was spelled "vi-VN" against a target of "vi" — and a
            # listener nobody counts is a listener nobody synthesises for.
            if is_same_language(lang.decode() if isinstance(lang, bytes) else lang, target_lang)
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
        generation_config = self._generation_config(translation)

        cache_key = self._cache_key(
            speaker_id=translation.speaker_id,
            target_lang=translation.target_lang,
            # Two different voice_ids must never share a cache entry even when
            # voice_type matches (e.g. two distinct "preference" picks) — the concrete
            # voice_id, not just the type, is part of what was actually rendered.
            text=text,
            voice_mode=f"{voice_type}:{voice_id}",
            # Same words, same voice, said differently is DIFFERENT AUDIO. Without this the
            # first rendering of a phrase would be replayed for every later one, and a speaker
            # who said "okay" calmly and then shouted it would be dubbed identically both
            # times — the cache would quietly undo the whole feature.
            generation_config=generation_config,
        )

        if self.tts_settings.cache_enabled:
            cached_audio = await self.redis.get(cache_key)
            if cached_audio:
                if voice_key:
                    # Extra voice variant — LiveKit only, never a second billing event
                    # for content already billed via the default variant's publish.
                    cached_bytes = (
                        cached_audio.encode("utf-8")
                        if isinstance(cached_audio, str)
                        else cached_audio
                    )
                    await self._publish_livekit_only(translation, cached_bytes, voice_key)
                else:
                    cached_bytes = (
                        cached_audio.encode("utf-8")
                        if isinstance(cached_audio, str)
                        else cached_audio
                    )
                    await self._publish_result(
                        translation=translation,
                        audio_bytes=cached_bytes,
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
            audio_bytes, duration_ms, resolved_voice_id = await self._synthesize_sentence(
                translation=translation,
                text=text,
                voice_id=voice_id,
                voice_key=voice_key,
                generation_config=generation_config,
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
        # Already measured, and until now only ever attached to a published message. This is the
        # stage B2 clocked at p95 8.54s while STT and translation both stayed under 1.5s — kept
        # apart from the cumulative pipeline number so a slow Cartesia call and a queue building
        # behind the per-key lock are two readings rather than one.
        await self.redis.record_latency("tts_synthesis", synthesis_latency_ms)
        self._observe_dub_fit(translation, duration_ms)

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
            # Empty when the speaker's delivery was not measured — which is what makes
            # "is prosody actually reaching Cartesia in production?" answerable from the logs
            # instead of by inspection.
            generation_config=generation_config or None,
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
        # WT-371 #9: what the clone currently in use was built from, so a later clip can be
        # recognised as better. Absent until this speaker has been cloned in this process.
        cloned_score: dict[tuple[str, str], float] = {}
        upgrades_used: dict[tuple[str, str], int] = {}

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
                        #
                        # Asked in the form that can recover the routes from Redis and that
                        # reports WHICH no it is: production ran with zero cloned voices and
                        # every dub on the default catalog voice, and this branch — the one that
                        # swallows the whole meeting — said nothing at all on the way past.
                        consented, consent_reason = await self.voice_clone_consent_state(
                            chunk.meeting_id, chunk.speaker_id
                        )
                        if not consented:
                            self._note_clone_state(key, consent_reason)
                            buffers.pop(key, None)
                            buffer_seconds.pop(key, None)
                            buffer_lang.pop(key, None)
                            cloned_score.pop(key, None)
                            upgrades_used.pop(key, None)
                            continue

                        # WT-371 #9: this used to be `if already cloned: continue` — the worker
                        # stopped listening the moment it had any clone at all, so the voice was
                        # locked to whatever register the speaker opened the meeting in. Change
                        # your tone, or crack your voice, and the clone stopped being you.
                        #
                        # It keeps listening now, but only while an upgrade is still allowed, so a
                        # speaker whose clone is already good costs nothing beyond the buffer.
                        if await self._get_voice_id(chunk.meeting_id, chunk.speaker_id):
                            if (
                                upgrades_used.get(key, 0)
                                >= self.tts_settings.voice_clone_max_upgrades
                            ):
                                self._note_clone_state(key, "cloned_upgrades_exhausted")
                                buffers.pop(key, None)
                                buffer_seconds.pop(key, None)
                                buffer_lang.pop(key, None)
                                continue
                            # A clone made by ANOTHER replica, or before this process started, has
                            # no local score. Treat it as good enough to keep rather than racing to
                            # replace something we cannot compare against.
                            if key not in cloned_score:
                                self._note_clone_state(key, "cloned_elsewhere_kept")
                                continue

                        buffers.setdefault(key, bytearray()).extend(chunk.audio_data)
                        # PCM 16-bit mono: 2 bytes per sample
                        duration_s = len(chunk.audio_data) / 2 / max(chunk.sample_rate, 1)
                        buffer_seconds[key] = buffer_seconds.get(key, 0.0) + duration_s
                        buffer_lang[key] = chunk.language

                        if buffer_seconds[key] >= self.tts_settings.voice_clone_min_seconds:
                            # The clip is only a reference if it is worth referring to.
                            #
                            # This used to clone the first N seconds unconditionally, and
                            # _get_voice_id short-circuits, so a microphone check became the
                            # speaker's voice for the entire meeting. Now a clip that fails the
                            # same bar the upload page enforces is not cloned — the oldest audio
                            # slides out and the speaker gets another go, which costs nothing
                            # because they are still talking.
                            assessment = assess_clone_sample(bytes(buffers[key]), chunk.sample_rate)
                            previous_score = cloned_score.get(key)
                            is_upgrade = previous_score is not None
                            # An upgrade has to EARN the disruption: re-cloning changes the voice
                            # people are currently listening to, and small score differences are
                            # noise in the pitch estimator rather than a better reference.
                            if not assessment.accepted:
                                worth_cloning = False
                            elif previous_score is None:
                                worth_cloning = True
                            else:
                                worth_cloning = (
                                    assessment.score
                                    >= previous_score + self.tts_settings.voice_clone_upgrade_margin
                                )
                            if worth_cloning:
                                audio_snapshot = bytes(buffers.pop(key))
                                del buffer_seconds[key]
                                clone_lang = _clone_language(buffer_lang.pop(key, "en"))
                                cloned_score[key] = assessment.score
                                if is_upgrade:
                                    upgrades_used[key] = upgrades_used.get(key, 0) + 1
                                self.logger.info(
                                    "voice_clone_sample_accepted",
                                    speaker_id=chunk.speaker_id,
                                    seconds=round(
                                        buffer_seconds.get(key, 0.0)
                                        or self.tts_settings.voice_clone_min_seconds,
                                        1,
                                    ),
                                    active_speech_ratio=round(assessment.active_speech_ratio, 3),
                                    pitch_semitones=round(assessment.pitch_semitone_range, 2),
                                    score=round(assessment.score, 3),
                                    upgrade=is_upgrade,
                                )
                                self._note_clone_state(key, "cloning")
                                asyncio.create_task(
                                    self._clone_and_cache(
                                        chunk.meeting_id,
                                        chunk.speaker_id,
                                        audio_snapshot,
                                        clone_lang,
                                    )
                                )
                            elif assessment.accepted:
                                # Usable, but no better than the clone already in use. Slide the
                                # window on and keep listening — the speaker may yet say something
                                # that covers more of their range.
                                self._trim_clone_buffer(
                                    key, buffers, buffer_seconds, chunk.sample_rate
                                )
                            else:
                                # Logged at info, not warning: a rejected clip is the gate doing
                                # its job, and a speaker who has not said anything usable yet is
                                # an ordinary state, not a fault.
                                self.logger.info(
                                    "voice_clone_sample_rejected",
                                    speaker_id=chunk.speaker_id,
                                    reason=assessment.reason,
                                    rms=round(assessment.rms, 4),
                                    active_speech_ratio=round(assessment.active_speech_ratio, 3),
                                )
                                self._trim_clone_buffer(
                                    key, buffers, buffer_seconds, chunk.sample_rate
                                )
                    except Exception:
                        self.logger.exception("audio_chunk_processing_error")
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("audio_consumer_error")
                await asyncio.sleep(2)

    def _note_clone_state(self, key: tuple[str, str], reason: str) -> None:
        """Say why this speaker is not on a cloned voice — once per change, not once per chunk.

        Audio chunks arrive continuously for the whole meeting, so logging at each of these
        branches directly would bury the pipeline in one line per chunk per speaker and get the
        level turned back down within a day. Only a CHANGE is news.

        This exists because production ran with `voice:*` empty and 97 of 97 dubbed segments on
        `voice_type=default`, and not one line in any log said why. Every exit before the clone
        call returned in silence, so the difference between "nobody opted in" and "this worker
        never learned the room's routes" was invisible from outside.
        """
        if getattr(self, "_clone_state", None) is None:
            self._clone_state: dict[tuple[str, str], str] = {}
        if self._clone_state.get(key) == reason:
            return
        self._clone_state[key] = reason
        self.logger.info(
            "voice_clone_state",
            meeting_id=key[0],
            speaker_id=key[1],
            reason=reason,
        )

    def _trim_clone_buffer(
        self,
        key: tuple[str, str],
        buffers: dict[tuple[str, str], bytearray],
        buffer_seconds: dict[tuple[str, str], float],
        sample_rate: int,
    ) -> None:
        """Drop the oldest audio so a rejected clip does not block the next attempt forever.

        A sliding window, not a reset. Resetting would throw away the speech that arrived while
        the clip was being judged, and in a room where every window is marginal it would mean
        never assembling a usable one. Keeping everything is the other failure: a speaker in a
        noisy office would hold the whole meeting in memory and still never clone.
        """
        buffer = buffers.get(key)
        if buffer is None or sample_rate <= 0:
            return

        max_bytes = int(self.tts_settings.voice_clone_max_buffer_seconds * sample_rate * 2)
        if max_bytes <= 0 or len(buffer) <= max_bytes:
            return

        overflow = len(buffer) - max_bytes
        del buffer[:overflow]
        buffer_seconds[key] = len(buffer) / 2 / sample_rate

    async def _clone_and_cache(
        self, meeting_id: str, speaker_id: str, audio_bytes: bytes, language: str = "en"
    ) -> None:
        """Clone voice via Cartesia and cache voice_id in Redis."""
        label = f"speaker-{speaker_id[:8]}-{meeting_id[:8]}"
        try:
            voice_id = await self._require_cartesia().clone_voice(
                audio_bytes,
                label,
                language,
            )
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

    def _generation_config(
        self, translation: TranslationResultMessage
    ) -> dict[str, float | str] | None:
        """Cartesia's delivery controls for this line, or None to say nothing about delivery.

        None is the important case and it is the common one: the STT worker omits prosody
        whenever it could not honestly measure it (a speaker it has not heard enough of, a
        chunk that was mostly silence), and this returns None again whenever the feature is
        off. In every one of those cases the call is byte-for-byte the call this worker made
        before prosody existed.
        """
        if not self.tts_settings.prosody_enabled:
            return None

        envelope = translation.prosody
        if envelope is None:
            return None

        # Rebuilt rather than passed around as a Delivery: the wire format is deliberately
        # plain numbers (schemas.py knows nothing about numpy or about prosody's vocabulary),
        # and this is the one place that translates it back.
        arousal: Arousal = (
            envelope.arousal if envelope.arousal in ("low", "neutral", "high") else "neutral"  # type: ignore[assignment]
        )
        valence: Valence | None = (
            envelope.valence  # type: ignore[assignment]
            if envelope.valence in ("negative", "neutral", "positive")
            else None
        )

        return to_generation_config(
            Delivery(
                pitch_lift=envelope.pitch_lift,
                pitch_variation=envelope.pitch_variation,
                energy_ratio=envelope.energy_ratio,
                rate_ratio=envelope.rate_ratio,
                arousal=arousal,
            ),
            valence,
            # Isochrony. The centre this speaker's tempo is applied AROUND, learned from how
            # their previous dubs actually ran against the clock — see shared/isochrony.py.
            # Exactly 1.0 until a fit is established, which is byte-for-byte the previous
            # behaviour. Their own rate_ratio still multiplies through it, so somebody who
            # genuinely slowed down still sounds like they slowed down, inside a slot that fits.
            speed_center=isochrony.speed_center(self._dub_fit(translation)),
        )

    def _fit_key(self, translation: TranslationResultMessage) -> tuple[str, str, str]:
        """Fit is per (meeting, speaker, target language). Not global: how much longer a dub
        runs is a property of the language pair and of how this person talks, and pooling a
        terse speaker with a discursive one would centre both on neither."""
        return (translation.meeting_id, translation.speaker_id, translation.target_lang)

    def _dub_fit(self, translation: TranslationResultMessage) -> isochrony.DubFit:
        fits: dict[tuple[str, str, str], isochrony.DubFit] = getattr(self, "_dub_fits", {})
        return fits.get(self._fit_key(translation), isochrony.NO_FIT)

    def _observe_dub_fit(self, translation: TranslationResultMessage, dub_ms: int) -> None:
        """Accumulate this sentence's dub, and compare the WHOLE turn once the turn is over.

        The comparison is turn against turn. `start_ms`/`end_ms` describe the whole spoken turn,
        so weighing one sentence's dub against them would report a fit of about 1/N for an
        N-sentence turn and drive the controller to speak everybody faster and faster. The
        sentences are summed and the total is what gets folded in on `is_final_chunk`.
        """
        if not self.tts_settings.prosody_enabled:
            return

        key = self._fit_key(translation)
        pending: dict[tuple[str, str, str], int] = getattr(self, "_turn_dub_ms", None) or {}
        self._turn_dub_ms = pending
        pending[key] = pending.get(key, 0) + max(0, dub_ms)

        if not translation.is_final_chunk:
            return

        turn_dub_ms = pending.pop(key, 0)
        source_ms = translation.end_ms - translation.start_ms
        if source_ms <= 0:
            return

        fits: dict[tuple[str, str, str], isochrony.DubFit] = getattr(self, "_dub_fits", None) or {}
        self._dub_fits = fits
        fits[key] = isochrony.observe(fits.get(key, isochrony.NO_FIT), source_ms, turn_dub_ms)

    @staticmethod
    def _cache_key(
        speaker_id: str,
        target_lang: str,
        text: str,
        voice_mode: str,
        generation_config: dict[str, float | str] | None = None,
    ) -> str:
        normalized = " ".join(text.casefold().split())
        material = f"{speaker_id}|{target_lang}|{normalized}|{voice_mode}"
        if generation_config:
            # Appended only when there ARE delivery controls, so a line with none hashes to
            # exactly the key it hashed to before prosody existed and the warm cache survives
            # the deploy. Sorted so two configs with the same content but a different insertion
            # order share one entry instead of rendering twice.
            material += "|" + json.dumps(generation_config, sort_keys=True, separators=(",", ":"))
        return f"tts:cache:{hashlib.sha256(material.encode()).hexdigest()}"

    def _require_cartesia(self) -> CartesiaSynthesizer:
        if self.cartesia is None:
            raise RuntimeError("Cartesia synthesizer is not loaded")
        return self.cartesia
