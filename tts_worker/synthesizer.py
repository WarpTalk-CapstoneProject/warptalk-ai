"""Cartesia TTS synthesizer — voice cloning + synthesis via Cartesia API.

Replaces XTTS v2 (CPML non-commercial license) and Edge-TTS (no voice cloning).
TTFA: 40ms (Sonic Turbo). Voice cloning: 10-15s audio sample → voice_id via API.
"""

from __future__ import annotations

import io
from typing import Any, cast

from cartesia import AsyncCartesia

from shared.lang import base_language
from shared.logger import get_logger
from tts_worker.prosody_context import SENTENCE_TIMEOUT_SECONDS, ProsodyContext

logger = get_logger(__name__)


class CartesiaSynthesizer:
    """Cartesia Sonic Turbo synthesizer with voice cloning.

    Voice cloning workflow:
        1. Buffer 10-15s of raw speaker audio (handled by TTSWorker)
        2. Call clone_voice() → returns voice_id (cached in Redis)
        3. All subsequent synthesize() calls use that voice_id
    """

    provider_name = "cartesia"

    def __init__(
        self,
        api_key: str,
        model: str = "sonic-3.5",
        sample_rate: int = 44100,
        speed: str = "fast",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        # "fast" by default, not "normal".
        #
        # A dub is not a narration: it has to fit inside the gap the original speaker left,
        # and the listener is waiting on it before the conversation can move. At "normal" the
        # translated line consistently finished well after the speaker had moved on, which
        # reads as the system being slow even when the pipeline latency is fine.
        #
        # Cartesia accepts only "slow" | "normal" | "fast" (cartesia.types.ModelSpeed), so
        # this is the whole of the available range, not a tuned number.
        self.speed = speed
        self._client: AsyncCartesia | None = None

    async def load(self) -> None:
        self._client = AsyncCartesia(api_key=self.api_key)
        logger.info("cartesia_ready", model=self.model, sample_rate=self.sample_rate)

    async def open_prosody_context(
        self,
        *,
        context_id: str,
        language: str,
        voice_id: str | None = None,
    ) -> tuple[ProsodyContext, Any]:
        """A single prosodic thread for one spoken turn — see tts_worker/prosody_context.py.

        Returns the context AND the connection that owns it, because the caller has to keep the
        connection alive for the whole turn and close it afterwards; a context outliving its
        socket is just a closed socket with extra steps.

        Raw PCM rather than a WAV container: this is a stream, so there is no total length to
        put in a header up front. ProsodyContext re-wraps each sentence in the 44-byte header
        the publish path expects, so nothing downstream can tell the difference.
        """
        client = self._require_client()
        connection = await client.tts.websocket_connect().enter()
        context = connection.context(
            context_id=context_id,
            # Belt and braces with ProsodyContext's own asyncio.timeout: this one is the SDK's
            # and may abort a wedged read closer to the socket, the other one is ours and is
            # what actually guarantees the caller gets an exception it can fall back from.
            timeout=SENTENCE_TIMEOUT_SECONDS,
            model_id=self.model,
            voice=cast(Any, {"id": voice_id or self._default_voice_id(language)}),
            language=cast(Any, language),
            output_format=cast(
                Any,
                {
                    "container": "raw",
                    "sample_rate": self.sample_rate,
                    "encoding": "pcm_s16le",
                },
            ),
        )
        return ProsodyContext(context, self.sample_rate), connection

    async def clone_voice(
        self, audio_bytes: bytes, speaker_label: str, language: str = "en"
    ) -> str:
        """Clone a speaker's voice from raw audio.

        Args:
            audio_bytes: Raw audio bytes (WAV/PCM), minimum ~10s
            speaker_label: Human-readable label for the cloned voice
            language: Speaker's source language (Cartesia's `language` param is
                required by the installed SDK; "auto"/unsupported hints are the
                caller's job to normalize before calling this)

        Returns:
            Cartesia voice_id to use in subsequent synthesize() calls
        """
        audio_io = io.BytesIO(audio_bytes)

        # cartesia==3.3.0's AsyncVoicesResource.clone() has no `enhance` kwarg
        # (that was a stale/pre-GA param name) and requires `language`.
        client = self._require_client()
        voice = await client.voices.clone(
            clip=audio_io,
            name=speaker_label,
            language=language,
        )
        voice_id: str = voice.id
        logger.info("voice_cloned", label=speaker_label, voice_id=voice_id)
        return voice_id

    async def synthesize(
        self,
        text: str,
        language: str,
        voice_id: str | None = None,
        generation_config: dict[str, float | str] | None = None,
    ) -> tuple[bytes, int, str]:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize
            language: ISO 639-1 language code (e.g. "en", "vi")
            voice_id: Cartesia voice_id from clone_voice(); None uses Cartesia default
            generation_config: Cartesia's delivery controls — {"speed": float in [0.6, 1.5],
                "volume": float, "emotion": str}. Built from the speaker's measured prosody
                (shared/prosody.py) and omitted entirely when nothing was measured, so an
                unmeasured utterance is synthesized exactly as it was before this existed.

        Returns:
            Tuple of (wav_bytes, duration_ms, resolved_voice_id). resolved_voice_id is
            always a real Cartesia voice id — the cloned one if `voice_id` was given,
            otherwise whichever built-in default this call actually used. Callers need
            this even in the default case: it's the only place that knows which id was
            used, and it's required to record which voice actually produced a given
            audio_dubbings row (see transcript.audio_dubbings.provider_voice_id).
        """
        resolved_voice_id = voice_id or self._default_voice_id(language)

        if not text.strip():
            return b"", 0, resolved_voice_id

        voice: dict[str, str] = {"id": resolved_voice_id}

        # cartesia-py 3.x's tts.bytes() is an async method whose awaited result is itself an
        # AsyncIterator[bytes] (it streams chunks) — never a plain bytes object, despite this
        # code's prior assumption. Needs BOTH the await (to get the stream) AND an async-for
        # (to drain it) — awaiting alone previously raised "object of type 'async_generator'
        # has no len()"; async-for alone (without the await) raises "'async for' requires an
        # object with __aiter__ method, got coroutine". Cartesia was unreachable/misconfigured
        # before now, so neither mistake had ever been exercised end-to-end.
        client = self._require_client()
        # `speed` (the ModelSpeed enum below) and `generation_config["speed"]` are two separate
        # inputs and both are sent. The enum measurably does nothing on sonic-3.5 — four renders
        # each of slow/normal/fast gave 6120/6200/5960ms medians against a 5680–6560ms spread —
        # but it is not inert on every Cartesia model, so dropping it would be a silent
        # behaviour change for anyone running one where it works.
        extra: dict[str, Any] = {}
        if generation_config:
            extra["generation_config"] = cast(Any, generation_config)

        stream = await client.tts.bytes(
            model_id=self.model,
            transcript=text,
            voice=cast(Any, voice),
            output_format=cast(
                Any,
                {
                    "container": "wav",
                    "sample_rate": self.sample_rate,
                    "encoding": "pcm_s16le",
                },
            ),
            language=language,
            speed=cast(Any, self.speed),
            **extra,
        )
        chunks: list[bytes] = [chunk async for chunk in stream]
        audio_bytes: bytes = b"".join(chunks)

        # WAV header is 44 bytes; remaining = PCM samples (16-bit mono)
        pcm_bytes = max(0, len(audio_bytes) - 44)
        duration_ms = int(pcm_bytes / 2 / self.sample_rate * 1000)

        return audio_bytes, duration_ms, resolved_voice_id

    @staticmethod
    def _default_voice_id(language: str) -> str:
        """Built-in Cartesia voice IDs per language — last-resort fallback used only
        when list_voices() below can't be reached or returns nothing for this
        language (never fabricate additional IDs beyond these two confirmed-working
        ones; anything else must come from Cartesia's real /voices API)."""
        defaults = {
            "en": "694f9389-aac1-45b6-b726-9d9369183238",  # Cartesia "Barbershop Man"
            "vi": "5619d38c-cf51-4d8e-9575-48f61a280413",  # Cartesia Vietnamese voice
        }
        # base_language first. This dict is keyed by primary subtag, and the lookup used the
        # tag verbatim — so "vi-VN" missed "vi" and fell through to the English default. A
        # Vietnamese-only meeting then spoke English, which is the report. The fallback is
        # meant for a language nobody has a voice for, not for a spelling of one we do.
        return defaults.get(base_language(language), defaults["en"])

    async def list_voices(
        self, language: str, limit: int = 12, max_scanned: int = 2000
    ) -> list[dict[str, Any]]:
        """Public library voices for `language`, from Cartesia's real /voices API.

        The SDK's voices.list() has no `language` filter param — it returns Cartesia's
        whole public library, so this scans pages client-side and keeps only matches.

        `max_scanned` is a runaway guard, NOT a budget: it must stay comfortably above
        the real library size or it silently starves whole languages. It used to be 300
        against a library of ~843 voices ordered with English first (417 of them), and
        the first Vietnamese voice sits at position 459 — so `vi` was unreachable, the
        control bar's picker stayed empty in Vietnamese, and _hashed_default_voice_id
        always fell back to the single hardcoded `vi` id (every speaker got one voice).
        Scanning the whole library costs ~9 pages of 100 and the result is cached in
        Redis for 6h (see tts_worker._get_voice_catalog), so the walk is rare.

        Best-effort: any failure (network, auth, SDK shape change) returns [] rather
        than raising — every caller must treat an empty result as "fall back to
        _default_voice_id()", never as a reason to fail synthesis.
        """
        voices: list[dict[str, Any]] = []
        scanned = 0
        try:
            client = self._require_client()
            async for voice in client.voices.list(is_owner=False, limit=100):
                scanned += 1
                # Same reason as _default_voice_id: Cartesia keys its library by primary
                # subtag, so comparing a full tag matched nothing and starved the catalog —
                # which then fell back to _default_voice_id and produced English anyway.
                if base_language(voice.language or "") == base_language(language):
                    voices.append(
                        {
                            "id": voice.id,
                            "name": voice.name,
                            "gender": voice.gender or "",
                        }
                    )
                    if len(voices) >= limit:
                        break
                if scanned >= max_scanned:
                    logger.warning(
                        "cartesia_list_voices_scan_capped",
                        language=language,
                        scanned=scanned,
                        found=len(voices),
                    )
                    break
        except Exception:
            logger.exception("cartesia_list_voices_failed", language=language)
            return []
        return voices

    async def list_owned_voices(self, max_scanned: int = 5000) -> list[dict[str, Any]]:
        """Voices this ACCOUNT owns — the ones we created, not the public library.

        `is_owner=True` is the whole difference from `list_voices`, which asks for the
        opposite. That one is looking for something to speak with, so it filters by language
        and stops at a handful; this one is looking for something to DELETE, so it filters by
        nothing and returns `created_at` and `name` — the caller decides what is garbage and
        cannot do that without seeing every candidate.

        Best-effort, same as `list_voices`: any failure returns [] rather than raising, which
        makes the sweep a no-op for this cycle instead of taking the worker down.
        """
        voices: list[dict[str, Any]] = []
        try:
            client = self._require_client()
            async for voice in client.voices.list(is_owner=True, limit=100):
                voices.append(
                    {
                        "id": voice.id,
                        "name": voice.name or "",
                        "created_at": voice.created_at,
                    }
                )
                if len(voices) >= max_scanned:
                    logger.warning("cartesia_list_owned_voices_scan_capped", scanned=len(voices))
                    break
        except Exception:
            logger.exception("cartesia_list_owned_voices_failed")
            return []
        return voices

    async def delete_voice(self, voice_id: str) -> bool:
        """Remove one voice from this account. True when it is gone.

        Best-effort by the same rule as the two list calls: a failure here leaves a voice in
        the account until the next sweep, which is exactly the state the sweep started from.
        Nothing a cleanup path does is worth raising into its caller.

        A voice that is already gone answers 404 and counts as failure here, which is
        harmless — the sweep will not see it again to retry.
        """
        try:
            await self._require_client().voices.delete(voice_id)
        except Exception:
            logger.warning("cartesia_delete_voice_failed", voice_id=voice_id, exc_info=True)
            return False
        logger.info("cartesia_voice_deleted", voice_id=voice_id)
        return True

    async def rename_voice(self, voice_id: str, name: str) -> bool:
        """Rename one voice in this account. True when the new name is stored.

        NOT best-effort, unlike `delete_voice` and the list calls beside it, and the difference
        decides whether a voice leaks. The orphan sweep judges a voice by its NAME
        (`_IN_MEETING_VOICE_PREFIX`), so this rename is the only thing that takes a carried-over
        clone out of the sweep's sights. Its caller must not hand the voice to AuthService unless
        this returned True: a stored profile pointing at a voice still named `speaker-` is a row
        that goes dead in 24 hours, and the person it belongs to would hear a stranger.

        So the answer is reported honestly and the caller decides, rather than being swallowed
        into a shrug the way a cleanup failure can be.
        """
        try:
            await self._require_client().voices.update(voice_id, name=name)
        except Exception:
            logger.warning(
                "cartesia_rename_voice_failed", voice_id=voice_id, name=name, exc_info=True
            )
            return False
        logger.info("cartesia_voice_renamed", voice_id=voice_id, name=name)
        return True

    def _require_client(self) -> AsyncCartesia:
        if self._client is None:
            raise RuntimeError("Cartesia synthesizer is not loaded")
        return self._client
