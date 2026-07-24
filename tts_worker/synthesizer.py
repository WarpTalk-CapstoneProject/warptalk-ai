"""Cartesia TTS synthesizer — voice cloning + synthesis via Cartesia API.

Replaces XTTS v2 (CPML non-commercial license) and Edge-TTS (no voice cloning).
TTFA: 40ms (Sonic Turbo). Voice cloning: 10-15s audio sample → voice_id via API.
"""

from __future__ import annotations

import io

from shared.logger import get_logger

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
        model: str = "sonic-turbo",
        sample_rate: int = 44100,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self._client = None

    async def load(self) -> None:
        from cartesia import AsyncCartesia

        self._client = AsyncCartesia(api_key=self.api_key)
        logger.info("cartesia_ready", model=self.model, sample_rate=self.sample_rate)

    async def clone_voice(self, audio_bytes: bytes, speaker_label: str, language: str = "en") -> str:
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
        voice = await self._client.voices.clone(
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
    ) -> tuple[bytes, int, str]:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize
            language: ISO 639-1 language code (e.g. "en", "vi")
            voice_id: Cartesia voice_id from clone_voice(); None uses Cartesia default

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

        voice: dict = {"id": resolved_voice_id}

        # cartesia-py 3.x's tts.bytes() is an async method whose awaited result is itself an
        # AsyncIterator[bytes] (it streams chunks) — never a plain bytes object, despite this
        # code's prior assumption. Needs BOTH the await (to get the stream) AND an async-for
        # (to drain it) — awaiting alone previously raised "object of type 'async_generator'
        # has no len()"; async-for alone (without the await) raises "'async for' requires an
        # object with __aiter__ method, got coroutine". Cartesia was unreachable/misconfigured
        # before now, so neither mistake had ever been exercised end-to-end.
        stream = await self._client.tts.bytes(
            model_id=self.model,
            transcript=text,
            voice=voice,
            output_format={
                "container": "wav",
                "sample_rate": self.sample_rate,
                "encoding": "pcm_s16le",
            },
            language=language,
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
        return defaults.get(language, defaults["en"])

    async def list_voices(self, language: str, limit: int = 12, max_scanned: int = 300) -> list[dict]:
        """Public library voices for `language`, from Cartesia's real /voices API.

        The SDK's voices.list() has no `language` filter param — it returns Cartesia's
        whole public library (600+ voices across 40+ languages), so this scans pages
        client-side and keeps only matches, capped at `max_scanned` total items looked
        at so a rare language can't force an unbounded scan of the whole catalog.

        Best-effort: any failure (network, auth, SDK shape change) returns [] rather
        than raising — every caller must treat an empty result as "fall back to
        _default_voice_id()", never as a reason to fail synthesis.
        """
        voices: list[dict] = []
        scanned = 0
        try:
            async for voice in self._client.voices.list(is_owner=False, limit=100):
                scanned += 1
                if voice.language == language:
                    voices.append({
                        "id": voice.id,
                        "name": voice.name,
                        "gender": voice.gender or "",
                    })
                    if len(voices) >= limit:
                        break
                if scanned >= max_scanned:
                    break
        except Exception:
            logger.exception("cartesia_list_voices_failed", language=language)
            return []
        return voices
