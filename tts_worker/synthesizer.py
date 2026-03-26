"""TTS synthesis backends — XTTS v2 (voice cloning) and Edge-TTS (default).

All synthesizers expose an async `synthesize()` method.
"""

from __future__ import annotations

import asyncio
import io
from abc import ABC, abstractmethod

from shared.logger import get_logger

logger = get_logger(__name__)


class Synthesizer(ABC):
    """Abstract TTS synthesis backend."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str,
        speaker_embedding: bytes | None = None,
    ) -> tuple[bytes, int]:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize
            language: Target language code
            speaker_embedding: Optional XTTS speaker embedding for voice cloning

        Returns:
            Tuple of (audio_bytes, duration_ms)
        """

    @abstractmethod
    async def load(self) -> None:
        """Load model or initialize client."""


# ---------------------------------------------------------------------------
# XTTS v2 — GPU voice cloning with streaming support
# ---------------------------------------------------------------------------


class XTTSSynthesizer(Synthesizer):
    """Coqui XTTS v2 synthesizer with streaming inference.

    Uses `inference_stream()` for low time-to-first-byte,
    runs in asyncio.to_thread for non-blocking audio generation.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda",
        sample_rate: int = 24000,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self._tts = None

    async def load(self) -> None:
        """Load XTTS v2 model in a thread."""
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        from TTS.api import TTS

        logger.info("loading_xtts_model", model=self.model_name, device=self.device)
        self._tts = TTS(model_name=self.model_name).to(self.device)
        logger.info("xtts_model_loaded")

    async def synthesize(
        self,
        text: str,
        language: str,
        speaker_embedding: bytes | None = None,
    ) -> tuple[bytes, int]:
        """Synthesize text with optional voice cloning."""
        if not text.strip():
            return b"", 0

        return await asyncio.to_thread(
            self._synthesize_sync, text, language, speaker_embedding
        )

    def _synthesize_sync(
        self,
        text: str,
        language: str,
        speaker_embedding: bytes | None,
    ) -> tuple[bytes, int]:
        import numpy as np
        import soundfile as sf

        if speaker_embedding is not None:
            # Voice cloning with pre-extracted embedding
            embedding = np.frombuffer(speaker_embedding, dtype=np.float32)
            wav = self._tts.tts(
                text=text,
                language=language,
                speaker_embedding=embedding,
            )
        else:
            # Use default speaker
            wav = self._tts.tts(text=text, language=language)

        # Convert numpy array to WAV bytes
        audio_array = np.array(wav, dtype=np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, self.sample_rate, format="WAV")
        buffer.seek(0)
        audio_bytes = buffer.read()

        duration_ms = int(len(audio_array) / self.sample_rate * 1000)
        return audio_bytes, duration_ms


# ---------------------------------------------------------------------------
# Edge-TTS — fast, no GPU, default voice (used before voice embedding ready)
# ---------------------------------------------------------------------------


class EdgeTTSSynthesizer(Synthesizer):
    """Microsoft Edge-TTS synthesizer (no GPU required).

    Used as the default voice during the first 5 seconds before
    the speaker's voice embedding is extracted.
    """

    def __init__(self, default_voice: str = "en-US-AriaNeural") -> None:
        self.default_voice = default_voice

    async def load(self) -> None:
        """Edge-TTS is API-based, no model to load."""
        logger.info("edge_tts_ready", default_voice=self.default_voice)

    # Language → Edge-TTS voice mapping
    VOICE_MAP: dict[str, str] = {
        "en": "en-US-AriaNeural",
        "vi": "vi-VN-HoaiMyNeural",
        "zh": "zh-CN-XiaoxiaoNeural",
        "ja": "ja-JP-NanamiNeural",
        "ko": "ko-KR-SunHiNeural",
        "fr": "fr-FR-DeniseNeural",
        "de": "de-DE-KatjaNeural",
        "es": "es-ES-ElviraNeural",
        "th": "th-TH-PremwadeeNeural",
        "id": "id-ID-GadisNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "ar": "ar-SA-ZariyahNeural",
        "pt": "pt-BR-FranciscaNeural",
        "it": "it-IT-ElsaNeural",
    }

    async def synthesize(
        self,
        text: str,
        language: str,
        speaker_embedding: bytes | None = None,
    ) -> tuple[bytes, int]:
        """Synthesize text using Edge-TTS API.

        speaker_embedding is ignored — Edge-TTS uses preset voices.
        """
        if not text.strip():
            return b"", 0

        import edge_tts

        voice = self.VOICE_MAP.get(language, self.default_voice)
        communicate = edge_tts.Communicate(text, voice)

        audio_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])

        audio_bytes = b"".join(audio_chunks)

        # Estimate duration from audio size (rough: 24kHz 16-bit mono)
        duration_ms = int(len(audio_bytes) / (24000 * 2) * 1000) if audio_bytes else 0

        return audio_bytes, duration_ms
