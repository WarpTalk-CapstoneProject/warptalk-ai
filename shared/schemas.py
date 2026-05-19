"""Pydantic message schemas — contracts between workers.

Each schema provides `to_redis()` / `from_redis()` for Redis Streams
serialization. Field names align with backend TranscriptSegmentDto.
"""

from __future__ import annotations

import base64
import time
import uuid

from pydantic import BaseModel, Field


class AudioChunkMessage(BaseModel):
    """Gateway → STT Worker.

    Represents a raw audio chunk from a meeting participant.
    """

    __slots__ = ()

    meeting_id: str
    speaker_id: str
    chunk_index: int
    audio_data: bytes  # Raw audio bytes (WAV/PCM)
    language: str = "auto"  # Source language hint or 'auto' for detection
    sample_rate: int = 16000
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    model_config = {"arbitrary_types_allowed": True}

    def to_redis(self) -> dict[str, str]:
        """Serialize to Redis Stream fields (all str values)."""
        return {
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "chunk_index": str(self.chunk_index),
            "audio_data": base64.b64encode(self.audio_data).decode("ascii"),
            "language": self.language,
            "sample_rate": str(self.sample_rate),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> AudioChunkMessage:
        """Deserialize from Redis Stream fields."""
        d = _decode_dict(data)
        return cls(
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            chunk_index=int(d["chunk_index"]),
            audio_data=base64.b64decode(d["audio_data"]),
            language=d.get("language", "auto"),
            sample_rate=int(d.get("sample_rate", "16000")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class STTResultMessage(BaseModel):
    """STT Worker → Translation Worker + AI Assistant.

    Represents a transcribed text segment.
    """

    __slots__ = ()

    segment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    meeting_id: str
    speaker_id: str
    text: str
    language: str  # Detected source language code (e.g. 'en', 'vi')
    confidence: float = 0.0
    start_ms: int = 0  # Segment start time relative to meeting
    end_ms: int = 0  # Segment end time relative to meeting
    chunk_index: int = 0
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "segment_id": self.segment_id,
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "text": self.text,
            "language": self.language,
            "confidence": str(self.confidence),
            "start_ms": str(self.start_ms),
            "end_ms": str(self.end_ms),
            "chunk_index": str(self.chunk_index),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> STTResultMessage:
        d = _decode_dict(data)
        return cls(
            segment_id=d.get("segment_id", str(uuid.uuid4())),
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            text=d["text"],
            language=d["language"],
            confidence=float(d.get("confidence", "0.0")),
            start_ms=int(d.get("start_ms", "0")),
            end_ms=int(d.get("end_ms", "0")),
            chunk_index=int(d.get("chunk_index", "0")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class TranslationResultMessage(BaseModel):
    """Translation Worker → TTS Worker.

    Represents a translated text segment ready for speech synthesis.
    """

    __slots__ = ()

    translation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: str
    meeting_id: str
    speaker_id: str
    original_text: str
    translated_text: str
    source_lang: str  # e.g. 'en'
    target_lang: str  # e.g. 'vi'
    confidence: float = 0.0
    start_ms: int = 0
    end_ms: int = 0
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "translation_id": self.translation_id,
            "segment_id": self.segment_id,
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "confidence": str(self.confidence),
            "start_ms": str(self.start_ms),
            "end_ms": str(self.end_ms),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> TranslationResultMessage:
        d = _decode_dict(data)
        return cls(
            translation_id=d.get("translation_id", str(uuid.uuid4())),
            segment_id=d["segment_id"],
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            original_text=d["original_text"],
            translated_text=d["translated_text"],
            source_lang=d["source_lang"],
            target_lang=d["target_lang"],
            confidence=float(d.get("confidence", "0.0")),
            start_ms=int(d.get("start_ms", "0")),
            end_ms=int(d.get("end_ms", "0")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class TTSResultMessage(BaseModel):
    """TTS Worker → Gateway.

    Represents synthesized audio to send back to meeting participants.
    """

    __slots__ = ()

    segment_id: str
    meeting_id: str
    speaker_id: str
    audio_data: bytes  # Synthesized audio bytes (WAV)
    duration_ms: int = 0  # Audio duration in milliseconds
    voice_type: str = "default"  # 'default' | 'cloned'
    target_lang: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    model_config = {"arbitrary_types_allowed": True}

    def to_redis(self) -> dict[str, str]:
        return {
            "segment_id": self.segment_id,
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "audio_data": base64.b64encode(self.audio_data).decode("ascii"),
            "duration_ms": str(self.duration_ms),
            "voice_type": self.voice_type,
            "target_lang": self.target_lang,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> TTSResultMessage:
        d = _decode_dict(data)
        return cls(
            segment_id=d["segment_id"],
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            audio_data=base64.b64decode(d["audio_data"]),
            duration_ms=int(d.get("duration_ms", "0")),
            voice_type=d.get("voice_type", "default"),
            target_lang=d.get("target_lang", ""),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_dict(data: dict[bytes | str, bytes | str]) -> dict[str, str]:
    """Decode Redis byte keys/values to str."""
    return {
        (k.decode() if isinstance(k, bytes) else k): (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in data.items()
    }
