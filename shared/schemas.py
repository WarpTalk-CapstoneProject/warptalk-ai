"""Pydantic message schemas — contracts between workers.

Each schema provides `to_redis()` / `from_redis()` for Redis Streams
serialization. Field names align with backend TranscriptSegmentDto.
"""

from __future__ import annotations

import base64
import time
import uuid
from decimal import Decimal

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
    source_runtime: str = "web"  # 'web' | 'desktop'
    vad_confidence: float = 0.0
    speech_start_ms: int = 0
    speech_end_ms: int = 0
    input_lufs: float = 0.0
    noise_suppression_enabled: bool = False
    is_final_chunk: bool = False
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
            "source_runtime": self.source_runtime,
            "vad_confidence": str(self.vad_confidence),
            "speech_start_ms": str(self.speech_start_ms),
            "speech_end_ms": str(self.speech_end_ms),
            "input_lufs": str(self.input_lufs),
            "noise_suppression_enabled": _bool_to_redis(self.noise_suppression_enabled),
            "is_final_chunk": "1" if self.is_final_chunk else "0",
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> AudioChunkMessage:
        """Deserialize from Redis Stream fields."""
        d = _decode_dict(data)
        return cls(
            meeting_id=d.get("meeting_id") or d["translation_room_id"],
            speaker_id=d["speaker_id"],
            chunk_index=int(d["chunk_index"]),
            audio_data=base64.b64decode(d["audio_data"]),
            language=d.get("language", "auto"),
            sample_rate=int(d.get("sample_rate", "16000")),
            source_runtime=d.get("source_runtime", "web"),
            vad_confidence=float(d.get("vad_confidence", "0.0")),
            speech_start_ms=int(d.get("speech_start_ms", "0")),
            speech_end_ms=int(d.get("speech_end_ms", "0")),
            input_lufs=float(d.get("input_lufs", "0.0")),
            noise_suppression_enabled=_redis_to_bool(
                d.get("noise_suppression_enabled", "false")
            ),
            is_final_chunk=d.get("is_final_chunk") == "1",
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
    is_final_chunk: bool = False
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
            "is_final_chunk": "1" if self.is_final_chunk else "0",
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
            is_final_chunk=d.get("is_final_chunk") == "1",
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class TranslationResultMessage(BaseModel):
    """Translation Worker → TTS Worker.

    Represents a translated text segment ready for speech synthesis.
    """

    __slots__ = ()

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
    is_final_chunk: bool = False
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    # OpenAITranslator.model — needed by TranscriptService to populate
    # transcript.translation_contents.translator_model (NOT NULL).
    translator_model: str = ""
    # The originating STTResultMessage.segment_id, BEFORE the "-{target_lang}-c{idx}"
    # suffix this message's own segment_id gets (see translation_worker._translate_and_publish).
    # Lets a consumer (gateway/frontend) join a translation back to the exact transcript
    # bubble it belongs to instead of guessing from the suffixed id — the two ids only
    # coincided by string-prefix luck before this field existed, and that's what let
    # original/translated text drift into separate, unmerged bubbles.
    source_segment_id: str = ""
    # Position of this sentence within the source STT segment's sentence split — 0 for
    # the first (and usually only) sentence. A consumer uses this to APPEND rather than
    # overwrite when one STT segment yields more than one translated chunk.
    chunk_index: int = 0

    def to_redis(self) -> dict[str, str]:
        return {
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
            "is_final_chunk": "1" if self.is_final_chunk else "0",
            "timestamp_ms": str(self.timestamp_ms),
            "translator_model": self.translator_model,
            "source_segment_id": self.source_segment_id,
            "chunk_index": str(self.chunk_index),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> TranslationResultMessage:
        d = _decode_dict(data)
        return cls(
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
            is_final_chunk=d.get("is_final_chunk") == "1",
            timestamp_ms=int(d.get("timestamp_ms", "0")),
            translator_model=d.get("translator_model", ""),
            source_segment_id=d.get("source_segment_id", ""),
            chunk_index=int(d.get("chunk_index", "0")),
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
    char_count: int = 0  # Number of synthesized text characters sent to the TTS provider
    voice_type: str = "default"  # 'default' | 'cloned'
    voice_mode: str = "standard"  # 'standard' | 'blended' | 'cloned' | 'caption_only'
    clone_strength: float = 0.0
    anchor_provider: str = ""
    clone_provider: str = ""
    # Cartesia voice id actually used — set even for voice_type='default'.
    provider_voice_id: str = ""
    render_location: str = "server"
    cache_key: str = ""
    cache_hit: bool = False
    synthesis_latency_ms: int = 0
    conversion_latency_ms: int = 0
    fallback_reason: str = ""
    target_lang: str = ""
    is_final_chunk: bool = False
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    model_config = {"arbitrary_types_allowed": True}

    def to_redis(self) -> dict[str, str]:
        return {
            "segment_id": self.segment_id,
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "audio_data": base64.b64encode(self.audio_data).decode("ascii"),
            "duration_ms": str(self.duration_ms),
            "char_count": str(self.char_count),
            "voice_type": self.voice_type,
            "voice_mode": self.voice_mode,
            "clone_strength": str(self.clone_strength),
            "anchor_provider": self.anchor_provider,
            "clone_provider": self.clone_provider,
            "provider_voice_id": self.provider_voice_id,
            "render_location": self.render_location,
            "cache_key": self.cache_key,
            "cache_hit": _bool_to_redis(self.cache_hit),
            "synthesis_latency_ms": str(self.synthesis_latency_ms),
            "conversion_latency_ms": str(self.conversion_latency_ms),
            "fallback_reason": self.fallback_reason,
            "target_lang": self.target_lang,
            "is_final_chunk": "1" if self.is_final_chunk else "0",
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
            char_count=int(d.get("char_count", "0")),
            voice_type=d.get("voice_type", "default"),
            voice_mode=d.get("voice_mode", "standard"),
            clone_strength=float(d.get("clone_strength", "0.0")),
            anchor_provider=d.get("anchor_provider", ""),
            clone_provider=d.get("clone_provider", ""),
            provider_voice_id=d.get("provider_voice_id", ""),
            render_location=d.get("render_location", "server"),
            cache_key=d.get("cache_key", ""),
            cache_hit=_redis_to_bool(d.get("cache_hit", "false")),
            synthesis_latency_ms=int(d.get("synthesis_latency_ms", "0")),
            conversion_latency_ms=int(d.get("conversion_latency_ms", "0")),
            fallback_reason=d.get("fallback_reason", ""),
            target_lang=d.get("target_lang", ""),
            is_final_chunk=d.get("is_final_chunk") == "1",
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class AIUsageMessage(BaseModel):
    """OpenAI usage event for billing settlement.

    Translation/assistant workers publish one event per provider call or per completed
    assistant turn. Billing can resolve subscription/workspace from room_id when
    workspace_id is not available at publish time.
    """

    __slots__ = ()

    workspace_id: str = ""
    room_id: str
    user_id: str = ""
    charge_type: str
    model: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    source_lang: str = ""
    target_lang: str = ""
    idempotency_key: str
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "room_id": self.room_id,
            "user_id": self.user_id,
            "charge_type": self.charge_type,
            "model": self.model,
            "prompt_tokens": str(self.prompt_tokens),
            "cached_tokens": str(self.cached_tokens),
            "completion_tokens": str(self.completion_tokens),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "idempotency_key": self.idempotency_key,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> AIUsageMessage:
        d = _decode_dict(data)
        return cls(
            workspace_id=d.get("workspace_id", ""),
            room_id=d["room_id"],
            user_id=d.get("user_id", ""),
            charge_type=d["charge_type"],
            model=d["model"],
            prompt_tokens=int(d.get("prompt_tokens", "0")),
            cached_tokens=int(d.get("cached_tokens", "0")),
            completion_tokens=int(d.get("completion_tokens", "0")),
            source_lang=d.get("source_lang", ""),
            target_lang=d.get("target_lang", ""),
            idempotency_key=d["idempotency_key"],
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class ProviderUsageMessage(BaseModel):
    """Provider-side non-token usage event for billing settlement."""

    __slots__ = ()

    workspace_id: str = ""
    room_id: str
    user_id: str = ""
    charge_type: str
    provider: str
    model: str
    quantity: Decimal
    unit: str
    idempotency_key: str
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "workspace_id": self.workspace_id,
            "room_id": self.room_id,
            "user_id": self.user_id,
            "charge_type": self.charge_type,
            "provider": self.provider,
            "model": self.model,
            "quantity": str(self.quantity),
            "unit": self.unit,
            "idempotency_key": self.idempotency_key,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> ProviderUsageMessage:
        d = _decode_dict(data)
        return cls(
            workspace_id=d.get("workspace_id", ""),
            room_id=d["room_id"],
            user_id=d.get("user_id", ""),
            charge_type=d["charge_type"],
            provider=d["provider"],
            model=d["model"],
            quantity=Decimal(d.get("quantity", "0")),
            unit=d["unit"],
            idempotency_key=d["idempotency_key"],
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class ChatRequestMessage(BaseModel):
    """AssistantService (.NET) → ChatAssistantWorker.

    One user turn to answer, with the prior conversation history needed for context.
    bearer_token is the caller's own "Bearer eyJ..." header, forwarded so tool calls hit
    sibling services' existing authenticated endpoints — never a privileged bypass.
    """

    __slots__ = ()

    request_id: str
    conversation_id: str
    workspace_id: str
    user_id: str
    bearer_token: str = ""
    history_json: str = "[]"  # JSON array of {"role": ..., "content": ...}
    # JSON {"pageType", "entityId", "workspaceId", "snapshot"} or "" if none.
    page_context_json: str = ""
    # JSON array of {"entityType", "entityId", "label", "workspaceId"} or "" if none.
    mentions_json: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "bearer_token": self.bearer_token,
            "history_json": self.history_json,
            "page_context_json": self.page_context_json,
            "mentions_json": self.mentions_json,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> ChatRequestMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            conversation_id=d["conversation_id"],
            workspace_id=d["workspace_id"],
            user_id=d["user_id"],
            bearer_token=d.get("bearer_token", ""),
            history_json=d.get("history_json", "[]"),
            page_context_json=d.get("page_context_json", ""),
            mentions_json=d.get("mentions_json", ""),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class ChatResultMessage(BaseModel):
    """ChatAssistantWorker → AssistantService (.NET).

    One event within a chat turn — a streamed text chunk, a tool-call lifecycle event, or
    the final completed/failed outcome. `type` discriminates which other fields are meaningful.
    """

    __slots__ = ()

    request_id: str
    conversation_id: str
    type: str  # "chunk" | "tool_call_started" | "tool_call_completed" | "completed" | "failed"
    content: str = ""
    tool_name: str = ""
    tool_status: str = ""
    tool_calls_json: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_status": self.tool_status,
            "tool_calls_json": self.tool_calls_json,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: dict[bytes | str, bytes | str]) -> ChatResultMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            conversation_id=d["conversation_id"],
            type=d["type"],
            content=d.get("content", ""),
            tool_name=d.get("tool_name", ""),
            tool_status=d.get("tool_status", ""),
            tool_calls_json=d.get("tool_calls_json", ""),
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


def _bool_to_redis(value: bool) -> str:
    return "true" if value else "false"


def _redis_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}
