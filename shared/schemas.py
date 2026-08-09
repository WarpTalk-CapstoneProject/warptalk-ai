"""Pydantic message schemas — contracts between workers.

Each schema provides `to_redis()` / `from_redis()` for Redis Streams
serialization. Field names align with backend TranscriptSegmentDto.
"""

from __future__ import annotations

import base64
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

# stt_worker/model.py's explicit fallback for a realtime event that exposed no token logprobs
# (`float(seg.get("avg_logprob", -1.0))`). It is a sentinel meaning "no confidence was reported",
# not a measurement — WT-277: every consumer must turn it into unknown/NULL rather than storing it
# as ordinary data. Mirrored in the backend by WarpTalk.Shared.ModelConfidence.UnknownSentinel.
STT_UNKNOWN_CONFIDENCE = -1.0


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
    def from_redis(cls, data: Mapping[Any, Any]) -> AudioChunkMessage:
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
            noise_suppression_enabled=_redis_to_bool(d.get("noise_suppression_enabled", "false")),
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
    def from_redis(cls, data: Mapping[Any, Any]) -> STTResultMessage:
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
    # WT-278: NOT a translation quality score — this field was called `confidence` and carried the
    # upstream STT segment's avg_logprob, a measurement of how clearly the *audio* was heard.
    # OpenAITranslator returns no score of its own, so there is nothing here that describes the
    # translation. Renamed so no consumer (TranscriptService persists it; the gateway/frontend read
    # this stream) can mistake it for one. If a real translation quality signal is ever built
    # (back-translation, COMET), it gets its own field; do not reuse this one.
    #
    # WT-277: Optional, and omitted from to_redis() when None, so "the source segment had no usable
    # confidence" travels as an absent field rather than as a fabricated number. Consumers store
    # NULL for it.
    source_stt_confidence: float | None = None
    start_ms: int = 0
    end_ms: int = 0
    is_final_chunk: bool = False
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    # OpenAITranslator.model — TranscriptService persists this into the NOT NULL
    # transcript.translation_contents.translator_model column.
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
        payload = {
            "segment_id": self.segment_id,
            "meeting_id": self.meeting_id,
            "speaker_id": self.speaker_id,
            "original_text": self.original_text,
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "start_ms": str(self.start_ms),
            "end_ms": str(self.end_ms),
            "is_final_chunk": "1" if self.is_final_chunk else "0",
            "timestamp_ms": str(self.timestamp_ms),
            "translator_model": self.translator_model,
            "source_segment_id": self.source_segment_id,
            "chunk_index": str(self.chunk_index),
        }
        # Redis stream fields are strings, so "unknown" cannot be encoded as a value — omit the
        # field entirely. Consumers treat an absent field as NULL (WT-277); writing "None" or a
        # placeholder number here is exactly the failure this ticket removed.
        if self.source_stt_confidence is not None:
            payload["source_stt_confidence"] = str(self.source_stt_confidence)
        return payload

    @classmethod
    def from_redis(
        cls,
        data: Mapping[Any, Any],
    ) -> TranslationResultMessage:
        d = _decode_dict(data)
        return cls(
            segment_id=d["segment_id"],
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            original_text=d["original_text"],
            translated_text=d["translated_text"],
            source_lang=d["source_lang"],
            target_lang=d["target_lang"],
            source_stt_confidence=optional_confidence(d.get("source_stt_confidence")),
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
    voice_type: str = "default"  # 'default' | 'cloned'
    voice_mode: str = "standard"  # 'standard' | 'blended' | 'cloned' | 'caption_only'
    clone_strength: float = 0.0
    anchor_provider: str = ""
    clone_provider: str = ""
    provider_voice_id: str = (
        ""  # Cartesia voice id actually used — set even for voice_type='default'
    )
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
    def from_redis(cls, data: Mapping[Any, Any]) -> TTSResultMessage:
        d = _decode_dict(data)
        return cls(
            segment_id=d["segment_id"],
            meeting_id=d["meeting_id"],
            speaker_id=d["speaker_id"],
            audio_data=base64.b64decode(d["audio_data"]),
            duration_ms=int(d.get("duration_ms", "0")),
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
    origin: str = "assistant"
    bearer_token: str = ""
    history_json: str = "[]"  # JSON array of {"role": ..., "content": ...}
    page_context_json: str = (
        ""  # JSON {"pageType", "entityId", "workspaceId", "snapshot"} or "" if none
    )
    mentions_json: str = (
        ""  # JSON array of {"entityType", "entityId", "label", "workspaceId"} or "" if none
    )
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "conversation_id": self.conversation_id,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "origin": self.origin,
            "bearer_token": self.bearer_token,
            "history_json": self.history_json,
            "page_context_json": self.page_context_json,
            "mentions_json": self.mentions_json,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> ChatRequestMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            conversation_id=d["conversation_id"],
            workspace_id=d["workspace_id"],
            user_id=d["user_id"],
            origin=d.get("origin", "assistant"),
            bearer_token=d.get("bearer_token", ""),
            history_json=d.get("history_json", "[]"),
            page_context_json=d.get("page_context_json", ""),
            mentions_json=d.get("mentions_json", ""),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class SummaryRequestMessage(BaseModel):
    """Backend → SummaryTemplateWorker: re-summarise a finished meeting.

    Carries no transcript. The worker fetches the SAVED transcript itself, because the
    in-memory accumulator AIAssistantWorker summarises from is gone once the meeting ends —
    and gone again on every restart. Re-reading the stored segments is also what makes the
    citations line up: they are the same segments the meeting page renders.
    """

    __slots__ = ()

    request_id: str
    room_id: str
    workspace_id: str
    template_key: str = "general"
    bearer_token: str = ""
    target_languages_json: str = "[]"
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "room_id": self.room_id,
            "workspace_id": self.workspace_id,
            "template_key": self.template_key,
            "bearer_token": self.bearer_token,
            "target_languages_json": self.target_languages_json,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> SummaryRequestMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            room_id=d["room_id"],
            workspace_id=d.get("workspace_id", ""),
            template_key=d.get("template_key", "general"),
            bearer_token=d.get("bearer_token", ""),
            target_languages_json=d.get("target_languages_json", "[]"),
            timestamp_ms=int(d.get("timestamp_ms", 0) or 0),
        )


class SummaryResultMessage(BaseModel):
    """SummaryTemplateWorker → backend: the regenerated summary, or why there is none."""

    __slots__ = ()

    request_id: str
    room_id: str
    template_key: str
    status: str  # "completed" | "failed"
    content_json: str = ""
    error: str = ""
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "request_id": self.request_id,
            "room_id": self.room_id,
            "template_key": self.template_key,
            "status": self.status,
            "content_json": self.content_json,
            "error": self.error,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> SummaryResultMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            room_id=d["room_id"],
            template_key=d.get("template_key", "general"),
            status=d.get("status", "failed"),
            content_json=d.get("content_json", ""),
            error=d.get("error", ""),
            timestamp_ms=int(d.get("timestamp_ms", 0) or 0),
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
    origin: str = "assistant"
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
            "origin": self.origin,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_status": self.tool_status,
            "tool_calls_json": self.tool_calls_json,
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> ChatResultMessage:
        d = _decode_dict(data)
        return cls(
            request_id=d["request_id"],
            conversation_id=d["conversation_id"],
            type=d["type"],
            origin=d.get("origin", "assistant"),
            content=d.get("content", ""),
            tool_name=d.get("tool_name", ""),
            tool_status=d.get("tool_status", ""),
            tool_calls_json=d.get("tool_calls_json", ""),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


class SuggestionResultMessage(BaseModel):
    """SuggestionWorker → Gateway (.NET) → SignalR "AiSuggestionReceived".

    Published onto the existing `ai_assistant:results` stream rather than a new one, so
    it shares the gateway consumer group already draining that stream. `type` is what
    lets AiResultConsumerService route it: anything other than "suggestion" keeps
    flowing to the legacy "AiAssistantResult" event untouched.

    `content` (not "text") deliberately matches the field name the gateway already reads
    off this stream for summaries and action items, so both branches parse the same key.

    `segment_id` is the STT segment that provoked the suggestion — the frontend anchors
    the strip to the transcript bubble containing that id. Note the bubble may have
    merged several segments into one utterance, so the id here is not necessarily the
    bubble's own id; resolving that is the client's job.
    """

    __slots__ = ()

    meeting_id: str
    segment_id: str
    category: str  # "clarification" | "term" | "action" | "correction" | "fact"
    content: str
    type: str = "suggestion"
    detail: str = ""
    confidence: float = 0.0
    language: str = ""
    # Tokens spent across BOTH the decide and generate calls for this suggestion.
    # billing_worker charges it as one AI_ASSISTANT usage record — see migration
    # 017-15-07-2026-translation-cluster-finalize.sql, which already lists AI_ASSISTANT
    # among the valid charge types, so no new charge type is introduced here.
    token_count: int = 0
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_redis(self) -> dict[str, str]:
        return {
            "meeting_id": self.meeting_id,
            "segment_id": self.segment_id,
            "category": self.category,
            "content": self.content,
            "type": self.type,
            "detail": self.detail,
            "confidence": str(self.confidence),
            "language": self.language,
            "token_count": str(self.token_count),
            "timestamp_ms": str(self.timestamp_ms),
        }

    @classmethod
    def from_redis(cls, data: Mapping[Any, Any]) -> SuggestionResultMessage:
        d = _decode_dict(data)
        return cls(
            meeting_id=d["meeting_id"],
            segment_id=d["segment_id"],
            category=d["category"],
            content=d["content"],
            type=d.get("type", "suggestion"),
            detail=d.get("detail", ""),
            confidence=float(d.get("confidence", "0.0")),
            language=d.get("language", ""),
            token_count=int(d.get("token_count", "0")),
            timestamp_ms=int(d.get("timestamp_ms", "0")),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def optional_confidence(raw: str | None) -> float | None:
    """Parse a confidence field, returning None for every flavour of "not reported".

    WT-277: absent, blank, unparsable, non-finite, or the STT_UNKNOWN_CONFIDENCE sentinel all mean
    the producer told us nothing. They must stay distinguishable from a real score all the way to
    the database, so they collapse to None here rather than to a default number.
    """
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value == STT_UNKNOWN_CONFIDENCE:
        return None
    return value


def _decode_dict(data: Mapping[Any, Any]) -> dict[str, str]:
    """Decode Redis byte keys/values to str."""
    return {
        (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
        for k, v in data.items()
    }


def _bool_to_redis(value: bool) -> str:
    return "true" if value else "false"


def _redis_to_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}
