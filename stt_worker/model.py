"""OpenAI Realtime API (gpt-realtime-whisper) STT wrapper.

Replaces the old gpt-4o-mini-transcribe REST-per-chunk call (~2.1-2.6s/chunk incl. a
full HTTP round-trip) with a persistent WebSocket "transcription" session per
(meeting_id, speaker_id), reused across every chunk from that speaker for the life of
the room. Paying the ~1s connection handshake once per speaker instead of once per
chunk, plus the Realtime API's own lower per-chunk transcription latency, is where the
latency win comes from — confirmed empirically (not from docs, which disagreed with
the real wire protocol in several places): commit -> first partial delta ~0.8s,
commit -> completed transcript ~1.8s, on a ~4s utterance.
"""

from __future__ import annotations

import asyncio
import base64
import io
import re
import time
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from shared.logger import get_logger
from shared.text_utils import split_into_sentences

logger = get_logger(__name__)

# gpt-realtime-whisper rejects session.audio.input.format.rate below this.
REALTIME_SAMPLE_RATE = 24000

# Realtime sessions are per (meeting_id, speaker_id) and outlive a single chunk, but
# nothing currently signals "this room/speaker is done" to this worker — sweep ones
# that haven't been used in a while rather than leaking connections for the process
# lifetime.
SESSION_IDLE_TIMEOUT_S = 300.0

# Guard against OpenAI never sending a completed/error event for a commit.
TRANSCRIBE_EVENT_TIMEOUT_S = 15.0

# Vietnamese diacritical corrections — gpt-4o-mini-transcribe also makes these
_VI_CORRECTIONS: dict[str, str] = {
    "lu trữ": "lưu trữ",
    "luu trữ": "lưu trữ",
    "sử lý": "xử lý",
    "sữ lý": "xử lý",
    "ứng dung": "ứng dụng",
    "giáo diện": "giao diện",
    "trinh bày": "trình bày",
    "hệ thong": "hệ thống",
    "dử liệu": "dữ liệu",
    "du liệu": "dữ liệu",
    "phan mềm": "phần mềm",
    "chức nang": "chức năng",
    "thiet kế": "thiết kế",
    "cơ sỡ": "cơ sở",
    "trien khai": "triển khai",
    "yêu câu": "yêu cầu",
    "hoan thành": "hoàn thành",
    "quản ly": "quản lý",
    "bào cáo": "báo cáo",
    "tính nang": "tính năng",
    "cap nhật": "cập nhật",
    "nguời": "người",
    "đuợc": "được",
    "cuộc hop": "cuộc họp",
    "trinh chiếu": "trình chiếu",
}


def _fix_vietnamese(text: str) -> str:
    result = text
    for wrong, right in _VI_CORRECTIONS.items():
        lower = result.lower()
        idx = lower.find(wrong)
        while idx != -1:
            result = result[:idx] + right + result[idx + len(wrong):]
            lower = result.lower()
            idx = lower.find(wrong, idx + len(right))
    return result


@dataclass(slots=True)
class TranscribedSegment:
    text: str
    language: str
    confidence: float
    start_ms: int
    end_ms: int


# OpenAI full-language-name → ISO 639-1 code (returned when language=None)
_LANG_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "vietnamese": "vi",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "german": "de",
    "spanish": "es",
    "thai": "th",
    "indonesian": "id",
    "russian": "ru",
    "arabic": "ar",
    "portuguese": "pt",
    "italian": "it",
    "malay": "ms",
}

_ALLOWED_LANGUAGES = {"vi", "en"}

_HALLUCINATIONS = {
    "thank you", "thanks for watching", "bye", "bye bye",
    "good night", "oh", "you", "yeah", "okay",
    "thanks for watching!", "thank you.", "good night.",
    "bye.", "bye-bye.", "oh.", "you.", "yeah.", "okay.",
    "fuck", "fuck.", "hmm", "hmm.", "i'm",
    "subscribe", "like and subscribe",
    "see you all later", "see you all later.",
    "cảm ơn mọi người", "cảm ơn các bạn đã theo dõi",
    "hãy subscribe cho kênh", "xin chào",
    "cảm ơn các bạn đã xem video",
    "đăng ký kênh", "nhấn nút đăng ký",
    "cuộc họp tiếng việt, có thể xen tiếng anh",
    "cuộc họp tiếng anh",
    "đây là cuộc họp bằng tiếng việt",
    "nói", "ừ", "à", "ađe", "ade",
    ".", "..", "...",
}

_HALLUCINATION_SUBSTRINGS = [
    "subscribe", "đăng ký kênh", "theo dõi kênh",
    "la la school", "xem video", "bỏ lỡ",
    "ủng hộ kênh", "hẹn gặp lại", "chào mừng",
    "ghiền mì gõ", "video tiếp theo", "video hấp dẫn",
]


def _wav_duration_seconds(audio_bytes: bytes, fallback_sample_rate: int) -> float:
    """Best-effort duration from WAV header — the Realtime API's completed event
    doesn't return segment timing either, so this is the only source of an end_ms
    estimate."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            return w.getnframes() / float(w.getframerate() or fallback_sample_rate)
    except Exception:
        return 0.0


def _wav_to_pcm16(audio_bytes: bytes) -> tuple[bytes, int]:
    """Extract raw 16-bit mono PCM samples + sample rate from a WAV container."""
    with wave.open(io.BytesIO(audio_bytes), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def _resample_pcm16(pcm_bytes: bytes, orig_rate: int, target_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit mono PCM.

    No anti-aliasing filter — fine for the 16kHz -> 24kHz upsample this is actually
    used for (gpt-realtime-whisper rejects anything below 24kHz); would need a proper
    filter if ever used to downsample.
    """
    if orig_rate == target_rate or not pcm_bytes:
        return pcm_bytes

    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    target_len = int(len(samples) * target_rate / orig_rate)
    orig_idx = np.arange(len(samples))
    target_idx = np.linspace(0, len(samples) - 1, num=target_len)
    resampled = np.interp(target_idx, orig_idx, samples.astype(np.float64))
    return resampled.astype(np.int16).tobytes()


def _normalize_language(lang: str) -> str:
    """Normalize OpenAI language output to ISO 639-1 code."""
    lower = lang.lower()
    return _LANG_NAME_TO_CODE.get(lower, lower[:2] if len(lower) > 2 else lower)


def _filter_segments(
    segments_raw: list[dict],
    detected_language: str,
    chunk_offset_ms: int,
) -> list[TranscribedSegment]:
    lang_code = _normalize_language(detected_language)

    if lang_code not in _ALLOWED_LANGUAGES:
        logger.debug("filtered_wrong_language", detected=detected_language, code=lang_code)
        return []

    results: list[TranscribedSegment] = []
    seen_texts: set[str] = set()

    for seg in segments_raw:
        text = seg.get("text", "").strip()
        if not text:
            continue

        avg_logprob = seg.get("avg_logprob", -1.0) or -1.0
        no_speech = seg.get("no_speech_prob", 0.0) or 0.0
        text_lower = text.lower().rstrip('.!,')

        if no_speech > 0.6:
            logger.debug("filtered_no_speech", text=text, no_speech_prob=round(no_speech, 2))
            continue

        if avg_logprob < -1.0:
            logger.debug("filtered_low_confidence", text=text, logprob=round(avg_logprob, 2))
            continue

        if text_lower in _HALLUCINATIONS:
            logger.debug("filtered_hallucination", text=text)
            continue

        if any(sub in text_lower for sub in _HALLUCINATION_SUBSTRINGS):
            logger.debug("filtered_hallucination_substring", text=text)
            continue

        words = text_lower.replace(",", "").split()
        if len(words) >= 4 and len(set(words)) <= 2:
            logger.debug("filtered_repetition", text=text[:50])
            continue

        if re.search(r'(.)\1{3,}', text_lower):
            logger.debug("filtered_char_repetition", text=text[:50])
            continue

        if text_lower in seen_texts:
            logger.debug("filtered_duplicate", text=text)
            continue
        seen_texts.add(text_lower)

        corrected = _fix_vietnamese(text) if lang_code == "vi" else text
        if corrected != text:
            logger.info("spelling_corrected", original=text, corrected=corrected)

        results.append(
            TranscribedSegment(
                text=corrected,
                language=lang_code,
                confidence=round(avg_logprob, 4),
                start_ms=chunk_offset_ms + int(seg.get("start", 0.0) * 1000),
                end_ms=chunk_offset_ms + int(seg.get("end", 0.0) * 1000),
            )
        )

    return results


class OpenAISTT:
    """OpenAI Realtime API (gpt-realtime-whisper) wrapper.

    Fully async — call `await transcribe()` directly, no asyncio.to_thread needed.
    Keeps one WebSocket "transcription" session open per (meeting_id, speaker_id),
    reused across chunks so only the FIRST chunk from a speaker pays the ~1s
    connection handshake.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-realtime-whisper") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None
        # (meeting_id, speaker_id) -> {"manager": ..., "conn": ..., "last_used": float}
        self._sessions: dict[tuple[str, str], dict] = {}

    async def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI STT")

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_stt_ready", model=self.model)

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
        chunk_offset_ms: int = 0,
        meeting_id: str = "",
        speaker_id: str = "",
        on_early_segment: Callable[[TranscribedSegment], Awaitable[None]] | None = None,
    ) -> list[TranscribedSegment]:
        """Transcribe raw audio bytes via the OpenAI Realtime API.

        Args:
            audio_bytes: WAV-container audio bytes (from Redis stream)
            sample_rate: Sample rate of the audio
            language: ISO 639-1 hint or None for auto-detect (not currently passed to
                the session — see note in _create_session)
            chunk_offset_ms: Timestamp offset to add to segment times
            meeting_id, speaker_id: key the reused realtime session for this speaker
            on_early_segment: called with each complete sentence as soon as it's
                detected in the Realtime API's incremental `.delta` transcription
                events — i.e. BEFORE the whole audio chunk finishes transcribing. Lets
                translation/TTS start on sentence 1 while the rest of the chunk is
                still being transcribed, instead of waiting for the single
                end-of-chunk result this method used to return. Sentences delivered
                this way are excluded from the returned list (they've already been
                handed off); only the still-incomplete trailing fragment comes back
                normally once the `.completed` event arrives.

        Returns:
            Filtered list of transcribed segments (the trailing fragment not already
            handed to on_early_segment, or everything if on_early_segment is None)
        """
        if not audio_bytes:
            return []

        lang_arg = language if language and language != "auto" else None
        detected_language = lang_arg or "unknown"

        async def _emit_early(sentence_text: str) -> None:
            if on_early_segment is None:
                return
            segs = _filter_segments(
                [{
                    "text": sentence_text,
                    "start": 0.0,
                    "end": 0.0,
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0,
                }],
                detected_language,
                chunk_offset_ms,
            )
            for seg in segs:
                await on_early_segment(seg)

        on_sentence = _emit_early if on_early_segment is not None else None

        pcm, orig_rate = _wav_to_pcm16(audio_bytes)
        pcm_24k = _resample_pcm16(pcm, orig_rate, REALTIME_SAMPLE_RATE)

        key = (meeting_id, speaker_id)
        try:
            text = await self._transcribe_via_session(key, pcm_24k, on_sentence)
        except Exception:
            logger.warning("realtime_session_retry", meeting_id=meeting_id, speaker_id=speaker_id)
            self._sessions.pop(key, None)
            try:
                text = await self._transcribe_via_session(key, pcm_24k, on_sentence)
            except Exception as e:
                logger.error("openai_stt_error", error=str(e))
                raise

        if not text.strip():
            return []

        duration_s = _wav_duration_seconds(audio_bytes, sample_rate)
        segments_dicts = [{
            "text": text.strip(),
            "start": 0.0,
            "end": duration_s,
            "avg_logprob": 0.0,
            "no_speech_prob": 0.0,
        }]

        return _filter_segments(segments_dicts, detected_language, chunk_offset_ms)

    async def _transcribe_via_session(
        self,
        key: tuple[str, str],
        pcm_24k: bytes,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        session = await self._get_or_create_session(key)
        conn = session["conn"]

        # ~100ms frames — matches how a real streaming producer would feed audio in;
        # a single huge append also works, this is just a conservative message size.
        frame_bytes = REALTIME_SAMPLE_RATE * 2 // 10
        for i in range(0, len(pcm_24k), frame_bytes):
            frame = pcm_24k[i : i + frame_bytes]
            await conn.input_audio_buffer.append(audio=base64.b64encode(frame).decode())

        await conn.input_audio_buffer.commit()
        session["last_used"] = time.monotonic()

        async def _collect() -> str:
            buffer = ""
            flushed = ""
            async for event in conn:
                etype = getattr(event, "type", "")
                if etype == "conversation.item.input_audio_transcription.delta":
                    if on_sentence is None:
                        continue
                    buffer += getattr(event, "delta", "") or ""
                    if not buffer:
                        continue
                    ends_clean = buffer[-1] in ".!?"
                    sentences = split_into_sentences(buffer)
                    flush_count = len(sentences) if ends_clean else len(sentences) - 1
                    if flush_count > 0:
                        for sentence in sentences[:flush_count]:
                            await on_sentence(sentence)
                            flushed += sentence + " "
                        buffer = "" if ends_clean else sentences[-1]
                elif etype == "conversation.item.input_audio_transcription.completed":
                    final_text = (getattr(event, "transcript", "") or "").strip()
                    flushed_stripped = flushed.strip()
                    if not flushed_stripped:
                        return final_text
                    if final_text.startswith(flushed_stripped):
                        return final_text[len(flushed_stripped):].strip()
                    # Model revised something inside the already-flushed prefix — we
                    # can't safely recompute the diff (would risk re-publishing text
                    # that was already billed/translated). Drop the trailing part
                    # rather than risk a duplicate charge or duplicate translation.
                    logger.warning(
                        "stt_delta_final_mismatch",
                        flushed=flushed_stripped[:60],
                        final=final_text[:60],
                    )
                    return ""
                elif etype == "error":
                    raise RuntimeError(f"realtime_transcription_error: {event}")
            raise RuntimeError("realtime_connection_closed_before_completed")

        return await asyncio.wait_for(_collect(), timeout=TRANSCRIBE_EVENT_TIMEOUT_S)

    async def _get_or_create_session(self, key: tuple[str, str]) -> dict:
        self._sweep_idle_sessions()

        cached = self._sessions.get(key)
        if cached is not None:
            return cached

        manager = self._client.realtime.connect(extra_query={"intent": "transcription"})
        conn = await manager.__aenter__()

        # NOTE: passing input_audio_transcription.language here would let us carry the
        # per-chunk language hint through like the old REST call did — left out for
        # now since the exact field name/schema for a language hint on transcription
        # sessions wasn't verified against the live API (several field names in the
        # public docs did not match the actual wire protocol; only what's below was
        # empirically confirmed to work). Falls back to auto-detect.
        await conn.session.update(session={
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
                    "transcription": {"model": self.model},
                    "turn_detection": None,
                }
            },
        })

        session = {"manager": manager, "conn": conn, "last_used": time.monotonic()}
        self._sessions[key] = session
        logger.info("realtime_session_opened", meeting_id=key[0], speaker_id=key[1])
        return session

    def _sweep_idle_sessions(self) -> None:
        now = time.monotonic()
        stale = [k for k, s in self._sessions.items() if now - s["last_used"] > SESSION_IDLE_TIMEOUT_S]
        for k in stale:
            session = self._sessions.pop(k)
            asyncio.create_task(self._close_session(session))
            logger.info("realtime_session_idle_closed", meeting_id=k[0], speaker_id=k[1])

    @staticmethod
    async def _close_session(session: dict) -> None:
        try:
            await session["manager"].__aexit__(None, None, None)
        except Exception:
            logger.exception("realtime_session_close_error")
