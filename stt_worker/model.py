"""OpenAI gpt-4o-mini-transcribe STT wrapper.

Replaces mlx-whisper: zero GPU infra, Linux-deployable, demo cost < $10.
Latency vs self-host: +100–200ms/utterance — not perceptible in meeting context.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

from shared.logger import get_logger

logger = get_logger(__name__)

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
    """OpenAI gpt-4o-mini-transcribe wrapper.

    Fully async — call `await transcribe()` directly, no asyncio.to_thread needed.
    """

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini-transcribe") -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

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
    ) -> list[TranscribedSegment]:
        """Transcribe raw audio bytes via OpenAI API.

        Args:
            audio_bytes: Raw audio bytes (WAV/PCM from Redis stream)
            sample_rate: Sample rate of the audio
            language: ISO 639-1 hint or None for auto-detect
            chunk_offset_ms: Timestamp offset to add to segment times

        Returns:
            Filtered list of transcribed segments
        """
        if not audio_bytes:
            return []

        audio_io = io.BytesIO(audio_bytes)
        audio_io.name = "audio.wav"

        lang_arg = language if language and language != "auto" else None

        try:
            result = await self._client.audio.transcriptions.create(
                model=self.model,
                file=audio_io,
                response_format="verbose_json",
                language=lang_arg,
                temperature=0.0,
            )
        except Exception as e:
            logger.error("openai_stt_error", error=str(e))
            raise

        detected_language = getattr(result, "language", "unknown") or "unknown"
        raw_segments = getattr(result, "segments", None) or []

        # Normalize SDK segment objects to plain dicts
        segments_dicts: list[dict] = []
        for seg in raw_segments:
            try:
                d = seg.model_dump() if hasattr(seg, "model_dump") else dict(seg)
            except Exception:
                d = {
                    "text": getattr(seg, "text", ""),
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "avg_logprob": getattr(seg, "avg_logprob", -1.0),
                    "no_speech_prob": getattr(seg, "no_speech_prob", 0.0),
                }
            segments_dicts.append(d)

        return _filter_segments(segments_dicts, detected_language, chunk_offset_ms)
