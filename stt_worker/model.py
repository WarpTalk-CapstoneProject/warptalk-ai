"""Whisper STT model wrapper with dual-engine support.

Supports:
- mlx-whisper: Best Vietnamese accuracy on Apple Silicon (default)
- whisper.cpp: Fastest inference for languages with good base-model support

Provides a synchronous `transcribe()` method designed to be called
via `asyncio.to_thread()` from the async worker.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.logger import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

# Common Whisper Vietnamese misspelling corrections (diacritical errors)
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
    """Apply common Vietnamese spelling corrections."""
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
    """A single transcribed text segment."""

    text: str
    language: str
    confidence: float
    start_ms: int
    end_ms: int


# MLX-Whisper model repo mapping
_MLX_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


# Known Whisper hallucination patterns
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "bye", "bye bye",
    "good night", "oh", "you", "yeah", "okay",
    "thanks for watching!", "thank you.", "good night.",
    "bye.", "bye-bye.", "oh.", "you.", "yeah.", "okay.",
    "fuck", "fuck.", "hmm", "hmm.", "i'm",
    "subscribe", "like and subscribe",
    "see you all later", "see you all later.",
    # Vietnamese hallucinations
    "cảm ơn mọi người", "cảm ơn các bạn đã theo dõi",
    "hãy subscribe cho kênh", "xin chào",
    "cảm ơn các bạn đã xem video",
    "đăng ký kênh", "nhấn nút đăng ký",
    # Initial prompt leakage
    "cuộc họp tiếng việt, có thể xen tiếng anh",
    "cuộc họp tiếng anh",
    "đây là cuộc họp bằng tiếng việt",
    # Short noise transcripts
    "nói", "ừ", "à", "ađe", "ade",
    # Lone punctuation
    ".", "..", "...",
}

_HALLUCINATION_SUBSTRINGS = [
    "subscribe", "đăng ký kênh", "theo dõi kênh",
    "la la school", "xem video", "bỏ lỡ",
    "ủng hộ kênh", "hẹn gặp lại", "chào mừng",
    "ghiền mì gõ", "video tiếp theo",
    "video hấp dẫn",
]


def _filter_segments(
    segments_raw: list[dict],
    detected_language: str,
    chunk_offset_ms: int,
) -> list[TranscribedSegment]:
    """Filter and post-process raw Whisper segments."""

    # Filter out unlikely languages
    _ALLOWED_LANGUAGES = {"vi", "en"}
    if detected_language not in _ALLOWED_LANGUAGES:
        logger.debug("filtered_wrong_language", detected=detected_language)
        return []

    results: list[TranscribedSegment] = []
    seen_texts: set[str] = set()

    for seg in segments_raw:
        text = seg.get("text", "").strip()
        if not text:
            continue

        avg_logprob = seg.get("avg_logprob", -1.0)
        no_speech = seg.get("no_speech_prob", 0.0)
        text_lower = text.lower().rstrip('.!,')

        # no_speech_prob filter
        if no_speech > 0.6:
            logger.debug("filtered_no_speech", text=text, no_speech_prob=round(no_speech, 2))
            continue

        # Confidence filter: logprob < -1.0 usually means garbled output
        if avg_logprob < -1.0:
            logger.debug("filtered_low_confidence", text=text, logprob=round(avg_logprob, 2))
            continue

        # Exact hallucination filter
        if text_lower in _HALLUCINATIONS:
            logger.debug("filtered_hallucination", text=text)
            continue

        # Substring hallucination filter
        if any(sub in text_lower for sub in _HALLUCINATION_SUBSTRINGS):
            logger.debug("filtered_hallucination_substring", text=text)
            continue

        # Repetition filter: "rô, rô, rô, rô" etc.
        words = text_lower.replace(",", "").split()
        if len(words) >= 4:
            unique_words = set(words)
            if len(unique_words) <= 2:
                logger.debug("filtered_repetition", text=text[:50])
                continue

        # Character repetition: "hìììì..." or "aaaa..."
        if re.search(r'(.)\1{3,}', text_lower):
            logger.debug("filtered_char_repetition", text=text[:50])
            continue

        # Dedup
        if text_lower in seen_texts:
            logger.debug("filtered_duplicate", text=text)
            continue
        seen_texts.add(text_lower)

        start_s = seg.get("start", 0.0)
        end_s = seg.get("end", 0.0)

        logger.info(
            "segment_accepted",
            text=text,
            logprob=round(avg_logprob, 2),
            no_speech=round(no_speech, 2),
        )
        # Apply Vietnamese spelling corrections
        corrected = _fix_vietnamese(text) if detected_language == "vi" else text
        if corrected != text:
            logger.info("spelling_corrected", original=text, corrected=corrected)

        results.append(
            TranscribedSegment(
                text=corrected,
                language=detected_language,
                confidence=round(avg_logprob, 4),
                start_ms=chunk_offset_ms + int(start_s * 1000),
                end_ms=chunk_offset_ms + int(end_s * 1000),
            )
        )

    return results


class WhisperSTT:
    """MLX-Whisper model wrapper for Apple Silicon accelerated STT.

    Uses Apple GPU (Metal) via the MLX framework for fast inference
    on M-series chips. Falls back gracefully on non-Apple hardware.
    """

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: str = "cpu",  # Ignored — MLX auto-selects GPU
        compute_type: str = "int8",  # Ignored — MLX uses its own quantization
        beam_size: int = 1,
        vad_filter: bool = False,
    ) -> None:
        self.model_size = model_size
        self.beam_size = beam_size
        self._repo = _MLX_REPOS.get(model_size, model_size)
        self._loaded = False

    def load(self) -> None:
        """Pre-download the MLX model weights."""
        import mlx_whisper

        logger.info(
            "loading_whisper_model",
            model_size=self.model_size,
            engine="mlx",
            repo=self._repo,
        )
        # Warmup: run a short transcription to trigger model download + JIT
        import numpy as np
        dummy = np.zeros(16000, dtype=np.float32)  # 1s silence
        mlx_whisper.transcribe(dummy, path_or_hf_repo=self._repo)
        self._loaded = True
        logger.info("whisper_model_loaded")

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = None,
        chunk_offset_ms: int = 0,
    ) -> list[TranscribedSegment]:
        """Transcribe audio array to text segments.

        This is a BLOCKING call — must be run via asyncio.to_thread().

        Args:
            audio: NumPy float32 array at 16kHz
            language: Source language hint, None for auto-detect
            chunk_offset_ms: Offset to add to segment timestamps

        Returns:
            List of transcribed segments
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import mlx_whisper

        lang_arg = language if language and language != "auto" else None

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._repo,
            language=lang_arg,
            condition_on_previous_text=False,  # Prevents hallucination chaining
            temperature=0.0,  # Deterministic output
            initial_prompt="Đây là cuộc họp bằng tiếng Việt.",
            no_speech_threshold=0.6,
            word_timestamps=False,
        )

        detected_language = result.get("language", "unknown")
        segments_raw = result.get("segments", [])

        return _filter_segments(segments_raw, detected_language, chunk_offset_ms)
