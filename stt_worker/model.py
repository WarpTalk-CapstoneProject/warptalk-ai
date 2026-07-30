"""OpenAI Realtime transcription STT wrapper.

Uses a persistent WebSocket transcription session per
(meeting_id, speaker_id), reused across every chunk from that speaker for the life of
the room. Production defaults to the accuracy-first gpt-transcribe model and supplies
the room's expected languages, glossary keywords, and bounded meeting context.
"""

from __future__ import annotations

import asyncio
import base64
import re
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI

from shared.config import STTSettings
from shared.logger import get_logger
from shared.text_utils import split_into_sentences

logger = get_logger(__name__)

# Mirrors STTSettings.model — the value production code actually runs with
# (stt_worker/worker.py always passes it explicitly). Sourcing the default from here
# instead of a second hardcoded literal keeps direct/test instantiation in sync with
# config.py without anyone having to remember to update both places.
_DEFAULTS = STTSettings()

# Realtime transcription PCM input uses 24 kHz.
REALTIME_SAMPLE_RATE = 24000

# Realtime sessions are per (meeting_id, speaker_id) and outlive a single chunk, but
# nothing currently signals "this room/speaker is done" to this worker — sweep ones
# that haven't been used in a while rather than leaking connections for the process
# lifetime.
SESSION_IDLE_TIMEOUT_S = 300.0

# Guard against OpenAI never sending a completed/error event for a commit.
TRANSCRIBE_EVENT_TIMEOUT_S = 15.0

# Conservative Vietnamese diacritical corrections for recurrent model errors.
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
            result = result[:idx] + right + result[idx + len(wrong) :]
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

# Fallback allow-list used ONLY when a room has not declared its language set yet
# (no speak_languages published) AND the speaker has no per-speaker hint. The real
# allow-list is the meeting's declared languages, passed into _filter_segments — see
# STTWorker._get_room_languages. Hard-coding vi/en here was the root cause of the
# "nói không ra transcript" bug: any speaker whose profile language was neither vi nor
# en had every segment dropped.
_DEFAULT_ALLOWED_LANGUAGES = {"vi", "en"}

# Languages written in Latin script. The cross-script hallucination guard
# (_FOREIGN_SCRIPT_RE) only makes sense for these: a speaker we know is speaking a
# Latin-script language never legitimately produces Thai/Hangul/CJK/Kana. For a speaker
# whose declared language IS one of those scripts, that same text is correct output and
# must NOT be filtered — applying the guard unconditionally is what would silently drop
# a legitimate Japanese/Korean/Chinese/Thai speaker.
_LATIN_SCRIPT_LANGUAGES = {
    "en",
    "vi",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "id",
    "ms",
    "nl",
    "pl",
    "tr",
    "sv",
    "da",
    "no",
    "fi",
    "cs",
    "sk",
    "hr",
    "ro",
    "hu",
}

_VI_CHAR_RE = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)


def _guess_language_from_text(text: str, allowed: set[str] | None = None) -> str:
    """gpt-realtime-whisper's completed event carries no language field (unlike
    the old REST Whisper response this replaced), so there is no real per-chunk
    detection signal. This only runs on the fallback path where the speaker has NO
    registered profile language (chunk.language == "auto") — when a hint IS present it
    is authoritative and this is never called, which is what stops the per-chunk
    language flip-flop.

    Constrain the guess to the meeting's declared language set so we don't label a
    segment 'vi' in a room where nobody speaks Vietnamese: diacritics ⇒ 'vi' only if the
    room actually allows 'vi', else prefer 'en', else any deterministic member of the
    declared set."""
    if _VI_CHAR_RE.search(text) and (not allowed or "vi" in allowed):
        return "vi"
    if allowed:
        if "en" in allowed:
            return "en"
        return sorted(allowed)[0]
    return "en"


_FOREIGN_SCRIPT_RE = re.compile(
    r"[฀-๿"  # Thai
    r"가-힣"  # Hangul syllables
    r"一-鿿"  # CJK unified ideographs
    r"぀-ヿ]"  # Hiragana/Katakana
)

# Even the fastest human speech in any language doesn't clear ~30 chars/sec, so this
# pair (< 0.5s of real audio, yet > 20 chars of text) only fires on the classic
# Whisper-family hallucination pattern: a full phrase invented over near-silence/noise.
_MIN_SPEECH_SECONDS_FOR_LONG_TEXT = 0.5
_MAX_CHARS_FOR_SHORT_AUDIO = 20

_HALLUCINATIONS = {
    "thank you",
    "thanks for watching",
    "bye",
    "bye bye",
    "good night",
    "oh",
    "you",
    "yeah",
    "okay",
    "thanks for watching!",
    "thank you.",
    "good night.",
    "bye.",
    "bye-bye.",
    "oh.",
    "you.",
    "yeah.",
    "okay.",
    "fuck",
    "fuck.",
    "hmm",
    "hmm.",
    "i'm",
    "subscribe",
    "like and subscribe",
    "see you all later",
    "see you all later.",
    "cảm ơn mọi người",
    "cảm ơn các bạn đã theo dõi",
    "hãy subscribe cho kênh",
    "xin chào",
    "cảm ơn các bạn đã xem video",
    "đăng ký kênh",
    "nhấn nút đăng ký",
    "cuộc họp tiếng việt, có thể xen tiếng anh",
    "cuộc họp tiếng anh",
    "đây là cuộc họp bằng tiếng việt",
    "nói",
    "ừ",
    "à",
    "ađe",
    "ade",
    ".",
    "..",
    "...",
}

_HALLUCINATION_SUBSTRINGS = [
    "subscribe",
    "đăng ký kênh",
    "theo dõi kênh",
    "la la school",
    "xem video",
    "bỏ lỡ",
    "ủng hộ kênh",
    "hẹn gặp lại",
    "chào mừng",
    "ghiền mì gõ",
    "video tiếp theo",
    "video hấp dẫn",
]

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_KEYWORD_ENUMERATION_SPLIT_RE = re.compile(r"[,;\n]+")
_MIN_KEYWORD_ECHO_TERMS = 6
_MIN_KEYWORD_ECHO_RATIO = 0.75


def _normalized_sentences(text: str) -> list[str]:
    return [
        " ".join(sentence.casefold().split()).rstrip(".!?,")
        for sentence in _SENTENCE_BOUNDARY_RE.split(text.strip())
        if sentence.strip()
    ]


def _pcm16_duration_seconds(audio_bytes: bytes, sample_rate: int) -> float:
    """Duration of raw 16-bit mono PCM — the Realtime API's completed event
    doesn't return segment timing either, so this is the only source of an end_ms
    estimate."""
    if not sample_rate:
        return 0.0
    return (len(audio_bytes) // 2) / float(sample_rate)


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


def _expected_languages(
    primary_language: str | None,
    allowed_languages: set[str] | None,
) -> list[str]:
    """Build a stable Realtime language hint without suppressing English code-switching."""
    ordered: list[str] = []

    def add(language: str | None) -> None:
        if not language or language == "auto":
            return
        normalized = _normalize_language(language)
        if normalized and normalized not in ordered:
            ordered.append(normalized)

    add(primary_language)
    for language in sorted(allowed_languages or ()):
        add(language)
    # Product meetings frequently embed English product and engineering terms in an
    # otherwise non-English utterance. Advertising English as expected prevents the
    # model from forcing those terms into a phonetic translation of the primary language.
    if ordered:
        add("en")
    return ordered


def _normalized_keywords(keywords: list[str] | None) -> list[str]:
    """Bound and de-duplicate provider keyword hints while preserving display casing."""
    result: list[str] = []
    seen: set[str] = set()
    for keyword in keywords or ():
        cleaned = " ".join(keyword.split())[:100]
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if len(result) >= 100:
            break
    return result


def _is_keyword_enumeration_echo(text: str, keywords: list[str] | None) -> bool:
    """Detect the provider reciting structured vocabulary hints as spoken content.

    Keep ordinary speech that happens to mention glossary terms. The production failure
    was structurally different: a long comma-separated enumeration where nearly every
    item was one of the exact provider keywords.
    """
    normalized_keywords = {
        " ".join(keyword.casefold().split()).strip(" .!?:;,")
        for keyword in (keywords or ())
        if keyword.strip()
    }
    if len(normalized_keywords) < _MIN_KEYWORD_ECHO_TERMS:
        return False

    items = [
        " ".join(item.casefold().split()).strip(" .!?:;,")
        for item in _KEYWORD_ENUMERATION_SPLIT_RE.split(text)
        if item.strip(" .!?:;,")
    ]
    if len(items) < _MIN_KEYWORD_ECHO_TERMS:
        return False

    matched = sum(item in normalized_keywords for item in items)
    return (
        matched >= _MIN_KEYWORD_ECHO_TERMS
        and matched / len(items) >= _MIN_KEYWORD_ECHO_RATIO
    )


def _filter_segments(
    segments_raw: list[dict[str, Any]],
    detected_language: str,
    chunk_offset_ms: int,
    allowed_languages: set[str] | None = None,
    real_duration_s: float | None = None,
    context_prompt: str | None = None,
    keywords: list[str] | None = None,
    min_avg_logprob: float = -0.7,
) -> list[TranscribedSegment]:
    language_known = detected_language != "unknown"
    lang_code = _normalize_language(detected_language) if language_known else None

    # The languages this meeting is allowed to produce — the distinct set of languages
    # its participants declared they speak (from their profile settings, published to
    # Redis; see STTWorker._get_room_languages). This replaces the old hard-coded vi/en
    # allow-list, honoring the design where a meeting declares which languages it will
    # contain instead of a single source/target pair.
    allowed = {_normalize_language(lang) for lang in (allowed_languages or ())}
    if not allowed:
        allowed = set(_DEFAULT_ALLOWED_LANGUAGES)
    # The speaker's OWN declared language is always allowed. STT is pinned to it on the
    # session (see _get_or_create_session), so filtering it back out here is exactly the
    # bug that produced no transcript at all for a non-vi/en speaker.
    if lang_code:
        allowed.add(lang_code)

    if language_known and lang_code not in allowed:
        logger.debug(
            "filtered_wrong_language",
            detected=detected_language,
            code=lang_code,
            allowed=sorted(allowed),
        )
        return []

    # The cross-script guard is only valid when the speaker's declared language is
    # Latin-script; for a declared CJK/Thai/Hangul language that script is legitimate.
    #
    # When the per-utterance language is unknown (speaker joined with speak_language left
    # on "auto" — see TranslationRoomHub.SetSpeakLanguage/JoinTranslationRoom — so there's
    # no session-level hint at all), still apply the guard IF every language this meeting
    # has actually declared (allowed, computed above) is Latin-script: nobody in the room
    # is expected to produce Thai/Hangul/CJK/Kana regardless of which Latin-script language
    # this particular speaker turns out to be using, so a hallucinated foreign-script
    # segment is still safe to drop. If the room has a real CJK/Thai speaker declared,
    # skip the guard here — there's no way to tell whose script is legitimate without a
    # known per-segment language, and dropping could silently eat that speaker's real
    # transcript.
    apply_foreign_script_guard = (language_known and lang_code in _LATIN_SCRIPT_LANGUAGES) or (
        not language_known
        and bool(allowed)
        and all(lang in _LATIN_SCRIPT_LANGUAGES for lang in allowed)
    )

    results: list[TranscribedSegment] = []
    seen_texts: set[str] = set()
    prompt_lines = {
        " ".join(line.casefold().split()).rstrip(".!?,")
        for line in (context_prompt or "").splitlines()
        if len(" ".join(line.split())) >= 12
    }

    for seg in segments_raw:
        text = seg.get("text", "").strip()
        if not text:
            continue

        # A speaker we already know is speaking a Latin-script language never
        # legitimately produces Thai/Hangul/CJK/Kana characters, so this is a
        # high-precision way to catch the residual cross-script hallucination — and to
        # enforce "no language mixing": a segment whose script doesn't match the
        # speaker's one declared language is dropped rather than emitted.
        if apply_foreign_script_guard and _FOREIGN_SCRIPT_RE.search(text):
            logger.debug("filtered_foreign_script", text=text)
            continue

        # Realtime completed events expose token logprobs when explicitly requested in
        # the session include list. transcribe() averages those into avg_logprob; -1.0
        # remains the compatibility fallback for an older event with no logprobs.
        avg_logprob = float(seg.get("avg_logprob", -1.0))
        no_speech = seg.get("no_speech_prob", 0.0) or 0.0
        text_lower = text.lower().rstrip(".!,")

        # Realtime transcription can repeat contextual prompt text on silence/noise.
        # Reject an exact long-line echo regardless of confidence. Also reject a long
        # partial prompt phrase only when its confidence is marginal: production copied
        # "WarpTalk transcript engineering" out of a longer title at -0.6066. A clear
        # speaker genuinely saying the title remains valid, and short glossary terms
        # ("backend", "gRPC") are never treated as prompt fragments.
        normalized_text = " ".join(text.casefold().split()).rstrip(".!?,")
        is_exact_prompt_echo = normalized_text in prompt_lines
        is_marginal_prompt_fragment = (
            len(normalized_text) >= 18
            and avg_logprob < -0.35
            and any(normalized_text in prompt_line for prompt_line in prompt_lines)
        )
        if is_exact_prompt_echo or is_marginal_prompt_fragment:
            logger.debug("filtered_prompt_echo", text=text[:80])
            continue

        # gpt-transcribe accepts structured keyword hints but does not currently expose
        # token confidence. On marginal audio it can recite the entire comma-separated
        # glossary verbatim, yielding the -1.0 compatibility sentinel and bypassing the
        # confidence gate below. Match the *enumeration structure*, not isolated terms,
        # so natural code-switched speech such as "deploy the backend API" remains valid.
        if _is_keyword_enumeration_echo(text, keywords):
            logger.warning(
                "filtered_keyword_enumeration_echo",
                text=text[:80],
                keyword_count=len(keywords or ()),
            )
            continue

        if no_speech > 0.6:
            logger.debug("filtered_no_speech", text=text, no_speech_prob=round(no_speech, 2))
            continue

        # -1.0 is the compatibility sentinel for old completed events that lacked
        # logprobs. Current sessions request real token logprobs; a real value below
        # the calibrated boundary is marginal audio and must not become an off-topic
        # plausible-looking caption.
        if avg_logprob != -1.0 and avg_logprob < min_avg_logprob:
            logger.debug("filtered_low_confidence", text=text, logprob=round(avg_logprob, 2))
            continue

        # Unlike no_speech/avg_logprob above, this IS a real signal — only passed for the
        # trailing fragment in transcribe() (the whole chunk's actual PCM duration via
        # _pcm16_duration_seconds), never for early-emitted sentences (which have no real
        # per-sentence timing to check against, see _emit_early). No human produces this
        # many characters, in any language, in this little audio — the classic
        # Whisper-family failure mode of a full sentence hallucinated onto near-silence.
        if (
            real_duration_s is not None
            and real_duration_s < _MIN_SPEECH_SECONDS_FOR_LONG_TEXT
            and len(text) > _MAX_CHARS_FOR_SHORT_AUDIO
        ):
            logger.debug(
                "filtered_text_too_long_for_audio",
                text=text[:60],
                duration_s=round(real_duration_s, 2),
            )
            continue

        if text_lower in _HALLUCINATIONS:
            logger.debug("filtered_hallucination", text=text)
            continue

        if any(sub in text_lower for sub in _HALLUCINATION_SUBSTRINGS):
            logger.debug("filtered_hallucination_substring", text=text)
            continue

        # A real production failure returned a high-confidence collage made by copying
        # the same three previous utterances over and over. Token logprob cannot catch
        # clear background vocals, so reject repeated *sentences* before they recursively
        # enter the prompt and grow on every following chunk.
        normalized_sentences = _normalized_sentences(text)
        substantial_sentences = [
            sentence for sentence in normalized_sentences if len(sentence) >= 8
        ]
        if len(substantial_sentences) >= 3:
            sentence_counts = Counter(substantial_sentences)
            repeated_sentence_count = sum(
                count - 1 for count in sentence_counts.values() if count > 1
            )
            if repeated_sentence_count >= 2:
                logger.debug(
                    "filtered_repeated_sentence_collage",
                    text=text[:80],
                    repeated_sentences=repeated_sentence_count,
                )
                continue

        # Whisper-family hallucination on marginal/noisy audio tends to loop a
        # word or short phrase rather than collapsing to 1-2 distinct words total
        # (e.g. "Nora, Nuang Nora Va Nuang Nora" — 3 distinct words, still garbage) —
        # a plain distinct-word-count check misses that, so gate on how dominant the
        # single most-repeated word is instead.
        words = text_lower.replace(",", "").split()
        if len(words) >= 4:
            most_common_count = Counter(words).most_common(1)[0][1]
            if most_common_count >= max(2, len(words) // 2):
                logger.debug("filtered_repetition", text=text[:50])
                continue

        if re.search(r"(.)\1{3,}", text_lower):
            logger.debug("filtered_char_repetition", text=text[:50])
            continue

        if text_lower in seen_texts:
            logger.debug("filtered_duplicate", text=text)
            continue
        seen_texts.add(text_lower)

        seg_lang = lang_code or _guess_language_from_text(text, allowed)
        corrected = _fix_vietnamese(text) if seg_lang == "vi" else text
        if corrected != text:
            logger.info("spelling_corrected", original=text, corrected=corrected)

        results.append(
            TranscribedSegment(
                text=corrected,
                language=seg_lang,
                confidence=round(avg_logprob, 4),
                start_ms=chunk_offset_ms + int(seg.get("start", 0.0) * 1000),
                end_ms=chunk_offset_ms + int(seg.get("end", 0.0) * 1000),
            )
        )

    return results


class OpenAISTT:
    """OpenAI Realtime transcription wrapper.

    Fully async — call `await transcribe()` directly, no asyncio.to_thread needed.
    Keeps one WebSocket "transcription" session open per (meeting_id, speaker_id),
    reused across chunks so only the FIRST chunk from a speaker pays the ~1s
    connection handshake.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = _DEFAULTS.model,
        noise_reduction: str = _DEFAULTS.noise_reduction,
        min_avg_logprob: float = _DEFAULTS.min_avg_logprob,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.noise_reduction = noise_reduction
        self.min_avg_logprob = min_avg_logprob
        self._client: AsyncOpenAI | None = None
        # (meeting_id, speaker_id) -> {"manager": ..., "conn": ..., "last_used": float}
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        self._warm_sessions: deque[dict[str, Any]] = deque()

    async def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI STT")

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_stt_ready", model=self.model)

    async def warm_up(self, pool_size: int = 4) -> None:
        """Open reusable transcription sockets before the first participant speaks."""
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI STT is not loaded")

        async def _open() -> dict[str, Any]:
            manager = client.realtime.connect(extra_query={"intent": "transcription"})
            conn = await manager.__aenter__()
            return {"manager": manager, "conn": conn}

        opened = await asyncio.gather(
            *(_open() for _ in range(max(0, pool_size))),
            return_exceptions=True,
        )
        warm_sessions = getattr(self, "_warm_sessions", None)
        if warm_sessions is None:
            warm_sessions = deque()
            self._warm_sessions = warm_sessions
        for result in opened:
            if isinstance(result, Exception):
                logger.warning("stt_warm_session_failed", error=str(result))
            else:
                warm_sessions.append(result)
        logger.info("stt_realtime_pool_warmed", connections=len(warm_sessions))

    async def close(self) -> None:
        sessions = list(getattr(self, "_sessions", {}).values())
        getattr(self, "_sessions", {}).clear()
        warm_sessions = list(getattr(self, "_warm_sessions", ()))
        getattr(self, "_warm_sessions", deque()).clear()
        await asyncio.gather(
            *(self._close_session(session) for session in sessions + warm_sessions),
            return_exceptions=True,
        )

    async def prepare_session(
        self,
        meeting_id: str,
        speaker_id: str,
        *,
        language: str | None,
        prompt: str | None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
    ) -> None:
        """Claim/configure a warm socket when a participant publishes their mic track."""
        await self._get_or_create_session(
            (meeting_id, speaker_id),
            language=language,
            prompt=prompt,
            allowed_languages=allowed_languages,
            keywords=keywords,
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
        chunk_offset_ms: int = 0,
        meeting_id: str = "",
        speaker_id: str = "",
        prompt: str | None = None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
        on_early_segment: Callable[[TranscribedSegment], Awaitable[None]] | None = None,
        on_speculative_segment: Callable[[TranscribedSegment], Awaitable[None]] | None = None,
    ) -> list[TranscribedSegment]:
        """Transcribe raw audio bytes via the OpenAI Realtime API.

        Args:
            audio_bytes: WAV-container audio bytes (from Redis stream)
            sample_rate: Sample rate of the audio
            language: ISO 639-1 hint or None for auto-detect (fed to the session as
                input_audio_transcription.language — see _get_or_create_session)
            chunk_offset_ms: Timestamp offset to add to segment times
            meeting_id, speaker_id: key the reused realtime session for this speaker
            allowed_languages: the meeting's declared language set (from participants'
                profile speak-languages). Segments outside it are dropped; the speaker's
                own hint language is always kept. None ⇒ fall back to the default set.
            prompt: free-text contextual-biasing hint (glossary/key terms for this
                meeting) fed to the session once at creation. Steers the model toward
                domain vocabulary and away from hallucinating — see _get_or_create_session.
            on_early_segment: called with each complete sentence as soon as it's
                detected in the Realtime API's incremental `.delta` transcription
                events — i.e. BEFORE the whole audio chunk finishes transcribing. Lets
                translation/TTS start on sentence 1 while the rest of the chunk is
                still being transcribed, instead of waiting for the single
                end-of-chunk result this method used to return. Sentences delivered
                this way are excluded from the returned list (they've already been
                handed off); only the still-incomplete trailing fragment comes back
                normally once the `.completed` event arrives.
            on_speculative_segment: called at the same early sentence boundary but never
                removes text from the final return. The callback may warm a downstream
                translation cache; only the completed-event result is published.

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
                [
                    {
                        "text": sentence_text,
                        "start": 0.0,
                        "end": 0.0,
                        "avg_logprob": 0.0,
                        "no_speech_prob": 0.0,
                    }
                ],
                detected_language,
                chunk_offset_ms,
                allowed_languages,
                context_prompt=prompt,
                keywords=keywords,
                min_avg_logprob=getattr(self, "min_avg_logprob", -0.7),
            )
            for seg in segs:
                await on_early_segment(seg)

        async def _emit_speculative(sentence_text: str) -> None:
            if on_speculative_segment is None:
                return
            segs = _filter_segments(
                [
                    {
                        "text": sentence_text,
                        "start": 0.0,
                        "end": 0.0,
                        "avg_logprob": 0.0,
                        "no_speech_prob": 0.0,
                    }
                ],
                detected_language,
                chunk_offset_ms,
                allowed_languages,
                context_prompt=prompt,
                keywords=keywords,
                min_avg_logprob=getattr(self, "min_avg_logprob", -0.7),
            )
            for seg in segs:
                await on_speculative_segment(seg)

        if on_early_segment is not None:
            on_sentence = _emit_early
            exclude_emitted_from_final = True
        elif on_speculative_segment is not None:
            on_sentence = _emit_speculative
            exclude_emitted_from_final = False
        else:
            on_sentence = None
            exclude_emitted_from_final = True

        pcm_24k = _resample_pcm16(audio_bytes, sample_rate, REALTIME_SAMPLE_RATE)

        key = (meeting_id, speaker_id)
        try:
            text, avg_logprob = await self._transcribe_via_session(
                key,
                pcm_24k,
                on_sentence,
                lang_arg,
                prompt,
                allowed_languages,
                keywords,
                exclude_emitted_from_final=exclude_emitted_from_final,
            )
        except Exception:
            logger.warning("realtime_session_retry", meeting_id=meeting_id, speaker_id=speaker_id)
            self._sessions.pop(key, None)
            try:
                text, avg_logprob = await self._transcribe_via_session(
                    key,
                    pcm_24k,
                    on_sentence,
                    lang_arg,
                    prompt,
                    allowed_languages,
                    keywords,
                    exclude_emitted_from_final=exclude_emitted_from_final,
                )
            except Exception as e:
                logger.error("openai_stt_error", error=str(e))
                raise

        if not text.strip():
            return []

        duration_s = _pcm16_duration_seconds(audio_bytes, sample_rate)
        segments_dicts = [
            {
                "text": text.strip(),
                "start": 0.0,
                "end": duration_s,
                "avg_logprob": avg_logprob,
                "no_speech_prob": 0.0,
            }
        ]

        # duration_s is the WHOLE chunk's audio, while `text` here is only the trailing
        # fragment not already flushed via on_early_segment — so this over-estimates the
        # audio actually backing this specific text, making the check in _filter_segments
        # more lenient than perfectly accurate. That's the safe direction: it only ever
        # risks missing a hallucination, never dropping real trailing speech.
        return _filter_segments(
            segments_dicts,
            detected_language,
            chunk_offset_ms,
            allowed_languages,
            real_duration_s=duration_s,
            context_prompt=prompt,
            keywords=keywords,
            min_avg_logprob=getattr(self, "min_avg_logprob", -0.7),
        )

    async def _transcribe_via_session(
        self,
        key: tuple[str, str],
        pcm_24k: bytes,
        on_sentence: Callable[[str], Awaitable[None]] | None = None,
        language: str | None = None,
        prompt: str | None = None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
        *,
        exclude_emitted_from_final: bool = True,
    ) -> tuple[str, float]:
        session = await self._get_or_create_session(
            key,
            language,
            prompt,
            allowed_languages,
            keywords,
        )
        conn = session["conn"]

        # Ingress has already assembled a VAD-bounded speech utterance.
        # Sending that as ten-to-fifteen separately awaited 100ms websocket messages
        # added pure transport overhead before the model could start. Keep a conservative
        # 2s raw-PCM cap for unusually long replay/test chunks; production uses one append.
        append_bytes = REALTIME_SAMPLE_RATE * 2 * 2
        for i in range(0, len(pcm_24k), append_bytes):
            frame = pcm_24k[i : i + append_bytes]
            await conn.input_audio_buffer.append(audio=base64.b64encode(frame).decode())

        await conn.input_audio_buffer.commit()
        session["last_used"] = time.monotonic()

        async def _collect() -> tuple[str, float]:
            buffer = ""
            flushed = ""
            # gpt-realtime-whisper occasionally gets stuck on trailing silence/noise and
            # emits the same short sentence over and over in the delta stream instead of
            # ever reaching "completed" — with no per-chunk confidence signal to catch
            # this after the fact (see _filter_segments), each repeat gets flushed early
            # and independently translated/spoken, which sounds like TTS stuck in a loop.
            # Cut the turn short once the same sentence repeats 3x in a row.
            last_sentence: str | None = None
            repeat_count = 0
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
                            normalized = sentence.strip().casefold()
                            if normalized and normalized == last_sentence:
                                repeat_count += 1
                            else:
                                repeat_count = 0
                            last_sentence = normalized
                            if repeat_count >= 2:
                                logger.warning(
                                    "stt_repetition_loop_detected", sentence=sentence[:60]
                                )
                                raise RuntimeError("stt_repetition_loop_detected")
                            await on_sentence(sentence)
                            flushed += sentence + " "
                        buffer = "" if ends_clean else sentences[-1]
                elif etype == "conversation.item.input_audio_transcription.completed":
                    final_text = (getattr(event, "transcript", "") or "").strip()
                    token_logprobs = [
                        float(value)
                        for item in (getattr(event, "logprobs", None) or [])
                        if (value := (
                            item.get("logprob")
                            if isinstance(item, dict)
                            else getattr(item, "logprob", None)
                        ))
                        is not None
                    ]
                    avg_logprob = (
                        sum(token_logprobs) / len(token_logprobs) if token_logprobs else -1.0
                    )
                    if not exclude_emitted_from_final:
                        return final_text, avg_logprob
                    flushed_stripped = flushed.strip()
                    if not flushed_stripped:
                        return final_text, avg_logprob
                    if final_text.startswith(flushed_stripped):
                        return final_text[len(flushed_stripped) :].strip(), avg_logprob
                    # Model revised something inside the already-flushed prefix — we
                    # can't safely recompute the diff (would risk re-publishing text
                    # that was already billed/translated). Drop the trailing part
                    # rather than risk a duplicate charge or duplicate translation.
                    logger.warning(
                        "stt_delta_final_mismatch",
                        flushed=flushed_stripped[:60],
                        final=final_text[:60],
                    )
                    return "", avg_logprob
                elif etype == "error":
                    raise RuntimeError(f"realtime_transcription_error: {event}")
            raise RuntimeError("realtime_connection_closed_before_completed")

        return await asyncio.wait_for(_collect(), timeout=TRANSCRIBE_EVENT_TIMEOUT_S)

    def _session_payload(
        self,
        language: str | None,
        prompt: str | None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        # input_audio_transcription.language is a real, documented field (ISO-639-1
        # hint): "improves accuracy and latency". Telling the model the speaker's
        # registered language outright (see livekit_ingress_worker's speak_languages
        # lookup) stops it auto-detecting — and sometimes hallucinating a completely
        # different script mid-sentence.
        #
        # transcription.prompt is a free-text contextual-biasing hint, supported by
        # gpt-4o-transcribe (NOT by gpt-realtime-whisper, which is why we switched).
        # Seeded with this meeting's glossary/key terms, it steers the model toward
        # the expected domain vocabulary — the research-backed hallucination-reduction
        # lever (arXiv 2410.18363).
        transcription_config: dict[str, Any] = {"model": self.model}
        is_next_generation_transcribe = self.model in {
            "gpt-transcribe",
            "gpt-live-transcribe",
        }
        if is_next_generation_transcribe:
            languages = _expected_languages(language, allowed_languages)
            if languages:
                transcription_config["languages"] = languages
            normalized_keywords = _normalized_keywords(keywords)
            if normalized_keywords:
                transcription_config["keywords"] = normalized_keywords
        elif language:
            transcription_config["language"] = language
        if prompt:
            transcription_config["prompt"] = prompt

        input_config: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE},
            "transcription": transcription_config,
            "turn_detection": None,
        }
        # Runs before VAD/the model ever see the audio — "improves VAD and turn
        # detection accuracy (reducing false positives) and model performance" per
        # OpenAI's own docs. self.noise_reduction == "off" omits the field entirely
        # (server default is no noise reduction).
        if self.noise_reduction and self.noise_reduction != "off":
            input_config["noise_reduction"] = {"type": self.noise_reduction}

        payload: dict[str, Any] = {
            "type": "transcription",
            "audio": {"input": input_config},
        }
        # The earlier 4o transcription models expose token logprobs. The newer
        # gpt-transcribe family currently does not expose confidence scores, and sending
        # this include selector makes some Realtime API versions reject the session.
        if not is_next_generation_transcribe:
            payload["include"] = ["item.input_audio_transcription.logprobs"]
        return payload

    async def _get_or_create_session(
        self,
        key: tuple[str, str],
        language: str | None = None,
        prompt: str | None = None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
    ) -> dict[str, Any]:
        self._sweep_idle_sessions()

        languages = tuple(_expected_languages(language, allowed_languages))
        normalized_keywords = tuple(_normalized_keywords(keywords))
        cached = self._sessions.get(key)
        is_next_generation_transcribe = self.model in {
            "gpt-transcribe",
            "gpt-live-transcribe",
        }
        if (
            cached is not None
            and not is_next_generation_transcribe
            and cached.get("language") != language
        ):
            logger.info(
                "stt_session_language_changed",
                meeting_id=key[0],
                speaker_id=key[1],
                old_language=cached.get("language"),
                new_language=language,
            )
            try:
                await cached["manager"].__aexit__(None, None, None)
            except Exception:
                logger.debug(
                    "stt_session_close_on_language_change_failed",
                    meeting_id=key[0],
                    speaker_id=key[1],
                )
            self._sessions.pop(key, None)
            cached = None
        if cached is not None:
            config_changed = (
                cached.get("language") != language
                or cached.get("prompt") != prompt
                or cached.get("languages") != languages
                or cached.get("keywords") != normalized_keywords
            )
            if config_changed:
                await cached["conn"].session.update(
                    session=cast(
                        Any,
                        self._session_payload(
                            language,
                            prompt,
                            allowed_languages,
                            list(normalized_keywords),
                        ),
                    )
                )
                cached.update(
                    language=language,
                    prompt=prompt,
                    languages=languages,
                    keywords=normalized_keywords,
                )
            return cached

        client = self._client
        if client is None:
            raise RuntimeError("OpenAI STT is not loaded")
        warm_sessions = getattr(self, "_warm_sessions", None)
        if warm_sessions:
            warm = warm_sessions.popleft()
            manager = warm["manager"]
            conn = warm["conn"]
        else:
            manager = client.realtime.connect(extra_query={"intent": "transcription"})
            conn = await manager.__aenter__()

        try:
            await conn.session.update(
                session=cast(
                    Any,
                    self._session_payload(
                        language,
                        prompt,
                        allowed_languages,
                        list(normalized_keywords),
                    ),
                )
            )
        except Exception:
            if not language and not prompt and not normalized_keywords:
                raise
            # Fail safe to a bare config rather than losing the session entirely if
            # this API version rejects the language or prompt field for some reason.
            logger.warning(
                "session_optional_fields_rejected",
                has_language=bool(language),
                has_prompt=bool(prompt),
                has_keywords=bool(normalized_keywords),
            )
            await conn.session.update(session=cast(Any, self._session_payload(None, None)))

        session = {
            "manager": manager,
            "conn": conn,
            "last_used": time.monotonic(),
            "language": language,
            "prompt": prompt,
            "languages": languages,
            "keywords": normalized_keywords,
        }
        self._sessions[key] = session
        logger.info(
            "realtime_session_opened",
            meeting_id=key[0],
            speaker_id=key[1],
            language=language or "auto",
            has_prompt=bool(prompt),
            keyword_count=len(normalized_keywords),
        )
        return session

    def _sweep_idle_sessions(self) -> None:
        now = time.monotonic()
        stale = [
            k for k, s in self._sessions.items() if now - s["last_used"] > SESSION_IDLE_TIMEOUT_S
        ]
        for k in stale:
            session = self._sessions.pop(k)
            asyncio.create_task(self._close_session(session))
            logger.info("realtime_session_idle_closed", meeting_id=k[0], speaker_id=k[1])

    @staticmethod
    async def _close_session(session: dict[str, Any]) -> None:
        try:
            await session["manager"].__aexit__(None, None, None)
        except Exception:
            logger.exception("realtime_session_close_error")
