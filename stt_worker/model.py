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
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, cast

import numpy as np
from openai import AsyncOpenAI

from shared.config import STTSettings
from shared.lang import base_language
from shared.logger import get_logger
from shared.schemas import STT_UNKNOWN_CONFIDENCE
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

# Every `filtered_*` line in this module logs at INFO, not DEBUG, and that is deliberate.
#
# Production runs at LOG_LEVEL=INFO, so a discard logged at DEBUG is a discard nobody can
# see. Measured on 15 Aug: 276 `inference_complete` produced 223 `segment_transcribed` —
# 53 utterances, 19%, removed with no record of which filter took them or why. The report
# that followed was "có vài câu nói không bắt được transcript nên bị bỏ qua luôn", and there
# was nothing to answer it with.
#
# This is the same defect the tts_worker header describes: an exit that swallows content
# in silence is indistinguishable from the feature being switched off, and the thresholds
# above cannot be calibrated against evidence that was never written down. The volume is
# not a concern — these are tens of lines a day, one per DISCARDED utterance, not per chunk.
#
# Which session options each model actually accepts, learned at runtime.
#
# These replace a hardcoded model allow-list, and the difference is not cosmetic. The old
# check was:
#
#     is_next_generation_transcribe = self.model in {"gpt-transcribe", "gpt-live-transcribe"}
#
# and it decided TWO unrelated things at once: whether to send structured context
# (`languages` plural + `keywords`) and whether to request token logprobs. Production runs
# gpt-4o-mini-transcribe, which is in neither family, so it silently received NO keywords
# at all — a workspace glossary curated specifically to stop "Codex" being transcribed as
# "cô đích" was published to Redis, read by this worker, and then dropped here. The two
# capabilities are also genuinely independent: a model may accept both, and folding them
# into one flag made "send keywords" and "keep confidence scores" look mutually exclusive
# when nothing says they are.
#
# Both default to optimistic. Guessing wrong costs one rejected session update per model
# per process, which _get_or_create_session catches, records here, and never repeats.
# Seeded with what this codebase already learned the hard way; everything absent is
# assumed supported until the API says otherwise. Structured context has NO seed on
# purpose — the whole point is that an unlisted model should try keywords rather than
# silently go without them.
_LOGPROBS_UNSUPPORTED_SEED = {
    # Sending the logprobs include selector to these makes some Realtime API versions
    # reject the entire session, so do not spend a failed round trip rediscovering it.
    "gpt-transcribe": False,
    "gpt-live-transcribe": False,
}

_STRUCTURED_CONTEXT_SUPPORT: dict[str, bool] = {}
_LOGPROBS_SUPPORT: dict[str, bool] = dict(_LOGPROBS_UNSUPPORTED_SEED)


def _supports_structured_context(model: str) -> bool:
    """Whether `model` takes the plural `languages` hint and `keywords`."""
    return _STRUCTURED_CONTEXT_SUPPORT.get(model, True)


def _supports_logprobs(model: str) -> bool:
    """Whether `model` accepts the transcription-logprobs include selector."""
    return _LOGPROBS_SUPPORT.get(model, True)


def _demote_capability_from_error(model: str, error_text: str) -> str | None:
    """
    Record a capability the Realtime API rejected on the STREAM rather than on session.update.

    Returns the name of what was demoted, or None if the error is about something else — an
    audio problem, a disconnect, a timeout — none of which say anything about what this model
    accepts, and demoting on those would strip the language hint off a model that handles it.

    Matching is on the parameter name inside the API's own wording ("The 'languages' parameter
    is not supported for this model."). Deliberately narrow: a broad `"languages" in error`
    would match a transcript that happens to contain the word.
    """
    lowered = error_text.lower()
    if "not supported" not in lowered and "invalid_parameter" not in lowered:
        return None

    if "'languages'" in lowered or '"languages"' in lowered or "'keywords'" in lowered:
        _STRUCTURED_CONTEXT_SUPPORT[model] = False
        return "structured_context"

    if "logprob" in lowered:
        _LOGPROBS_SUPPORT[model] = False
        return "logprobs"

    return None


def reset_capability_memo() -> None:
    """Forget everything learned at runtime, back to the seeds. For tests."""
    _STRUCTURED_CONTEXT_SUPPORT.clear()
    _LOGPROBS_SUPPORT.clear()
    _LOGPROBS_SUPPORT.update(_LOGPROBS_UNSUPPORTED_SEED)


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


# Vietnamese-UNIQUE characters only.
#
# The previous version of this class listed the whole Vietnamese alphabet, including
# à á è é ì í ò ó ù ú â ê ô ã õ — which is precisely the accent inventory of Spanish,
# French, Portuguese and Italian. It was therefore not a Vietnamese detector but a
# "has a Latin diacritic" detector that claimed every hit for Vietnamese: the Spanish
# sentence "Deberíamos terminar esta parte" was labelled 'vi' on the strength of its í,
# and French "regardé" on its é. A wrong language label is not cosmetic — it becomes the
# translation worker's source_lang and then the dubbed voice's language.
#
# What is left is the set no other Latin-script language uses: the horn vowels (ơ ư),
# breve a (ă), bar d (đ), and every vowel carrying an under-dot or hook-above tone mark,
# plus the circumflex vowels that carry a Vietnamese tone on top.
_VI_UNIQUE_CHAR_RE = re.compile(
    r"[ăằắặẳẵ"  # breve a + tones
    r"ầấậẩẫ"  # circumflex a + Vietnamese tones
    r"ềếệểễ"  # circumflex e + tones
    r"ồốộổỗ"  # circumflex o + tones
    r"ơờớợởỡ"  # horn o
    r"ưừứựửữ"  # horn u
    r"ạảẹẻịỉọỏụủ"  # under-dot / hook-above
    r"ẽĩũỳỵỷỹ"  # tilde/grave-dot finals not shared with Romance
    r"đ]",  # bar d
    re.IGNORECASE,
)

# Per-script character classes. These were already present as one combined regex used
# only to DROP text; splitting them means the same information can also IDENTIFY it.
# Script is the strongest language signal available here and it is nearly free: no model
# call, no ambiguity, and no way to mistake Hangul for French.
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ]")
_KANA_RE = re.compile(r"[぀-ヿ]")
_HAN_RE = re.compile(r"[一-鿿]")
_THAI_RE = re.compile(r"[฀-๿]")
_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿԀ-ԯ]")
_HEBREW_RE = re.compile(r"[֐-׿]")
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")

# Which writing systems each language is actually written in.
#
# A room that declared its languages has declared its writing systems too, and that is a
# far stronger filter than a blocklist of the scripts somebody happened to think of. The
# blocklist this replaced named Thai, Hangul, CJK and Kana — so Arabic, which is what
# production actually produced in a Vietnamese/Japanese room, was never on it.
#
# Absent from this table means Latin, which covers every remaining language the product
# offers.
_LANGUAGE_SCRIPTS: dict[str, frozenset[str]] = {
    "ja": frozenset({"kana", "han"}),
    "zh": frozenset({"han"}),
    "ko": frozenset({"hangul", "han"}),
    "th": frozenset({"thai"}),
    "ar": frozenset({"arabic"}),
    "he": frozenset({"hebrew"}),
    "ru": frozenset({"cyrillic"}),
    "uk": frozenset({"cyrillic"}),
    "bg": frozenset({"cyrillic"}),
    "hi": frozenset({"devanagari"}),
    "mr": frozenset({"devanagari"}),
    "el": frozenset({"greek"}),
}

_SCRIPT_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("hangul", _HANGUL_RE),
    ("kana", _KANA_RE),
    ("han", _HAN_RE),
    ("thai", _THAI_RE),
    ("arabic", _ARABIC_RE),
    ("cyrillic", _CYRILLIC_RE),
    ("hebrew", _HEBREW_RE),
    ("devanagari", _DEVANAGARI_RE),
    ("greek", _GREEK_RE),
)


def _scripts_in(text: str) -> set[str]:
    """Non-Latin writing systems present in `text`.

    Latin is not reported and never filtered on. It is the script every code-switched
    English product term arrives in — "deploy the backend API" inside a Vietnamese
    sentence is the thing this product is for, not a defect to be filtered out.
    """
    return {name for name, pattern in _SCRIPT_DETECTORS if pattern.search(text)}


def _allowed_scripts(languages: set[str]) -> set[str]:
    scripts: set[str] = set()
    for language in languages:
        scripts |= _LANGUAGE_SCRIPTS.get(base_language(language), frozenset())
    return scripts


def _detect_script_language(text: str) -> str | None:
    """Language implied by the writing system, or None for Latin script.

    Deliberately NOT constrained to the room's declared language set. Script evidence is
    strong enough to stand on its own: text in Hangul is Korean whatever the room said it
    would contain, and mislabelling it to fit the declared set is how a Korean speaker
    ends up tagged as Japanese.
    """
    if _HANGUL_RE.search(text):
        return "ko"
    if _KANA_RE.search(text):
        # Japanese mixes kana with kanji; the presence of ANY kana settles it against
        # Chinese, which uses Han characters alone.
        return "ja"
    if _THAI_RE.search(text):
        return "th"
    if _HAN_RE.search(text):
        # Han with no kana. Japanese written purely in kanji exists but is vanishingly
        # rare in conversational speech, so Chinese is the better call.
        return "zh"
    return None


def _detect_unambiguous_language(text: str) -> str | None:
    """The language the TEXT proves, or None when the text proves nothing.

    Two signals, both of which this module already documents as unambiguous:
    a non-Latin writing system, and the Vietnamese-unique character class.

    WHY BOTH, IN ONE PLACE
      `_detect_script_language` returns None for Latin script — it says so in its own
      docstring — so it can never separate Vietnamese from English. The evidence that CAN
      separate them lived only inside `_guess_language_from_text`, which by design runs
      only when the speaker declared no language at all.

      So for a speaker who declared one, the declaration was the last word even against
      proof to the contrary. A participant whose profile said `en` and who spoke Vietnamese
      for a whole meeting had every segment labelled `en`; the translator then ran en→vi
      over text that was already Vietnamese, and the "translation" was nonsense.

      A declaration is a statement of intent, and people get it wrong — this exact meeting
      is the evidence. Absence of evidence is a reason to trust the declaration.
      CONTRADICTION is not.

    Returns None for ordinary Latin text, where a declaration remains the best information
    available and must keep winning.
    """
    script_language = _detect_script_language(text)
    if script_language:
        return script_language

    # The character class no other Latin-script language uses — see _VI_UNIQUE_CHAR_RE for
    # why it is this narrow, and for the Spanish/French false positives that made it so.
    if _VI_UNIQUE_CHAR_RE.search(text):
        return "vi"

    return None


def _guess_language_from_text(text: str, allowed: set[str] | None = None) -> str:
    """gpt-realtime-whisper's completed event carries no language field (unlike
    the old REST Whisper response this replaced), so there is no real per-chunk
    detection signal. This only runs on the fallback path where the speaker has NO
    registered profile language (chunk.language == "auto") — when a hint IS present it
    is authoritative and this is never called, which is what stops the per-chunk
    language flip-flop.

    Evidence is used strongest-first:

    1. Writing system. Unambiguous, and not filtered through `allowed` — see
       _detect_script_language.
    2. Vietnamese-unique diacritics. Also unambiguous now that the character class no
       longer collides with the Romance languages.
    3. Only then the weak fallback, which does respect `allowed`: with no evidence at
       all, prefer English, else a deterministic member of the declared set.

    Step 3 is a guess and is documented as one. Steps 1 and 2 are not.
    """
    script_language = _detect_script_language(text)
    if script_language:
        return script_language

    if _VI_UNIQUE_CHAR_RE.search(text):
        return "vi"

    if allowed:
        if "en" in allowed:
            return "en"
        return sorted(allowed)[0]
    return "en"


# Even the fastest human speech in any language doesn't clear ~30 chars/sec, so this
# pair (< 0.5s of real audio, yet > 20 chars of text) only fires on the classic
# Whisper-family hallucination pattern: a full phrase invented over near-silence/noise.
# How many CONSECUTIVE segments must contradict a speaker's declared language before the
# session is re-pinned to what they are actually speaking. Two, not one: the evidence itself is
# unambiguous, so this guards only against a stray mis-transcription, and every extra segment of
# patience is another chunk decoded with the wrong language.
_LANGUAGE_OVERRIDE_SEGMENTS = 2

_MIN_SPEECH_SECONDS_FOR_LONG_TEXT = 0.5
_MAX_CHARS_FOR_SHORT_AUDIO = 20

# Two blocklists, not one, because the original single list conflated two very different
# things and silently deleted meeting speech as a result.
#
# _HALLUCINATIONS_ALWAYS is training-data bleed: phrases a Whisper-family model emits
# because its training corpus is full of video outros, plus echoes of our own session
# prompt. Nobody says these in a work meeting at any confidence, so text alone is enough.
_HALLUCINATIONS_ALWAYS = {
    "thanks for watching",
    "thanks for watching!",
    "subscribe",
    "like and subscribe",
    "cảm ơn các bạn đã theo dõi",
    "hãy subscribe cho kênh",
    "cảm ơn các bạn đã xem video",
    "đăng ký kênh",
    "nhấn nút đăng ký",
    "cuộc họp tiếng việt, có thể xen tiếng anh",
    "cuộc họp tiếng anh",
    "đây là cuộc họp bằng tiếng việt",
    "ađe",
    "ade",
    ".",
    "..",
    "...",
    # The other four meeting-scope languages. Until these were added, ja/ko/fr/es had NO
    # training-data-bleed protection at all: tools/stt_filter_audit.py --by-language
    # measured 100% of their hallucinations surviving, against 0% for en/vi. The gap was
    # not a judgement call about those languages, just the absence of anyone writing the
    # list — which is exactly why the confidence-based guards matter more than these do.
    #
    # Japanese. The first entry is the single most reported Whisper hallucination in any
    # language; it appears on silence in essentially every long Japanese transcription.
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "チャンネル登録お願いします",
    "チャンネル登録よろしくお願いします",
    # Korean.
    "시청해주셔서 감사합니다",
    "시청해 주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    "구독 좋아요 알림설정",
    # French.
    "merci d'avoir regardé",
    "merci d'avoir regardé cette vidéo",
    "n'oubliez pas de vous abonner",
    "abonnez-vous à la chaîne",
    # Spanish.
    "gracias por ver el video",
    "gracias por ver este video",
    "no olvides suscribirte",
    "suscríbete al canal",
    # Chinese — not a meeting-scope language, but _detect_script_language can now label a
    # segment 'zh', so the same bleed can reach a transcript.
    "感谢观看",
    "谢谢观看",
    "请订阅",
}

# _HALLUCINATIONS_IF_MARGINAL is the dangerous half. These strings ARE hallucinated onto
# silence — but they are also, verbatim, the most common things a person actually says in
# a meeting: agreement, thinking noises, and the greetings that open and close the call.
# "okay", "yeah", "ừ", "à", "xin chào" and "cảm ơn mọi người" were all being deleted
# unconditionally, so a clearly-spoken "Okay." never reached the transcript, and a
# Vietnamese meeting lost both its opening and its closing line every single time.
#
# Text cannot tell the two cases apart; the audio evidence can. Drop these only when the
# model was unsure (see _BLOCKLIST_MARGINAL_LOGPROB), and keep them when it was confident.
_HALLUCINATIONS_IF_MARGINAL = {
    "thank you",
    "thank you.",
    "bye",
    "bye.",
    "bye bye",
    "bye-bye.",
    "good night",
    "good night.",
    "oh",
    "oh.",
    "you",
    "you.",
    "yeah",
    "yeah.",
    "okay",
    "okay.",
    "hmm",
    "hmm.",
    "i'm",
    "fuck",
    "fuck.",
    "see you all later",
    "see you all later.",
    "cảm ơn mọi người",
    "xin chào",
    "nói",
    "ừ",
    "à",
    # Direct analogues of "okay"/"thank you" above in the other four languages: the model
    # hallucinates these onto silence AND they are among the commonest things actually
    # said in a meeting. Kept deliberately SHORT — only bare acknowledgements and bare
    # thanks. A longer real sentence such as "はい、わかりました。" or "Sí, de acuerdo."
    # never matches, because this list is exact-match, not substring.
    "はい",
    "ええ",
    "ありがとうございました",
    "ありがとうございます",
    "네",
    "예",
    "감사합니다",
    "oui",
    "merci",
    "d'accord",
    "au revoir",
    "sí",
    "si",
    "gracias",
    "vale",
    "adiós",
}

# Kept for compatibility with anything reading the old name, and for tests that assert on
# the union. Membership alone no longer decides a drop — see the two sets above.
_HALLUCINATIONS = _HALLUCINATIONS_ALWAYS | _HALLUCINATIONS_IF_MARGINAL

# Above this average token logprob, a blocklisted string is taken as genuinely spoken.
# Chosen by sweeping tools/stt_filter_audit.py over a corpus built to production's real
# language mix: it is the point where every clearly-spoken acknowledgement survives while
# the same strings hallucinated onto near-silence (which land near -0.65) still do not.
_BLOCKLIST_MARGINAL_LOGPROB = -0.45

# Fallback evidence for models that return NO token logprobs.
#
# _session_payload only requests the logprobs include-selector for models outside the
# gpt-transcribe family, so avg_logprob arrives as the STT_UNKNOWN_CONFIDENCE sentinel on
# exactly the models worth moving to (they are the ones that accept `keywords` and the
# plural `languages` hint). Treating the sentinel as "marginal" would hand the blocklist
# back its old power to delete "Okay", "Ừ" and "Xin chào" the moment the model changed —
# a fix that silently depends on one model is not a fix.
#
# These two signals are independent of the model's confidence AND of the writing system,
# which the character-count guards are not.
_BLOCKLIST_NO_SPEECH_MARGINAL = 0.3
_BLOCKLIST_MIN_SPEECH_SECONDS = 0.4


def _is_marginal_for_blocklist(
    avg_logprob: float,
    no_speech_prob: float,
    real_duration_s: float | None,
) -> bool:
    """Whether the audio evidence is weak enough to believe a blocklisted string.

    Confidence first when the model gives it. Otherwise fall back to how much speech
    there was and how likely the frame was silence.

    When a model supplies none of the three, this returns False — the blocklist is left
    enforcing only _HALLUCINATIONS_ALWAYS. That is deliberate: with no evidence either
    way, keeping a real "Okay." costs a line a human can see and correct, while dropping
    it costs meeting content nobody can recover. Production correction data shows content
    loss is the failure actually happening, so the default leans that way.
    """
    if avg_logprob != STT_UNKNOWN_CONFIDENCE:
        return avg_logprob < _BLOCKLIST_MARGINAL_LOGPROB
    if no_speech_prob >= _BLOCKLIST_NO_SPEECH_MARGINAL:
        return True
    if real_duration_s is not None and real_duration_s < _BLOCKLIST_MIN_SPEECH_SECONDS:
        return True
    return False


# Minimum share of DISTINCT words before a segment reads as a repetition loop rather than
# ordinary speech. Swept in tools/stt_filter_audit.py: real utterances in both languages
# sit at 0.75 and above (natural doubling included), while genuine loops land at 0.5 and
# below, so anything in 0.55-0.65 separates them — 0.6 takes the middle.
_MIN_DISTINCT_WORD_RATIO = 0.6

_HALLUCINATION_SUBSTRINGS_ALWAYS = [
    "subscribe",
    "đăng ký kênh",
    "theo dõi kênh",
    "la la school",
    "xem video",
    "ủng hộ kênh",
    "ghiền mì gõ",
    "video tiếp theo",
    "video hấp dẫn",
]

# Same trap as above, at substring level — and worse, because these drop the WHOLE segment
# for containing the phrase anywhere. "chào mừng" (welcome) and "hẹn gặp lại" (see you
# again) are ordinary meeting speech: "Chào mừng mọi người đến với buổi họp hôm nay" was
# being discarded in full.
_HALLUCINATION_SUBSTRINGS_IF_MARGINAL = [
    "bỏ lỡ",
    "hẹn gặp lại",
    "chào mừng",
]

_HALLUCINATION_SUBSTRINGS = [
    *_HALLUCINATION_SUBSTRINGS_ALWAYS,
    *_HALLUCINATION_SUBSTRINGS_IF_MARGINAL,
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
    return matched >= _MIN_KEYWORD_ECHO_TERMS and matched / len(items) >= _MIN_KEYWORD_ECHO_RATIO


def _normalize_overheard_text(text: str) -> str:
    """One spelling for comparing what STT heard against what the room's TTS was told to say.

    The same recipe the prompt-echo guard uses on its side of the comparison, exported so the
    worker normalizes translate:results lines identically — two normalizers that drift is a
    guard that silently stops matching.
    """
    return " ".join(text.casefold().split()).rstrip(".!?,")


# THE ROOM HEARING ITS OWN TRANSLATION BACK.
#
# A listener hears the far side's dub through their speakers, their microphone re-captures it,
# and the pipeline transcribes it as new speech — attributed to the LISTENER, in the dub's
# language. Browser echo cancellation is modelled on a near-field human talker and synthesized
# speech through laptop speakers routinely defeats it; the client half of the defence
# (half-duplex-mic.tsx in warptalk-web) gates on the tab running current code and on LiveKit's
# isSpeaking signal, and production meeting "Hieu Clone" (21 Aug, 01a0202b…) shows what gets
# through regardless: 77 English segments in ten minutes credited to a participant who had not
# spoken — each one, line for line, the English dub of the other speaker's Vietnamese. Those
# segments were then re-translated, re-synthesized, billed, and taught the language-override
# loop that the listener speaks English.
#
# So the guard of last resort sits where every path converges: a segment whose text matches a
# line the room's own TTS was just told to speak, in that dub's language, is the room hearing
# itself, and it is dropped before it can become transcript, translation, or language evidence.
# The texts to compare against come from translate:results, fetched and windowed by the worker
# (STTWorker._get_recent_dub_texts).
#
# Length floors rise with match looseness: "yeah." said near a dub that also said "Yeah." is
# ordinary back-channel, not echo, and dropping real speech is the one failure mode this guard
# must not have.
_DUB_ECHO_MIN_EXACT_CHARS = 6
_DUB_ECHO_MIN_PARTIAL_CHARS = 10
_DUB_ECHO_MIN_FUZZY_CHARS = 12
_DUB_ECHO_FUZZY_RATIO = 0.85


def _matches_recent_dub(
    normalized_text: str,
    recent_dub_texts: Sequence[str],
) -> bool:
    """Whether this segment re-captures a dub line the room just played.

    `recent_dub_texts` holds translated_text lines, already normalized with
    _normalize_overheard_text and already windowed for recency by the caller.

    DELIBERATELY NO LANGUAGE CONDITION. The segment's language label is the one thing echo
    corrupts: an English dub re-captured from a speaker declared `vi` resolves to `vi` —
    Latin text carries no unambiguous evidence, so the declaration wins — and a guard gated
    on the label matching the dub's language misses exactly the production case it was built
    from. Cross-language coincidence needs no gate either: text in one language does not
    fuzzy-match text in another, so the text comparison already separates them.
    """
    if not normalized_text:
        return False
    for dub_text in recent_dub_texts:
        if not dub_text:
            continue
        if len(normalized_text) >= _DUB_ECHO_MIN_EXACT_CHARS and normalized_text == dub_text:
            return True
        if len(normalized_text) >= _DUB_ECHO_MIN_PARTIAL_CHARS and (
            normalized_text in dub_text or dub_text in normalized_text
        ):
            return True
        if (
            len(normalized_text) >= _DUB_ECHO_MIN_FUZZY_CHARS
            and len(dub_text) >= _DUB_ECHO_MIN_FUZZY_CHARS
            and SequenceMatcher(None, normalized_text, dub_text).ratio() >= _DUB_ECHO_FUZZY_RATIO
        ):
            return True
    return False


def _filter_segments(
    segments_raw: list[dict[str, Any]],
    detected_language: str,
    chunk_offset_ms: int,
    allowed_languages: set[str] | None = None,
    real_duration_s: float | None = None,
    context_prompt: str | None = None,
    keywords: list[str] | None = None,
    min_avg_logprob: float = -0.7,
    min_avg_logprob_by_language: dict[str, float] | None = None,
    recent_dub_texts: Sequence[str] | None = None,
) -> list[TranscribedSegment]:
    language_known = detected_language != "unknown"
    lang_code = _normalize_language(detected_language) if language_known else None

    # The languages this meeting is allowed to produce — the distinct set of languages
    # its participants declared they speak (from their profile settings, published to
    # Redis; see STTWorker._get_room_languages). This replaces the old hard-coded vi/en
    # allow-list, honoring the design where a meeting declares which languages it will
    # contain instead of a single source/target pair.
    # Whether the room actually TOLD us its languages, as opposed to us assuming vi/en.
    # The distinction matters for the cross-script guard below: an assumed all-Latin
    # allow-list is not evidence that the room is all-Latin.
    languages_declared = bool(allowed_languages)
    allowed = {_normalize_language(lang) for lang in (allowed_languages or ())}
    if not allowed:
        allowed = set(_DEFAULT_ALLOWED_LANGUAGES)
    # The speaker's OWN declared language is always allowed. STT is pinned to it on the
    # session (see _get_or_create_session), so filtering it back out here is exactly the
    # bug that produced no transcript at all for a non-vi/en speaker.
    if lang_code:
        allowed.add(lang_code)

    # There WAS a "reject the whole batch if its language is not allowed" check here. It
    # could never fire: `lang_code` is the speaker's declared language and the two lines
    # above had just added it to `allowed`, so the condition was always false. Worse, it
    # read as the thing enforcing "only the room's languages" while enforcing nothing —
    # which is why a room configured for vi + ja was showing Arabic. Enforcement is the
    # script allow-list below, which tests the TEXT rather than a label that was assigned
    # from the speaker's profile before anyone looked at what they said.

    # WHAT THE ROOM MAY CONTAIN, AS AN ALLOW-LIST
    #
    # A room that declared vi + ja declared two writing systems: Latin and Japanese. Text
    # in any other is not that room's speech, whoever the model thinks was talking. So the
    # scripts of the DECLARED languages are what a segment is checked against.
    #
    # This replaces a blocklist — Thai, Hangul, CJK, Kana — which failed in both
    # directions at once, and production hit both:
    #
    #   Arabic was never on the list, so it was never dropped. A Vietnamese/Japanese room
    #   showed Arabic transcript lines.
    #
    #   The list armed only for a Latin-declared speaker, so a speaker registered as
    #   Vietnamese who READ JAPANESE ALOUD had every kanji segment deleted as "foreign
    #   script" — in a room that had declared Japanese. That is the report "đọc Kanji thì
    #   không bắt transcript", and it was this guard doing it.
    #
    # Latin is never filtered on: it is the script code-switched English arrives in, and
    # "deploy the backend API" inside a Vietnamese sentence is the product working.
    #
    # PRECEDENCE: the room, then the speaker, then nothing.
    #
    #   The room declared its languages   -> those scripts. This is what fixes the report:
    #                                        a vi-registered speaker reading Japanese in a
    #                                        room that declared ja is speaking one of the
    #                                        room's languages, and the speaker's own
    #                                        profile must not overrule the room's set.
    #   Only the speaker's is known       -> that speaker's scripts. The room's set is
    #                                        cached for 15s and can be briefly empty; in
    #                                        that window the speaker's declaration is the
    #                                        only evidence there is, and it is better than
    #                                        none.
    #   Neither                           -> filter nothing. An ASSUMED vi/en allow-list
    #                                        once armed this guard and deleted a real
    #                                        Japanese speaker's entire transcript in a room
    #                                        that had simply not finished announcing
    #                                        itself. Losing a speaker beats letting a rare
    #                                        cross-script hallucination through.
    if languages_declared:
        permitted_scripts: set[str] | None = _allowed_scripts(allowed)
    elif lang_code:
        permitted_scripts = _allowed_scripts({lang_code})
    else:
        permitted_scripts = None

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

        # The room said which languages it contains. A writing system belonging to none of
        # them is not this room's speech.
        if permitted_scripts is not None:
            foreign_scripts = _scripts_in(text) - permitted_scripts
            if foreign_scripts:
                logger.info(
                    "filtered_foreign_script",
                    text=text[:80],
                    scripts=sorted(foreign_scripts),
                    declared=sorted(allowed),
                )
                continue

        # Realtime completed events expose token logprobs when explicitly requested in
        # the session include list. transcribe() averages those into avg_logprob;
        # STT_UNKNOWN_CONFIDENCE (-1.0) remains the compatibility fallback for an older
        # event with no logprobs. WT-277: it is a sentinel, not a score — consumers map it
        # back to NULL rather than persisting it. It is still used as a real number by the
        # local quality gates below (a segment with no logprobs is treated as marginal),
        # which is why it is not None here.
        avg_logprob = float(seg.get("avg_logprob", STT_UNKNOWN_CONFIDENCE))
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
            logger.info("filtered_prompt_echo", text=text[:80])
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
            logger.info("filtered_no_speech", text=text, no_speech_prob=round(no_speech, 2))
            continue

        # Resolved here, before the confidence gate, because the gate itself is
        # per-language: models are measurably less confident in some languages than
        # others at identical audio quality, so one shared floor discards more real
        # speech from the languages it already handles worst.
        # Writing system first, declared language second.
        #
        # `lang_code` is what the SPEAKER REGISTERED, not what the model heard — the
        # Realtime completed event carries no language field, so every segment used to be
        # labelled with the speaker's profile language no matter what they actually said.
        # A Vietnamese-registered speaker reading Japanese produced segments labelled `vi`,
        # which then went to the translator as vi→ja and came back as nonsense, and which
        # no language filter could ever reject because the label was allowed by
        # construction.
        #
        # Script evidence does not have that problem: kana is Japanese whatever the
        # profile says. It is used when present and the declared language is kept
        # otherwise, so Latin-script speech — where script proves nothing — behaves
        # exactly as before.
        # Evidence, then the declaration, then a guess.
        #
        # This used to read `_detect_script_language(text) or lang_code or ...`, which looks
        # like the same order but is not: _detect_script_language is blind to Latin script, so
        # for a vi/en room the first term was ALWAYS None and the declaration always won.
        # _detect_unambiguous_language adds the Vietnamese-unique evidence that was previously
        # locked inside the no-declaration fallback path.
        seg_lang = (
            _detect_unambiguous_language(text)
            or lang_code
            or _guess_language_from_text(text, allowed)
        )
        # Before the contradiction log below, deliberately: an echoed dub is exactly a segment
        # whose language contradicts the declaration, and letting it write that log line is the
        # confusion this guard exists to remove.
        if recent_dub_texts and _matches_recent_dub(normalized_text, recent_dub_texts):
            logger.info(
                "filtered_dub_echo",
                text=text[:80],
                language=seg_lang,
            )
            continue
        if lang_code and seg_lang != lang_code:
            # Worth a line in the log: the speaker's profile and their actual speech disagree,
            # which is a setup mistake they cannot see and which degrades their transcription
            # badly — STT is pinned to the declared language on the session.
            logger.info(
                "stt_declared_language_contradicted",
                speaker_declared=lang_code,
                text_evidence=seg_lang,
            )
        language_floor = (min_avg_logprob_by_language or {}).get(
            base_language(seg_lang),
            min_avg_logprob,
        )

        # -1.0 is the compatibility sentinel for old completed events that lacked
        # logprobs. Current sessions request real token logprobs; a real value below
        # the calibrated boundary is marginal audio and must not become an off-topic
        # plausible-looking caption.
        if avg_logprob != STT_UNKNOWN_CONFIDENCE and avg_logprob < language_floor:
            logger.info(
                "filtered_low_confidence",
                text=text,
                logprob=round(avg_logprob, 2),
                language=seg_lang,
                floor=language_floor,
            )
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
            logger.info(
                "filtered_text_too_long_for_audio",
                text=text[:60],
                duration_s=round(real_duration_s, 2),
            )
            continue

        # A blocklisted string is only evidence of hallucination when the AUDIO evidence
        # is weak — see _is_marginal_for_blocklist, which falls back to no-speech
        # probability and speech duration on models that expose no confidence at all.
        blocklist_is_marginal = _is_marginal_for_blocklist(
            avg_logprob,
            no_speech,
            real_duration_s,
        )

        if text_lower in _HALLUCINATIONS_ALWAYS:
            logger.info("filtered_hallucination", text=text)
            continue

        if blocklist_is_marginal and text_lower in _HALLUCINATIONS_IF_MARGINAL:
            logger.info(
                "filtered_hallucination_marginal",
                text=text,
                logprob=round(avg_logprob, 2),
            )
            continue

        if any(sub in text_lower for sub in _HALLUCINATION_SUBSTRINGS_ALWAYS):
            logger.info("filtered_hallucination_substring", text=text)
            continue

        if blocklist_is_marginal and any(
            sub in text_lower for sub in _HALLUCINATION_SUBSTRINGS_IF_MARGINAL
        ):
            logger.info(
                "filtered_hallucination_substring_marginal",
                text=text,
                logprob=round(avg_logprob, 2),
            )
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
                logger.info(
                    "filtered_repeated_sentence_collage",
                    text=text[:80],
                    repeated_sentences=repeated_sentence_count,
                )
                continue

        # Whisper-family hallucination on marginal/noisy audio loops a word or short
        # phrase (e.g. "Nora, Nuang Nora Va Nuang Nora").
        #
        # This used to gate on the most-repeated word's count against len(words) // 2,
        # which was wrong in BOTH directions and measurably so:
        #   "Ừ ừ đúng rồi"                    -> 4 words, top count 2, threshold 2 -> DROPPED
        #   "Nora Nuang Nora Va Nuang Nora Va Nuang" -> 8 words, top 3, threshold 4 -> KEPT
        # Natural doubling ("ừ ừ", "no no", "rất rất") is normal in both Vietnamese and
        # English and was being deleted, while the actual loop it was written to catch
        # walked straight through as soon as it ran long enough.
        #
        # The separating signal is lexical VARIETY, not the top word's count: real speech
        # keeps introducing new words, a loop does not.
        words = text_lower.replace(",", "").split()
        if len(words) >= 4:
            distinct_ratio = len(set(words)) / len(words)
            if distinct_ratio < _MIN_DISTINCT_WORD_RATIO:
                logger.info(
                    "filtered_repetition",
                    text=text[:50],
                    distinct_ratio=round(distinct_ratio, 2),
                )
                continue

        if re.search(r"(.)\1{3,}", text_lower):
            logger.info("filtered_char_repetition", text=text[:50])
            continue

        if text_lower in seen_texts:
            logger.info("filtered_duplicate", text=text)
            continue
        seen_texts.add(text_lower)

        # base_language: the repair is for the Vietnamese language, not for one spelling of
        # its tag. STT returning "vi-VN" skipped it entirely.
        corrected = _fix_vietnamese(text) if base_language(seg_lang) == "vi" else text
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
        min_avg_logprob_by_language: dict[str, float] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.noise_reduction = noise_reduction
        self.min_avg_logprob = min_avg_logprob
        self.min_avg_logprob_by_language = dict(min_avg_logprob_by_language or {})
        self._client: AsyncOpenAI | None = None
        # (meeting_id, speaker_id) -> {"manager": ..., "conn": ..., "last_used": float}
        self._sessions: dict[tuple[str, str], dict[str, Any]] = {}
        # WHAT THIS SPEAKER IS ACTUALLY SPEAKING, when it contradicts what they declared.
        #
        # STT is pinned to the declared language on the session, so a participant who picks
        # the wrong one in their profile gets a whole meeting of mistranscription and no sign
        # of why: the audio is Vietnamese, the decoder is told to expect English, and the words
        # come back fluent-looking and wrong. It is a setup mistake, invisible to the person
        # making it, and it cannot be fixed by anything downstream — once the decoder has
        # mis-heard a word, the word is gone.
        #
        # So the evidence is allowed to correct the declaration, after it has been consistent.
        # `_language_evidence` counts consecutive contradicting segments; `_language_override`
        # is what actually replaces the pinned language once the count is met.
        self._language_evidence: dict[tuple[str, str], tuple[str, int]] = {}
        # (spoken language, the declaration it corrects). The declaration is part of the entry
        # because the override only holds while that declaration stands — a fresh pick in the
        # meeting bar releases it (see transcribe).
        self._language_override: dict[tuple[str, str], tuple[str, str | None]] = {}
        self._warm_sessions: deque[dict[str, Any]] = deque()
        # How many warm sockets to keep ready. Set by warm_up() and used by
        # _schedule_warm_refill to replace every socket a speaker claims.
        self._warm_target = 0
        self._warm_refill_task: asyncio.Task[None] | None = None

    async def load(self) -> None:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI STT")

        self._client = AsyncOpenAI(api_key=self.api_key)
        logger.info("openai_stt_ready", model=self.model)

    async def _open_warm_socket(self) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise RuntimeError("OpenAI STT is not loaded")
        manager = client.realtime.connect(extra_query={"intent": "transcription"})
        conn = await manager.__aenter__()
        return {"manager": manager, "conn": conn}

    async def warm_up(self, pool_size: int = 4) -> None:
        """Open reusable transcription sockets before the first participant speaks."""
        if self._client is None:
            raise RuntimeError("OpenAI STT is not loaded")

        # Remembered so the pool can be refilled later. Without this the pool was a
        # one-shot allocation: see _refill_warm_sessions.
        self._warm_target = max(0, pool_size)

        opened = await asyncio.gather(
            *(self._open_warm_socket() for _ in range(self._warm_target)),
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

    def _schedule_warm_refill(self) -> None:
        """Top the warm pool back up, off the caller's critical path.

        The pool used to be filled exactly once, at worker startup, and every claimed
        socket was simply gone: _get_or_create_session popped one and nothing ever put
        one back. So the first four speakers a process ever saw got an instant session
        and everyone after them — every later participant, every later meeting, for the
        rest of the process's life — paid the full ~1-2s Realtime handshake on their
        FIRST utterance. That is the delay a user feels as "I joined, I spoke, and the
        transcript took a moment to appear", and it got worse the longer a deployment
        stayed up.

        Refilling in the background rather than inline keeps the claim itself instant.
        """
        if getattr(self, "_warm_target", 0) <= 0 or self._client is None:
            return
        existing = getattr(self, "_warm_refill_task", None)
        if existing is not None and not existing.done():
            return
        self._warm_refill_task = asyncio.create_task(self._refill_warm_sessions())

    async def _refill_warm_sessions(self) -> None:
        warm_sessions = getattr(self, "_warm_sessions", None)
        if warm_sessions is None:
            warm_sessions = deque()
            self._warm_sessions = warm_sessions

        while len(warm_sessions) < self._warm_target:
            try:
                warm_sessions.append(await self._open_warm_socket())
            except Exception as exc:  # noqa: BLE001 - a provider hiccup must not kill the worker
                # Stop rather than spin: the next claim schedules another attempt, so a
                # provider outage costs cold handshakes, never a reconnect loop.
                logger.warning("stt_warm_refill_failed", error=str(exc))
                return
        logger.debug("stt_warm_pool_refilled", connections=len(warm_sessions))

    async def close(self) -> None:
        # Stop refilling before draining, or the task races the shutdown and reopens
        # sockets nobody will ever close.
        self._warm_target = 0
        refill = getattr(self, "_warm_refill_task", None)
        if refill is not None and not refill.done():
            refill.cancel()
            with suppress(asyncio.CancelledError):
                await refill

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
        noise_reduction: str | None = None,
    ) -> None:
        """Claim/configure a warm socket when a participant publishes their mic track."""
        await self._get_or_create_session(
            (meeting_id, speaker_id),
            language=language,
            prompt=prompt,
            allowed_languages=allowed_languages,
            keywords=keywords,
            noise_reduction=noise_reduction,
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
        noise_reduction: str | None = None,
        on_early_segment: Callable[[TranscribedSegment], Awaitable[None]] | None = None,
        on_speculative_segment: Callable[[TranscribedSegment], Awaitable[None]] | None = None,
        streamed_epoch: int | None = None,
        recent_dub_texts: Sequence[str] | None = None,
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

        # An override earned from this speaker's own speech outranks their profile. It is only
        # ever set after several consecutive segments carried unambiguous evidence of another
        # language (see _learn_language_evidence), and it re-pins the realtime session through
        # the ordinary language-change path in _get_or_create_session.
        # getattr, not attribute access: instances built without __init__ are a supported
        # shape here — see the same guard on _warm_sessions — and this must not be the thing
        # that decides whether transcription runs at all.
        lang_arg = self._apply_language_override((meeting_id, speaker_id), lang_arg)

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
                min_avg_logprob_by_language=getattr(self, "min_avg_logprob_by_language", None),
                recent_dub_texts=recent_dub_texts,
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
                min_avg_logprob_by_language=getattr(self, "min_avg_logprob_by_language", None),
                recent_dub_texts=recent_dub_texts,
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
                noise_reduction=noise_reduction,
                streamed_epoch=streamed_epoch,
            )
        except Exception as first_error:
            # A capability the API rejected ASYNCHRONOUSLY has to be learned here, because the
            # degradation ladder in _apply_session_config cannot see it. That ladder catches an
            # exception from `session.update`; the Realtime API accepts the update, says nothing,
            # and then rejects every transcription on the stream with an `error` event — which
            # arrives as the RuntimeError being handled right now. The memo was therefore written
            # as "supported", the retry below rebuilt an identical session, and it failed
            # identically. Production ran 304 chunks through that loop in 45 minutes without one
            # word of transcript: every audio chunk in every meeting, silently, for as long as the
            # model was configured.
            demoted = _demote_capability_from_error(self.model, str(first_error))
            if demoted:
                logger.warning(
                    "stt_capability_demoted_from_stream_error",
                    model=self.model,
                    unsupported=demoted,
                    meeting_id=meeting_id,
                )
            logger.warning("realtime_session_retry", meeting_id=meeting_id, speaker_id=speaker_id)
            self._sessions.pop(key, None)
            # No streamed_epoch on this path, deliberately: the line above threw the session
            # away, and with it the buffer the frames were appended to. This retry must send the
            # audio itself, which is what omitting the epoch makes it do.
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
                    noise_reduction=noise_reduction,
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
        segments = _filter_segments(
            segments_dicts,
            detected_language,
            chunk_offset_ms,
            allowed_languages,
            real_duration_s=duration_s,
            context_prompt=prompt,
            keywords=keywords,
            min_avg_logprob=getattr(self, "min_avg_logprob", -0.7),
            # Was missing here while the early/speculative paths passed it — the per-language
            # floor never applied to the completed path, the one place with real logprobs.
            min_avg_logprob_by_language=getattr(self, "min_avg_logprob_by_language", None),
            recent_dub_texts=recent_dub_texts,
        )
        # Learned from the COMPLETED path only. Early and speculative segments are provisional
        # by construction, and re-pinning a session on a guess that a later completed event
        # withdraws would be worse than the mislabelling this exists to fix.
        self._learn_language_evidence((meeting_id, speaker_id), lang_arg, segments)
        return segments

    def _apply_language_override(
        self,
        key: tuple[str, str],
        declared: str | None,
    ) -> str | None:
        """The language to pin this chunk's session to: the declaration, unless a learned
        override still corrects it.

        A FRESH DECLARATION RELEASES THE OVERRIDE.

        The override corrects one specific claim — "declared en while audibly speaking vi" —
        and it used to outlive that claim: once learned it was permanent for the meeting, and
        nothing a person did could take their microphone back.

        Production meeting 01a00a34 (16 Aug) is the whole story. A speaker joined declared en,
        spoke Vietnamese, and the override correctly re-pinned them to vi. They then
        deliberately picked English in the meeting bar and STARTED SPEAKING ENGLISH — and every
        English sentence ("I still hear Vietnamese, so I ask in English", "Hello, can you hear
        me?") kept coming back labelled vi, because the learning loop returns early while an
        override exists and English text carries none of the unambiguous evidence that could
        contradict it. Downstream, their vi-listening partner got NO translation at all: source
        vi, target vi, dropped as same-language. The user's deliberate choice was unrecoverable.

        So the override is scoped to the declaration it corrected. The moment the declared
        language differs from the one the override was learned against, the person has said
        something NEW about themselves, and that statement gets the same initial trust a
        join-time declaration does. If they are still actually speaking something else, the
        evidence loop simply re-learns — two unambiguous segments, same as the first time.
        """
        overrides: dict[tuple[str, str], tuple[str, str | None]] | None = getattr(
            self, "_language_override", None
        )
        if not overrides:
            return declared

        entry = overrides.get(key)
        if not entry:
            return declared

        learned, corrected_declaration = entry

        if declared != corrected_declaration:
            overrides.pop(key, None)
            evidence = getattr(self, "_language_evidence", None)
            if evidence is not None:
                evidence.pop(key, None)
            logger.info(
                "stt_language_override_released",
                meeting_id=key[0],
                speaker_id=key[1],
                was_speaking=learned,
                old_declaration=corrected_declaration,
                new_declaration=declared,
            )
            return declared

        if learned != declared:
            logger.info(
                "stt_language_override_applied",
                meeting_id=key[0],
                speaker_id=key[1],
                declared=declared,
                speaking=learned,
            )
            return learned

        return declared

    def _learn_language_evidence(
        self,
        key: tuple[str, str],
        declared: str | None,
        segments: list[TranscribedSegment],
    ) -> None:
        """Let a speaker's actual speech correct the language they declared.

        Only unambiguous evidence counts — a non-Latin writing system, or the Vietnamese-unique
        character class — so ordinary Latin text never moves this. `_filter_segments` has
        already resolved each segment's language through `_detect_unambiguous_language`, so a
        label that differs from `declared` IS that evidence.

        CONSECUTIVE, not cumulative. One contradicting segment in an otherwise consistent
        meeting is far more likely to be a stray mis-transcription than a person switching
        language mid-sentence, and re-pinning the session on it would trade a rare wrong label
        for a whole chunk of wrong audio. Any segment that agrees with the declaration resets
        the count to zero.
        """
        if not declared or not segments:
            return

        # Lazily created for the same reason transcribe() reads them with getattr: an instance
        # may exist without __init__ having run.
        if getattr(self, "_language_evidence", None) is None:
            self._language_evidence = {}
        if getattr(self, "_language_override", None) is None:
            self._language_override = {}

        if key in self._language_override:
            return

        for segment in segments:
            if segment.language == declared:
                self._language_evidence.pop(key, None)
                continue

            previous_language, count = self._language_evidence.get(key, (segment.language, 0))
            count = count + 1 if previous_language == segment.language else 1
            self._language_evidence[key] = (segment.language, count)

            if count >= _LANGUAGE_OVERRIDE_SEGMENTS:
                # Stored WITH the declaration it corrects, not alone. transcribe() drops the
                # override the moment the speaker declares something new — an override that
                # outlived its declaration is how a deliberate mid-meeting switch to English
                # stayed pinned to Vietnamese for the rest of the meeting (room 01a00a34).
                self._language_override[key] = (segment.language, declared)
                self._language_evidence.pop(key, None)
                logger.warning(
                    "stt_language_override_learned",
                    meeting_id=key[0],
                    speaker_id=key[1],
                    declared=declared,
                    speaking=segment.language,
                    after_segments=count,
                )
                return

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
        noise_reduction: str | None = None,
        exclude_emitted_from_final: bool = True,
        streamed_epoch: int | None = None,
    ) -> tuple[str, float]:
        session = await self._get_or_create_session(
            key,
            language,
            prompt,
            allowed_languages,
            keywords,
            noise_reduction=noise_reduction,
        )
        conn = session["conn"]

        # ALREADY IN THE BUFFER? Then commit it rather than sending it twice.
        #
        # When STT_STREAMING_ENABLED is on, the frames of this turn were appended by
        # `append_streamed_audio` while the speaker was still talking — so by the time this runs
        # the model has already heard the utterance and the commit below is all that is left.
        # That is the entire latency win: what used to be "send five seconds, then wait for the
        # model to hear it" becomes "say go".
        #
        # The epoch check is what makes it safe. `append_streamed_audio` returns the epoch its
        # audio landed in, and a session recreated since then — a language change, an idle
        # sweep, a restart — took that buffer with it. A mismatch therefore falls through to the
        # ordinary append below, which is exactly the behaviour this method had before streaming
        # existed. Wrong here means slow; it never means silent.
        already_buffered = streamed_epoch is not None and int(session.get("epoch", 0)) == int(
            streamed_epoch
        )

        if not already_buffered:
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
                        if (
                            value := (
                                item.get("logprob")
                                if isinstance(item, dict)
                                else getattr(item, "logprob", None)
                            )
                        )
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

    async def _degrade_session_config(
        self,
        conn: Any,
        language: str | None,
        prompt: str | None,
        allowed_languages: set[str] | None,
        keywords: list[str],
        noise_reduction: str | None = None,
    ) -> None:
        """Find the richest session config this model accepts, and remember it.

        Tried in order, keeping as much context as possible at each rung:

            1. drop structured context (`languages` + `keywords`), keep prompt + logprobs
            2. also drop the logprobs selector
            3. bare config — the previous behaviour's only option

        Whatever succeeds is recorded per model, so a process pays this at most once
        rather than re-deriving it for every speaker in every room.
        """
        attempts = [
            ("structured_context", {"structured_context": False, "logprobs": True}),
            ("logprobs", {"structured_context": False, "logprobs": False}),
        ]
        for rejected, flags in attempts:
            try:
                await conn.session.update(
                    session=cast(
                        Any,
                        self._session_payload(
                            language,
                            prompt,
                            allowed_languages,
                            keywords,
                            noise_reduction=noise_reduction,
                            # Named rather than splatted: **flags is dict[str, bool] and would
                            # otherwise be a candidate for every keyword parameter, including the
                            # string one added for per-room noise reduction.
                            structured_context=flags["structured_context"],
                            logprobs=flags["logprobs"],
                        ),
                    )
                )
            except Exception:
                continue

            _STRUCTURED_CONTEXT_SUPPORT[self.model] = bool(flags["structured_context"])
            _LOGPROBS_SUPPORT[self.model] = bool(flags["logprobs"])
            logger.warning(
                "stt_session_capability_downgraded",
                model=self.model,
                unsupported=rejected,
                structured_context=flags["structured_context"],
                logprobs=flags["logprobs"],
                keyword_count=len(keywords),
            )
            return

        # Nothing optional survived. Keep the session rather than lose the speaker.
        logger.warning(
            "session_optional_fields_rejected",
            model=self.model,
            has_language=bool(language),
            has_prompt=bool(prompt),
            has_keywords=bool(keywords),
        )
        _STRUCTURED_CONTEXT_SUPPORT[self.model] = False
        _LOGPROBS_SUPPORT[self.model] = False
        await conn.session.update(
            session=cast(
                Any,
                self._session_payload(
                    None,
                    None,
                    structured_context=False,
                    logprobs=False,
                ),
            )
        )

    def _session_payload(
        self,
        language: str | None,
        prompt: str | None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
        # Before the `*` on purpose: _degrade_session_config forwards the capability flags as
        # **dict[str, bool], and a keyword-only str parameter sits inside that splat's target set.
        noise_reduction: str | None = None,
        *,
        structured_context: bool | None = None,
        logprobs: bool | None = None,
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
        is_next_generation_transcribe = (
            _supports_structured_context(self.model)
            if structured_context is None
            else structured_context
        )
        wants_logprobs = _supports_logprobs(self.model) if logprobs is None else logprobs
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
        #
        # WT-427: per ROOM, falling back to the worker's default. One meeting is a headset at a
        # desk and the next is a laptop across a table, and the same setting cannot be right for
        # both: the deployment default is "off" precisely because a second denoising pass
        # distorted clean close-mic speech in replay tests, while a room being picked up from two
        # metres away needs exactly that pass. A single env var made this an all-or-nothing choice
        # for the whole platform.
        effective_noise_reduction = (
            noise_reduction if noise_reduction is not None else self.noise_reduction
        )
        if effective_noise_reduction and effective_noise_reduction != "off":
            input_config["noise_reduction"] = {"type": effective_noise_reduction}

        payload: dict[str, Any] = {
            "type": "transcription",
            "audio": {"input": input_config},
        }
        # Independent of structured context above: a model may well accept BOTH the
        # keyword hints and the logprobs selector, and collapsing the two into one
        # model-family flag is what made "send keywords" and "keep confidence" look
        # mutually exclusive. Sending this selector to a model that rejects it fails the
        # whole session update, which _get_or_create_session degrades and memoises.
        if wants_logprobs:
            payload["include"] = ["item.input_audio_transcription.logprobs"]
        return payload

    async def _get_or_create_session(
        self,
        key: tuple[str, str],
        language: str | None = None,
        prompt: str | None = None,
        allowed_languages: set[str] | None = None,
        keywords: list[str] | None = None,
        noise_reduction: str | None = None,
    ) -> dict[str, Any]:
        self._sweep_idle_sessions()

        languages = tuple(_expected_languages(language, allowed_languages))
        normalized_keywords = tuple(_normalized_keywords(keywords))
        cached = self._sessions.get(key)
        # Deliberately NOT the optimistic capability memo, unlike _session_payload. Being
        # wrong about keywords costs a hint that does not help; being wrong here leaves a
        # session pinned to the wrong language for the rest of the meeting, mistranscribing
        # silently. So this stays on models we have actually confirmed handle several
        # languages in one session.
        is_next_generation_transcribe = self.model in {
            "gpt-transcribe",
            "gpt-live-transcribe",
        }
        if (
            cached is not None
            and not is_next_generation_transcribe
            # A session prepared before the speaker's language was known carries None.
            # Reopening it would throw away the very handshake the prewarm paid for and
            # hand the speaker back the delay it existed to remove — the config update
            # below pins it just as well.
            and cached.get("language") is not None
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
                # In the comparison, not merely in the payload. A room that switches to far-field
                # mid-meeting keeps a live socket, and a value the update never notices changed is
                # a setting that silently does nothing until the session happens to be recycled.
                or cached.get("noise_reduction") != noise_reduction
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
                            noise_reduction=noise_reduction,
                        ),
                    )
                )
                cached.update(
                    language=language,
                    prompt=prompt,
                    languages=languages,
                    keywords=normalized_keywords,
                    noise_reduction=noise_reduction,
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
            # Replace what we just took, in the background, so the NEXT speaker is also
            # instant. Without this the pool drained permanently after four claims.
            self._schedule_warm_refill()
        else:
            manager = client.realtime.connect(extra_query={"intent": "transcription"})
            conn = await manager.__aenter__()
            # Empty pool means the refill has not caught up (or has never run) — ask for
            # one now so this speaker is the last to pay the handshake.
            self._schedule_warm_refill()

        try:
            await conn.session.update(
                session=cast(
                    Any,
                    self._session_payload(
                        language,
                        prompt,
                        allowed_languages,
                        list(normalized_keywords),
                        noise_reduction=noise_reduction,
                    ),
                )
            )
        except Exception:
            if not language and not prompt and not normalized_keywords:
                raise
            # Step DOWN one capability at a time instead of collapsing straight to a bare
            # config. The old behaviour threw away the language hint, the prompt and the
            # keywords together on any rejection, so one unsupported field cost all three
            # — and it did so on every new session, having learned nothing.
            await self._degrade_session_config(
                conn,
                language,
                prompt,
                allowed_languages,
                list(normalized_keywords),
                noise_reduction,
            )

        # Bumped per CREATED session, never on a config update — an update keeps the same
        # socket and therefore the same input_audio_buffer, while a new session throws that
        # buffer away. Audio streamed into one epoch must never be committed under another;
        # see append_streamed_audio.
        self._session_epoch = getattr(self, "_session_epoch", 0) + 1
        session = {
            "manager": manager,
            "conn": conn,
            "epoch": self._session_epoch,
            "last_used": time.monotonic(),
            "language": language,
            "prompt": prompt,
            "languages": languages,
            "keywords": normalized_keywords,
            "noise_reduction": noise_reduction,
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

    async def append_streamed_audio(
        self,
        key: tuple[str, str],
        pcm_bytes: bytes,
        sample_rate: int,
    ) -> int | None:
        """Push one frame of live speech into this speaker's OPEN session buffer.

        Returns the session epoch the audio landed in, or None when there was nothing to append
        to. The epoch is the whole safety mechanism: `transcribe` will only commit without
        re-sending the audio if the session it resolves is still that same one. A session
        recreated in between (a language change, an idle sweep, a restart) took the buffer with
        it, so the caller falls back to sending the audio the way it always did — which is why
        the worst case of every failure in this path is the latency this feature exists to
        remove, never a lost sentence.

        DELIBERATELY DOES NOT CREATE A SESSION. Creating one here would mean guessing this
        speaker's language, prompt and keywords from a frame — and a session pinned to the wrong
        language transcribes the rest of the meeting badly and silently. `process` owns session
        lifecycle; this only ever borrows one that already exists, which in practice the
        track-published prewarm has already opened.
        """
        session = getattr(self, "_sessions", {}).get(key)
        if session is None:
            return None

        try:
            pcm_24k = _resample_pcm16(pcm_bytes, sample_rate, REALTIME_SAMPLE_RATE)
            await session["conn"].input_audio_buffer.append(
                audio=base64.b64encode(pcm_24k).decode()
            )
        except Exception:
            # A frame that does not land is not an error anybody needs to act on: the closed
            # utterance still carries the whole turn. Debug, because this fires per 96ms window.
            logger.debug("stt_stream_append_failed", meeting_id=key[0], exc_info=True)
            return None

        session["last_used"] = time.monotonic()
        return int(session.get("epoch", 0))

    async def discard_streamed_audio(self, key: tuple[str, str]) -> None:
        """Throw away whatever has been appended but never committed for this speaker.

        Called when a turn ends without a chunk to commit it — an utterance too short to
        publish, a lost frame, an ingress that died mid-turn. Without it those samples stay in
        the buffer and are transcribed as the opening of the NEXT turn, which is a defect that
        compounds: every abandoned fragment makes the following utterance wronger.
        """
        session = getattr(self, "_sessions", {}).get(key)
        if session is None:
            return
        try:
            await session["conn"].input_audio_buffer.clear()
        except Exception:
            logger.debug("stt_stream_clear_failed", meeting_id=key[0], exc_info=True)

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
