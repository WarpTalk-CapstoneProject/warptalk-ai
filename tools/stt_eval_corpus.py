"""Labelled utterance corpus for auditing stt_worker's post-transcription filters.

Composition mirrors what production actually contains, measured 2026-08-10 over the
1,422 non-system segments in warptalk_transcript.transcript_segments:

    English            778 (55%)
    Vietnamese         644 (45%)
    of those Vietnamese, 304 (47%) carry an embedded ASCII technical term

so a corpus that is mostly clean monolingual English would flatter the pipeline in a
way the real product never sees.

`expected` is the verdict a CORRECT pipeline should reach for the utterance, judged on
the text alone:

    KEEP  — real speech; dropping it loses meeting content
    DROP  — genuine model garbage; keeping it pollutes the transcript

Anything the filters do that disagrees with `expected` is an error with a direction:
a KEEP that gets dropped is a DELETION (silent content loss — the failure mode the
production correction data points at), a DROP that survives is an INSERTION.

These are deliberately written as text-level cases, not audio. They exercise the
filter chain in stt_worker/model.py, which is pure and therefore testable without an
API key, audio, or a live meeting. Audio-level accuracy is a separate question that
needs recordings — see tools/stt_filter_audit.py's module docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

KEEP = "KEEP"
DROP = "DROP"


@dataclass(frozen=True)
class Utterance:
    text: str
    language: str
    expected: str
    # Average token logprob the model would plausibly return for this utterance. Clean
    # close-mic speech sits near 0.0; marginal or noisy speech trends toward -0.7, which
    # is where production's distribution actually ends.
    avg_logprob: float
    # Why this case exists — printed beside any disagreement so a failure is readable.
    note: str = ""
    # Seconds of audio, only needed by the short-audio/long-text guard.
    duration_s: float = 3.0
    tags: tuple[str, ...] = field(default_factory=tuple)


# --- Vietnamese, monolingual -------------------------------------------------------

VIETNAMESE: list[Utterance] = [
    Utterance("Chúng ta cần chốt deadline cho phần này trước thứ sáu.", "vi", KEEP, -0.08),
    Utterance("Em nghĩ là phương án thứ hai hợp lý hơn.", "vi", KEEP, -0.12),
    Utterance("Anh gửi lại tài liệu cho em sau buổi họp nhé.", "vi", KEEP, -0.15),
    Utterance("Vâng.", "vi", KEEP, -0.22, note="short acknowledgement, must survive"),
    Utterance("Dạ em hiểu rồi ạ.", "vi", KEEP, -0.18),
    Utterance(
        "Cái này thì mình chưa có số liệu cụ thể, để em kiểm tra lại rồi báo anh.",
        "vi",
        KEEP,
        -0.31,
    ),
    Utterance(
        "Rất rất tốt, phần này làm ổn rồi.",
        "vi",
        KEEP,
        -0.28,
        note="natural Vietnamese intensifier repetition — must not read as a repetition loop",
        tags=("repetition-risk",),
    ),
    Utterance(
        "Không không, ý em không phải như vậy.",
        "vi",
        KEEP,
        -0.30,
        note="natural negation doubling — same repetition-guard risk",
        tags=("repetition-risk",),
    ),
    Utterance(
        "Ừ ừ đúng rồi.",
        "vi",
        KEEP,
        -0.42,
        note="short + doubled + marginal confidence: three guards at once",
        tags=("repetition-risk",),
    ),
    Utterance(
        "Mình review lại phần đó vào sáng mai được không?",
        "vi",
        KEEP,
        -0.24,
    ),
    Utterance("Thôi để anh làm phần đó cho.", "vi", KEEP, -0.35),
    Utterance("Cảm ơn mọi người đã tham gia buổi họp hôm nay.", "vi", KEEP, -0.09),
]

# --- Vietnamese with embedded English (47% of real vi segments) --------------------

CODE_SWITCHED: list[Utterance] = [
    Utterance(
        "Mình deploy lên staging trước rồi test lại nhé.",
        "vi",
        KEEP,
        -0.19,
        tags=("code-switch",),
    ),
    Utterance(
        "Cái API này trả về lỗi bốn trăm lẻ một.",
        "vi",
        KEEP,
        -0.26,
        tags=("code-switch",),
    ),
    Utterance(
        "Em đã fix cái bug ở phần authentication rồi anh ạ.",
        "vi",
        KEEP,
        -0.21,
        tags=("code-switch",),
    ),
    Utterance(
        "Phần backend cần refactor lại cái repository pattern.",
        "vi",
        KEEP,
        -0.33,
        tags=("code-switch",),
    ),
    Utterance(
        "Anh merge cái pull request đó vào branch main giúp em.",
        "vi",
        KEEP,
        -0.29,
        tags=("code-switch",),
    ),
    Utterance(
        "Con Redis nó bị timeout khi mở nhiều room cùng lúc.",
        "vi",
        KEEP,
        -0.38,
        tags=("code-switch",),
    ),
    Utterance(
        "Mình dùng WebSocket cho realtime, còn REST cho phần settings.",
        "vi",
        KEEP,
        -0.44,
        note="marginal confidence + heavy technical terms — the hardest real case",
        tags=("code-switch", "marginal"),
    ),
    Utterance(
        "Cái latency của translation worker khoảng tám trăm mili giây.",
        "vi",
        KEEP,
        -0.52,
        note="marginal, and overlaps glossary/prompt vocabulary",
        tags=("code-switch", "marginal", "prompt-overlap"),
    ),
    Utterance(
        "Kubernetes nó tự restart cái pod đó rồi.",
        "vi",
        KEEP,
        -0.47,
        tags=("code-switch", "marginal"),
    ),
    Utterance(
        "Em push code lên rồi, anh pull về xem thử.",
        "vi",
        KEEP,
        -0.36,
        tags=("code-switch",),
    ),
]

# --- English, monolingual ----------------------------------------------------------

ENGLISH: list[Utterance] = [
    Utterance("Let's move the deadline to next Friday.", "en", KEEP, -0.07),
    Utterance("I think the second approach makes more sense.", "en", KEEP, -0.11),
    Utterance("Okay.", "en", KEEP, -0.25, note="short acknowledgement"),
    Utterance("Yeah that works for me.", "en", KEEP, -0.19),
    Utterance("Can you share the document after the meeting?", "en", KEEP, -0.13),
    Utterance(
        "The translation worker is timing out on the realtime socket.",
        "en",
        KEEP,
        -0.34,
        tags=("prompt-overlap",),
    ),
    Utterance(
        "We should probably cache that response instead of recomputing it every time.",
        "en",
        KEEP,
        -0.41,
        tags=("marginal",),
    ),
    Utterance("No no, that's not what I meant.", "en", KEEP, -0.32, tags=("repetition-risk",)),
    Utterance(
        "So so basically the pipeline drops the segment.",
        "en",
        KEEP,
        -0.45,
        note="natural disfluency doubling at marginal confidence",
        tags=("repetition-risk", "marginal"),
    ),
    Utterance("Let me check the logs and get back to you.", "en", KEEP, -0.16),
    Utterance("That's a good point, let's write it down.", "en", KEEP, -0.23),
    Utterance("I'll take that action item.", "en", KEEP, -0.28),
    Utterance("Could you repeat that? I didn't catch the last part.", "en", KEEP, -0.39),
    Utterance("We're running about ten minutes over.", "en", KEEP, -0.21),
]

# --- The blocklist collision -------------------------------------------------------
#
# _HALLUCINATIONS in stt_worker/model.py blocks these strings outright, on text alone.
# The list is a YouTube-transcription blocklist ("thanks for watching", "like and
# subscribe", "đăng ký kênh") that also happens to contain the most common things anyone
# says in an actual meeting. Each pair below is the SAME text at clear confidence and at
# hallucination-grade confidence: a correct filter must separate them, and a text-only
# blocklist provably cannot.

BLOCKLIST_COLLISION: list[Utterance] = [
    Utterance("Okay.", "en", KEEP, -0.09, note="clearly spoken agreement", tags=("blocklist",)),
    Utterance("Yeah.", "en", KEEP, -0.11, note="clearly spoken agreement", tags=("blocklist",)),
    Utterance(
        "Thank you.",
        "en",
        KEEP,
        -0.10,
        note="how a real meeting ends",
        tags=("blocklist",),
    ),
    Utterance("Bye.", "en", KEEP, -0.13, note="how a real meeting ends", tags=("blocklist",)),
    Utterance("Hmm.", "en", KEEP, -0.30, note="genuine thinking noise", tags=("blocklist",)),
    Utterance("Ừ", "vi", KEEP, -0.14, note="the commonest Vietnamese ack", tags=("blocklist",)),
    Utterance("À", "vi", KEEP, -0.16, note="Vietnamese realisation particle", tags=("blocklist",)),
    Utterance(
        "Xin chào",
        "vi",
        KEEP,
        -0.12,
        note="how a real Vietnamese meeting OPENS",
        tags=("blocklist",),
    ),
    Utterance(
        "Cảm ơn mọi người",
        "vi",
        KEEP,
        -0.11,
        note="how a real Vietnamese meeting CLOSES",
        tags=("blocklist",),
    ),
    # The same strings as actual hallucinations: marginal confidence, almost no audio.
    Utterance(
        "Okay.",
        "en",
        DROP,
        -0.68,
        duration_s=0.2,
        note="hallucinated onto silence — must still be caught",
        tags=("blocklist", "hallucination"),
    ),
    Utterance(
        "Thank you.",
        "en",
        DROP,
        -0.66,
        duration_s=0.2,
        note="hallucinated onto silence",
        tags=("blocklist", "hallucination"),
    ),
    Utterance(
        "Cảm ơn mọi người",
        "vi",
        DROP,
        -0.65,
        duration_s=0.2,
        note="hallucinated onto silence",
        tags=("blocklist", "hallucination"),
    ),
]


# --- The other four meeting languages ----------------------------------------------
#
# languages.ts offers SIX meeting-scope languages: vi, en, ja, ko, fr, es. Everything
# above this point exercises two of them. These cases exist to answer a question the
# vi/en corpus structurally cannot: does the filter chain treat the other four as
# first-class, or only as "not Vietnamese"?
#
# Each language gets real meeting speech AND the video-outro hallucination that the
# Whisper family is documented to emit in that language — the exact pattern
# _HALLUCINATIONS_ALWAYS blocks for English and Vietnamese.

MULTILINGUAL: list[Utterance] = [
    # Japanese
    Utterance(
        "この部分は金曜日までに終わらせましょう。",
        "ja",
        KEEP,
        -0.14,
        tags=("ja",),
    ),
    Utterance("はい、わかりました。", "ja", KEEP, -0.19, tags=("ja",)),
    Utterance(
        "ご視聴ありがとうございました",
        "ja",
        DROP,
        -0.58,
        note="THE canonical Japanese Whisper hallucination — no blocklist entry exists",
        tags=("ja", "hallucination"),
    ),
    # Korean
    Utterance("이 부분은 금요일까지 끝내야 합니다.", "ko", KEEP, -0.17, tags=("ko",)),
    Utterance("네, 알겠습니다.", "ko", KEEP, -0.21, tags=("ko",)),
    Utterance(
        "구독과 좋아요 부탁드립니다",
        "ko",
        DROP,
        -0.56,
        note="Korean YouTube outro — no blocklist entry exists",
        tags=("ko", "hallucination"),
    ),
    # French
    Utterance(
        "On devrait finir cette partie avant vendredi.",
        "fr",
        KEEP,
        -0.16,
        tags=("fr",),
    ),
    Utterance("Oui, d'accord.", "fr", KEEP, -0.20, tags=("fr",)),
    Utterance(
        "merci d'avoir regardé cette vidéo",
        "fr",
        DROP,
        -0.57,
        note="French YouTube outro — no blocklist entry exists",
        tags=("fr", "hallucination"),
    ),
    # Spanish
    Utterance(
        "Deberíamos terminar esta parte antes del viernes.",
        "es",
        KEEP,
        -0.15,
        tags=("es",),
    ),
    Utterance("Sí, de acuerdo.", "es", KEEP, -0.22, tags=("es",)),
    Utterance(
        "gracias por ver el video",
        "es",
        DROP,
        -0.59,
        note="Spanish YouTube outro — no blocklist entry exists",
        tags=("es", "hallucination"),
    ),
]

# --- Bare acknowledgements in the other four languages ------------------------------
#
# The counterpart to BLOCKLIST_COLLISION. Adding "はい", "네", "oui" and "sí" to the
# confidence-gated blocklist buys protection against the commonest hallucination in each
# language, but it also puts the commonest MEETING WORD in each language behind a gate.
# These pairs are what proves the gate actually separates them rather than just deleting
# the word — which is the mistake the English/Vietnamese list made for months.

MULTILINGUAL_ACKS: list[Utterance] = [
    Utterance("はい", "ja", KEEP, -0.12, note="clearly spoken yes", tags=("ja", "blocklist")),
    Utterance(
        "はい", "ja", DROP, -0.64, duration_s=0.2, note="on silence", tags=("ja", "blocklist")
    ),
    Utterance("네", "ko", KEEP, -0.15, note="clearly spoken yes", tags=("ko", "blocklist")),
    Utterance(
        "네", "ko", DROP, -0.62, duration_s=0.2, note="on silence", tags=("ko", "blocklist")
    ),
    Utterance("Oui", "fr", KEEP, -0.13, note="clearly spoken yes", tags=("fr", "blocklist")),
    Utterance(
        "Oui", "fr", DROP, -0.63, duration_s=0.2, note="on silence", tags=("fr", "blocklist")
    ),
    Utterance("Sí", "es", KEEP, -0.14, note="clearly spoken yes", tags=("es", "blocklist")),
    Utterance(
        "Sí", "es", DROP, -0.61, duration_s=0.2, note="on silence", tags=("es", "blocklist")
    ),
    # Longer real sentences that merely BEGIN with a blocklisted token must never match:
    # the list is exact-match, and these are the regression guard for that.
    Utterance("はい、わかりました。", "ja", KEEP, -0.35, tags=("ja", "blocklist")),
    Utterance("네, 알겠습니다.", "ko", KEEP, -0.38, tags=("ko", "blocklist")),
    Utterance("Oui, tout à fait d'accord.", "fr", KEEP, -0.40, tags=("fr", "blocklist")),
    Utterance("Sí, estoy de acuerdo con eso.", "es", KEEP, -0.42, tags=("es", "blocklist")),
]


# --- Genuine garbage the filters SHOULD remove -------------------------------------

GARBAGE: list[Utterance] = [
    Utterance(
        "Nora, Nuang Nora Va Nuang Nora Va Nuang",
        "vi",
        DROP,
        -0.66,
        note="classic Whisper-family repetition loop on noise",
        tags=("hallucination",),
    ),
    Utterance(
        "aaaaaaaa",
        "vi",
        DROP,
        -0.61,
        note="character repetition",
        tags=("hallucination",),
    ),
    Utterance(
        "This is a fully formed plausible sentence hallucinated onto near silence.",
        "en",
        DROP,
        -0.30,
        duration_s=0.3,
        note="long text on 0.3s of audio — only the duration guard can catch this",
        tags=("hallucination",),
    ),
    Utterance(
        "the the the the the the",
        "en",
        DROP,
        -0.64,
        note="dominant-word repetition",
        tags=("hallucination",),
    ),
    Utterance(
        "Hãy subscribe cho kênh để không bỏ lỡ video tiếp theo",
        "vi",
        DROP,
        -0.55,
        note="YouTube-training-data bleed — the blocklist's legitimate purpose",
        tags=("hallucination",),
    ),
]


def full_corpus() -> list[Utterance]:
    """Every case, in production's own language proportions."""
    return [
        *ENGLISH,
        *VIETNAMESE,
        *CODE_SWITCHED,
        *BLOCKLIST_COLLISION,
        *MULTILINGUAL,
        *MULTILINGUAL_ACKS,
        *GARBAGE,
    ]


def vi_en_only() -> list[Utterance]:
    """The two languages the pipeline was actually built for."""
    return [*ENGLISH, *VIETNAMESE, *CODE_SWITCHED, *BLOCKLIST_COLLISION, *GARBAGE]


def real_speech(corpus: list[Utterance] | None = None) -> list[Utterance]:
    """Only the utterances a correct pipeline must preserve."""
    return [u for u in (corpus or full_corpus()) if u.expected == KEEP]
