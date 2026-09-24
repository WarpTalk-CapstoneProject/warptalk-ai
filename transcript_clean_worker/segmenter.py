"""Grouping STT segments into sentences (WT-716). Pure: no Redis, no clock, no model.

THE PROBLEM THIS SOLVES
    A clean transcript line is a SENTENCE. An STT segment is not: the recogniser cuts on
    silence, so one sentence routinely arrives as two or three segments —

        [ja]  "えーと、あのー、来週の会議ですが"   +500ms   "火曜日に変更になりました"

    — and one segment can carry two finished sentences ("Okay. Let's start."). Publishing one
    clean line per segment would give the reader the recogniser's breathing pattern instead of
    the speaker's sentences.

THE RULES, AND WHY EACH ONE IS THERE
    MERGE while the line is UNFINISHED and the pause is short. "Unfinished" is not just
    "no full stop": gpt-live-transcribe punctuates every segment, so "I think that." ends with
    one and is obviously mid-sentence. A line is therefore unfinished when it has no terminal
    mark OR ends on a continuation word (en "and/because/to", vi "thì/mà/để", ja "が/けど/ので"),
    which is the same signal `punctuation._ja_is_unfinished` uses to decide not to add a "。".

    FLUSH on: a finished line followed by anything, a real turn change, silence
    (`idle_flush_ms`), length (`max_sentence_ms`), or the end of the meeting.

    A BACKCHANNEL DOES NOT END SOMEBODY'S SENTENCE. "ừ", "はい", "mm-hmm" from a listener lands
    in the middle of the speaker's sentence constantly; treating it as a turn change would cut
    every second sentence in the meeting in half. It is emitted as its own line (or dropped, if
    the prepass says it was nothing but filler) and the speaker's buffer is left alone.

    NEVER SPLIT INSIDE A SEGMENT. `segment_ids` must map cleanly back to the raw record — a
    clean line is the thing corrections and the transcript→video mark point are keyed on — so a
    buffer is only ever cut at a segment boundary. A segment holding "Okay. Let's start." stays
    one line rather than becoming two lines that both claim the same raw segment.

    NEVER COMPLETE A SENTENCE. A line cut by length or by the meeting ending is published as it
    stands. Not even an ellipsis: this whole feature is deletion-only, and "…" is a character
    the speaker did not say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shared.disfluency import FLAG_FILLER_ONLY, normalize_key
from shared.disfluency.lexicon_ja import CONTINUATION_PARTICLES as JA_CONTINUATION_PARTICLES
from shared.disfluency.normalize import resolve_language
from shared.disfluency.tokenize import TERMINALS, ja_morphology_available, tokenize_spans

# Words a line cannot honestly end on. They are what somebody says on the way INTO the rest of
# the sentence, so a terminal mark after one of them is the recogniser's punctuation model
# guessing, not the speaker stopping.
_EN_CONTINUATION = frozenset(
    {"and", "but", "because", "so", "to", "the", "of", "which", "that", "a", "an"}
)
_VI_CONTINUATION = frozenset(
    normalize_key(w, "vi")
    for w in ("thì", "là", "mà", "nên", "để", "nhưng", "vì", "với", "của", "và")
)
# て/で are in here and deliberately NOT in `lexicon_ja.CONTINUATION_PARTICLES`: that set decides
# whether to add a "。" and "ちょっと待って" is a complete request, while here the question is
# only whether the NEXT segment continues this one, and after a て-form it usually does.
_JA_CONTINUATION = frozenset(JA_CONTINUATION_PARTICLES) | {
    "て",
    "で",
    "から",
    "たら",
    "と",
    "けど",
    "が",
    "し",
    "ば",
    "ので",
}

# Short answers and acknowledgements. Not a filler list — these are meaningful lines and are
# published as their own sentences; the set exists only to answer "did the turn really change?".
_BACKCHANNELS = {
    "en": frozenset(
        {
            "yeah",
            "yep",
            "yes",
            "yup",
            "ok",
            "okay",
            "right",
            "sure",
            "exactly",
            "mm-hmm",
            "mhm",
            "uh-huh",
            "hmm",
            "mm",
            "no",
            "true",
        }
    ),
    "vi": frozenset(
        normalize_key(w, "vi")
        for w in ("ừ", "ừm", "ờ", "dạ", "vâng", "đúng", "rồi", "ok", "okay", "được", "không")
    ),
    "ja": frozenset(
        normalize_key(w, "ja")
        for w in ("はい", "ええ", "うん", "そう", "なるほど", "はいはい", "うんうん", "ああ")
    ),
}
_BACKCHANNEL_MAX_TOKENS = 3

# Capitalisation is not a word, so lowering it back down when two segments are joined is allowed
# where deleting a word would not be — but only for words that cannot be a name. "Anh" is a
# pronoun AND a name in Vietnamese; the cost of being wrong is a lower-case name, the cost of
# not doing it at all is a capital letter in the middle of every merged sentence.
_JOIN_LOWERCASE = {
    "en": frozenset(
        {
            "a", "an", "and", "are", "as", "at", "be", "because", "but", "can", "for", "he",
            "her", "his", "if", "in", "is", "it", "its", "of", "on", "or", "our", "she", "so",
            "that", "the", "their", "then", "there", "they", "this", "to", "we", "were", "what",
            "when", "which", "will", "with", "would", "you", "your",
        }
    ),
    "vi": frozenset(
        normalize_key(w, "vi")
        for w in (
            "thì", "là", "mà", "và", "nhưng", "vì", "để", "với", "của", "cái", "này", "đó",
            "sẽ", "đã", "đang", "có", "không", "rồi", "cho", "tôi", "mình", "chúng", "nó",
            "khi", "nếu", "nên", "trong", "trên", "ở",
        )
    ),
}

# Terminal marks a merge may drop. "?" and "!" are never dropped: the recogniser heard an
# intonation to write them, and deleting one would lose a question — the exact thing invariant
# I2_question_marker protects.
_SOFT_TERMINALS = (".", "。", "．")


@dataclass(frozen=True, slots=True)
class CleanSegment:
    """One final STT segment, with whatever the deterministic prepass made of it.

    `clean_text` is "" for a filler-only segment ("Ummm"), exactly as `prepass` reports it —
    the segment still belongs to the sentence's `segment_ids`, it just contributes no text.
    `raw_text` is never edited; it is what the LLM tier is shown and what the invariants are
    checked against.
    """

    segment_id: str
    speaker_id: str
    language: str
    raw_text: str
    clean_text: str
    flags: frozenset[str] = frozenset()
    start_ms: int = 0
    end_ms: int = 0
    arrived_at_ms: int = 0

    @property
    def is_filler_only(self) -> bool:
        return FLAG_FILLER_ONLY in self.flags or not self.clean_text.strip()


@dataclass(frozen=True, slots=True)
class CleanSentence:
    """One transcript line: the segments it was built from, and both renderings of it."""

    speaker_id: str
    language: str
    segments: tuple[CleanSegment, ...]
    raw_text: str
    prepass_text: str
    reason: str  # why it was flushed: finished | turn_change | idle | max_length | meeting_end
    flags: frozenset[str] = frozenset()

    @property
    def segment_ids(self) -> list[str]:
        return [segment.segment_id for segment in self.segments]

    @property
    def start_ms(self) -> int:
        return self.segments[0].start_ms if self.segments else 0

    @property
    def end_ms(self) -> int:
        return self.segments[-1].end_ms if self.segments else 0

    @property
    def is_empty(self) -> bool:
        """Nothing survived the prepass — every segment was filler. Publish nothing."""
        return not self.prepass_text.strip()


# --- text helpers (pure, shared with the LLM tier) --------------------------------------------


def _lang(segment_language: str, text: str) -> str:
    return resolve_language(segment_language, text) or "en"


def _last_token_key(text: str, language: str) -> str:
    tokens = [t for t in tokenize_spans(text, language) if not t.punct]
    return tokens[-1].key if tokens else ""


def ends_with_continuation(text: str, language: str) -> bool:
    """Whether `text` ends on a word that hands over to the rest of the sentence."""
    stripped = text.strip().rstrip("".join(TERMINALS) + " 　")
    if not stripped:
        return False
    lang = _lang(language, stripped)
    if lang == "ja":
        if not ja_morphology_available():
            # The fallback tokenizer cannot see morphemes, so a suffix test is all there is.
            # It over-fires on words ENDING in a particle ("ちょっと"), which costs a merge that
            # should not have happened — cheaper than the reverse, and only in an image built
            # without fugashi.
            return stripped.endswith(tuple(_JA_CONTINUATION))
        morphemes = [t for t in tokenize_spans(stripped, "ja") if not t.punct]
        if not morphemes:
            return False
        last = morphemes[-1]
        return last.pos1 == "助詞" and last.text in _JA_CONTINUATION
    key = _last_token_key(stripped, lang)
    return key in (_EN_CONTINUATION if lang == "en" else _VI_CONTINUATION)


def is_finished(text: str, language: str) -> bool:
    """Whether `text` reads as a complete sentence that nothing is going to continue."""
    stripped = text.strip().rstrip(" 　\"'”’」』)")
    if not stripped:
        return False
    if stripped[-1] not in TERMINALS:
        return False
    return not ends_with_continuation(stripped, language)


def is_backchannel(text: str, language: str, flags: frozenset[str] = frozenset()) -> bool:
    """Whether this segment is an acknowledgement rather than a turn of its own.

    Filler-only (per the prepass) counts, and so does a short line made entirely of
    backchannel words. Length is capped because "yes, and that is why we should wait" is a
    turn, not a backchannel, and starts with the same word.
    """
    if FLAG_FILLER_ONLY in flags:
        return True
    lang = _lang(language, text)
    tokens = [t for t in tokenize_spans(text, lang) if not t.punct]
    if not tokens or len(tokens) > _BACKCHANNEL_MAX_TOKENS:
        return False
    vocabulary = _BACKCHANNELS.get(lang, frozenset())
    return all(token.key in vocabulary for token in tokens)


def _strip_soft_terminal(text: str, language: str) -> str:
    stripped = text.rstrip()
    while stripped.endswith(_SOFT_TERMINALS):
        stripped = stripped[:-1].rstrip()
    if language == "ja":
        stripped = stripped.rstrip("、，,")
    return stripped


def _decapitalize(text: str, language: str) -> str:
    """Lower a sentence-initial capital that a merge has moved into mid-sentence."""
    lowercase = _JOIN_LOWERCASE.get(language)
    if not lowercase or not text[:1].isupper():
        return text
    head = text.split(" ", 1)[0].strip(",.?!;:")
    if head.isupper():  # an acronym, not a sentence start
        return text
    if normalize_key(head, language) not in lowercase:
        return text
    return text[:1].lower() + text[1:]


def join_texts(pieces: list[str], language: str) -> str:
    """Join the pieces of one sentence back together, punctuation-only.

    Adds and removes PUNCTUATION and case, never a word, so the result stays a deletion of the
    raw text under invariant I1. Two things happen here: a full stop the recogniser put at a
    place the speaker did not stop is dropped, and Japanese gets a "、" where the first piece
    ended on a conjunctive particle — "来週の会議ですが" + "火曜日に..." reads as one sentence
    only with the comma, and reads as two with a 。.
    """
    lang = resolve_language(language, " ".join(pieces)) or language
    out = ""
    for piece in pieces:
        text = piece.strip()
        if not text:
            continue
        if not out:
            out = text
            continue
        head = _strip_soft_terminal(out, lang)
        if not head:
            out = text
            continue
        if lang == "ja":
            separator = "、" if ends_with_continuation(head, lang) else ""
        else:
            separator = " "
        # Only a piece that really was mid-sentence gets its capital lowered. A "?" or "!"
        # survives the join (see `_SOFT_TERMINALS`), and what follows one is a new sentence
        # whose capital is correct where it stands.
        joined = text if head[-1:] in TERMINALS else _decapitalize(text, lang)
        out = head + separator + joined
    return out


# --- the segmenter ----------------------------------------------------------------------------


@dataclass
class _Buffer:
    speaker_id: str
    segments: list[CleanSegment] = field(default_factory=list)

    @property
    def language(self) -> str:
        return self.segments[0].language if self.segments else ""

    @property
    def last_arrival_ms(self) -> int:
        return self.segments[-1].arrived_at_ms if self.segments else 0

    @property
    def duration_ms(self) -> int:
        if not self.segments:
            return 0
        return max(0, self.segments[-1].end_ms - self.segments[0].start_ms)


class SentenceSegmenter:
    """Sentence assembly for ONE meeting. Feed it segments, take sentences out.

    Every method returns the sentences the call produced, in order, and the caller publishes
    them; nothing is buffered on the way out. There is one open buffer at a time — the speaker
    who currently holds the floor — because a genuine turn change closes the previous speaker's
    line anyway and a backchannel deliberately does not open one.
    """

    def __init__(
        self,
        *,
        merge_gap_ms: int = 1200,
        max_sentence_ms: int = 30000,
        idle_flush_ms: int = 4000,
    ) -> None:
        self.merge_gap_ms = merge_gap_ms
        self.max_sentence_ms = max_sentence_ms
        self.idle_flush_ms = idle_flush_ms
        self._buffer: _Buffer | None = None

    # -- input ---------------------------------------------------------------------------

    def add(self, segment: CleanSegment) -> list[CleanSentence]:
        """Take one final STT segment; return whatever sentences that completed."""
        out: list[CleanSentence] = []
        buffer = self._buffer

        if buffer is not None and buffer.speaker_id != segment.speaker_id:
            if is_backchannel(segment.raw_text, segment.language, segment.flags):
                # Somebody said "ừ" while another person is mid-sentence. Their line is left
                # open; the acknowledgement becomes a line of its own unless it was pure filler.
                if not segment.is_filler_only:
                    out.append(self._make([segment], reason="backchannel"))
                return out
            out.extend(self._flush(reason="turn_change"))
            buffer = None

        if buffer is not None and not self._continues(buffer, segment):
            out.extend(self._flush(reason="finished"))
            buffer = None

        if buffer is None:
            buffer = _Buffer(speaker_id=segment.speaker_id)
            self._buffer = buffer
        buffer.segments.append(segment)

        if buffer.duration_ms >= self.max_sentence_ms:
            out.extend(self._flush(reason="max_length"))
        return out

    def flush_idle(self, now_ms: int) -> list[CleanSentence]:
        """Close a line whose speaker has stopped talking. `now_ms` is the caller's clock."""
        buffer = self._buffer
        if buffer is None or not buffer.segments:
            return []
        if now_ms - buffer.last_arrival_ms < self.idle_flush_ms:
            return []
        return self._flush(reason="idle")

    def flush(self, reason: str = "meeting_end") -> list[CleanSentence]:
        """Close whatever is open — the meeting ended, or the room went terminal."""
        return self._flush(reason=reason)

    @property
    def is_empty(self) -> bool:
        return self._buffer is None or not self._buffer.segments

    # -- internals -----------------------------------------------------------------------

    def _continues(self, buffer: _Buffer, segment: CleanSegment) -> bool:
        gap_ms = segment.start_ms - buffer.segments[-1].end_ms
        if gap_ms >= self.merge_gap_ms:
            return False
        # Measured on the RAW text: the prepass has already normalised terminal punctuation onto
        # its own output, so its version of a mid-sentence fragment ends with a full stop the
        # speaker did not say. The recogniser's punctuation is the only evidence of where the
        # speaker actually stopped.
        combined = join_texts([s.raw_text for s in buffer.segments], buffer.language)
        return not is_finished(combined, buffer.language)

    def _flush(self, *, reason: str) -> list[CleanSentence]:
        buffer = self._buffer
        self._buffer = None
        if buffer is None or not buffer.segments:
            return []
        return [
            self._make(group, reason=reason)
            for group in _split_at_finished_boundaries(buffer.segments, buffer.language)
        ]

    def _make(self, segments: list[CleanSegment], *, reason: str) -> CleanSentence:
        language = segments[0].language
        flags: set[str] = set()
        for segment in segments:
            flags |= set(segment.flags)
        return CleanSentence(
            speaker_id=segments[0].speaker_id,
            language=language,
            segments=tuple(segments),
            raw_text=join_texts([s.raw_text for s in segments], language),
            prepass_text=join_texts([s.clean_text for s in segments], language),
            reason=reason,
            flags=frozenset(flags),
        )


def _split_at_finished_boundaries(
    segments: list[CleanSegment], language: str
) -> list[list[CleanSegment]]:
    """Cut a buffer wherever a SEGMENT BOUNDARY falls after a completed sentence.

    Normally there is nothing to cut: a buffer only ever grew because it was unfinished. The
    split exists for the paths that append without asking — the max-length and meeting-end
    flushes — so a line that did contain "…. And then" is not published as one run-on. Boundaries
    inside a segment are never considered; see the module docstring.
    """
    groups: list[list[CleanSegment]] = []
    current: list[CleanSegment] = []
    for segment in segments:
        current.append(segment)
        combined = join_texts([s.raw_text for s in current], language)
        if is_finished(combined, language) and segment is not segments[-1]:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups
