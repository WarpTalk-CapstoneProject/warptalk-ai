"""The LLM tier of clean transcript: a model that may only DELETE (WT-716).

THE PROTOCOL, AND WHY IT IS INDICES AND NOT TEXT
    Asked to "clean up" a line, a model returns a nicer line — one where a mishearing has been
    repaired into something plausible, a half-finished thought has been finished, and a word
    nobody said has appeared. `translation_worker/transcript_guardian.py` argues at length why
    that is worse than a visibly messy transcript, and it checks the answer afterwards. This
    stage removes the opportunity instead: the model is shown NUMBERED TOKENS and answers with
    the indices to delete, so the only thing it can express is a deletion.

    The text is then rebuilt here, from the raw string, by removing those token spans. Every
    word in the output came out of the input by construction; `check_invariants` still runs
    afterwards, because the model can still choose a bad deletion (a "not", a number, the
    question particle) and that is a different failure from an invented word.

WHAT IT IS FOR THAT THE PREPASS CANNOT DO
    The deterministic prepass deletes what a lexicon can recognise without understanding the
    sentence. It deliberately stops at everything else and says so with `escalate`:

        "Monday, I mean Tuesday"        — which half survives is meaning, not formatting
        "họp thứ hai, à không, thứ ba"  — same, in Vietnamese
        "赤じゃなくて青がいい"            — looks identical, and is NOT a repair: it is a contrast

    Only a model can tell those apart, and each of them is a sentence where the reader ends up
    with the wrong fact if we guess.

WHEN THE ANSWER IS DISCARDED
    Invariant violation, a deletion ratio over the cap, an index that is not a token, an empty
    result, a timeout, malformed JSON — and, for a claimed self-repair, a negation deleted
    outside the repair marker or a number deleted with nothing of its kind put back. In every
    case revision 0 — the prepass line, already published — stands. The clean tier is a polish
    on a line the reader already has, so the right failure mode is "no second revision", never
    "no line".
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from shared.disfluency import (
    check_invariants,
    lexicon_en,
    lexicon_ja,
    lexicon_vi,
    normalize_key,
    normalize_terminal_punctuation,
    protected_indices,
)
from shared.disfluency.invariants import I2_NEGATION_COUNT, I2_NUMBER_COUNT
from shared.disfluency.normalize import resolve_language
from shared.disfluency.tokenize import COMMAS, Token, tokenize_spans
from shared.logger import get_logger
from shared.openai_options import completion_options
from shared.provider_calls import observed_openai_http_client
from transcript_clean_worker.config import TranscriptCleanSettings

logger = get_logger(__name__)

# Rejection reasons, so a log line (and a future metric) names the same thing every time.
REJECT_BAD_JSON = "bad_json"
REJECT_BAD_INDEX = "bad_index"
REJECT_EMPTY = "empty_result"
REJECT_RATIO = "delete_ratio"
REJECT_TIMEOUT = "timeout"
REJECT_CALL_FAILED = "call_failed"
REJECT_NO_CHANGE = "no_change"
REJECT_NEGATION_OUTSIDE_MARKER = "negation_outside_marker"
REJECT_NUMBER_WITHOUT_REPLACEMENT = "number_without_replacement"
REJECT_PROTECTED_TOKEN = "protected_token_deleted"
REJECT_REPARANDUM_LONGER_THAN_REPAIR = "reparandum_longer_than_repair"

# A VERIFIED self-repair is allowed to delete more than an ordinary clean-up, because that is
# the shape of the thing: "họp thứ hai, à không, thứ ba" is four of seven words, and "We ship on
# Monday, I mean Tuesday" is three of seven. The ordinary cap exists to catch a model that is
# summarising; a repair whose marker ("I mean", "à không", "じゃなくて") is inside one contiguous
# deleted span is not summarising. Without this, the cap would reject exactly the case this tier
# was added for. What still holds for a repair: I1 (nothing invented) and the question marker.
#
# 0.7 and not the 0.8 this started at. The number is a guess either way, so it is set at the
# smallest value that still covers the repairs we have actually seen — a whole SENTENCE that is
# four fifths reparandum is more likely a model summarising than a speaker correcting themselves,
# and when the two readings are that close the ruling on this ticket says to keep the messier,
# truer line. `TranscriptCleanSettings.self_repair_max_delete_ratio` makes it an env var, so this
# can be tightened further in production without shipping code.
_SELF_REPAIR_MAX_DELETE_RATIO = 0.7

_OPENING = frozenset('([{“‘「『"')
_CLOSING = frozenset(",.?!;:)]}…”’、。？！」』%")

# The markers that make a deletion a SELF-REPAIR rather than an ordinary tidy-up. The model is
# asked for the judgement, but its `self_repair` flag is only believed when one of these is
# actually inside what it deleted — a flag is free to set, and this is the one flag that changes
# how a reader understands the line (it says the speaker corrected themselves).
#
# Taken from the prepass lexicons rather than retyped: those are the phrases the rule tier
# ESCALATES on, so the two tiers cannot end up with different ideas of what a repair looks like.
_SELF_REPAIR_MARKERS: dict[str, tuple[str, ...]] = {
    "en": tuple(" ".join(phrase) for phrase in lexicon_en.SELF_REPAIR_AFTER_COMMA) + ("actually",),
    "vi": tuple(" ".join(phrase) for phrase in lexicon_vi.SELF_REPAIR_MARKERS),
    "ja": tuple(lexicon_ja.SELF_REPAIR_MARKERS),
}


def _marker_phrases(lang: str) -> tuple[tuple[str, ...], ...]:
    """The markers as comparison keys, longest first.

    Longest first because the marker span is what licenses a deleted negation below, and the
    long markers are the ones that CONTAIN the negation: matching "à không" before "à", or
    "không phải" before "à không", is the difference between "the negation is part of the
    marker" and "the model quietly deleted a không".
    """
    phrases = [
        (normalize_key(marker, lang),)
        if lang == "ja"
        else tuple(normalize_key(word, lang) for word in marker.split())
        for marker in _SELF_REPAIR_MARKERS.get(lang, ())
    ]
    phrases.sort(key=lambda phrase: -sum(len(key) for key in phrase))
    return tuple(phrases)


_MARKER_PHRASES: dict[str, tuple[tuple[str, ...], ...]] = {
    lang: _marker_phrases(lang) for lang in ("en", "vi", "ja")
}

# Mirrors `shared.disfluency.invariants._JA_NEGATION_RE`, which is private and owned by another
# task on this ticket — so it is restated rather than imported or edited. The one difference: the
# shared pattern anchors a bare "ず" on the punctuation that follows it in the SENTENCE, and this
# one is matched against a single token key, where end-of-token is the same evidence.
_JA_NEGATION_RE = re.compile(r"ない|ません|なかっ|ず$")

# WHAT COUNTS AS "A NUMBER" FOR THE REPLACEMENT RULE, AND WHY IT IS LOCAL.
#
# `shared.disfluency` counts digits and number words (en NUMBER_WORDS, vi NUMBER_SYLLABLES, ja
# NUMERAL_CHARS) and nothing else — "Monday" and "May" are ordinary words to it. But the repair
# this tier exists for is precisely "Monday, I mean Tuesday": a weekday swapped for a weekday. To
# tell a REPLACEMENT repair from a plain loss of fact, this stage has to know that those two
# words are the same kind of thing, so the calendar vocabulary lives here, beside the rule that
# needs it, instead of in shared/** (owned elsewhere on this ticket).
#
# Vietnamese weekdays and months are built out of number syllables ("thứ hai", "tháng ba") that
# shared already counts, so only "chủ nhật" needs an entry — handled positionally below.
_KIND_NUMBER = "number"
_KIND_WEEKDAY = "weekday"
_KIND_MONTH = "month"

_EN_WEEKDAYS = frozenset(
    {
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "mon",
        "tue",
        "tues",
        "wed",
        "thu",
        "thur",
        "thurs",
        "fri",
        "sat",
        "sun",
    }
)
# "may" and "march" are also an ordinary verb and an ordinary noun. They stay in: mistaking a
# modal for a month can only make this stage REFUSE a deletion, which is the cheap direction.
_EN_MONTHS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
# Ordinals are how a date or a position is said out loud ("the third, sorry, the fourth"), and
# shared's NUMBER_WORDS has only the cardinals.
_EN_ORDINALS = frozenset(
    {
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "eleventh",
        "twelfth",
    }
)
_JA_WEEKDAYS = frozenset(
    normalize_key(word, "ja")
    for word in (
        "月曜",
        "火曜",
        "水曜",
        "木曜",
        "金曜",
        "土曜",
        "日曜",
        "月曜日",
        "火曜日",
        "水曜日",
        "木曜日",
        "金曜日",
        "土曜日",
        "日曜日",
    )
)
_VI_SUNDAY = (normalize_key("chủ", "vi"), normalize_key("nhật", "vi"))


def _quantity_kind(tokens: list[Token], index: int, lang: str) -> str | None:
    """Which KIND of fact this token carries — a number, a weekday, a month — or None.

    "Kind" is the whole point: a repair replaces a fact with another fact of the same kind, and
    a deletion that removes the only weekday in the line is not a repair, it is a fact going
    missing. The classes are deliberately coarse; a digit and a number word are one kind, because
    "three, I mean 4" is the same repair written two ways.
    """
    key = tokens[index].key
    if lang == "ja":
        if any(char in lexicon_ja.NUMERAL_CHARS for char in key):
            return _KIND_NUMBER
        return _KIND_WEEKDAY if key in _JA_WEEKDAYS else None
    if any(char.isdigit() for char in key):
        return _KIND_NUMBER
    if lang == "vi":
        if key in lexicon_vi.NUMBER_SYLLABLES:
            return _KIND_NUMBER
        # "chủ nhật" is the one Vietnamese weekday with no number syllable in it; "nhật" on its
        # own is a language ("tiếng Nhật"), so it only counts next to its "chủ".
        neighbours = (
            key == _VI_SUNDAY[0]
            and index + 1 < len(tokens)
            and tokens[index + 1].key == _VI_SUNDAY[1]
        ) or (key == _VI_SUNDAY[1] and index > 0 and tokens[index - 1].key == _VI_SUNDAY[0])
        return _KIND_WEEKDAY if neighbours else None
    if key in lexicon_en.NUMBER_WORDS or key in _EN_ORDINALS:
        return _KIND_NUMBER
    if key in _EN_WEEKDAYS:
        return _KIND_WEEKDAY
    return _KIND_MONTH if key in _EN_MONTHS else None


def _is_negation(token: Token, lang: str) -> bool:
    if lang == "en":
        return token.key in lexicon_en.NEGATIONS or token.key.endswith("n't")
    if lang == "vi":
        return token.key in lexicon_vi.NEGATIONS
    return bool(_JA_NEGATION_RE.search(token.key))


_SYSTEM_PROMPT = """You clean one line of a live meeting transcript. You may ONLY DELETE tokens.

You are given the line as numbered tokens. Answer with JSON:
{"delete": [indices to delete], "self_repair": true|false}

DELETE:
- filler sounds and hesitations (um, uh, ờ, ừm, えーと, あのー)
- stutters and repeated words the speaker restarted on
- abandoned false starts
- a self-correction's WRONG half together with its marker: in "Monday, I mean Tuesday" delete
  "Monday", "I", "mean" and keep "Tuesday"; set self_repair to true when you do this

NEVER DELETE:
- negations (not, no, never, không, chưa, ない, ません) or any number, date or name
- question particles and question words (chưa, à, hả, か, っけ)
- politeness and honorifics (ạ, nhé, です, ます)
- discourse words that carry meaning in this sentence ("actually" in "it actually failed")
- a contrast, which is not a self-correction: "赤じゃなくて青がいい" ("not red, blue") keeps
  every token, and so does "not Monday but Tuesday"

WHEN UNSURE, DELETE LESS. This is the rule that outranks every other rule here. A line that
still has a filler in it is untidy; a line that lost a word the speaker said is a record of a
meeting that did not happen, and it will be translated and read out in another language. If you
cannot tell whether a span is an abandoned false start or something the speaker meant, LEAVE IT
IN. If you cannot tell whether two halves are a correction or a contrast, LEAVE THEM BOTH IN.
Never delete a word only because the sentence would read better without it.

If nothing should be deleted, answer {"delete": [], "self_repair": false}. That is a correct
and common answer, not a failure to do the job. Never invent an index."""

# Four, and no more. Measured behaviour on this task: the more deletions a model is shown, the
# more it finds — a longer example list reliably pushed it into removing meaningful words. Two of
# the four delete nothing, on purpose: the empty answer has to look as normal as the other two.
_FEW_SHOT = """Examples (tokens are shown as index:token):
1. 0:um 1:so 2:we 3:we 4:should 5:ship 6:it -> {"delete": [0, 3], "self_repair": false}
2. 0:họp 1:thứ 2:hai 3:à 4:không 5:thứ 6:ba -> {"delete": [1, 2, 3, 4], "self_repair": true}
3. 0:赤 1:じゃなくて 2:青 3:が 4:いい -> {"delete": [], "self_repair": false}
4. 0:not 1:monday 2:but 3:tuesday -> {"delete": [], "self_repair": false}
   (a contrast, not a correction — when it could be either, delete nothing)"""


@dataclass(frozen=True, slots=True)
class CleanedSentence:
    """An accepted LLM answer: the rebuilt line and whether it resolved a self-repair."""

    text: str
    self_repair: bool
    deleted_indices: tuple[int, ...]


def _language_of(raw: str, language: str) -> str:
    return resolve_language(language, raw) or "en"


def lexical_tokens(raw: str, language: str) -> list[Token]:
    """The tokens the indices in the protocol refer to: words, no punctuation."""
    return [t for t in tokenize_spans(raw, _language_of(raw, language)) if not t.punct]


def prepass_deletion_indices(raw: str, prepass_text: str, language: str) -> list[int]:
    """Which token indices the deterministic prepass removed, recovered by alignment.

    Sent to the model as a SUGGESTION, not as a decision: it is the cheap tier's answer and
    having it in the prompt stops the model relitigating obvious fillers, which is where its
    attention is worth spending.
    """
    lang = _language_of(raw, language)
    raw_tokens = lexical_tokens(raw, lang)
    clean_keys = [t.key for t in lexical_tokens(prepass_text, lang)]
    removed: list[int] = []
    cursor = 0
    for index, token in enumerate(raw_tokens):
        if cursor < len(clean_keys) and clean_keys[cursor] == token.key:
            cursor += 1
        else:
            removed.append(index)
    # The alignment failed (the prepass text is not a subsequence of the raw one, which the
    # prepass's own invariants should have prevented). Suggest nothing rather than nonsense.
    return removed if cursor == len(clean_keys) else []


def apply_deletions(raw: str, language: str, indices: list[int]) -> str:
    """Rebuild `raw` without the tokens at `indices`. Punctuation and case only, otherwise.

    Deleting a word orphans the punctuation that belonged to it, so a comma whose left-hand
    neighbour was deleted goes with it ("Monday, I mean Tuesday" must not leave a leading
    comma), as does a comma with nothing kept before it. Nothing else is touched: the commas
    the recogniser put between words that both survive are the speaker's.
    """
    lang = _language_of(raw, language)
    tokens = tokenize_spans(raw, lang)
    lexical_positions = [i for i, t in enumerate(tokens) if not t.punct]
    deleted: set[int] = set()
    for index in indices:
        if 0 <= index < len(lexical_positions):
            deleted.add(lexical_positions[index])

    for position, token in enumerate(tokens):
        if position in deleted or not token.punct or token.text not in COMMAS:
            continue
        left = position - 1
        kept_before = any(i not in deleted and not tokens[i].punct for i in range(position))
        if not kept_before or (left >= 0 and left in deleted):
            deleted.add(position)

    parts: list[str] = []
    previous: int | None = None
    for position, token in enumerate(tokens):
        if position in deleted:
            continue
        if previous is None:
            parts.append(token.text)
            previous = position
            continue
        gap_deleted = any(i in deleted for i in range(previous + 1, position))
        if not gap_deleted:
            separator = " " if token.start > tokens[previous].end else ""
        elif lang == "ja":
            separator = ""
        elif token.punct and token.text[:1] in _CLOSING:
            separator = ""
        elif tokens[previous].punct and tokens[previous].text[-1:] in _OPENING:
            separator = ""
        else:
            separator = " "
        parts.append(separator + token.text)
        previous = position

    text = "".join(parts).strip().lstrip("".join(COMMAS)).strip()
    if lang in ("en", "vi") and text[:1].islower():
        head = text.split(" ", 1)[0]
        if not any(c.isupper() for c in head):
            text = text[:1].upper() + text[1:]
    return normalize_terminal_punctuation(text, lang)


def self_repair_marker_span(raw: str, language: str, indices: list[int]) -> tuple[int, ...]:
    """The token indices the correction marker itself occupies, or () when none was deleted.

    Which tokens, not just whether — because the marker span is what licenses the narrow
    exceptions below. "à không" and "no wait" ARE partly a negation, so a repair cannot be
    verified without allowing that one negation to disappear; every other negation in the
    deleted span is a word the speaker said and is not covered by anything.

    The contiguity of the deletion is checked by the caller: a marker inside the deleted span IS
    the evidence that the span was a reparandum, and the lexicons here are the same ones the
    prepass escalates on, so the two tiers cannot disagree about what a repair looks like.
    """
    lang = _language_of(raw, language)
    tokens = lexical_tokens(raw, lang)
    ordered = [index for index in sorted(set(indices)) if 0 <= index < len(tokens)]
    keys = [tokens[index].key for index in ordered]

    if lang == "ja":
        # No spaces to align on, so the match is over the concatenated keys and the span is every
        # token the matched character range touches.
        bounds: list[tuple[int, int]] = []
        cursor = 0
        for key in keys:
            bounds.append((cursor, cursor + len(key)))
            cursor += len(key)
        surface = "".join(keys)
        for phrase in _MARKER_PHRASES["ja"]:
            start = surface.find(phrase[0])
            if start < 0:
                continue
            end = start + len(phrase[0])
            return tuple(
                ordered[position]
                for position, (low, high) in enumerate(bounds)
                if low < end and high > start
            )
        return ()

    for phrase in _MARKER_PHRASES.get(lang, ()):
        width = len(phrase)
        for start in range(len(keys) - width + 1):
            if tuple(keys[start : start + width]) == phrase:
                return tuple(ordered[start : start + width])
    return ()


def has_self_repair_marker(raw: str, language: str, indices: list[int]) -> bool:
    """Whether what was deleted contains a correction marker ("I mean", "à không", "じゃなくて")."""
    return bool(self_repair_marker_span(raw, language, indices))


def negations_deleted_outside_marker(
    tokens: list[Token], lang: str, indices: list[int], marker_span: tuple[int, ...]
) -> list[str]:
    """The negations this deletion removes that are NOT part of the correction marker.

    Empty is the only acceptable answer. A negation inside "à không" / "không phải" / "no wait"
    / "じゃなくて" goes because the marker goes, and that is a word the speaker used to CANCEL a
    statement, not to make one. A negation anywhere else in the span is the statement itself.
    """
    inside = set(marker_span)
    return [
        tokens[index].text
        for index in indices
        if index not in inside and _is_negation(tokens[index], lang)
    ]


def quantity_without_replacement(tokens: list[Token], lang: str, indices: list[int]) -> str | None:
    """The kind of fact this deletion removes without putting another of that kind back.

    A repair REPLACES: "thứ hai, à không, thứ ba" still has a number in it afterwards, and
    "Monday, I mean Tuesday" still has a weekday. That surviving token is the evidence that the
    speaker was correcting a fact rather than dropping one, so it is required — and required
    AFTER the deleted span, because that is where the repairing half of a repair lives. A
    deletion that leaves no number where there was one is a number going missing, whatever the
    model called it.
    """
    last = indices[-1]
    deleted_kinds: set[str] = set()
    for index in indices:
        kind = _quantity_kind(tokens, index, lang)
        if kind is not None:
            deleted_kinds.add(kind)
    if not deleted_kinds:
        return None
    for index in range(last + 1, len(tokens)):
        deleted_kinds.discard(_quantity_kind(tokens, index, lang) or "")
    return sorted(deleted_kinds)[0] if deleted_kinds else None


class LLMCleaner:
    """Runs the deletion-index protocol against OpenAI, and refuses anything it cannot verify."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 8.0,
        max_delete_ratio: float = 0.4,
        self_repair_max_delete_ratio: float | None = None,
        concurrency: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_delete_ratio = max_delete_ratio
        # Resolved from the settings object when the caller does not pass one, so
        # TRANSCRIPT_CLEAN_SELF_REPAIR_MAX_DELETE_RATIO is live in production today: the worker's
        # construction site (worker.load_model) belongs to another task on this ticket and is not
        # touched here. An explicit argument always wins, which is how the tests pin it.
        self.self_repair_max_delete_ratio = (
            TranscriptCleanSettings().self_repair_max_delete_ratio
            if self_repair_max_delete_ratio is None
            else self_repair_max_delete_ratio
        )
        self.temperature = temperature
        self._client: AsyncOpenAI | None = None
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        # Counted rather than published: this repo's workers have no HTTP surface, and the
        # numbers that matter here (how often the model is refused, and why) belong beside the
        # log line that already says it. Read by the worker's shutdown log.
        self.rejections: dict[str, int] = {}

    @property
    def is_available(self) -> bool:
        return self._client is not None

    async def load(self) -> None:
        if not self.api_key:
            # Not fatal: the prepass tier still cleans every line. See config.api_key.
            logger.warning("transcript_clean_llm_disabled", reason="no_api_key")
            return
        self._client = AsyncOpenAI(
            api_key=self.api_key, http_client=observed_openai_http_client("transcript-clean")
        )
        logger.info("transcript_clean_llm_loaded", model=self.model)

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.close()

    def _reject(self, reason: str, **fields: Any) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        logger.info("transcript_clean_llm_rejected", reason=reason, **fields)

    async def clean(
        self,
        raw: str,
        language: str,
        *,
        prepass_text: str | None = None,
        previous_line: str = "",
    ) -> CleanedSentence | None:
        """The model's cleaned version of `raw`, or None when there is nothing safe to publish."""
        client = self._client
        if client is None or not raw.strip():
            return None

        lang = _language_of(raw, language)
        tokens = lexical_tokens(raw, lang)
        if not tokens:
            return None

        suggestion = (
            prepass_deletion_indices(raw, prepass_text, lang) if prepass_text is not None else []
        )
        user_message = _render_request(raw, lang, tokens, suggestion, previous_line)

        try:
            async with self._semaphore:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + _FEW_SHOT},
                            {"role": "user", "content": user_message},
                        ],
                        response_format={"type": "json_object"},
                        **completion_options(self.model, 256, self.temperature),
                    ),
                    timeout=self.timeout_s,
                )
        except TimeoutError:
            self._reject(REJECT_TIMEOUT, model=self.model)
            return None
        except Exception as exc:
            self._reject(REJECT_CALL_FAILED, error=repr(exc))
            return None

        return self._verify(raw, lang, tokens, response, suggestion)

    def _verify(
        self,
        raw: str,
        lang: str,
        tokens: list[Token],
        response: Any,
        prepass_deleted: list[int],
    ) -> CleanedSentence | None:
        """The model's answer, or None — and None is the cheap answer, by design.

        THE RULING THIS FUNCTION IMPLEMENTS (WT-716)
            Faithfulness to what the speaker actually said outranks tidiness. Deleting too little
            is better than deleting too much, and deleting too little is also better than
            refusing to publish: a rejection here is not a missing line, it is the tier-1 wording
            standing as revision 0, which the reader already has. So every doubt in this function
            resolves the same way — refuse the polish, keep the line.

        WHY THAT MAKES THE SELF-REPAIR EXCEPTION THE DANGEROUS PART
            A verified self-repair is the one path that RELAXES a check, and the checks it used
            to relax wholesale were exactly the two that stop the transcript saying something the
            speaker did not: the negation count and the number count. A sentence wrongly
            classified as a repair could therefore lose a "không"/"not"/"ない" or a figure and
            come out meaning the opposite, in the transcript, in its translation, and in the
            minutes built on top of it. "Cleaned but wrong" is the one outcome this stage must
            never produce, and it is strictly worse than "not cleaned".

            So the exception is narrowed to the two shapes that make a repair a repair, and
            nothing else is forgiven:

            - a deleted negation is allowed ONLY when it is part of the correction marker itself
              ("à không", "không phải", "no wait", "じゃなくて"). The marker is how the speaker
              cancelled a statement; any other negation in the span is the statement.
            - a deleted number, weekday or month is allowed ONLY when the kept text still has one
              of the same kind after the span — a REPLACEMENT ("thứ hai, à không, thứ ba" keeps
              "thứ ba"). A fact that is removed and not replaced is a fact going missing.

            Anything outside those two shapes is rejected outright rather than downgraded to an
            ordinary clean-up, because a model that claimed a repair here was wrong about the
            sentence, not merely over-eager on one token.

        TWO MORE RULINGS FROM THE SAME "FAITHFULNESS FIRST" PRINCIPLE, MEASURED AGAINST THE REAL
        MODELS (WT-716, ai-t6)
            Running this stage against gpt-4.1-mini and gpt-4.1 on hard sentences found the same
            failure shape every time: the model deletes what the deterministic tier deliberately
            protected ("very very", "từ từ", "あの資料"), or resolves a self-repair by deleting
            more of the sentence than the repair ever needed to lose. Two checks close those:

            - `protected_indices` names the tokens tier 1 protects ON PRINCIPLE — reduplication,
              畳語, a tier-C look-alike. If the model's answer deletes ANY of them, the whole
              answer is rejected: a model that got a protected token wrong was not reasoning about
              this sentence correctly, so nothing else it said can be trusted either.
            - For a verified repair, the reparandum (what is deleted before the marker) may not
              outnumber the repair (what survives after it). "gửi cho anh Nam, à nhầm, anh Nam
              Anh" deleting "gửi cho anh Nam" (4) to keep "anh Nam Anh" (3) swallows words that
              were never part of the correction; "họp thứ hai, à không, thứ ba" deleting "thứ hai"
              (2) to keep "thứ ba" (2) is the repair working as intended.

            And once an answer clears every check above, the published deletion set is the UNION
            of this model's accepted indices and whatever the deterministic tier already deleted
            in revision 0 (`prepass_deleted`) — never a replacement. Revision 1 can only add to
            revision 0's deletions, so it can never come out MESSIER than the line the reader
            already has ("So we we need to finalize the budget" — the model cleaned the fillers
            but not tier 1's own stutter fix — is impossible once the two are unioned instead of
            one replacing the other). This union is safe without re-running the checks above: tier
            1 never deletes a number, a negation or anything protected, so adding its deletions to
            an already-verified answer cannot newly violate I1 (deletion-only, trivially) or I2
            (tier 1 touches neither the negations nor the numbers I2 counts).
        """
        try:
            payload = json.loads(response.choices[0].message.content or "{}")
            raw_indices = payload["delete"] if isinstance(payload, dict) else None
            if raw_indices is None:
                raw_indices = []
            indices = [int(index) for index in raw_indices]
            claims_repair = bool(payload.get("self_repair")) if isinstance(payload, dict) else False
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as exc:
            self._reject(REJECT_BAD_JSON, error=repr(exc))
            return None

        # A hallucinated index is not a rounding error — it means the model was not answering
        # about this sentence, so nothing it said about the other indices can be trusted either.
        if any(index < 0 or index >= len(tokens) for index in indices):
            self._reject(REJECT_BAD_INDEX, token_count=len(tokens), indices=indices)
            return None
        indices = sorted(set(indices))
        if not indices:
            self._reject(REJECT_NO_CHANGE)
            return None

        # RULE 1 (WT-716, ai-t6): tier 2 may never delete what tier 1 protected on principle.
        # A model that deleted a reduplication or a tier-C look-alike was wrong about the
        # sentence, not merely over-eager on that one token — so the whole answer is refused,
        # the same way a hallucinated index refuses the whole answer above.
        touched = protected_indices(raw, lang).intersection(indices)
        if touched:
            self._reject(REJECT_PROTECTED_TOKEN, indices=sorted(touched))
            return None

        # Contiguity is part of what makes a claimed repair believable: a reparandum plus its
        # marker is one run of tokens. It is required here because the flag also unlocks the
        # count-invariant exceptions below.
        contiguous = indices[-1] - indices[0] + 1 == len(indices)
        marker_span = (
            self_repair_marker_span(raw, lang, indices) if claims_repair and contiguous else ()
        )
        self_repair = bool(marker_span)
        cap = self.self_repair_max_delete_ratio if self_repair else self.max_delete_ratio
        if len(indices) > len(tokens) * cap:
            self._reject(
                REJECT_RATIO,
                deleted=len(indices),
                token_count=len(tokens),
                cap=cap,
                self_repair=self_repair,
            )
            return None

        text = apply_deletions(raw, lang, indices)
        if not text.strip():
            self._reject(REJECT_EMPTY)
            return None

        violations = check_invariants(raw, text, lang)
        if self_repair:
            # THE ONE PLACE THE COUNT INVARIANTS ARE RELAXED, and it has to be here.
            #
            # A self-repair is the speaker WITHDRAWING what they just said, and what they
            # withdraw is almost always a number or a negation: "họp thứ hai, à không, thứ ba"
            # loses the number "hai" and — because the Vietnamese repair marker is literally
            # "à không" — a negation too. "Monday, I mean Tuesday" loses a date. So I2's count
            # rules, which exist to catch a model quietly dropping a "not", call every correctly
            # resolved repair a violation, and the ticket's own example would be rejected.
            #
            # The exception is bought with the conditions checked above — the model said this was
            # a repair, a repair MARKER from the prepass lexicons is inside what it deleted, and
            # the deletion is one contiguous span — AND with the two narrow tests below, which
            # are what keep a misclassified sentence from flipping meaning. I1 still holds
            # absolutely, and a question marker still may not be lost.
            stray = negations_deleted_outside_marker(tokens, lang, indices, marker_span)
            if stray:
                self._reject(REJECT_NEGATION_OUTSIDE_MARKER, raw=raw, clean=text, negations=stray)
                return None
            missing_kind = quantity_without_replacement(tokens, lang, indices)
            if missing_kind is not None:
                self._reject(
                    REJECT_NUMBER_WITHOUT_REPLACEMENT, raw=raw, clean=text, kind=missing_kind
                )
                return None
            # RULE 3 (WT-716, ai-t6): the reparandum may not outnumber the repair. Counted in
            # tokens, excluding the marker itself from both sides — the reparandum is what is
            # deleted BEFORE the marker, the repair is what SURVIVES after it. A repair that
            # replaces "thứ hai" with "thứ ba" is one-for-one; "gửi cho anh Nam, à nhầm, anh Nam
            # Anh" deleting the whole "gửi cho anh Nam" to keep "anh Nam Anh" deletes a clause
            # that was never part of the correction, and the length alone gives it away.
            reparandum = sum(1 for index in indices if index < marker_span[0])
            repair = sum(
                1 for index in range(marker_span[-1] + 1, len(tokens)) if index not in indices
            )
            if reparandum > repair:
                self._reject(
                    REJECT_REPARANDUM_LONGER_THAN_REPAIR,
                    raw=raw,
                    clean=text,
                    reparandum=reparandum,
                    repair=repair,
                )
                return None
            violations = [
                violation
                for violation in violations
                if violation not in (I2_NEGATION_COUNT, I2_NUMBER_COUNT)
            ]
        if violations:
            self._reject(",".join(violations), raw=raw, clean=text)
            return None

        # RULE 2 (WT-716, ai-t6): tier 2 may only ADD to tier 1's deletions, never undo them.
        # See the docstring above for why this union needs no re-verification: tier 1 never
        # deletes a number, a negation, or anything `protected_indices` names, so it cannot
        # newly trip I1 or I2 on top of an already-accepted answer.
        final_indices = sorted(set(indices) | {i for i in prepass_deleted if 0 <= i < len(tokens)})
        if final_indices != indices:
            text = apply_deletions(raw, lang, final_indices)

        return CleanedSentence(
            text=text,
            self_repair=self_repair,
            deleted_indices=tuple(final_indices),
        )


def _render_request(
    raw: str,
    language: str,
    tokens: list[Token],
    suggestion: list[int],
    previous_line: str,
) -> str:
    numbered = " ".join(f"{index}:{token.text}" for index, token in enumerate(tokens))
    lines = [f"language: {language}"]
    if previous_line.strip():
        # One line, because the only thing context has to answer here is whether a turn-initial
        # "hmm"/"ừ"/"うん" is an answer (keep it) or a hesitation (delete it).
        lines.append(f"previous speaker said: {previous_line.strip()}")
    lines.append(f"line: {raw}")
    lines.append(f"tokens: {numbered}")
    lines.append(f"already removed by the rule tier (a suggestion, not a decision): {suggestion}")
    return "\n".join(lines)


__all__ = [
    "REJECT_BAD_INDEX",
    "REJECT_BAD_JSON",
    "REJECT_CALL_FAILED",
    "REJECT_EMPTY",
    "REJECT_NEGATION_OUTSIDE_MARKER",
    "REJECT_NO_CHANGE",
    "REJECT_NUMBER_WITHOUT_REPLACEMENT",
    "REJECT_PROTECTED_TOKEN",
    "REJECT_RATIO",
    "REJECT_REPARANDUM_LONGER_THAN_REPAIR",
    "REJECT_TIMEOUT",
    "CleanedSentence",
    "LLMCleaner",
    "apply_deletions",
    "has_self_repair_marker",
    "lexical_tokens",
    "negations_deleted_outside_marker",
    "prepass_deletion_indices",
    "quantity_without_replacement",
    "self_repair_marker_span",
]
