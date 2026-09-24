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
    result, a timeout, malformed JSON. In every case revision 0 — the prepass line, already
    published — stands. The clean tier is a polish on a line the reader already has, so the
    right failure mode is "no second revision", never "no line".
"""

from __future__ import annotations

import asyncio
import json
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
)
from shared.disfluency.invariants import I2_NEGATION_COUNT, I2_NUMBER_COUNT
from shared.disfluency.normalize import resolve_language
from shared.disfluency.tokenize import COMMAS, Token, tokenize_spans
from shared.logger import get_logger
from shared.openai_options import completion_options

logger = get_logger(__name__)

# Rejection reasons, so a log line (and a future metric) names the same thing every time.
REJECT_BAD_JSON = "bad_json"
REJECT_BAD_INDEX = "bad_index"
REJECT_EMPTY = "empty_result"
REJECT_RATIO = "delete_ratio"
REJECT_TIMEOUT = "timeout"
REJECT_CALL_FAILED = "call_failed"
REJECT_NO_CHANGE = "no_change"

# A VERIFIED self-repair is allowed to delete more than an ordinary clean-up, because that is
# the shape of the thing: "Monday, I mean Tuesday" is three of its four words, and "họp thứ hai,
# à không, thứ ba" is four of seven. The ordinary cap exists to catch a model that is
# summarising; a repair whose marker ("I mean", "à không", "じゃなくて") is inside one contiguous
# deleted span is not summarising. Without this, the cap would reject exactly the case this tier
# was added for. What still holds for a repair: I1 (nothing invented) and the question marker.
_SELF_REPAIR_MAX_DELETE_RATIO = 0.8

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

If nothing should be deleted, answer {"delete": [], "self_repair": false}. Never invent an
index. Deleting too little is always better than deleting too much."""

# Three, and no more. Measured behaviour on this task: the more deletions a model is shown, the
# more it finds — a longer example list reliably pushed it into removing meaningful words.
_FEW_SHOT = """Examples (tokens are shown as index:token):
1. 0:um 1:so 2:we 3:we 4:should 5:ship 6:it -> {"delete": [0, 3], "self_repair": false}
2. 0:họp 1:thứ 2:hai 3:à 4:không 5:thứ 6:ba -> {"delete": [1, 2, 3, 4], "self_repair": true}
3. 0:赤 1:じゃなくて 2:青 3:が 4:いい -> {"delete": [], "self_repair": false}"""


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


def deleted_surface(raw: str, language: str, indices: list[int]) -> str:
    """The normalised keys of the deleted tokens, as one string, for marker matching."""
    lang = _language_of(raw, language)
    tokens = lexical_tokens(raw, lang)
    keys = [tokens[i].key for i in sorted(indices) if 0 <= i < len(tokens)]
    return "".join(keys) if lang == "ja" else " " + " ".join(keys) + " "


def has_self_repair_marker(raw: str, language: str, indices: list[int]) -> bool:
    """Whether what was deleted contains a correction marker ("I mean", "à không", "じゃなくて").

    The contiguity of the deletion is not checked separately: a marker inside the deleted span
    IS the evidence that the span was a reparandum, and the lexicons here are the same ones the
    prepass escalates on, so the two tiers cannot disagree about what a repair looks like.
    """
    lang = _language_of(raw, language)
    surface = deleted_surface(raw, lang, indices)
    for marker in _SELF_REPAIR_MARKERS.get(lang, ()):
        key = normalize_key(marker, lang)
        if lang == "ja":
            if key in surface:
                return True
        else:
            phrase = " ".join(normalize_key(word, lang) for word in marker.split())
            if f" {phrase} " in surface:
                return True
    return False


class LLMCleaner:
    """Runs the deletion-index protocol against OpenAI, and refuses anything it cannot verify."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_s: float = 8.0,
        max_delete_ratio: float = 0.4,
        concurrency: int = 4,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_delete_ratio = max_delete_ratio
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
        self._client = AsyncOpenAI(api_key=self.api_key)
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

        return self._verify(raw, lang, tokens, response)

    def _verify(
        self, raw: str, lang: str, tokens: list[Token], response: Any
    ) -> CleanedSentence | None:
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

        # Contiguity is part of what makes a claimed repair believable: a reparandum plus its
        # marker is one run of tokens. It is required here because the flag also unlocks the
        # count-invariant exception below.
        self_repair = (
            claims_repair
            and indices[-1] - indices[0] + 1 == len(indices)
            and has_self_repair_marker(raw, lang, indices)
        )
        cap = _SELF_REPAIR_MAX_DELETE_RATIO if self_repair else self.max_delete_ratio
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
            # The exception is bought with three conditions, all checked above: the model said
            # this was a repair, a repair MARKER from the prepass lexicons is inside what it
            # deleted, and the deletion is one contiguous span (a reparandum is; a model
            # harvesting numbers from around the sentence is not). I1 still holds absolutely —
            # nothing may be invented — and a question marker still may not be lost.
            violations = [
                violation
                for violation in violations
                if violation not in (I2_NEGATION_COUNT, I2_NUMBER_COUNT)
            ]
        if violations:
            self._reject(",".join(violations), raw=raw, clean=text)
            return None

        return CleanedSentence(
            text=text,
            self_repair=self_repair,
            deleted_indices=tuple(indices),
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
    "REJECT_NO_CHANGE",
    "REJECT_RATIO",
    "REJECT_TIMEOUT",
    "CleanedSentence",
    "LLMCleaner",
    "apply_deletions",
    "has_self_repair_marker",
    "lexical_tokens",
    "prepass_deletion_indices",
]
