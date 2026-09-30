"""Tidying a same-language transcript without letting the model rewrite it.

WHAT THIS IS FOR
    When a listener's language matches the speaker's, the translation worker forwards the STT
    text unchanged (`passthrough`). That text is raw recogniser output: no sentence casing,
    little punctuation, and every filler the speaker actually said. A same-language pass can
    make it read like writing.

WHAT THIS REFUSES TO BE
    A repair tool. If the recogniser mis-heard a word, that word is gone — the audio is not
    available here and no amount of language modelling can recover it. Asked to "clean up" a
    garbled line, an LLM produces fluent, confident, invented text, which then flows into the
    summary, the knowledge index and the meeting record. A transcript that reads badly is a
    transcript people know to doubt; a transcript that reads well and is wrong is not.

HOW THAT IS ENFORCED, WITHOUT A DICTIONARY
    An instruction is not a guarantee, so `is_faithful` checks mechanically. The rule is
    deliberately structural rather than lexical:

        every word the model returns must already be in the original, in the same order.

    Formally: the polished token sequence must be a SUBSEQUENCE of the original's. Deletion is
    allowed and bounded; insertion, substitution and reordering are not possible at all.

    This replaced a per-language filler list, and is better on both counts. It needs no
    knowledge of what a filler is in any language — the model can drop "ờ", "kiểu", "à mà",
    "you know", a stutter, anything — while an inserted or altered word is rejected by
    construction rather than by a similarity score that a good paraphrase might slip past.
    The model is told what it may do; this is what makes it true.

WHY THE CHECK MOVED ONTO shared.disfluency (WT-716)
    The rule was right and its TOKENIZER was wrong for one of the three languages this product
    ships in. Splitting on whitespace makes a Japanese sentence exactly one token, so

        明日えーとリリースします  →  明日リリースします

    was not a subsequence of anything: the whole line changed, by the only measure available,
    and every legitimate Japanese clean-up was refused. Deleting the filler was impossible
    unless the recogniser happened to put spaces around it, which Japanese does not.

    `shared.disfluency.tokenize` segments Japanese into morphemes (fugashi/unidic-lite) and
    keeps the regex path for en/vi, and `check_invariants` states the whole rule — I1 is the
    subsequence rule above, and I2 adds what a subsequence check cannot see: a polish that
    deletes a "not", a number, or the particle that made the line a question is a subsequence
    and is still a different sentence. Those three failures are the ones a reader cannot spot,
    so the guardian now refuses them too. The ≤50% cap is unchanged.
"""

from __future__ import annotations

from shared.disfluency import check_invariants, normalize_key, tokenize
from shared.disfluency.normalize import resolve_language

# How much of an utterance may be dropped as filler. Generous enough for a hesitant sentence
# that is half "ừm à thì", tight enough that a model summarising instead of formatting fails —
# a summary keeps far less than half the words.
_MAX_DELETED_FRACTION = 0.5


def guardian_instruction(language: str) -> str:
    """The system instruction for a same-language tidy-up.

    No filler list. Naming six Vietnamese particles told a model that already knows them far
    more than it needed, and worse, it read as an exhaustive permission — the hesitations
    people actually produce are not a closed set. What matters is the boundary, so the
    instruction states the boundary.
    """
    return (
        "You are formatting a live meeting transcript. The text is already in the language your "
        "output must use — do NOT translate it.\n"
        "You may: fix capitalisation, add or correct punctuation, fix spacing, and delete filler "
        "sounds, stutters and repeated false starts.\n"
        "You may NOT: change any word, add any word, reorder words, complete an unfinished "
        "sentence, or guess what a garbled passage was meant to say. Leave anything you cannot "
        "read exactly as it is — a transcript that looks wrong is more useful than one that "
        "reads well and is wrong.\n"
        "If the text is already well formatted, return it unchanged. Return only the corrected "
        "text, with no commentary and no quotation marks."
    )


def _tokens(text: str, language: str) -> list[str]:
    """The words, stripped of everything the guardian is allowed to change.

    Case, punctuation and spacing go, because those are exactly what it MAY edit. Diacritics
    stay: in Vietnamese they are the difference between words, not decoration, and folding them
    away would let a model quietly strip the tone marks off an entire meeting — `normalize_key`
    keeps them for exactly that reason (it folds elongation and tone-mark PLACEMENT, not tone).
    """
    return [normalize_key(token, language) for token in tokenize(text, language)]


def _language_of(text: str, language: str) -> str:
    """The base language tag to tokenize and check with; script heuristic when unstated.

    `language` used to be accepted and ignored. It is read now because the tokenizer depends on
    it, and it stays optional because `stt_worker.second_pass` calls this without one — an
    unstated language is resolved from the text's script, and falls back to the Latin path.
    """
    return resolve_language(language, text) or "en"


def is_faithful(original: str, polished: str, language: str = "") -> bool:
    """Whether `polished` is `original` with only formatting changed."""
    if not polished.strip():
        return False

    lang = _language_of(original, language)
    original_tokens = _tokens(original, lang)
    if not original_tokens:
        return False

    # Nothing may be added, changed or moved — only dropped (I1) — and what is dropped may not
    # be a negation, a number or the question marker (I2).
    if check_invariants(original, polished, lang):
        return False

    deleted = len(original_tokens) - len(_tokens(polished, lang))
    return deleted <= len(original_tokens) * _MAX_DELETED_FRACTION


def choose_transcript(original: str, polished: str, language: str = "") -> str:
    """The text to publish: the tidy version when it is faithful, otherwise the original.

    The single place the decision is made, so no caller can accidentally publish an unchecked
    model output.
    """
    return polished.strip() if is_faithful(original, polished, language) else original
