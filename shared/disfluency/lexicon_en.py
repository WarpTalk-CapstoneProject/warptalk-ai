"""English disfluency lexicon (WT-716). Data only — the rules that use it live in prepass.py.

Tiers:
- A1: always deleted when a standalone token. Hyphenated words are single tokens, so the "uh"
  inside "uh-oh" is never seen as a filler.
- A2: deleted unless the turn is nothing but fillers/backchannels (a standalone "Hmm." answers
  something), and kept turn-initially when the previous turn was a question.
- B: discourse markers. NEVER deleted — "I like it" and "like, three people" differ only in
  context a lexicon cannot see — but they raise `escalate` so the LLM tier can look.
- C: backchannels / answers. Kept always, and never collapsed as repeats.

Filler tiers are stored squeezed (normalize.squeeze_key: "hmm" → "hm") because that is how
tokens are looked up against them.
"""

from __future__ import annotations

from shared.disfluency.normalize import squeeze_key

A1_FILLERS = frozenset(squeeze_key(w) for w in ("um", "uh", "erm", "er", "uhm"))
A2_FILLERS = frozenset(squeeze_key(w) for w in ("hmm", "hm", "mm", "ah"))

B_MARKERS: tuple[tuple[str, ...], ...] = (
    ("you", "know"),
    ("i", "mean"),
    ("kind", "of"),
    ("sort", "of"),
    ("like",),
    ("so",),
    ("well",),
    ("actually",),
    ("basically",),
)

C_KEEP = frozenset(
    {"uh-huh", "mm-hmm", "mhm", "uh-uh", "uh-oh", "yeah", "yep", "okay", "ok", "yes"}
)

# ≥2 consecutive copies collapse to one: these are the words people restart on.
STUTTER_FUNCTION_WORDS = frozenset(
    {
        "i",
        "we",
        "you",
        "they",
        "it",
        "the",
        "a",
        "an",
        "and",
        "to",
        "of",
        "in",
        "my",
        "this",
        "we're",
        "it's",
    }
)
# Any other word collapses only at this many copies or more.
CONTENT_REPEAT_MIN = 3

# Repeats that are grammar or idiom, not stutter. A run of any of these is protected whole
# ("very very very" included). "had had", "that that", "is is" are grammatical English.
PROTECTED_REPEAT_WORDS = frozenset(
    {
        "very",
        "no",
        "yeah",
        "bye",
        "so",
        "well",
        "had",
        "that",
        "is",
        "do",
        "walla",
        "really",
        "yes",
        "ha",
        "blah",
        "knock",
        "tick",
        "tock",
        "bang",
        "night",
    }
)

# A dangling "x-" is a restart only when "x" begins the next word; before these it is a
# suspended hyphen ("pre- and post-launch") and must stay.
SUSPENDED_HYPHEN_NEXT = frozenset({"and", "or", "to", "nor", "&"})

NEGATIONS = frozenset({"not", "no", "never"})

NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
        "thousand",
        "million",
        "billion",
        "percent",
        "double",
        "triple",
    }
)

# Self-repair after a comma ("Monday, I mean Tuesday"): which half survives is meaning, not
# formatting, so these only escalate.
SELF_REPAIR_AFTER_COMMA: tuple[tuple[str, ...], ...] = (
    ("i", "mean"),
    ("i", "meant"),
    ("sorry",),
    ("no", "wait"),
    ("or", "rather"),
)
