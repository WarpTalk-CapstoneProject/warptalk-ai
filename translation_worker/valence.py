"""Whether what was said was positive, negative, or neither — asked of the model that read it.

WHY THIS IS THE ONE PIECE THAT WAS LEFT OUT
    `shared/prosody.py` derives AROUSAL from sound: how activated the speaker was. It cannot
    derive valence, and says so — anger and delight look nearly identical on pitch and energy.
    So `to_generation_config` refuses to emit an emotion label without one, and today it never
    gets one: nothing in the pipeline reads the WORDS for sentiment. Every meeting is therefore
    dubbed with speed and volume but no emotion at all.

    Level 4's emotional half is blocked on exactly this. The plan deferred it because it looked
    like it cost a model round trip on the critical path — but the translation worker has
    already read the sentence, in a call that is already being made. Asking for one extra token
    at the end of that reply costs essentially nothing.

WHY A TRAILING MARKER AND NOT JSON
    The primary translation path is OpenAI Realtime, which returns free text; `response_format`
    and JSON mode are not available on it, and the Chat path is only the fallback. So whatever
    carries the label has to survive as plain text, and this module is the parser for it.

    The codebase already signals out-of-band on this exact return value — see
    `OUT_OF_MEETING_SCOPE` in translator.py — so this follows an idiom rather than inventing one.

THE INVARIANT: VALENCE MUST NEVER DAMAGE THE TRANSLATION
    The translation is what the listener hears; the label is a nicety. So the parse is
    deliberately lopsided:

      - A marker is recognised ONLY as one of three exact tokens, at the very end.
      - Anything else shaped like a marker is STRIPPED but yields no valence — a model that
        invents `⟦positive⟧` must not have it spoken aloud, and must not have it believed either.
      - No marker at all is the normal, expected case for older prompts and for the batch path.
        It yields the text unchanged and no valence, which is exactly today's behaviour.

    There is no input for which this returns text longer than it received, and none for which a
    malformed marker becomes a valence.
"""

from __future__ import annotations

import re
from typing import Literal

Valence = Literal["negative", "neutral", "positive"]

# Corner brackets rather than ASCII: they do not occur in ordinary speech in any of the seven
# supported languages, so a false positive on real dialogue is not a practical concern, and a
# stray one is far more likely to be the model than the speaker.
MARKERS: dict[str, Valence] = {
    "⟦+⟧": "positive",
    "⟦-⟧": "negative",
    "⟦=⟧": "neutral",
}

# Anything marker-SHAPED at the very end, bounded in length so a long tail of real text can
# never be swallowed. Used to clean up a malformed label; it never produces a valence.
_TRAILING_MARKER_SHAPE = re.compile(r"\s*⟦[^⟧]{0,24}⟧\s*$")

INSTRUCTION = (
    "\n\nAfter the translation, on the same line, append exactly one marker describing the "
    "SENTIMENT OF WHAT THE SPEAKER SAID — not of the translation, and not your opinion of it:\n"
    "⟦+⟧ if positive (agreement, praise, good news, enthusiasm)\n"
    "⟦-⟧ if negative (disagreement, a problem, criticism, bad news)\n"
    "⟦=⟧ if neither, or if you are unsure\n"
    "Use ⟦=⟧ whenever it is not clear. The marker is the last thing in your reply and nothing "
    "follows it. Never translate the marker, never explain it, and never use any other marker."
)


def split_valence(reply: str) -> tuple[str, Valence | None]:
    """Separate the translation from its sentiment marker.

    Returns `(text, valence)`. `valence` is None whenever nothing trustworthy was found, which
    the rest of the pipeline already treats correctly as NOT DETERMINED — distinct from
    "neutral", and the reason `to_generation_config` then sends no emotion label rather than
    guessing one from loudness alone.
    """
    if not reply:
        return reply, None

    stripped = reply.rstrip()

    for marker, valence in MARKERS.items():
        if stripped.endswith(marker):
            return stripped[: -len(marker)].rstrip(), valence

    # Marker-shaped but not a marker we know. Remove it — an invented `⟦positive⟧` reaching the
    # dub would be spoken aloud — but believe nothing about it.
    cleaned = _TRAILING_MARKER_SHAPE.sub("", stripped)
    if cleaned != stripped:
        return cleaned.rstrip(), None

    return reply, None


__all__ = ["INSTRUCTION", "MARKERS", "Valence", "split_valence"]
