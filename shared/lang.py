"""Language-tag comparison shared by the translation and TTS workers.

S6 — a same-language pair produced double audio. The pipeline compared language tags in
three different ways and none of them stopped a publish:

- translation_worker/worker.py compared them with `==` to decide `passthrough`, which only
  skipped the LLM call — the message was still built and published;
- the speculative prefetch and translator.py both compared BASE tags, and both merely
  declined to translate;
- tts_worker had no comparison at all, so it synthesized the "translation" and published a
  LiveKit interpreter track carrying the speaker's own words back to a listener who is
  already hearing that speaker's raw mic.

One helper, used at every decision point, so a new call site cannot pick a subtly different
rule. BASE tags, because that is the comparison the translator itself uses: with source
"en-US" and target "en-GB" the translator returns the text verbatim, so an exact-match test
would let a perfect echo through.
"""

from __future__ import annotations


def base_language(tag: str) -> str:
    """The primary subtag of a BCP-47-ish language tag: "en-US" -> "en", "vi" -> "vi"."""
    return tag.strip().split("-", 1)[0].casefold()


def is_same_language(left: str, right: str) -> bool:
    """Whether two language tags name the same language, ignoring region/script.

    An empty or missing tag names no language and therefore matches nothing — treating it
    as a match would silently suppress real translations whenever STT failed to report a
    language.
    """
    left_base = base_language(left or "")
    right_base = base_language(right or "")
    return bool(left_base) and left_base == right_base
