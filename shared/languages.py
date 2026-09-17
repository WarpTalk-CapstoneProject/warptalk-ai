"""ISO 639-1 code → the language's English name, for prompts that must NAME a language.

WHY A MODEL IS GIVEN THE NAME AND NOT THE CODE
    "Write in ja" is a tag; "Write in Japanese" is an instruction. The codes are what travels
    over the wire — rooms, requests and stored rows are all keyed by them — but a prompt is
    read, not parsed, and a two-letter tag is the part of a prompt a model is most likely to
    ignore or to echo back into its output as a label.

WHY THIS LIVES IN shared/
    translation_worker had this map to itself, and the summary path now needs the same one.
    A second copy would drift, and the copy that drifts is always the one nobody is looking at
    — the exact failure warptalk-web's languages.ts documents at length. So: one map, imported
    by both, and `translation_worker.translator._lang_name` is kept as a thin alias so its
    existing callers and tests do not have to move.

The rows mirror the `meeting`-scoped entries of warptalk-web's SUPPORTED_LANGUAGES. A code with
no entry here falls back to itself rather than raising: an unknown language is a worse prompt,
but a crashed worker is a missing summary.
"""

from __future__ import annotations

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "vi": "Vietnamese",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ru": "Russian",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
    "it": "Italian",
}


def language_name(iso_code: str) -> str:
    """The English name for a code, or the code itself when it is not one we know."""
    return LANGUAGE_NAMES.get(iso_code.split("-")[0].strip().lower(), iso_code)


def normalize_language_code(code: str | None) -> str:
    """Bare lowercase ISO 639-1, matching LanguageHelper.NormalizeLanguageCode on the backend.

    Both sides have to agree, because a room stores `vi-VN` while a summary request carries
    `vi`, and a mismatch here is a summary silently written in the wrong language.
    """
    if not code:
        return ""
    return code.strip().split("-")[0].lower()


def known_language_code(code: str | None) -> str:
    """The normalized code when it is one we can name, otherwise "" (nobody chose).

    WT-703. `language_name` falls back to its input, which is right for a translation prompt
    naming a code the web added before this map caught up, and wrong for any string that
    arrived from outside: a summary language of "klingon" — or a sentence of instructions —
    went into the system prompt verbatim. Anything this map cannot name is therefore treated
    as no choice at all, so an unrecognised value can only ever cost the reader the language
    they wanted, never put its own text in front of the model.
    """
    normalized = normalize_language_code(code)
    return normalized if normalized in LANGUAGE_NAMES else ""
