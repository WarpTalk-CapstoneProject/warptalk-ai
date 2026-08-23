"""Turning a speaker id into something a summary can say out loud. WT-529.

The live summariser accumulates ``(speaker_id, text, timestamp)`` straight off ``stt:results``,
where ``speaker_id`` is the LiveKit participant identity — ``speaker-019f0d00-0de0-7000-9000-
000000000003``. ``format_transcript_line`` put it in the prompt verbatim, the model wrote it into
its prose, and the meeting page rendered summaries and action items attributing decisions to a
uuid.

The model is not doing anything wrong. It is repeating the only name it was given.

TWO WAYS OUT, AND BOTH ARE USED
    A room's real names live in the ``meeting:{id}:speaker_names`` Redis HASH, written one field
    per participant by ``livekit_ingress_worker`` as it sees them join. When that map has the
    speaker, the summary says the person.

    A hash rather than a JSON blob, deliberately: the ingress worker writes each participant as
    it meets them, and two replicas doing read-modify-write on one JSON string would lose
    whichever name landed second.

    When it does not — the key expired, the room service never wrote it, a bridge guest who was
    never on the roster — the speaker becomes ``Speaker 1``, ``Speaker 2``, numbered by first
    appearance in the transcript.

    Falling back to a pseudonym rather than the raw id is deliberate. A uuid in a summary is not
    a degraded name, it is noise that makes the summary unusable; ``Speaker 2`` is readable,
    stable for the length of the meeting, and — this is the part that matters — claims nothing
    false about who anybody is. The numbering is per meeting and derived from the transcript, so
    it never leaks an id and never mislabels one person as another.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: LiveKit identities are published with this prefix; the map may be keyed either way.
_IDENTITY_PREFIX = "speaker-"


def parse_speaker_names(raw: Mapping[Any, Any] | bytes | str | None) -> dict[str, str]:
    """The room's ``{speaker_id: display name}`` map, or an empty one.

    Takes the hash as ``hgetall`` returns it (bytes keys and values), and still accepts a JSON
    string so a caller holding one — a test, or a future producer — does not need a second
    function.

    Never raises. A missing key, an expired key, a half-written value or a shape this function has
    never seen all mean the same thing to the caller — no names — and that case is already handled
    by the pseudonyms. Failing the summary over it would trade an ugly summary for no summary.
    """
    if not raw:
        return {}

    parsed: Any = raw
    if isinstance(raw, (bytes, str)):
        try:
            text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            parsed = json.loads(text)
        except Exception:
            return {}

    if not isinstance(parsed, Mapping):
        return {}

    names: dict[str, str] = {}
    for key, value in parsed.items():
        try:
            name = value.decode("utf-8") if isinstance(value, bytes) else value
            speaker_id = key.decode("utf-8") if isinstance(key, bytes) else key
        except Exception:
            continue
        if not speaker_id or not isinstance(name, str) or not name.strip():
            continue
        names[str(speaker_id)] = name.strip()
    return names


class SpeakerNamer:
    """Resolves speaker ids to names, assigning stable pseudonyms to the ones it cannot.

    Stateful on purpose: ``Speaker 2`` has to mean the same person on every line of one
    transcript, so the numbering lives with the meeting being formatted rather than being
    recomputed per line.
    """

    def __init__(self, names: Mapping[str, str] | None = None) -> None:
        self._names = dict(names or {})
        self._pseudonyms: dict[str, str] = {}

    def __call__(self, speaker_id: str) -> str:
        return self.name_for(speaker_id)

    def name_for(self, speaker_id: str) -> str:
        speaker_id = (speaker_id or "").strip()
        if not speaker_id:
            # Nothing to key a pseudonym on either — two blank ids are not evidence of two
            # people, so they share one label rather than inflating the count.
            return "Unknown speaker"

        known = self._known(speaker_id)
        if known:
            return known

        if speaker_id not in self._pseudonyms:
            self._pseudonyms[speaker_id] = f"Speaker {len(self._pseudonyms) + 1}"
        return self._pseudonyms[speaker_id]

    def _known(self, speaker_id: str) -> str | None:
        """The published name, matched with or without the identity prefix.

        Both forms are tried because the two ends disagree about it historically: the STT stream
        carries the LiveKit identity (``speaker-<uuid>``) while a roster keyed by user id carries
        the bare uuid. Accepting either costs one dictionary lookup and removes a whole class of
        "the map was there and matched nothing".
        """
        direct = self._names.get(speaker_id)
        if direct:
            return direct

        if speaker_id.startswith(_IDENTITY_PREFIX):
            bare = speaker_id[len(_IDENTITY_PREFIX) :]
            stripped = self._names.get(bare)
            if stripped:
                return stripped
        else:
            prefixed = self._names.get(f"{_IDENTITY_PREFIX}{speaker_id}")
            if prefixed:
                return prefixed

        return None
