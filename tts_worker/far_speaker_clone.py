"""Cloning the voice of ONE consenting person on the Meet side of a bridge room (WT-933).

WHAT THIS ADDS TO WT-932
    Everyone in the Meet call is published under one stand-in speaker_id, and WT-932 gave each
    caption name its own STOCK voice. A Meet-side person may now consent, for themselves, to be
    dubbed in a clone of their own voice. Voice is biometric data and the mixed feed carries
    several people, so every rule here is about never taking, and never using, a voice on a
    guess:

    * only the person's own consent counts (a Redis hash the backend writes, see CONSENTS_KEY);
    * audio is kept for cloning only when the caption hints are unanimous about who is speaking;
    * the clone is spoken only once the attribution has been unanimous several sentences running.

    The feature is behind TTS_FAR_SPEAKER_CLONE_ENABLED and off by default.

WHAT LIVES HERE
    The pure parts and the in-memory state: name folding, the consent field, the Redis keys, the
    pending queue of stand-in chunks waiting for their caption hints, and the streak counter. The
    I/O (Redis, Cartesia, the capture loop) stays in tts_worker/worker.py.

THE NAME IS NEVER STORED OR LOGGED
    A display name from someone else's call is personal data. It is folded and hashed into the
    consent field the moment it is read, the hash is what keys every buffer, streak and Redis
    entry, and logs carry only its first FAR_HASH_LOG_CHARS characters.
"""

from __future__ import annotations

import asyncio
import hashlib
import unicodedata
from collections import deque
from dataclasses import dataclass, field

from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.far_speaker import CaptionHintTracker
from shared.schemas import AudioChunkMessage

#: Written by the backend, only ever READ here. field = consent_field(name), value = an ISO
#: timestamp. A field that is present is consent in force; withdrawal is an HDEL of the field.
#: `{room}` is the translation room id, the value every AI worker calls meeting_id.
CONSENTS_KEY = "translationRoom:{room}:far_speaker_clone_consents"

#: Written here. field = consent_field(name), value = the Cartesia voice cloned for that name.
#:
#: Deliberately NOT under `voice:{meeting}:{speaker}`, the key a native speaker's clone lives at:
#: nothing that reads a speaker's own voice (`_get_voice_id`) can ever be handed one of these, so
#: a Meet-side person's clone can not become "the stand-in's voice" and be spoken for everyone on
#: the far side. And keyed by the consent field, so one name can never read another's.
CLONES_KEY = "tts:far_speaker_clones:{room}"

#: The three numbers of the shared contract. Not settings: the backend and the consent page
#: describe this behaviour to the person consenting, so they change in the contract or not at all.
CLONE_MIN_CONFIDENCE = 1.0
CLONE_MIN_SPEECH_MS = 1500
CLONE_MIN_STREAK = 3

#: How much of the consent field a log line carries.
FAR_HASH_LOG_CHARS = 12

#: Stand-in chunks waiting for their caption hints, across all rooms. A bound, not a tuning knob:
#: at a few seconds per chunk and a two-second hold, a healthy worker has a handful. Overflow
#: drops the OLDEST, which costs a clone sample and nothing else.
MAX_PENDING_CHUNKS = 64

#: (meeting_id, consent field): one Meet-side person in one room.
FarKey = tuple[str, str]


def fold(name: str) -> str:
    """The contract's name normalisation: NFKC, `str.lower()`, whitespace collapsed and trimmed.

    `lower`, not `casefold`: the backend (C# `ToLowerInvariant`) and the web (`toLowerCase`) fold
    the same name into the same string, and the consent field is a hash of it. WT-932's stock
    voice key uses `casefold` and is deliberately left alone; the two never meet.
    """
    return " ".join(unicodedata.normalize("NFKC", name).lower().split())


def consent_field(name: str | None) -> str | None:
    """The 64-hex consent field for a Meet display name, or None for a name that folds to nothing.

    None is the contract's "empty is invalid: no consent can be recorded for it", so a caller
    that gets None has no consent to look for.
    """
    folded = fold(name or "")
    if not folded:
        return None
    return hashlib.sha256(f"{EXTERNAL_BRIDGE_SPEAKER_ID}:{folded}".encode()).hexdigest()


def consents_key(meeting_id: str) -> str:
    return CONSENTS_KEY.format(room=meeting_id)


def clones_key(meeting_id: str) -> str:
    return CLONES_KEY.format(room=meeting_id)


def far_hash(consent_field_value: str) -> str:
    """What a log line may say about a name."""
    return consent_field_value[:FAR_HASH_LOG_CHARS]


def chunk_pcm_ms(chunk: AudioChunkMessage) -> int:
    """How long the PCM in this chunk is. 16-bit mono, the arithmetic the clone buffer uses."""
    return int(len(chunk.audio_data) // 2 * 1000 / max(chunk.sample_rate, 1))


def chunk_speech_ms(chunk: AudioChunkMessage) -> int:
    """How much of this chunk is speech, for the 1.5 s capture gate.

    `speech_ms` is what VAD called speech, which is the honest number: the PCM is wrapped in
    pre-speech and hangover padding. An older ingress leaves it at 0 ("did not say"), and then
    the PCM length is the only measurement there is. Never more than the PCM itself, because the
    PCM is what would be cloned from: a streamed turn can carry a long `speech_ms` and no audio.
    """
    pcm_ms = chunk_pcm_ms(chunk)
    if chunk.speech_ms > 0:
        return min(chunk.speech_ms, pcm_ms)
    return pcm_ms


def sentence_duration_ms(start_ms: int, end_ms: int, chunk_duration_ms: int) -> int:
    """How much audio a dubbed sentence's caption name was judged over, for the 1.5 s use gate.

    The segment's own span when STT gave it one. A sentence published early carries
    start_ms == end_ms (stt_worker: in flash mode most lines arrive that way), and STT attributed
    its name over the WHOLE chunk in that case, so the chunk's duration is the matching number.
    This is the same choice stt_worker._review_far_side makes for the attribution window.
    """
    if end_ms > start_ms:
        return end_ms - start_ms
    return max(0, chunk_duration_ms)


def sentence_qualifies(confidence: float | None, duration_ms: int) -> bool:
    """The two contract conditions that are about the sentence itself."""
    return (
        confidence is not None
        and confidence >= CLONE_MIN_CONFIDENCE
        and duration_ms >= CLONE_MIN_SPEECH_MS
    )


@dataclass
class PendingChunk:
    """A stand-in chunk held until the caption hints for it can have arrived."""

    chunk: AudioChunkMessage
    #: `time.monotonic()` at which it may be attributed.
    due: float


@dataclass
class FarCloneState:
    """Everything the far-speaker clone keeps in memory, for one worker process."""

    pending: deque[PendingChunk] = field(default_factory=deque)
    drainer: asyncio.Task[None] | None = None
    tracker: CaptionHintTracker | None = None
    # Clone samples being assembled. Only ever holds audio that was attributed with certainty to
    # a name whose consent field was present at that moment.
    buffers: dict[FarKey, bytearray] = field(default_factory=dict)
    buffer_seconds: dict[FarKey, float] = field(default_factory=dict)
    buffer_lang: dict[FarKey, str] = field(default_factory=dict)
    # A clone call is in the air for this name; further audio is not buffered meanwhile.
    in_flight: set[FarKey] = field(default_factory=set)
    # The vendor refused in a way no later clip can change (plan, credits, credentials).
    refused: set[FarKey] = field(default_factory=set)
    # (meeting, consent field, target language) -> (consecutive qualifying sentences, the
    # segment_id last counted). See note_sentence.
    streaks: dict[tuple[str, str, str], tuple[int, str]] = field(default_factory=dict)
    # Rooms a stand-in has been seen in, which is every room that can hold a far clone. The
    # consent watcher walks these.
    watched: set[str] = field(default_factory=set)

    def hold(self, chunk: AudioChunkMessage, due: float) -> None:
        self.pending.append(PendingChunk(chunk, due))
        while len(self.pending) > MAX_PENDING_CHUNKS:
            self.pending.popleft()

    def drop_buffer(self, key: FarKey) -> None:
        self.buffers.pop(key, None)
        self.buffer_seconds.pop(key, None)
        self.buffer_lang.pop(key, None)

    def note_sentence(
        self, meeting_id: str, field_value: str, target_lang: str, segment_id: str, qualifies: bool
    ) -> int:
        """Count one dubbed sentence attributed to this name; returns the streak it leaves.

        PER TARGET LANGUAGE, because one spoken sentence arrives here once per language it was
        translated into. Counted per (meeting, name) alone, a room with three listening languages
        would reach "the third consecutive sentence" on the first one. Within one language each
        sentence is seen once and in order (the worker holds that key's lock).

        A redelivered message (same segment_id as the last one counted) does not count twice.
        A sentence that does not qualify takes the streak back to zero.
        """
        key = (meeting_id, field_value, target_lang)
        if not qualifies:
            self.streaks[key] = (0, segment_id)
            return 0
        count, last_segment = self.streaks.get(key, (0, ""))
        if segment_id and segment_id == last_segment:
            return count
        self.streaks[key] = (count + 1, segment_id)
        return count + 1

    def forget_room(self, meeting_id: str) -> None:
        """A room ended. Nothing of its audio or its people stays in this process."""
        self.pending = deque(p for p in self.pending if p.chunk.meeting_id != meeting_id)
        for key in [key for key in self.buffers if key[0] == meeting_id]:
            self.drop_buffer(key)
        self.in_flight = {key for key in self.in_flight if key[0] != meeting_id}
        self.refused = {key for key in self.refused if key[0] != meeting_id}
        for streak_key in [key for key in self.streaks if key[0] == meeting_id]:
            self.streaks.pop(streak_key, None)
        self.watched.discard(meeting_id)
        if self.tracker is not None:
            self.tracker.forget(meeting_id)
