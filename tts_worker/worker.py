"""TTS Worker — Consumes translated text, produces synthesized audio.

Pipeline:
    Redis Stream (translate:results:{meetingId})
    → Cartesia Sonic Turbo (default voice until clone ready, then cloned voice)
    → Redis Stream (tts:results:{meetingId})

Voice cloning:
    Background task buffers audio:chunks per speaker.
    Once voice_clone_min_seconds of audio collected → POST /voices/clone.
    voice_id cached in Redis: voice:{meeting_id}:{speaker_id}
    All synthesis calls after that use the cloned voice.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from shared import isochrony
from shared.base_worker import BaseWorker
from shared.config import TTSSettings
from shared.lang import base_language, is_same_language
from shared.prosody import SPEED_MAX, Arousal, Delivery, Valence, to_generation_config
from shared.schemas import AudioChunkMessage, TranslationResultMessage, TTSResultMessage
from tts_worker.clone_sample_quality import MAX_SAMPLE_SCORE, assess_clone_sample
from tts_worker.livekit_publisher import LiveKitTTSPublisher, TrackStream
from tts_worker.prosody_context import ProsodyContext, wav_header
from tts_worker.synthesizer import CartesiaSynthesizer

# Standard WAV header size for the pcm_s16le format CartesiaSynthesizer requests —
# used to strip the header before feeding audio into the LiveKit track (which wants
# raw PCM frames, not a WAV container).
_WAV_HEADER_BYTES = 44

# WT-396 — the hand-off for a recording somebody uploaded of themselves.
#
# AuthService cannot clone (no Cartesia key, by design) and this side cannot read the bucket the
# recording lands in (no storage credentials, also by design), so the audio and the answer travel
# through Redis — the same way the voice catalogue already does in the other direction.
#
# The sample key is suffixed with the profile id so WT-402's rule gives it a bounded lifetime
# automatically: anything whose last segment parses as a UUID expires. It is deleted as soon as
# the clone finishes either way, because these are biometric bytes and the TTL is the backstop
# rather than the plan.
_CLONE_REQUEST_STREAM = "voice:clone_requests"
_CLONE_SAMPLE_PREFIX = "voice:clone_sample:"
_CLONE_RESULT_PREFIX = "voice:clone_result:"
# The answer outlives the audio: somebody may upload and not open the page for days, and losing
# the id would mean paying Cartesia again for a voice we already made.
_CLONE_RESULT_TTL_SECONDS = 7 * 24 * 60 * 60

# The two kinds of voice this worker creates in the Cartesia account, told apart by the name it
# gives them. This is not cosmetic: `_sweep_orphan_voices` deletes on exactly this distinction,
# because the AI side has no database and no other way to know which voices something still
# points at.
#
#   {_IN_MEETING_VOICE_PREFIX}  cloned live from a meeting microphone. Reachable only through
#                               `voice:{meeting}:{speaker}`, which expires — so these become
#                               garbage on their own and are what the sweep collects.
#   {_UPLOAD_VOICE_PREFIX}      made from a recording somebody uploaded. AuthService stores the
#                               id in voice_profiles.provider_voice_id and a person chose it in
#                               the picker. NEVER swept: this side cannot see that table, so a
#                               deletion here is unrecoverable from here.
#
# Anything that later PROMOTES an in-meeting clone to a permanent profile must rename it to the
# upload prefix (cartesia voices.update takes `name`) as part of promoting it. That is the whole
# handshake — no extra bookkeeping, and the voice stops being sweepable the moment it stops
# being disposable.
_IN_MEETING_VOICE_PREFIX = "speaker-"
_UPLOAD_VOICE_PREFIX = "profile-"

# WT-D — "what will I actually sound like in a meeting?"
#
# Same shape as the clone hand-off above and for the same reason: the Cartesia key lives only on
# this side, so AuthService cannot render a sample itself and asks for one instead.
#
# THE RESULT KEY IS THE CACHE, AND THAT IS WHY IT IS NOT KEYED BY REQUEST
#     A preview of one voice in one language is the same audio every time anybody asks for it.
#     Keying by (voice, language) rather than by request id means the second person to press play
#     — and the same person pressing it again — is served from Redis with no Cartesia call at all.
#     The cost of a preview is therefore paid once per voice, not once per click.
_PREVIEW_REQUEST_STREAM = "voice:preview_requests"
_PREVIEW_RESULT_PREFIX = "voice:preview:"
# A rendered sample does not go stale: the voice it came from is immutable, and a voice that is
# deleted takes its preview with it at the next expiry. A day keeps repeat plays free without
# holding audio for a voice nobody has touched in a week.
_PREVIEW_RESULT_TTL_SECONDS = 24 * 60 * 60

# What the preview says, per language.
#
# It is spoken in the language being previewed rather than translated into it, because the point
# of the sample is to answer "is this me?" — and a listener judges that far better on a sentence
# in a language they read. The wording is deliberately ordinary: a dramatic line would be
# performed by the model and flatter the voice, which is the opposite of what this is for.
_PREVIEW_TEXT: dict[str, str] = {
    "en": "Hello, this is how I will sound to people listening in another language.",
    "vi": "Xin chào, đây là giọng của tôi khi mọi người nghe bằng ngôn ngữ khác.",
    "ja": "こんにちは。別の言語で聞いている人には、この声で聞こえます。",
    "ko": "안녕하세요. 다른 언어로 듣는 사람에게는 이 목소리로 들립니다.",
    "fr": "Bonjour, voici comment je sonnerai pour ceux qui écoutent dans une autre langue.",
    "es": "Hola, así es como sonaré para quienes escuchen en otro idioma.",
    "zh": "你好，这就是别人用其他语言收听时我的声音。",
}

# WT-B — a clone that outlives the meeting it was made in.
#
# WHY THE HAND-OFF EXISTS AT ALL
#     `voice:{meeting}:{speaker}` is keyed BY MEETING and expires, so the next meeting re-clones
#     from nothing and pays its first twenty seconds in a stock voice. Making the clone permanent
#     means a row in voice_profiles, and only AuthService can write one — this side has no
#     database, by the same design that keeps the Cartesia key over here.
#
# WHY THIS SIDE RENAMES BEFORE IT PUBLISHES
#     The sweep judges a voice by its name. A voice announced to AuthService while still named
#     `speaker-` is a row that points at something the sweep deletes within 24 hours, and the
#     person would open a later meeting in a voice that no longer exists. So the rename is the
#     promotion, the publish only reports it, and a failed rename publishes nothing at all —
#     leaving an ordinary in-meeting clone that the sweep collects exactly as before.
#
# WHY A STREAM AND NOT A KEY
#     There is no page load between the clone and the next meeting to pull an answer on, the way
#     `voice:clone_result:` is pulled when the profiles page opens. The fact has to arrive on its
#     own. A stream also survives AuthService being down for a deploy, which a pub/sub message
#     does not.
_CARRY_OVER_STREAM = "voice:auto_clone_ready"
# The payload is four small fields; this is a backlog for a deploy, not a queue.
_CARRY_OVER_STREAM_MAXLEN = 1_000

# The other direction: AuthService asking for a voice to be destroyed, because it holds the row
# and this side holds the key.
#
# Two callers, both of which MUST reach Cartesia or they are a lie. Withdrawing voice-clone
# consent has to actually delete the voice model, not merely stop using it — that is the promise
# the consent text makes, and it costs nothing to keep while the clone dies with the meeting but
# becomes a real one the moment a clone is kept. Replacing a carried-over clone with a better one
# has to delete the loser, or promoting clones simply moves the leak the sweep just closed to a
# place the sweep can no longer see (a `profile-` voice is never swept).
_VOICE_DELETE_STREAM = "voice:delete_requests"

# One replica sweeps per cycle. Held for the length of the interval, so a fleet-wide restart
# does not turn every replica's startup sweep into a duplicate walk of the whole account.
_ORPHAN_SWEEP_LOCK_KEY = "voice:orphan_sweep:lock"

# Cartesia's voices.clone() requires a concrete `language`, but AudioChunkMessage.language
# defaults to "auto" (STT does language auto-detection, not the audio-chunk producer) — so
# fall back to "en" for anything Cartesia's SDK wouldn't accept as a real language code.
_CARTESIA_SUPPORTED_LANGUAGES = {
    "en",
    "fr",
    "de",
    "es",
    "pt",
    "zh",
    "ja",
    "hi",
    "it",
    "ko",
    "nl",
    "pl",
    "ru",
    "sv",
    "tr",
    "tl",
    "bg",
    "ro",
    "ar",
    "cs",
    "el",
    "fi",
    "hr",
    "ms",
    "sk",
    "da",
    "ta",
    "uk",
    "hu",
    "no",
    "vi",
    "bn",
    "th",
    "he",
    "ka",
    "id",
    "te",
    "gu",
    "kn",
    "ml",
    "mr",
    "pa",
}


def _clone_language(hint: str) -> str:
    """The language to clone a voice IN, normalized to what Cartesia is keyed by.

    `base_language` first, for the third time in this codebase and for the same reason each
    time: `_CARTESIA_SUPPORTED_LANGUAGES` holds primary subtags, and a locale tag compared
    against it verbatim matches nothing. "vi-VN" therefore fell through to "en" and a Vietnamese
    speaker's clip was cloned as an English voice — the same class of bug that made
    `_default_voice_id` speak English in a Vietnamese-only meeting and that starved
    `list_voices` of every non-English language.

    "auto" still lands on "en", and that is the honest answer rather than a fix: it means the
    speak language never resolved (see participant-language-preference.ts, UNRESOLVED_LANGUAGE),
    so there is nothing better to say. It is worth knowing that the fallback is now only ever
    reached by a language we genuinely have no support for, not by a spelling of one we do.
    """
    normalized = base_language(hint)
    return normalized if normalized in _CARTESIA_SUPPORTED_LANGUAGES else "en"


def _decode_field(data: Mapping[Any, Any], key: str) -> str:
    raw = data.get(key)
    if raw is None:
        raw = data.get(key.encode())
    if raw is None:
        return ""
    return raw.decode() if isinstance(raw, bytes) else raw


@dataclass(slots=True)
class SynthesizedSentence:
    """What one sentence of synthesis produced, and what has already been done with it.

    This was a bare `(bytes, duration_ms, voice_id)` tuple until WT-397 made synthesis able to
    publish. Two of the fields below only mean anything because of that:

    `already_spoken` — the listener has ALREADY heard this audio; it went onto the LiveKit
        track chunk by chunk as Cartesia produced it. The caller must not publish it again. The
        bytes are still complete, because the cache, billing (tts:results) and the transcript
        all read them and none of them is audible.

    `first_audio_at` — `time.monotonic()` at the moment the first frame reached the track.
        THE NUMBER THIS FEATURE MOVES. `tts_synthesis` covers the whole call and now includes
        handing the audio over, which takes about as long as the sentence lasts — so it RISES
        with streaming on while the listener waits far less. Anyone reading the dashboard needs
        `tts_first_audio` next to it or v91 looks like a regression.
    """

    audio: bytes
    duration_ms: int
    voice_id: str
    already_spoken: bool = False
    first_audio_at: float | None = None


def _extract_tts_key(
    data: Mapping[Any, Any],
) -> tuple[str, str, str]:
    """Cheap (meeting_id, speaker_id, target_lang) extraction from a raw translate:results
    Redis entry, ahead of process()'s own full TranslationResultMessage parse — this is
    the unit of ordering _consume_loop must preserve (one LiveKit track per this triple)."""
    return (
        _decode_field(data, "meeting_id"),
        _decode_field(data, "speaker_id"),
        _decode_field(data, "target_lang"),
    )


# How far behind a dub may fall before it is read faster to catch up. WT-528.
#
# TWO WAYS TO BE BEHIND, AND THEY ARE THE SAME PROBLEM
#     * Out of order — this sentence belongs EARLIER in the speaker's timeline than one already
#       spoken. The per-key lock preserves the order messages arrive in, but that order is set by
#       when each translation finished upstream, and translation runs concurrently: a long
#       sentence loses the race to a short one spoken after it. Testers described exactly this —
#       "voice clone đoạn dưới trước rồi ngược lên đoạn trên".
#     * Simply slow — translation took a long time, so the dub starts well after the speaker
#       finished, and every sentence behind it inherits the delay because the key is serialized.
#
# Both mean the dub is running behind the conversation, and reading the backlog at normal pace
# keeps it behind. Speaking a late line FASTER is what lets the gap close instead of persist.
#
# ON_TIME_MS is a budget, not a target: a sentence translated within it was never behind, so no
# adjustment is made and the call is byte-for-byte what it was before this existed.
_CATCH_UP_ON_TIME_MS = 1500
# Lag at which the full boost applies. Between the two the ramp is linear, so a dub that is
# slightly behind is only slightly quicker — a step change would be audible as the pace jumping.
_CATCH_UP_FULL_LAG_MS = 4000
# Deliberately well under shared.prosody.SPEED_MAX (1.5). This multiplies the speaker's own
# measured rate, so it has to leave room for someone who already talks fast, and a dub that
# outruns comprehension has not caught up with anything.
_CATCH_UP_MAX_SPEED = 1.25


class TTSWorker(BaseWorker):
    """Text-to-Speech worker using Cartesia Sonic Turbo."""

    worker_name = "tts"
    input_stream = "translate:results"
    consumer_group = "tts-workers"
    _audio_consumer_group = "tts-audio-workers"
    # WT-396. Its own group, not the audio one: these are two unrelated backlogs, and sharing a
    # group would make a slow clone hold up live meeting audio behind it.
    _clone_request_group = "tts-upload-clone-workers"
    # Its own group again, by the same rule: a preview is somebody waiting on a button, and it
    # must not queue behind a clone or behind live meeting audio.
    _preview_request_group = "tts-preview-workers"
    # And again. A deletion is a promise being kept (withdrawn consent) and must not wait behind
    # a clone, a preview, or a meeting's audio.
    _voice_delete_group = "tts-voice-delete-workers"
    _running = True

    def __init__(
        self,
        tts_settings: TTSSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tts_settings = tts_settings or TTSSettings()
        self.cartesia: CartesiaSynthesizer | None = None
        self.livekit_publisher: LiveKitTTSPublisher | None = None
        # (meeting_id, speaker_id, target_lang) -> lock serializing that key's own
        # messages — see _consume_loop for why.
        self._key_locks: dict[tuple[str, str, str], asyncio.Lock] = {}
        # The furthest-along sentence this key has already SPOKEN, by its position in the
        # speaker's own timeline. Read and written only while that key's lock is held, so a
        # plain dict is safe for the same reason `_turns` below is.
        self._spoken_start_ms: dict[tuple[str, str, str], int] = {}
        # One in-flight spoken turn per (meeting, speaker, language, voice). The per-key lock
        # above is what makes a plain dict safe here: a key's sentences are processed one at a
        # time, so a turn can never be pushed into concurrently.
        self._turns: dict[tuple[str, ...], ProsodyContext] = {}
        self._turn_connections: dict[tuple[str, ...], Any] = {}
        # Isochrony state, per (meeting, speaker, target language): how this speaker's dubs have
        # been running against the clock, and the turn currently being accumulated.
        self._dub_fits: dict[tuple[str, str, str], isochrony.DubFit] = {}
        self._turn_dub_ms: dict[tuple[str, str, str], int] = {}

    async def load_model(self) -> None:
        self.cartesia = CartesiaSynthesizer(
            api_key=self.tts_settings.api_key,
            model=self.tts_settings.model,
            sample_rate=self.tts_settings.sample_rate,
            speed=self.tts_settings.speed,
        )
        await self.cartesia.load()
        # Before any consumer starts: the first sentence of the first turn is the one that
        # would otherwise pay the ~427ms dial, and it is also the one somebody is listening for.
        await self.cartesia.warm_up(pool_size=self.tts_settings.tts_warm_pool_size)
        self.livekit_publisher = LiveKitTTSPublisher(self.settings.livekit)
        asyncio.create_task(self._consume_audio_for_cloning())
        asyncio.create_task(self._consume_upload_clone_requests())
        asyncio.create_task(self._consume_preview_requests())
        asyncio.create_task(self._consume_voice_delete_requests())
        if self.tts_settings.orphan_voice_sweep_enabled:
            asyncio.create_task(self._sweep_orphan_voices())
        self.logger.info("tts_worker_ready", model=self.tts_settings.model)

    async def _consume_loop(self) -> None:
        """Dispatch process() concurrently across DIFFERENT (speaker, target_lang)
        keys, while keeping each key's OWN messages strictly ordered.

        BaseWorker's default _consume_loop awaits process() for one message before
        even reading the next, so — before this override — synthesizing speaker A's
        sentence into English fully blocked speaker B's sentence (a different speaker,
        a different LiveKit track, no shared state at all) from even STARTING its own
        Cartesia call. Dispatching concurrently is what lets concurrent speakers (and a
        single speaker's multiple target languages) be synthesized and dubbed in true
        parallel, matching how livekit_publisher already gives each (speaker,
        target_lang) its own independent WebRTC track.

        The per-key lock keeps messages for the SAME key (e.g. sentence 1 then sentence
        2 of one utterance, same speaker, same target_lang) processed one at a time, in
        order — Cartesia's per-call synthesis latency varies, so without this a later
        sentence could finish synthesizing first and get published to the shared track
        before an earlier one, playing the dub back in the wrong order. This trades a
        little same-key pipelining (sentence 2 can't start synthesizing until sentence
        1's audio has fully been pushed to the track) for guaranteed in-order playback —
        the right trade-off, since real speech itself paces how fast new same-key
        sentences even arrive.

        RedisStreamClient.consume_concurrent ties XACK to successful handler
        completion. Failed work remains pending for BaseWorker's reclaim/DLQ path.
        """
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )

        async def _run(message_id: bytes, data: dict[bytes, bytes]) -> None:
            key = _extract_tts_key(data)
            lock = self._key_locks.setdefault(key, asyncio.Lock())
            async with lock:
                await self._process_and_log_errors(message_id, data)

        while not self._shutdown_event.is_set():
            try:
                await self._recover_stale_messages()
                await self.redis.consume_concurrent(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    handler=_run,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=8,
                    concurrency=8,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                await asyncio.sleep(1.0)

    async def _cleanup(self) -> None:
        """Drain the Cartesia connection pool on shutdown.

        Without this a redeploy leaves however many sockets the pool held open on the vendor's
        side until they time out — small, but it is the kind of leak this worker has already
        been caught doing once with cloned voices.
        """
        cartesia = getattr(self, "cartesia", None)
        if cartesia is not None:
            await cartesia.close()

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        stale_keys = [key for key in self._key_locks if key[0] == room_id]
        for key in stale_keys:
            self._key_locks.pop(key, None)
        # Same lifetime as the locks. A room that ends and is somehow seen again must not judge
        # its first sentence as late against a timeline from the previous meeting.
        #
        # getattr for the same reason `_turns` below uses it: the tests build workers with
        # __new__ and never run __init__.
        spoken: dict[tuple[str, str, str], int] = getattr(self, "_spoken_start_ms", {})
        for key in [key for key in spoken if key[0] == room_id]:
            spoken.pop(key, None)
        # getattr, because the tests build workers with __new__ and never run __init__ — the
        # same guard the rest of this codebase uses for that pattern. A worker with no turns
        # dict has no turns to abandon.
        turns: dict[tuple[str, ...], ProsodyContext] = getattr(self, "_turns", {})
        for turn_key in [k for k in turns if k[0] == room_id]:
            turn = turns.pop(turn_key, None)
            if turn is not None:
                # Fire-and-forget: _cleanup_room is sync (it is called from the route-state
                # broadcast handler), and a room that has ended is not waiting on a socket.
                asyncio.create_task(turn.abandon())

    async def _synthesize_sentence(
        self,
        *,
        translation: TranslationResultMessage,
        text: str,
        voice_id: str | None,
        voice_key: str,
        generation_config: dict[str, float | str] | None,
    ) -> SynthesizedSentence:
        """One sentence of a turn, spoken in prosodic continuity with the ones before it.

        WT-371 follow-up / Level 4. A spoken turn is routinely split into several sentences
        (chunk_index > 0), and each used to be an independent one-shot generation with no memory
        of the one before it — so the model opened every sentence at its own default baseline and
        the dub came back as a list of separately-read sentences. Cartesia's contexts exist for
        exactly this; see tts_worker/prosody_context.py.

        Falls back to the proven one-shot path on ANY failure, and when the feature is off. That
        is not defensive padding: this WebSocket path has never run against the real API from
        this codebase, and a dub that fails is silence in a live meeting.

        See SynthesizedSentence for what comes back and why it is no longer just the audio.
        """
        synthesizer = self._require_cartesia()
        resolved_voice_id = voice_id or CartesiaSynthesizer._default_voice_id(
            translation.target_lang
        )

        if not self.tts_settings.prosody_continuity:
            audio_bytes, duration_ms, one_shot_voice_id = await synthesizer.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
                generation_config=generation_config,
            )
            return SynthesizedSentence(audio_bytes, duration_ms, one_shot_voice_id)

        # Keyed by voice as well as by speaker and language: a clone upgrade replaces the voice
        # mid-meeting (voice_clone_max_upgrades), and continuing a turn into a different voice
        # would be worse than the seam this removes.
        key = (
            translation.meeting_id,
            translation.speaker_id,
            translation.target_lang,
            voice_key,
            resolved_voice_id,
        )

        track: TrackStream | None = None
        try:
            turn = self._turns.get(key)
            # A context can retire itself without ever raising. `_collect` treats Cartesia's
            # `done` as an ordinary end of stream: it marks the context closed, breaks, and
            # returns the audio it collected — so `speak()` SUCCEEDS and the caller never
            # reaches the except branch that would have called `_end_turn`. The spent context
            # stayed in this map, and the next sentence for the same key fetched it, found it
            # not-None, and called `speak()` on it, which raised "ProsodyContext is closed".
            #
            # One wasted sentence per `done`, every time — no streaming and a full one-shot
            # re-synthesis, which is the p95 tail. Production 15 Aug, meeting 01a0033f: 12 of 47
            # sentences, up to 10.2s each, clustered exactly where the two speakers alternated.
            # Cartesia ends a context that has been idle, and with two people talking each
            # speaker's context idles while the other one speaks — so the more natural the
            # conversation, the more often this fired.
            #
            # Checked at acquisition rather than after `speak()` returns, because this is the
            # one place every reuse passes through: it covers the `done` path and any other
            # route to a closed context equally, instead of guarding the single case we know
            # about today.
            if turn is not None and turn.is_closed:
                await self._end_turn(key)
                turn = None
            if turn is None:
                turn, connection = await synthesizer.open_prosody_context(
                    context_id=f"{translation.speaker_id}:{translation.target_lang}:{voice_key}",
                    language=translation.target_lang,
                    voice_id=voice_id,
                )
                self._turns[key] = turn
                self._turn_connections[key] = connection

            # getattr, because some tests build workers with __new__ and never run __init__ —
            # the same guard the rest of this codebase uses for that pattern.
            publisher = getattr(self, "livekit_publisher", None)
            if publisher is None or not self.tts_settings.stream_to_livekit:
                audio_bytes, duration_ms = await turn.speak(text, generation_config)
                return SynthesizedSentence(audio_bytes, duration_ms, resolved_voice_id)

            async with publisher.stream(
                translation.meeting_id,
                translation.speaker_id,
                translation.target_lang,
                self.tts_settings.sample_rate,
                voice_key=voice_key,
            ) as track:
                audio_bytes, duration_ms = await turn.speak(
                    text, generation_config, on_pcm=track.feed
                )
            # Read AFTER the stream closed: the pump is still draining while speak() returns,
            # so asking inside the block would undercount what the listener actually heard.
            return SynthesizedSentence(
                audio_bytes,
                duration_ms,
                resolved_voice_id,
                already_spoken=track.spoken_bytes > 0,
                first_audio_at=track.first_audio_at,
            )
        except Exception:
            already_spoken = track is not None and track.spoken_bytes > 0
            self.logger.warning(
                "prosody_context_failed_falling_back",
                meeting_id=translation.meeting_id,
                already_spoken=already_spoken,
                exc_info=True,
            )
            await self._end_turn(key)
            audio_bytes, duration_ms, one_shot_voice_id = await synthesizer.synthesize(
                text=text,
                language=translation.target_lang,
                voice_id=voice_id,
                generation_config=generation_config,
            )
            if already_spoken:
                # THE ONE DECISION THIS FEATURE TURNS ON, recorded here rather than in a ticket.
                #
                # The context died after part of the sentence was already on the track. The
                # fallback re-synthesizes the WHOLE sentence, so playing it would speak the
                # opening words a second time. Three options were weighed:
                #
                #   1. don't play the fallback — the listener hears a truncated sentence
                #   2. play only the missing tail — needs a byte offset across two independent
                #      generations, which will not match at the seam
                #   3. only stream once a flush has already succeeded in this turn — narrows
                #      the window, never closes it
                #
                # (1), because the two failures are not equally bad. A cut-off sentence reads
                # as a dropout: the listener asks the speaker to repeat and the transcript
                # (built from the bytes returned below, which ARE complete) still has the whole
                # line. A sentence that stutters its own opening reads as the system being
                # broken, and there is nothing to fix it with.
                self.logger.warning(
                    "tts_fallback_suppressed_after_partial_stream",
                    meeting_id=translation.meeting_id,
                    speaker_id=translation.speaker_id,
                    spoken_bytes=track.spoken_bytes if track else 0,
                )
            return SynthesizedSentence(
                audio_bytes,
                duration_ms,
                one_shot_voice_id,
                already_spoken=already_spoken,
                first_audio_at=track.first_audio_at if track else None,
            )
        finally:
            # The turn ends where the SPEAKER stopped, not where a chunk boundary fell —
            # is_final_chunk is the only signal that carries that.
            if translation.is_final_chunk:
                await self._end_turn(key)

    async def _end_turn(self, key: tuple[str, ...]) -> None:
        turn = self._turns.pop(key, None)
        connection = self._turn_connections.pop(key, None)
        if turn is not None:
            await turn.aclose()
        if connection is not None:
            try:
                await connection.close()
            except Exception:
                self.logger.debug("prosody_connection_close_failed", exc_info=True)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Synthesize one translated text segment — into every DISTINCT voice this
        (speaker, target_lang) needs (see _resolve_voice_variants): the shared
        default/cloned track everyone hears absent a preference, plus one extra
        track per distinct voice a listener explicitly picked via SetVoicePreference.
        """
        translation = TranslationResultMessage.from_redis(data)
        text = translation.translated_text

        route_status = self._route_states.get(translation.meeting_id, "AUDIO_ROUTING_ACTIVE")
        if route_status == "PAUSED":
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - translation.timestamp_ms
        await self.redis.publish_telemetry(translation.meeting_id, self.worker_name, e2e_latency_ms)

        # S6. Never dub a listener back into the language the speaker is already speaking.
        # The listener is subscribed to BOTH the ai-interpreter track this would publish and
        # that speaker's raw mic, so synthesizing here plays the real voice and a synthetic
        # echo of the same sentence over each other.
        #
        # The producing side (translation_worker._get_target_languages) no longer builds
        # these messages, so in a healthy pipeline this never fires. It stays because this
        # worker is where the LiveKit track is actually published, and that is the last
        # place the echo can still be stopped: translate:results is a Redis stream that
        # outlives a deploy, so messages built by the previous revision are replayed into
        # this one, and any future producer inherits the guard for free rather than having
        # to remember it. Placed above the empty-text check so the final-chunk bookkeeping
        # below still runs — billing_worker and TranscriptRedisConsumerService key off
        # final_chunk_processed, and swallowing it would stall them on a silent segment.
        if is_same_language(translation.source_lang, translation.target_lang):
            self.logger.info(
                "same_language_synthesis_skipped",
                meeting_id=translation.meeting_id,
                speaker_id=translation.speaker_id,
                segment_id=translation.segment_id,
                lang=translation.target_lang,
            )
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

        if route_status == "TEXT_ONLY_MODE" or not text.strip():
            if translation.is_final_chunk:
                await self.redis.publish_system_event(
                    room_id=translation.meeting_id,
                    event_type="final_chunk_processed",
                    payload={"segmentId": translation.segment_id},
                )
            return

        variants = await self._resolve_voice_variants(
            translation.meeting_id, translation.speaker_id, translation.target_lang
        )

        # Once per message, not once per voice variant: the variants are the same sentence
        # rendered in different voices, so the second one must not be judged as arriving after
        # the first and read faster for it.
        lag_ms = self._catch_up_lag_ms(translation)
        if lag_ms > 0:
            self.logger.info(
                "dub_running_behind",
                meeting_id=translation.meeting_id,
                speaker_id=translation.speaker_id,
                target_lang=translation.target_lang,
                segment_id=translation.segment_id,
                lag_ms=lag_ms,
                start_ms=translation.start_ms,
                translation_latency_ms=translation.latency_ms,
            )

        for voice_id, voice_type, voice_key in variants:
            await self._synthesize_and_publish(
                translation, text, voice_id, voice_type, voice_key, lag_ms=lag_ms
            )

        # Exactly once per message regardless of how many voice variants rendered —
        # billing_worker/TranscriptRedisConsumerService key off this event, not off
        # per-variant synthesis.
        if translation.is_final_chunk:
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="final_chunk_processed",
                payload={"segmentId": translation.segment_id},
            )

    async def _resolve_voice_variants(
        self, meeting_id: str, speaker_id: str, target_lang: str
    ) -> list[tuple[str, str, str]]:
        """Every distinct (voice_id, voice_type, voice_key) this (speaker, target_lang)
        must be rendered into.

        Always includes exactly one "default" entry (voice_key="") — the speaker's own
        cloned voice if available, else a voice deterministically hashed from
        speaker_id out of that language's Cartesia catalog (so two un-cloned speakers
        dubbed into the same language sound different from each other by default,
        instead of both using Cartesia's single hardcoded fallback voice — the "A and B
        sound identical when they talk over each other" problem). This is the ONLY
        variant billed (see _synthesize_and_publish) and the only one with a backward-
        compatible LiveKit identity (ai-interpreter-{lang}-{speakerId}, unchanged).

        Plus — ONLY for a speaker who has no voice of their own — one extra entry per
        DISTINCT voice a listener explicitly chose via SetVoicePreference for this
        language (deduped: two listeners picking the same voice share one
        synthesis+track, same principle as _get_target_languages deduping identical
        listen-languages).

        VOICE IS ONE-DIRECTIONAL; ONLY THE LANGUAGE IS THE LISTENER'S
            Whose voice a dub is spoken in is the SPEAKER's decision, and a listener may
            not overrule it. What the listener chooses is which language they hear — and
            the same voice is rendered once per distinct target language, so A speaking
            Vietnamese with a cloned voice is heard by B in English IN A'S VOICE.

            It did not work that way. `_get_explicit_voice_choices` was applied to every
            speaker unconditionally, and the client accepts ONLY the preference track once
            a listener has one (see filtered-room-audio.tsx `dubbedSpeakerId`), so any
            listener who had ever picked a voice silently stopped hearing every cloned
            speaker in their own voice — while the speaker saw "My voice", watched the
            capture succeed, and had no way to learn it was being discarded.

            So the override now applies only where there is nothing of the speaker's to
            override: a speaker on the hashed catalogue default has expressed no
            preference about how they sound, and letting a listener pick a stand-in for
            them costs nobody anything.
        """
        # WT-396. The speaker's OWN choice wins over everything, including a voice cloned live in
        # this meeting: they went and picked one, and a live clone quietly overriding it is the
        # same class of bug as the pick never being read at all.
        #
        # Until now nothing read it. A person uploaded a recording of themselves, the UI listed
        # the profile as active, and the dub came back in a stock catalogue voice — because the
        # only voice this function ever looked for was one cloned from the meeting's microphone.
        chosen_voice_id = self.chosen_dub_voice(meeting_id, speaker_id)
        cloned_voice_id = (
            None if chosen_voice_id else await self._get_voice_id(meeting_id, speaker_id)
        )

        if chosen_voice_id:
            default_voice_id, default_voice_type = chosen_voice_id, "profile"
        elif cloned_voice_id:
            default_voice_id, default_voice_type = cloned_voice_id, "cloned"
        else:
            default_voice_id = await self._hashed_default_voice_id(
                target_lang, speaker_id, meeting_id
            )
            default_voice_type = "default"

        variants: list[tuple[str, str, str]] = [(default_voice_id, default_voice_type, "")]

        # The speaker owns how they sound. A clone or a deliberate pick is not a default
        # waiting to be overridden, and rendering a listener's alternative for one would be
        # paying Cartesia to throw the speaker's own voice away.
        if chosen_voice_id or cloned_voice_id:
            return variants

        explicit_choices = await self._get_explicit_voice_choices(meeting_id, target_lang)
        for voice_id in explicit_choices:
            if voice_id == default_voice_id:
                continue
            variants.append((voice_id, "preference", f"voice-{voice_id[:8]}"))

        return variants

    async def _get_voice_catalog(self, language: str) -> list[dict[str, Any]]:
        """Redis-cached (TTL) list of public Cartesia voices for a language.

        Falls back to [] on any cache/fetch problem — callers must fall back to
        CartesiaSynthesizer._default_voice_id() rather than fail synthesis.
        """
        cache_key = f"voice_catalog:{language}"
        cached = await self.redis.get(cache_key)
        if cached:
            try:
                raw = cached.decode() if isinstance(cached, bytes) else cached
                return cast(list[dict[str, Any]], json.loads(raw))
            except Exception:
                self.logger.warning("voice_catalog_cache_corrupt", language=language)

        voices = await self._require_cartesia().list_voices(
            language, limit=self.tts_settings.voice_catalog_size
        )
        if voices:
            await self.redis.set_with_ttl(
                cache_key, json.dumps(voices), self.tts_settings.voice_catalog_cache_ttl_seconds
            )
        return voices

    def _wav_duration_ms(self, audio_bytes: bytes) -> int:
        """How long this WAV actually plays for.

        Same arithmetic CartesiaSynthesizer.synthesize does on the bytes it just produced — 44
        bytes of header, the rest 16-bit mono PCM — so a cached rendering reports the identical
        duration to the render that filled the cache.

        Reads the sample rate off the synthesizer rather than assuming one: it is configurable,
        and a wrong rate here would misreport every cached line by the ratio between them.
        """
        cartesia = getattr(self, "cartesia", None)
        sample_rate = getattr(cartesia, "sample_rate", 0) or 44100
        pcm_bytes = max(0, len(audio_bytes) - 44)
        return int(pcm_bytes / 2 / sample_rate * 1000)

    @staticmethod
    def _persona_hash(speaker_id: str, salt: str = "") -> int:
        return int(hashlib.sha256(f"{salt}{speaker_id}".encode()).hexdigest(), 16)

    @classmethod
    def _persona_pool(cls, catalog: list[dict[str, Any]], speaker_id: str) -> list[dict[str, Any]]:
        """The voices this speaker may be heard in — the SAME kind of voice in every language.

        This is the half that fixes "Nhi hears him as female, I hear him as male". The choice of
        gender is made from the speaker id alone, so it is settled before any catalog is
        consulted and cannot vary with the listener's language.

        It does NOT claim to know the speaker's real gender — nothing upstream measures or asks
        for it, and guessing one from pitch is not something to do quietly. What it guarantees is
        that whatever they are assigned, everybody hears the same one. Matching a real gender
        needs a real signal (a declared preference, or their voice profile) and is a separate
        piece of work.

        Falls back to the whole catalog when a language offers only one gender, because there is
        then nothing to choose between and forcing the choice would just empty the pool.
        """
        by_gender: dict[str, list[dict[str, Any]]] = {}
        for voice in catalog:
            gender = str(voice.get("gender") or "").strip().lower()
            by_gender.setdefault(gender, []).append(voice)

        masculine = by_gender.get("masculine", [])
        feminine = by_gender.get("feminine", [])
        if not masculine or not feminine:
            return list(catalog)

        return masculine if cls._persona_hash(speaker_id, "gender:") % 2 == 0 else feminine

    @classmethod
    def _claim_voice(
        cls,
        catalog: list[dict[str, Any]],
        speaker_id: str,
        taken: set[str],
    ) -> dict[str, Any]:
        """This speaker's preferred voice, or the nearest free one after it."""
        pool = cls._persona_pool(catalog, speaker_id)
        start = cls._persona_hash(speaker_id) % len(pool)

        for offset in range(len(pool)):
            candidate = pool[(start + offset) % len(pool)]
            if str(candidate["id"]) not in taken:
                return candidate

        # Their own pool is exhausted. Spilling into the rest of the catalog keeps two people
        # distinguishable, which matters more than the gender being consistent for one of them.
        for offset in range(len(catalog)):
            candidate = catalog[(start + offset) % len(catalog)]
            if str(candidate["id"]) not in taken:
                return candidate

        # More speakers than the language has voices. Somebody has to share; the preference is
        # still deterministic, so at least it is the same sharing every time.
        return pool[start]

    @classmethod
    def _assign_voice(
        cls,
        catalog: list[dict[str, Any]],
        roster: list[str],
        speaker_id: str,
    ) -> dict[str, Any]:
        """Which voice `speaker_id` gets, given everybody else in the room. Pure and total.

        THE COLLISION THIS EXISTS TO PREVENT
            The previous rule was `catalog[hash(speaker) % len(catalog)]` with no reference to
            anyone else, so two speakers in a room of four routinely landed on one voice. In
            production on 18 Aug, Kỳ and Tú were assigned the SAME id in English (62ae83ad) AND
            the same id in Japanese (498e7f37) — every listener in either language heard two
            different people in one voice and could not tell who was speaking. In a conversation
            product that is worse than the wrong gender.

            Resolution walks the roster in sorted order and lets each speaker claim their
            preferred voice, probing forward when it is already taken. Deterministic, so every
            worker process and every language reaches the same answer without coordinating.

            A join can shift an assignment, but only for someone who both sorts after the joiner
            AND actually wanted the same voice. Stable assignment would need shared state; a rare
            change is the better trade against two people permanently sounding identical.
        """
        # Sorted so the answer does not depend on the order Cartesia's API happened to return,
        # nor on which worker warmed the cache.
        ordered_catalog = sorted(catalog, key=lambda voice: str(voice["id"]))

        taken: set[str] = set()
        for uid in roster:
            chosen = cls._claim_voice(ordered_catalog, uid, taken)
            if uid == speaker_id:
                return chosen
            taken.add(str(chosen["id"]))

        # Not on the roster — a speaker the languages hash has not caught up with. They still get
        # their own preference rather than nothing.
        return cls._claim_voice(ordered_catalog, speaker_id, set())

    async def _room_speaker_ids(self, meeting_id: str) -> set[str]:
        """Everyone the room currently knows about, from the same hash the language lookup uses."""
        try:
            raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:languages")
        except Exception:
            self.logger.warning("room_roster_unavailable", meeting_id=meeting_id, exc_info=True)
            return set()
        return {uid.decode() if isinstance(uid, bytes) else uid for uid in (raw or {})}

    async def _hashed_default_voice_id(
        self, language: str, speaker_id: str, meeting_id: str = ""
    ) -> str:
        """The stand-in voice this speaker is dubbed in, when they have no clone and no profile.

        Was `catalog[hash(speaker) % len(catalog)]`, which keyed the choice on the LISTENER's
        language — so one speaker had a different voice, and a different gender, for every
        listener — and ignored everyone else in the room, so two speakers collided routinely.
        See _persona_pool and _assign_voice for the two halves of the answer.
        """
        catalog = await self._get_voice_catalog(language)
        if not catalog:
            return CartesiaSynthesizer._default_voice_id(language)

        roster = await self._room_speaker_ids(meeting_id) if meeting_id else set()
        roster.add(speaker_id)
        return str(self._assign_voice(catalog, sorted(roster), speaker_id)["id"])

    async def _get_explicit_voice_choices(self, meeting_id: str, target_lang: str) -> set[str]:
        """Distinct voice_ids explicitly chosen (via TranslationRoomHub.
        SetVoicePreference) by listeners currently tuned to target_lang — cross-
        references the languages hash (who's listening in target_lang right now)
        against the voice_preferences hash (their chosen voice, if any). A listener
        who changes target_lang stops being counted here on their very next
        utterance, same as _get_target_languages already behaves for language itself.
        """
        languages_raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:languages")
        listeners_in_lang = {
            (uid.decode() if isinstance(uid, bytes) else uid)
            for uid, lang in (languages_raw or {}).items()
            # Listeners store whatever tag their picker gave them, so an exact match dropped
            # anyone whose choice was spelled "vi-VN" against a target of "vi" — and a
            # listener nobody counts is a listener nobody synthesises for.
            if is_same_language(lang.decode() if isinstance(lang, bytes) else lang, target_lang)
        }
        if not listeners_in_lang:
            return set()

        prefs_raw = await self.redis.hgetall(f"translationRoom:{meeting_id}:voice_preferences")
        choices: set[str] = set()
        for uid, voice_id in (prefs_raw or {}).items():
            user_id = uid.decode() if isinstance(uid, bytes) else uid
            if user_id not in listeners_in_lang:
                continue
            value = voice_id.decode() if isinstance(voice_id, bytes) else voice_id
            if value:
                choices.add(value)
        return choices

    async def _synthesize_and_publish(
        self,
        translation: TranslationResultMessage,
        text: str,
        voice_id: str,
        voice_type: str,
        voice_key: str,
        lag_ms: int = 0,
    ) -> None:
        # Catch-up is folded in HERE rather than inside _generation_config, because that method
        # answers "what did we measure about this speaker's delivery" and the answer is often
        # nothing. How far behind the dub is running is known either way, so it must be able to
        # produce a config where prosody produced None. See _with_catch_up.
        #
        # It also has to land before _cache_key below: the key already includes
        # generation_config, so a sped-up render and a normal one cannot collide, and a line
        # cached while the room was keeping up is not replayed at the wrong pace later.
        generation_config = self._with_catch_up(self._generation_config(translation), lag_ms)

        cache_key = self._cache_key(
            speaker_id=translation.speaker_id,
            target_lang=translation.target_lang,
            # Two different voice_ids must never share a cache entry even when
            # voice_type matches (e.g. two distinct "preference" picks) — the concrete
            # voice_id, not just the type, is part of what was actually rendered.
            text=text,
            voice_mode=f"{voice_type}:{voice_id}",
            # Same words, same voice, said differently is DIFFERENT AUDIO. Without this the
            # first rendering of a phrase would be replayed for every later one, and a speaker
            # who said "okay" calmly and then shouted it would be dubbed identically both
            # times — the cache would quietly undo the whole feature.
            generation_config=generation_config,
        )

        if self.tts_settings.cache_enabled:
            cached_audio = await self.redis.get(cache_key)
            if cached_audio:
                if voice_key:
                    # Extra voice variant — LiveKit only, never a second billing event
                    # for content already billed via the default variant's publish.
                    cached_bytes = (
                        cached_audio.encode("utf-8")
                        if isinstance(cached_audio, str)
                        else cached_audio
                    )
                    await self._publish_livekit_only(translation, cached_bytes, voice_key)
                else:
                    cached_bytes = (
                        cached_audio.encode("utf-8")
                        if isinstance(cached_audio, str)
                        else cached_audio
                    )
                    await self._publish_result(
                        translation=translation,
                        audio_bytes=cached_bytes,
                        # Derived from the audio, not hardcoded to 0. WT-528.
                        #
                        # A cache hit plays exactly as long as the render that filled the cache,
                        # so 0 was never true — it was "we did not bother to work it out", and it
                        # reached transcript.audio_dubbings.duration_ms as a fact. A third of the
                        # rows one production evening read 0 with status 'done', which is
                        # indistinguishable from audio that failed to synthesize, and it is what
                        # sent an investigation into the TTS pipeline looking for silence that
                        # was never there.
                        #
                        # It is also not cosmetic: _observe_dub_fit sums these to learn how much
                        # longer a dub runs than the speech it replaces, and isochrony centres
                        # every later sentence's speed on that fit. Feeding it zeros taught it
                        # that dubs take no time at all.
                        duration_ms=self._wav_duration_ms(cached_bytes),
                        voice_type=voice_type,
                        voice_key=voice_key,
                        provider_voice_id=voice_id,
                        cache_key=cache_key,
                        cache_hit=True,
                        # Genuinely zero — nothing was synthesized. Unlike the duration, this one
                        # is a real measurement of this call and 0 is the honest answer.
                        synthesis_latency_ms=0,
                    )
                return

        t0 = time.monotonic()
        try:
            sentence = await self._synthesize_sentence(
                translation=translation,
                text=text,
                voice_id=voice_id,
                voice_key=voice_key,
                generation_config=generation_config,
            )
        except Exception as e:
            # Carried the error and the voice and nothing else, so a failure could not be tied
            # to the sentence that failed: the one question worth asking of this line — WHICH
            # line went silent — was the one it could not answer.
            self.logger.error(
                "cartesia_synthesis_failed",
                error=str(e),
                meeting_id=translation.meeting_id,
                speaker_id=translation.speaker_id,
                segment_id=translation.segment_id,
                target_lang=translation.target_lang,
                text=text[:60],
                voice_type=voice_type,
            )
            await self.redis.publish_system_event(
                room_id=translation.meeting_id,
                event_type="tts_unavailable",
                payload={"error": str(e)},
            )
            return

        audio_bytes = sentence.audio
        duration_ms = sentence.duration_ms
        resolved_voice_id = sentence.voice_id
        already_spoken = sentence.already_spoken

        synthesis_latency_ms = int((time.monotonic() - t0) * 1000)
        # Already measured, and until now only ever attached to a published message. This is the
        # stage B2 clocked at p95 8.54s while STT and translation both stayed under 1.5s — kept
        # apart from the cumulative pipeline number so a slow Cartesia call and a queue building
        # behind the per-key lock are two readings rather than one.
        #
        # WT-397 CHANGED WHAT THIS COVERS. With streaming on, _synthesize_sentence does not
        # return until the audio has been handed to the track, which back-pressures to real
        # time — so this number now includes roughly the duration of the dub and is EXPECTED to
        # rise. It is still the right measure of "how long the worker was busy with this
        # sentence"; it is no longer a measure of how long anyone waited to hear it.
        await self.redis.record_latency("tts_synthesis", synthesis_latency_ms)
        if sentence.first_audio_at is not None:
            # What the listener actually experiences, and the only number that answers the
            # complaint this work came from. Same t0 as above, so the two are comparable.
            await self.redis.record_latency(
                "tts_first_audio", int((sentence.first_audio_at - t0) * 1000)
            )
        self._observe_dub_fit(translation, duration_ms)

        if audio_bytes:
            if voice_key:
                if not already_spoken:
                    await self._publish_livekit_only(translation, audio_bytes, voice_key)
            else:
                await self._publish_result(
                    translation=translation,
                    audio_bytes=audio_bytes,
                    duration_ms=duration_ms,
                    voice_type=voice_type,
                    voice_key=voice_key,
                    provider_voice_id=resolved_voice_id,
                    cache_key=cache_key,
                    cache_hit=False,
                    synthesis_latency_ms=synthesis_latency_ms,
                    # WT-397: streamed audio has already reached the track. tts:results still
                    # goes out below — billing and the transcript are driven by that message,
                    # not by the LiveKit push, and both must see every synthesized sentence.
                    publish_to_livekit=not already_spoken,
                )
            if self.tts_settings.cache_enabled:
                await self.redis.set_with_ttl(
                    cache_key, audio_bytes, self.tts_settings.cache_ttl_seconds
                )

        self.logger.info(
            "audio_synthesized",
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            # Closes the STT -> MT -> TTS chain: this is the id chunk_translated logged.
            segment_id=translation.segment_id,
            target_lang=translation.target_lang,
            voice_type=voice_type,
            voice_key=voice_key,
            # WHICH Cartesia voice actually spoke, not just whether we believe it was cloned.
            # voice_type is what this worker intended; this is what it used.
            provider_voice_id=resolved_voice_id,
            # THIS LINE IS LOGGED WHETHER OR NOT ANYTHING WAS SPOKEN. The publish above sits
            # behind `if audio_bytes:`, so a sentence Cartesia returned nothing for still
            # arrived here looking exactly like a success with duration_ms=0 — which is how a
            # dropped sentence stayed invisible while every other field said the pipeline was
            # healthy. `synthesized=False` is that case, stated rather than inferred.
            synthesized=bool(audio_bytes),
            # Streamed into the track during synthesis rather than pushed after it. Not a
            # failure — but it changes which publish branch ran, and therefore what to expect
            # on tts:results.
            already_spoken=already_spoken,
            duration_ms=duration_ms,
            synthesis_latency_ms=synthesis_latency_ms,
            text=text[:60],
            is_final=translation.is_final_chunk,
            # Empty when the speaker's delivery was not measured — which is what makes
            # "is prosody actually reaching Cartesia in production?" answerable from the logs
            # instead of by inspection.
            generation_config=generation_config or None,
        )

    async def _publish_result(
        self,
        translation: TranslationResultMessage,
        audio_bytes: bytes,
        duration_ms: int,
        voice_type: str,
        voice_key: str,
        provider_voice_id: str,
        cache_key: str,
        cache_hit: bool,
        synthesis_latency_ms: int,
        publish_to_livekit: bool = True,
    ) -> None:
        """Full publish: tts:results (billing_worker/TranscriptRedisConsumerService
        depend on this) + LiveKit track.

        ONLY called for the default/cloned variant (voice_key=""). An explicit-
        preference variant is a re-render of content already billed via this call for
        the same utterance — it must reach LiveKit (see _publish_livekit_only) but
        must NOT publish a second tts:results event, or billing_worker (which charges
        per tts:results message, keyed by segment_id+target_lang) would double-charge
        the workspace for one translated utterance just because a listener picked an
        alternate voice.
        """
        result = TTSResultMessage(
            segment_id=translation.segment_id,
            meeting_id=translation.meeting_id,
            speaker_id=translation.speaker_id,
            audio_data=audio_bytes,
            duration_ms=duration_ms,
            voice_type=voice_type,
            voice_mode=voice_type,
            clone_strength=1.0 if voice_type == "cloned" else 0.0,
            anchor_provider="cartesia",
            clone_provider="cartesia" if voice_type == "cloned" else "",
            provider_voice_id=provider_voice_id,
            render_location="server",
            cache_key=cache_key,
            cache_hit=cache_hit,
            synthesis_latency_ms=synthesis_latency_ms,
            # WT-396: "profile" is a voice the speaker CHOSE, so nothing fell back. Leaving the
            # old else-branch in place would stamp every deliberately picked voice
            # "voice_profile_not_ready" — the transcript would then report the exact opposite of
            # what happened, which is worse than reporting nothing.
            fallback_reason=(
                "" if voice_type in ("cloned", "profile") else "voice_profile_not_ready"
            ),
            target_lang=translation.target_lang,
            is_final_chunk=translation.is_final_chunk,
            timestamp_ms=translation.timestamp_ms,
        )
        await self.publish("tts:results", translation.meeting_id, result.to_redis())
        if publish_to_livekit:
            await self._publish_livekit_only(translation, audio_bytes, voice_key)

    async def _publish_livekit_only(
        self, translation: TranslationResultMessage, audio_bytes: bytes, voice_key: str
    ) -> None:
        """Push to this variant's LiveKit track only — no tts:results.

        Awaited (not fire-and-forget) so this key's _consume_loop lock genuinely
        covers the full publish, not just synthesis — otherwise a later sentence for
        the same key could start capturing to the track before an earlier one
        finishes, playing the dub back out of order. Safe to await unconditionally:
        publish_pcm() catches every internal failure itself and never raises.
        """
        pcm = audio_bytes[_WAV_HEADER_BYTES:] if len(audio_bytes) > _WAV_HEADER_BYTES else b""
        if pcm and self.livekit_publisher is not None:
            await self.livekit_publisher.publish_pcm(
                translation.meeting_id,
                translation.speaker_id,
                translation.target_lang,
                pcm,
                self.tts_settings.sample_rate,
                voice_key=voice_key,
            )

    async def _get_voice_id(self, meeting_id: str, speaker_id: str) -> str | None:
        """Return cached Cartesia voice_id for this speaker, or None.

        Re-checks consent on every call (not just before cloning) — if the speaker
        revokes voice clone consent mid-session, synthesis must fall back to the
        default voice immediately, even though a voice_id is still cached.
        """
        if not self.is_voice_clone_consented(meeting_id, speaker_id):
            return None
        cached = await self.redis.hget(f"voice:{meeting_id}:{speaker_id}", "voice_id")
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached
        # WT-B — a clone this speaker earned in an EARLIER meeting, from the route payload.
        #
        # Answered HERE rather than seeded into the Redis key at the first audio chunk, because
        # this is read by two independent consumers: the capture loop and the translation loop.
        # Seeding would leave a window between the first chunk and the write in which a dub had
        # already gone out in a stock voice — the exact twenty seconds of stranger's-voice this
        # feature exists to remove, merely made shorter.
        #
        # Below the cache on purpose. A clone made in THIS meeting is newer evidence about how
        # this person sounds today than one carried from a previous one, so live always wins.
        #
        # Behind the consent gate above, like everything else on this path: consent withdrawn
        # mid-meeting must drop the carried voice as immediately as it drops a live one.
        carried, _score = self.carried_clone(meeting_id, speaker_id)
        return carried

    async def _consume_upload_clone_requests(self) -> None:
        """Turn recordings people upload of themselves into provider voices (WT-396).

        THE BOUNDARY THIS EXISTS TO CROSS
            Cloning needs the Cartesia key, which only this side holds; the recording lives in a
            bucket only AuthService has credentials for. Neither half can do the other's part, so
            AuthService leaves the audio in Redis under `voice:clone_sample:{profile_id}` and a
            notice on this stream, and the answer goes back the same way.

            Before this, `CreateProfileAsync` ended at "bytes in a bucket, row marked active".
            Nothing anywhere could make a voice out of them, so an uploaded profile was listed as
            ready and every dub still came back in a stock catalogue voice.

        AN ANSWER IS ALWAYS WRITTEN, INCLUDING FOR FAILURE
            A missing answer is indistinguishable from one still being worked on, and AuthService
            renders both as "not usable yet" forever. A named failure is what lets the page say
            the recording could not be turned into a voice.
        """
        while self._running:
            try:
                async for msg_id, data in self.redis.consume(
                    stream=_CLONE_REQUEST_STREAM,
                    group=self._clone_request_group,
                    consumer=self._consumer_name,
                    block_ms=5000,
                    count=1,
                ):
                    await self._handle_upload_clone_request(data)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("upload_clone_consumer_error")
                await asyncio.sleep(1.0)

    async def _handle_upload_clone_request(self, data: dict[bytes, bytes]) -> None:
        profile_id = _decode_field(data, "profile_id")
        language = _decode_field(data, "language") or "en"
        if not profile_id:
            return

        sample_key = f"{_CLONE_SAMPLE_PREFIX}{profile_id}"
        result_key = f"{_CLONE_RESULT_PREFIX}{profile_id}"

        async def answer(voice_id: str | None, error: str | None) -> None:
            # Seven days, because somebody may upload and not open the page for a while, and
            # losing the id would mean paying Cartesia again for a voice we already made.
            await self.redis.set_with_ttl(
                result_key,
                json.dumps({"voiceId": voice_id, "provider": "cartesia", "error": error}),
                _CLONE_RESULT_TTL_SECONDS,
            )

        try:
            sample = await self.redis.get(sample_key)
            if not sample:
                # The audio outlived by its TTL, or the request was replayed after the sample was
                # collected. Said plainly rather than left pending forever.
                await answer(None, "the uploaded recording was no longer available to clone")
                return

            audio = sample.encode("utf-8") if isinstance(sample, str) else sample
            synthesizer = self._require_cartesia()
            voice_id = await synthesizer.clone_voice(
                audio,
                speaker_label=f"{_UPLOAD_VOICE_PREFIX}{profile_id[:8]}",
                language=_clone_language(language),
            )
            await answer(voice_id, None)
            self.logger.info(
                "uploaded_voice_cloned", profile_id=profile_id, bytes=len(audio), language=language
            )
        except Exception as exc:
            self.logger.exception("uploaded_voice_clone_failed", profile_id=profile_id)
            await answer(None, str(exc)[:200])
        finally:
            # The bytes are biometric data and there is no reason to keep them once we are done
            # with them, whichever way it went. The TTL is the backstop, not the plan.
            try:
                await self.redis.delete(sample_key)
            except Exception:
                self.logger.debug("clone_sample_delete_failed", profile_id=profile_id)

    async def _consume_preview_requests(self) -> None:
        """Render "what will I sound like?" samples for the voice-profiles page.

        WHY THIS SIDE RENDERS IT
            The same boundary the clone hand-off crosses: the Cartesia key is confined here, so
            AuthService can offer a play button but cannot produce the audio behind it.

        WHY IT MUST GO THROUGH synthesize() AND NOT A SIMPLER CALL
            The preview is only worth anything if it is the SAME rendering the meeting would
            produce. `CartesiaSynthesizer.synthesize` carries `speed="fast"`, which is not a
            default — it is a deliberate choice made because a dub has to fit the gap the speaker
            left, and it audibly changes the result. A preview rendered at Cartesia's normal speed
            would be a fair sample of a voice this product never plays.

            `generation_config` is passed as None, and that is correct rather than a shortcut: it
            is built from the speaker's measured prosody, there is no speaker here, and
            `_generation_config` already returns None for any utterance prosody could not be
            measured on. A preview is exactly that case, so it matches a real dub of one.
        """
        while self._running:
            try:
                async for _msg_id, data in self.redis.consume(
                    stream=_PREVIEW_REQUEST_STREAM,
                    group=self._preview_request_group,
                    consumer=self._consumer_name,
                    block_ms=5000,
                    count=1,
                ):
                    await self._handle_preview_request(data)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("preview_consumer_error")
                await asyncio.sleep(1.0)

    async def _handle_preview_request(self, data: dict[bytes, bytes]) -> None:
        voice_id = _decode_field(data, "voice_id")
        language = base_language(_decode_field(data, "language") or "en")
        if not voice_id:
            return

        result_key = f"{_PREVIEW_RESULT_PREFIX}{voice_id}:{language}"

        async def answer(audio: bytes | None, error: str | None) -> None:
            # An answer is always written, including for failure — the same rule the clone
            # hand-off follows. A missing key and a key that has not been written yet look
            # identical to the waiting request, so silence would render as "still loading"
            # until it timed out, for every retry, forever.
            await self.redis.set_with_ttl(
                result_key,
                json.dumps(
                    {
                        "audio": base64.b64encode(audio).decode("ascii") if audio else None,
                        "error": error,
                    }
                ),
                _PREVIEW_RESULT_TTL_SECONDS,
            )

        try:
            text = _PREVIEW_TEXT.get(language) or _PREVIEW_TEXT["en"]
            audio_bytes, duration_ms, _resolved = await self._require_cartesia().synthesize(
                text, language, voice_id
            )
            if not audio_bytes:
                await answer(None, "the provider returned no audio for this voice")
                return
            await answer(audio_bytes, None)
            self.logger.info(
                "voice_preview_rendered",
                voice_id=voice_id,
                language=language,
                duration_ms=duration_ms,
                bytes=len(audio_bytes),
            )
        except Exception as exc:
            self.logger.exception("voice_preview_failed", voice_id=voice_id, language=language)
            # Truncated because it goes on the wire to a person pressing a play button, and a
            # Cartesia stack trace is not a message for one.
            await answer(None, str(exc)[:200])

    async def _sweep_orphan_voices(self) -> None:
        """Delete in-meeting clones from the Cartesia account once nothing can reach them.

        WHAT LEAKS, AND WHY IT NEVER STOPPED
            `_clone_and_cache` creates a real voice in the account and records it at
            `voice:{meeting}:{speaker}` with a 12h TTL. When that key expires the voice is
            unreachable — and it is also still there, because nothing in this repository has
            ever called `voices.delete`. Every meeting leaked one voice per cloned speaker, and
            every upgrade (`voice_clone_max_upgrades`) leaked the one it replaced, mid-meeting,
            without even that pointer surviving to name it.

        WHY A SWEEP AND NOT A DELETE WHEN THE MEETING ENDS
            Three reasons, in order of how much they matter.

            A sweep collects what has ALREADY leaked. An end-of-meeting hook only ever stops
            the bleeding from now on, and the account is the state it is in.

            A sweep cannot delete a voice somebody is still speaking through. The end-of-meeting
            path runs off the route-status broadcast, which arrives on every replica and can
            arrive for a room that a slower replica is mid-sentence on; getting that wrong takes
            a live dub down, whereas getting the sweep wrong keeps a voice a few hours too long.

            A sweep has no bookkeeping to lose. Deleting on the way out means tracking every id
            created for a room, and a replica that dies takes its list with it — the leak this
            method exists to close, reintroduced one level up.

        HOW A VOICE IS JUDGED GARBAGE
            Its name and its age, and nothing else, because this service has no database. See
            `_IN_MEETING_VOICE_PREFIX`. An upload-made voice is named for its profile and is
            never touched here; anything else in the account was not made by this worker at all
            and is likewise left alone, which is the safe reading of a name we do not recognise.
        """
        interval = self.tts_settings.orphan_voice_sweep_interval_seconds
        while not self._shutdown_event.is_set():
            try:
                await self._sweep_orphan_voices_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let account housekeeping stop the worker. The next cycle retries, and
                # the failure mode of skipping one is the state this method started from.
                self.logger.exception("orphan_voice_sweep_failed")

            # Waiting on the shutdown event rather than sleeping it out, so a deploy is not held
            # up by however much of a six-hour interval happens to be left.
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=interval)
            except TimeoutError:
                pass

    async def _sweep_orphan_voices_once(self) -> None:
        """One pass. Returns quietly when another replica already has this cycle."""
        interval = self.tts_settings.orphan_voice_sweep_interval_seconds
        if not await self.redis.set_if_absent(
            _ORPHAN_SWEEP_LOCK_KEY, self._consumer_name, interval
        ):
            return

        cutoff = datetime.now(UTC) - timedelta(
            seconds=self.tts_settings.orphan_voice_min_age_seconds
        )
        synthesizer = self._require_cartesia()
        voices = await synthesizer.list_owned_voices()

        deleted = 0
        failed = 0
        too_young = 0
        for voice in voices:
            if not str(voice.get("name") or "").startswith(_IN_MEETING_VOICE_PREFIX):
                continue

            created_at = voice.get("created_at")
            if not isinstance(created_at, datetime):
                # No age means no way to know it is unreachable. Keeping a voice costs storage;
                # deleting one that a meeting is still speaking through costs the meeting.
                too_young += 1
                continue
            # A naive timestamp from the vendor is UTC — comparing it against an aware `cutoff`
            # raises rather than mis-sorting, so it is normalized rather than left to chance.
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if created_at > cutoff:
                too_young += 1
                continue

            if await synthesizer.delete_voice(str(voice["id"])):
                deleted += 1
            else:
                failed += 1

        self.logger.info(
            "orphan_voice_sweep_completed",
            owned=len(voices),
            deleted=deleted,
            failed=failed,
            too_young=too_young,
            min_age_seconds=self.tts_settings.orphan_voice_min_age_seconds,
        )

    async def _consume_audio_for_cloning(self) -> None:
        """Buffer raw audio per speaker; clone voice once enough is collected."""
        # {(meeting_id, speaker_id): accumulated_audio_bytes}
        buffers: dict[tuple[str, str], bytearray] = {}
        buffer_seconds: dict[tuple[str, str], float] = {}
        buffer_lang: dict[tuple[str, str], str] = {}
        # WT-371 #9: what the clone currently in use was built from, so a later clip can be
        # recognised as better. Absent until this speaker has been cloned in this process.
        cloned_score: dict[tuple[str, str], float] = {}
        upgrades_used: dict[tuple[str, str], int] = {}
        # WT-B: speakers whose carried-over clone has already been looked up. The lookup is a
        # read of the route snapshot rather than of Redis, but it must happen exactly once per
        # speaker: doing it per chunk would re-seed the bar after an upgrade had raised it and
        # walk the score back down to whatever the previous meeting managed.
        carried_seen: set[tuple[str, str]] = set()

        while self._running:
            try:
                async for _msg_id, data in self.redis.consume(
                    stream="audio:chunks",
                    group=self._audio_consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=5,
                ):
                    try:
                        chunk = AudioChunkMessage.from_redis(data)
                        key = (chunk.meeting_id, chunk.speaker_id)

                        # Consent gate: never buffer/clone a speaker's voice (biometric
                        # data) unless they have at least one current outgoing route with
                        # VoiceCloneEnabled = true. See base_worker.is_voice_clone_consented.
                        #
                        # Asked in the form that can recover the routes from Redis and that
                        # reports WHICH no it is: production ran with zero cloned voices and
                        # every dub on the default catalog voice, and this branch — the one that
                        # swallows the whole meeting — said nothing at all on the way past.
                        consented, consent_reason = await self.voice_clone_consent_state(
                            chunk.meeting_id, chunk.speaker_id
                        )
                        if not consented:
                            await self._note_clone_state(key, consent_reason)
                            buffers.pop(key, None)
                            buffer_seconds.pop(key, None)
                            buffer_lang.pop(key, None)
                            cloned_score.pop(key, None)
                            upgrades_used.pop(key, None)
                            # Discarded with the rest, so granting consent again re-seeds the
                            # carried bar. Left behind, a speaker who toggled the switch off and
                            # on would spend the remainder of the meeting with no bar at all and
                            # re-clone on their next acceptable clip.
                            carried_seen.discard(key)
                            continue

                        # WT-B: adopt the bar a previous meeting's clone set, once per speaker.
                        #
                        # Without this the carried voice is used but never measured against, so
                        # `previous_score is None` makes the very first acceptable clip of the
                        # meeting "worth cloning" — every meeting would re-clone at once, throw
                        # away a voice that may well have been better, and burn an upgrade doing
                        # it. Seeding the score is what turns the carried voice into something
                        # a new clip has to actually BEAT.
                        #
                        # Only the score is seeded. `upgrades_used` stays at zero deliberately:
                        # the budget is per meeting, and carrying it over would mean a speaker
                        # whose clone was replaced once could never improve it again.
                        if key not in carried_seen:
                            carried_seen.add(key)
                            carried_voice, carried_score = self.carried_clone(
                                chunk.meeting_id, chunk.speaker_id
                            )
                            if carried_voice and carried_score is not None:
                                cloned_score[key] = carried_score
                            if carried_voice:
                                await self._note_clone_state(
                                    key, "carried_over", score=carried_score
                                )

                        # WT-371 #9: this used to be `if already cloned: continue` — the worker
                        # stopped listening the moment it had any clone at all, so the voice was
                        # locked to whatever register the speaker opened the meeting in. Change
                        # your tone, or crack your voice, and the clone stopped being you.
                        #
                        # It keeps listening now, but only while an upgrade is still allowed, so a
                        # speaker whose clone is already good costs nothing beyond the buffer.
                        if await self._get_voice_id(chunk.meeting_id, chunk.speaker_id):
                            if (
                                upgrades_used.get(key, 0)
                                >= self.tts_settings.voice_clone_max_upgrades
                            ):
                                # Carries the score it settled on: this is the state a speaker
                                # is STUCK in, so it is the one that most needs to say whether
                                # the voice they are stuck with is a good likeness or a weak
                                # one. `cloned_elsewhere_kept` below stays scoreless because it
                                # genuinely has none — that clone was made by another replica.
                                await self._note_clone_state(
                                    key,
                                    "cloned_upgrades_exhausted",
                                    score=cloned_score.get(key),
                                )
                                buffers.pop(key, None)
                                buffer_seconds.pop(key, None)
                                buffer_lang.pop(key, None)
                                continue
                            # A clone made by ANOTHER replica, or before this process started, has
                            # no local score. Treat it as good enough to keep rather than racing to
                            # replace something we cannot compare against.
                            if key not in cloned_score:
                                await self._note_clone_state(key, "cloned_elsewhere_kept")
                                continue

                        # Stop measuring once no upgrade can possibly be earned.
                        #
                        # `worth_cloning` below is `score >= previous + upgrade_margin`, and
                        # `score` is capped at MAX_SAMPLE_SCORE by construction. A speaker whose
                        # clip scored 1.0 therefore needs 1.15 to improve on it, which no clip can
                        # ever reach — the comparison is unsatisfiable, not merely unlikely.
                        #
                        # Both production speakers in meeting 01a00547 scored exactly 1.0, so this
                        # is the ordinary case rather than an edge one. What happened next: the
                        # buffer kept growing (20.4 → 70.8 seconds before the 90s cap would have
                        # slid it) and `assess_clone_sample` re-ran over the WHOLE buffer on every
                        # single chunk — an FFT per 40ms frame, so ~3,500 of them per chunk per
                        # speaker at 70 seconds — to re-derive a verdict that was already decided.
                        #
                        # Dropping the buffer here is not a lost opportunity: there is nothing left
                        # to find. It is reported rather than done silently, for the same reason
                        # every other exit on this path is.
                        best_so_far = cloned_score.get(key)
                        if (
                            best_so_far is not None
                            and best_so_far + self.tts_settings.voice_clone_upgrade_margin
                            > MAX_SAMPLE_SCORE
                        ):
                            await self._note_clone_state(
                                key, "cloned_best_possible", score=best_so_far
                            )
                            buffers.pop(key, None)
                            buffer_seconds.pop(key, None)
                            buffer_lang.pop(key, None)
                            continue

                        buffers.setdefault(key, bytearray()).extend(chunk.audio_data)
                        # PCM 16-bit mono: 2 bytes per sample
                        duration_s = len(chunk.audio_data) / 2 / max(chunk.sample_rate, 1)
                        buffer_seconds[key] = buffer_seconds.get(key, 0.0) + duration_s
                        buffer_lang[key] = chunk.language

                        # WT-420: the bar needs something to fill with. Nothing before this
                        # reported that capture was even happening — "ủa nó ko tự thu hở" was the
                        # reasonable conclusion, and it was wrong the whole time.
                        await self._note_clone_state(
                            key,
                            "capturing",
                            seconds=buffer_seconds[key],
                            required_seconds=float(self.tts_settings.voice_clone_min_seconds),
                        )

                        if buffer_seconds[key] >= self.tts_settings.voice_clone_min_seconds:
                            # The clip is only a reference if it is worth referring to.
                            #
                            # This used to clone the first N seconds unconditionally, and
                            # _get_voice_id short-circuits, so a microphone check became the
                            # speaker's voice for the entire meeting. Now a clip that fails the
                            # same bar the upload page enforces is not cloned — the oldest audio
                            # slides out and the speaker gets another go, which costs nothing
                            # because they are still talking.
                            assessment = assess_clone_sample(bytes(buffers[key]), chunk.sample_rate)
                            previous_score = cloned_score.get(key)
                            is_upgrade = previous_score is not None
                            # An upgrade has to EARN the disruption: re-cloning changes the voice
                            # people are currently listening to, and small score differences are
                            # noise in the pitch estimator rather than a better reference.
                            if not assessment.accepted:
                                worth_cloning = False
                                # The last silent exit on this path, and the one that swallowed a
                                # whole meeting.
                                #
                                # WT-405 follow-up. Meeting 01a0033f: two speakers, four minutes,
                                # 46 dubbed sentences, every one on a stock catalog voice — and
                                # not a single clone-related line in the log. Consent had passed,
                                # so the gate above said nothing; the clip was refused here, which
                                # also said nothing; and the buffer slid and tried again, forever.
                                # From outside it is indistinguishable from cloning being switched
                                # off, which is exactly the report we got.
                                #
                                # The reason is the point. "too quiet" is a microphone, "too little
                                # speech" is a room, "clipped" is a gain setting — three different
                                # conversations with the user, and none of them can start from
                                # silence. Prefixed so a threshold that turns out to be wrong is
                                # tuned against measurements rather than guessed at.
                                await self._note_clone_state(
                                    key,
                                    f"clip_rejected:{assessment.reason}",
                                    active_speech_ratio=assessment.active_speech_ratio,
                                )
                            elif previous_score is None:
                                worth_cloning = True
                            else:
                                worth_cloning = (
                                    assessment.score
                                    >= previous_score + self.tts_settings.voice_clone_upgrade_margin
                                )
                            if worth_cloning:
                                audio_snapshot = bytes(buffers.pop(key))
                                del buffer_seconds[key]
                                clone_lang = _clone_language(buffer_lang.pop(key, "en"))
                                cloned_score[key] = assessment.score
                                if is_upgrade:
                                    upgrades_used[key] = upgrades_used.get(key, 0) + 1
                                self.logger.info(
                                    "voice_clone_sample_accepted",
                                    speaker_id=chunk.speaker_id,
                                    seconds=round(
                                        buffer_seconds.get(key, 0.0)
                                        or self.tts_settings.voice_clone_min_seconds,
                                        1,
                                    ),
                                    active_speech_ratio=round(assessment.active_speech_ratio, 3),
                                    pitch_semitones=round(assessment.pitch_semitone_range, 2),
                                    score=round(assessment.score, 3),
                                    upgrade=is_upgrade,
                                )
                                await self._note_clone_state(
                                    key,
                                    "cloning",
                                    score=assessment.score,
                                    active_speech_ratio=assessment.active_speech_ratio,
                                )
                                asyncio.create_task(
                                    self._clone_and_cache(
                                        chunk.meeting_id,
                                        chunk.speaker_id,
                                        audio_snapshot,
                                        clone_lang,
                                        # The rate the buffer was actually captured at. The WAV
                                        # header _clone_and_cache writes is only correct if it
                                        # matches, and a header that lies about the rate is how a
                                        # clone comes back chipmunked rather than refused.
                                        chunk.sample_rate,
                                        assessment.score,
                                    )
                                )
                            elif assessment.accepted:
                                # Usable, but no better than the clone already in use. Slide the
                                # window on and keep listening — the speaker may yet say something
                                # that covers more of their range.
                                self._trim_clone_buffer(
                                    key, buffers, buffer_seconds, chunk.sample_rate
                                )
                            else:
                                # Logged at info, not warning: a rejected clip is the gate doing
                                # its job, and a speaker who has not said anything usable yet is
                                # an ordinary state, not a fault.
                                self.logger.info(
                                    "voice_clone_sample_rejected",
                                    speaker_id=chunk.speaker_id,
                                    reason=assessment.reason,
                                    rms=round(assessment.rms, 4),
                                    active_speech_ratio=round(assessment.active_speech_ratio, 3),
                                )
                                self._trim_clone_buffer(
                                    key, buffers, buffer_seconds, chunk.sample_rate
                                )
                    except Exception:
                        self.logger.exception("audio_chunk_processing_error")
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("audio_consumer_error")
                await asyncio.sleep(2)

    async def _note_clone_state(
        self,
        key: tuple[str, str],
        reason: str,
        *,
        seconds: float | None = None,
        required_seconds: float | None = None,
        score: float | None = None,
        active_speech_ratio: float | None = None,
    ) -> None:
        """Say why this speaker is not on a cloned voice — once per change, not once per chunk.

        Audio chunks arrive continuously for the whole meeting, so logging at each of these
        branches directly would bury the pipeline in one line per chunk per speaker and get the
        level turned back down within a day. Only a CHANGE is news.

        This exists because production ran with `voice:*` empty and 97 of 97 dubbed segments on
        `voice_type=default`, and not one line in any log said why. Every exit before the clone
        call returned in silence, so the difference between "nobody opted in" and "this worker
        never learned the room's routes" was invisible from outside.

        WT-420: it now also PUBLISHES, because a log solved the wrong half of the problem. On
        15 Aug the whole team tried to hear a cloned voice, could not, and reported cloning as
        broken — while this method was writing `score: 1.0` into a log nobody in a meeting can
        read. Every one of these reasons is something the person at the microphone needs, and
        the worker is the only place that knows it.

        The stream carries the same facts as the log line, so the two cannot drift into
        disagreeing about the same speaker.
        """
        if getattr(self, "_clone_state", None) is None:
            self._clone_state: dict[tuple[str, str], str] = {}

        # Progress is bucketed into the dedupe key rather than excluded from it: `capturing`
        # changes on every chunk, so deduping on the reason alone would emit it once and freeze
        # the bar at its first value, while deduping on nothing would publish per chunk per
        # speaker — the exact volume this method exists to avoid.
        fingerprint = reason if seconds is None else f"{reason}:{int(seconds)}"
        if self._clone_state.get(key) == fingerprint:
            return
        self._clone_state[key] = fingerprint

        self.logger.info(
            "voice_clone_state",
            meeting_id=key[0],
            speaker_id=key[1],
            reason=reason,
        )

        payload: dict[str, Any] = {
            "meeting_id": key[0],
            "speaker_id": key[1],
            "reason": reason,
        }
        if seconds is not None:
            payload["seconds"] = round(seconds, 1)
        if required_seconds is not None:
            payload["required_seconds"] = round(required_seconds, 1)
        if score is not None:
            payload["score"] = round(score, 3)
        if active_speech_ratio is not None:
            payload["active_speech_ratio"] = round(active_speech_ratio, 3)

        try:
            await self.publish("voice:clone:state", key[0], payload)
        except Exception:
            # Never let telemetry take the clone path down with it. A speaker whose progress bar
            # stalls is a worse UI; a speaker whose voice stops being cloned because a Redis
            # write failed is a worse product.
            self.logger.warning("voice_clone_state_publish_failed", meeting_id=key[0])

    def _trim_clone_buffer(
        self,
        key: tuple[str, str],
        buffers: dict[tuple[str, str], bytearray],
        buffer_seconds: dict[tuple[str, str], float],
        sample_rate: int,
    ) -> None:
        """Drop the oldest audio so a rejected clip does not block the next attempt forever.

        A sliding window, not a reset. Resetting would throw away the speech that arrived while
        the clip was being judged, and in a room where every window is marginal it would mean
        never assembling a usable one. Keeping everything is the other failure: a speaker in a
        noisy office would hold the whole meeting in memory and still never clone.
        """
        buffer = buffers.get(key)
        if buffer is None or sample_rate <= 0:
            return

        max_bytes = int(self.tts_settings.voice_clone_max_buffer_seconds * sample_rate * 2)
        if max_bytes <= 0 or len(buffer) <= max_bytes:
            return

        overflow = len(buffer) - max_bytes
        del buffer[:overflow]
        buffer_seconds[key] = len(buffer) / 2 / sample_rate

    async def _clone_and_cache(
        self,
        meeting_id: str,
        speaker_id: str,
        audio_bytes: bytes,
        language: str = "en",
        sample_rate: int = 16000,
        score: float | None = None,
    ) -> None:
        """Clone voice via Cartesia and cache voice_id in Redis.

        `score` is the accepted clip's quality, carried through only so the terminal `cloned`
        state can publish it. Acceptance and quality are two different questions here:
        `assess_clone_sample` rejects on hard floors (level, speech ratio, energy variation),
        while a narrow or monotone delivery clears every floor and comes back with a LOW score
        on purpose — see clone_sample_quality's preamble. Without this the success state went
        out scoreless and the meeting UI could only say "Your voice is ready", in the same
        words, for a clip that covers the speaker's whole range and one that barely covers a
        note of it. "không phải cứ nói vào là ready đâu mà phải có bộ lọc" — the filter ran; it
        just had no way to say what it found.

        `audio_bytes` is the RAW PCM the clone buffer accumulated — headerless 16-bit mono
        samples, exactly what `assess_clone_sample` reads with `np.frombuffer(..., int16)`.
        Cartesia's /voices/clone takes an audio FILE, so it is wrapped in a RIFF header here.

        THIS IS WHY VOICE CLONING HAD NEVER RUN IN PRODUCTION. `voices.clone()` was being handed
        a naked PCM stream and could not decode it, so every in-meeting clone request failed at
        the vendor — 2153 dubs on record, every one of them `voice_type=default`, not a single
        `cloned` row ever written. And it failed INVISIBLY: this method only logged, so from the
        outside it was indistinguishable from cloning being switched off. The quality gate, the
        consent gate and the upgrade logic were all working perfectly the whole time; the clip
        that reached Cartesia simply was not a file.

        The upload path (`_handle_upload_clone_request`) never hit this because the bytes it
        sends come straight from a user-uploaded recording, which already has a container.
        """
        label = f"{_IN_MEETING_VOICE_PREFIX}{speaker_id[:8]}-{meeting_id[:8]}"
        key = (meeting_id, speaker_id)
        try:
            voice_id = await self._require_cartesia().clone_voice(
                wav_header(len(audio_bytes), sample_rate) + audio_bytes,
                label,
                language,
            )
            cache_key = f"voice:{meeting_id}:{speaker_id}"
            await self.redis.hset(cache_key, "voice_id", voice_id)
            # hset has no TTL of its own — without this the key lives in Redis forever.
            await self.redis.expire(cache_key, self.tts_settings.voice_clone_key_ttl_seconds)
            self.logger.info(
                "voice_cached",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                voice_id=voice_id,
            )
            await self._note_clone_state(key, "cloned", score=score)
            await self.redis.publish_system_event(
                room_id=meeting_id,
                event_type="voice_clone_ready",
                payload={"speakerId": speaker_id, "voiceId": voice_id},
            )
            await self._offer_carry_over(speaker_id, language, voice_id, score)
        except Exception as e:
            self.logger.error(
                "voice_clone_failed",
                meeting_id=meeting_id,
                speaker_id=speaker_id,
                error=str(e),
            )
            # The last silent exit on this path, and the one that hid the bug above for the whole
            # life of the feature. Every OTHER branch in _consume_audio_for_cloning publishes its
            # reason (WT-420), so the clone-state stream showed `capturing` → `cloning` → nothing,
            # forever: from the outside, identical to a clone still in flight. A vendor refusal is
            # exactly the state the person at the microphone needs to see, and the only place that
            # knows it is here.
            #
            # Truncated because it goes on the wire to a UI, and a Cartesia stack trace is not a
            # message for a person in a meeting.
            await self._note_clone_state(key, f"clone_failed:{str(e)[:120]}")

    async def _offer_carry_over(
        self,
        speaker_id: str,
        language: str,
        voice_id: str,
        score: float | None,
    ) -> None:
        """Promote a just-made in-meeting clone and tell AuthService it exists (WT-B).

        THE ORDER IS THE WHOLE DESIGN. Rename first, publish only if the rename took.

        The orphan sweep decides what to delete by NAME, so until this voice is renamed it is
        garbage with a 24-hour fuse. Publishing first would hand AuthService a row pointing at
        something the sweep destroys tomorrow, and the failure would surface a week later as a
        person opening a meeting in a voice that no longer exists — with the row still there,
        still looking correct, naming an id Cartesia has never heard of. Renaming first means
        the worst case is the state we were already in: an ordinary in-meeting clone, swept on
        schedule, and the speaker re-clones next meeting exactly as they do today.

        SCORE MAY BE ABSENT AND MUST TRAVEL THAT WAY. It is the bar a later clip has to beat, and
        a missing bar is "not measured", never zero — a fabricated 0 grades as the worst possible
        sample and invites replacement by anything at all. Sent as an empty field so the far side
        stores NULL rather than parsing a number out of nothing.

        Never raises. A carry-over that does not happen costs the speaker a re-clone in their
        next meeting, which is today's behaviour; letting it throw would take down the clone that
        just succeeded.
        """
        try:
            promoted_name = f"{_UPLOAD_VOICE_PREFIX}{speaker_id[:8]}-{language}"
            if not await self._require_cartesia().rename_voice(voice_id, promoted_name):
                # Deliberately not an exception path: the clone itself is fine and in use for
                # this meeting. Only the carrying-over is lost, and the voice stays sweepable,
                # which is the state that keeps the account honest.
                self.logger.warning(
                    "voice_carry_over_skipped_rename_failed",
                    speaker_id=speaker_id,
                    voice_id=voice_id,
                )
                return

            await self.redis.publish(
                _CARRY_OVER_STREAM,
                {
                    # The speaker id IS the auth user id — the same value routes carry as
                    # SourceUserId, which is how `chosen_dub_voice` matches them.
                    "user_id": speaker_id,
                    "language": language,
                    "voice_id": voice_id,
                    "score": "" if score is None else f"{score:.3f}",
                },
            )
            self.logger.info(
                "voice_carry_over_offered",
                speaker_id=speaker_id,
                language=language,
                voice_id=voice_id,
                score=None if score is None else round(score, 3),
            )
        except Exception:
            self.logger.warning(
                "voice_carry_over_failed", speaker_id=speaker_id, voice_id=voice_id, exc_info=True
            )

    async def _consume_voice_delete_requests(self) -> None:
        """Destroy voices AuthService has decided must stop existing (WT-B).

        WHY THIS SIDE, AND WHY IT IS NOT THE SWEEP
            AuthService owns the row and this side owns the Cartesia key, so a deletion it
            decides on can only be carried out here. The sweep cannot serve this: it deletes by
            name and age, and every voice named here is a PROMOTED one the sweep is specifically
            built never to touch.

        THE TWO CALLERS
            Withdrawing voice-clone consent, which has to delete the voice model rather than
            merely stop using it — the consent text says the voice "stops being used", and while
            a clone died with its meeting that was the whole of the promise. Keeping the clone
            makes deletion the rest of it.

            And replacing a carried-over clone with a better one, which must destroy the loser
            or promotion becomes a leak in a place the sweep has been told to leave alone.
        """
        while self._running:
            try:
                async for _msg_id, data in self.redis.consume(
                    stream=_VOICE_DELETE_STREAM,
                    group=self._voice_delete_group,
                    consumer=self._consumer_name,
                    block_ms=5000,
                    count=10,
                ):
                    voice_id = _decode_field(data, "voice_id")
                    if not voice_id:
                        continue
                    # The answer is not reported back. A voice that will not delete is retried by
                    # nothing, and that is deliberate: the row is already gone on the far side, so
                    # there is nobody left to tell, and the sweep cannot reach a `profile-` name.
                    # It is logged loudly instead, because for the consent case it is the
                    # difference between a promise kept and a promise broken.
                    if await self._require_cartesia().delete_voice(voice_id):
                        self.logger.info("voice_deleted_on_request", voice_id=voice_id)
                    else:
                        self.logger.error(
                            "voice_delete_on_request_failed",
                            voice_id=voice_id,
                            reason=_decode_field(data, "reason") or "unspecified",
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("voice_delete_consumer_error")
                await asyncio.sleep(1.0)

    def _generation_config(
        self, translation: TranslationResultMessage
    ) -> dict[str, float | str] | None:
        """Cartesia's delivery controls for this line, or None to say nothing about delivery.

        None is the important case and it is the common one: the STT worker omits prosody
        whenever it could not honestly measure it (a speaker it has not heard enough of, a
        chunk that was mostly silence), and this returns None again whenever the feature is
        off. In every one of those cases the call is byte-for-byte the call this worker made
        before prosody existed.
        """
        if not self.tts_settings.prosody_enabled:
            return None

        envelope = translation.prosody
        if envelope is None:
            return None

        # Rebuilt rather than passed around as a Delivery: the wire format is deliberately
        # plain numbers (schemas.py knows nothing about numpy or about prosody's vocabulary),
        # and this is the one place that translates it back.
        arousal: Arousal = (
            envelope.arousal if envelope.arousal in ("low", "neutral", "high") else "neutral"  # type: ignore[assignment]
        )
        valence: Valence | None = (
            envelope.valence  # type: ignore[assignment]
            if envelope.valence in ("negative", "neutral", "positive")
            else None
        )

        return to_generation_config(
            Delivery(
                pitch_lift=envelope.pitch_lift,
                pitch_variation=envelope.pitch_variation,
                energy_ratio=envelope.energy_ratio,
                rate_ratio=envelope.rate_ratio,
                arousal=arousal,
            ),
            valence,
            # Isochrony. The centre this speaker's tempo is applied AROUND, learned from how
            # their previous dubs actually ran against the clock — see shared/isochrony.py.
            # Exactly 1.0 until a fit is established, which is byte-for-byte the previous
            # behaviour. Their own rate_ratio still multiplies through it, so somebody who
            # genuinely slowed down still sounds like they slowed down, inside a slot that fits.
            speed_center=isochrony.speed_center(self._dub_fit(translation)),
        )

    def _catch_up_lag_ms(self, translation: TranslationResultMessage) -> int:
        """How far behind the conversation this sentence is, in milliseconds. WT-528.

        Records this sentence's position as the furthest spoken for its key, so the NEXT one is
        judged against it. Called once per message, from process(), under that key's lock — which
        is what makes the read-then-write safe without a second lock of its own.

        Zero is the normal answer and it means "make no adjustment".
        """
        # getattr + assign back, matching `_turns`: the tests build workers with __new__ and
        # never run __init__, and this one is written to as well as read.
        timeline: dict[tuple[str, str, str], int] = getattr(self, "_spoken_start_ms", None) or {}
        self._spoken_start_ms = timeline

        key = self._fit_key(translation)
        spoken = timeline.get(key)

        # Out of order: a sentence that belongs before one already spoken. Its whole gap is lag,
        # because everything between the two has already been said and it is arriving into a
        # conversation that has moved on.
        out_of_order_ms = 0 if spoken is None else max(0, spoken - translation.start_ms)

        # Simply slow. Only the part of the translation that overran the budget counts: a sentence
        # inside it was never behind, and charging the whole duration as lag would speed up every
        # dub in a healthy meeting.
        overrun_ms = max(0, (translation.latency_ms or 0) - _CATCH_UP_ON_TIME_MS)

        # The larger, not the sum. They are two measurements of the same delay from different
        # ends — a slow translation is usually also what pushed the sentence out of order — and
        # adding them would double-count it.
        timeline[key] = max(spoken or 0, translation.start_ms)
        return max(out_of_order_ms, overrun_ms)

    @staticmethod
    def _with_catch_up(
        generation_config: dict[str, float | str] | None, lag_ms: int
    ) -> dict[str, float | str] | None:
        """Raise the delivery speed in proportion to how far behind the dub is. WT-528.

        Returns the config UNCHANGED when there is no lag, including None — a meeting that is
        keeping up makes exactly the Cartesia call it made before this existed, and prosody's own
        rule that silence is the honest thing to say about unmeasured delivery is preserved.

        When there IS lag the config may be created from nothing, because being behind is a fact
        about this dub that is known regardless of whether the speaker's prosody was measurable.
        """
        if lag_ms <= 0:
            return generation_config

        ramp = min(1.0, lag_ms / _CATCH_UP_FULL_LAG_MS)
        multiplier = 1.0 + ramp * (_CATCH_UP_MAX_SPEED - 1.0)

        base = dict(generation_config) if generation_config else {}
        # Multiplies whatever speed was already decided — the speaker's own measured rate through
        # isochrony's centre — rather than replacing it, so a slow talker who is behind still
        # sounds like a slow talker, just less far behind.
        current_speed = base.get("speed", 1.0)
        speed = min(SPEED_MAX, float(cast(float, current_speed)) * multiplier)
        base["speed"] = round(speed, 3)
        return base

    def _fit_key(self, translation: TranslationResultMessage) -> tuple[str, str, str]:
        """Fit is per (meeting, speaker, target language). Not global: how much longer a dub
        runs is a property of the language pair and of how this person talks, and pooling a
        terse speaker with a discursive one would centre both on neither."""
        return (translation.meeting_id, translation.speaker_id, translation.target_lang)

    def _dub_fit(self, translation: TranslationResultMessage) -> isochrony.DubFit:
        fits: dict[tuple[str, str, str], isochrony.DubFit] = getattr(self, "_dub_fits", {})
        return fits.get(self._fit_key(translation), isochrony.NO_FIT)

    def _observe_dub_fit(self, translation: TranslationResultMessage, dub_ms: int) -> None:
        """Accumulate this sentence's dub, and compare the WHOLE turn once the turn is over.

        The comparison is turn against turn. `start_ms`/`end_ms` describe the whole spoken turn,
        so weighing one sentence's dub against them would report a fit of about 1/N for an
        N-sentence turn and drive the controller to speak everybody faster and faster. The
        sentences are summed and the total is what gets folded in on `is_final_chunk`.
        """
        if not self.tts_settings.prosody_enabled:
            return

        key = self._fit_key(translation)
        pending: dict[tuple[str, str, str], int] = getattr(self, "_turn_dub_ms", None) or {}
        self._turn_dub_ms = pending
        pending[key] = pending.get(key, 0) + max(0, dub_ms)

        if not translation.is_final_chunk:
            return

        turn_dub_ms = pending.pop(key, 0)
        source_ms = translation.end_ms - translation.start_ms
        if source_ms <= 0:
            return

        fits: dict[tuple[str, str, str], isochrony.DubFit] = getattr(self, "_dub_fits", None) or {}
        self._dub_fits = fits
        fits[key] = isochrony.observe(fits.get(key, isochrony.NO_FIT), source_ms, turn_dub_ms)

    @staticmethod
    def _cache_key(
        speaker_id: str,
        target_lang: str,
        text: str,
        voice_mode: str,
        generation_config: dict[str, float | str] | None = None,
    ) -> str:
        normalized = " ".join(text.casefold().split())
        material = f"{speaker_id}|{target_lang}|{normalized}|{voice_mode}"
        if generation_config:
            # Appended only when there ARE delivery controls, so a line with none hashes to
            # exactly the key it hashed to before prosody existed and the warm cache survives
            # the deploy. Sorted so two configs with the same content but a different insertion
            # order share one entry instead of rendering twice.
            material += "|" + json.dumps(generation_config, sort_keys=True, separators=(",", ":"))
        return f"tts:cache:{hashlib.sha256(material.encode()).hexdigest()}"

    def _require_cartesia(self) -> CartesiaSynthesizer:
        if self.cartesia is None:
            raise RuntimeError("Cartesia synthesizer is not loaded")
        return self.cartesia
