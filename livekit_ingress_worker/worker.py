"""LiveKit Ingress Worker — Connects to LiveKit and extracts audio chunks.

Uses Silero VAD as a standalone gatekeeper to only publish audio chunks
that contain actual speech, preventing Whisper from seeing silence/noise.
"""

import asyncio
import copy
import json
import random
import time
import uuid
from collections import deque
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from livekit import api, rtc
from redis.exceptions import RedisError

from livekit_ingress_worker.audio_archive import MeetingAudioArchive, describe
from livekit_ingress_worker.near_field_gate import NearFieldGate
from shared.base_worker import BaseWorker
from shared.control_markers import is_external_bridge_speaker
from shared.object_storage import ObjectStorage, ObjectStorageSettings
from shared.schemas import (
    STT_FRAME_STREAM,
    STT_FRAME_STREAM_MAXLEN,
    AudioChunkMessage,
    AudioFrameMessage,
)

SILERO_VAD_REPOSITORY = "snakers4/silero-vad:v6.2.1"

# How long one VAD decision covers. Three exact Silero frames (3 x 32ms = 96ms) is the
# smallest window that can carry any majority at all.
VAD_WINDOW_FRAMES = 3

# How many of those frames must clear the threshold for the window to count as speech.
#
# WT-371 #7. This used to BE the window length — `VAD_WINDOW_SAMPLES = 512 * MIN_VAD_SPEECH_FRAMES`
# with the check `speech_frames >= MIN_VAD_SPEECH_FRAMES`. One constant set both, so the rule was
# never "some evidence"; it was UNANIMITY. Every frame in the window had to clear the threshold,
# and the knob could not express anything else: raising it grew the window by exactly as much as
# it raised the bar.
#
# Unanimity is the wrong rule for the start of an utterance, which is exactly where speech is
# least confident — a breath before the first vowel, an unvoiced consonant, a word begun partway
# through a window. One such frame rejected the whole 96ms, so onsets were routinely classified as
# silence and detection waited for the next window. Which is the reported symptom: speech register-
# ing late, and registering better when there is background noise keeping every frame's
# probability up.
#
# A majority still rejects the isolated spike the original docstring was written to reject.
MIN_VAD_SPEECH_FRAMES = 2

VAD_WINDOW_SAMPLES = 512 * VAD_WINDOW_FRAMES
VAD_WINDOW_MS = VAD_WINDOW_SAMPLES * 1000 // 16000

# Bot participant identities used elsewhere in this pipeline — this worker's own
# "AIBot_{room}" (join_room, below) and the TTS publisher's "ai-interpreter-{lang}"
# (tts_worker/livekit_publisher.py). Without this filter, subscribing to the AI
# interpreter's own synthesized-speech track feeds it straight back into STT ->
# translation -> TTS, a feedback loop that compounds hallucination/repetition.
_AI_BOT_IDENTITY_PREFIXES = ("AIBot_", "ai-interpreter-")


# WT-269 — reconnect governance.
#
# This worker used to answer every meeting.track_published by disconnecting the whole
# room and rejoining it. A track publish is routine and frequent: MeetingRoomService
# fires one synthetic event on every JoinMeetingAsync AND LiveKit's own track_published
# webhook fires another, so an ordinary four-person meeting produced a continuous stream
# of connect/disconnect pairs against one LiveKit project. LiveKit Cloud answered with
# HTTP 429 in every region and no meeting could carry media at all.
#
# Reconnecting is now a last resort (only when the connection we hold is genuinely gone),
# serialised per room, and always backed off.
_RECONNECT_BASE_DELAY_S = 1.0
_RECONNECT_MAX_DELAY_S = 60.0
# A 429 is LiveKit telling us to stop, not a transient to retry through — retrying at the
# normal cadence is exactly what keeps the limit in place. Start a rate-limited room above
# the ordinary cap so the project gets real breathing room before we dial again.
_RATE_LIMITED_DELAY_S = 120.0
# Every worker replica must not come off backoff on the same tick.
_RECONNECT_JITTER_RATIO = 0.25
# More connects than this to a single room inside the window is a storm, not churn. Say so
# at error level: the reason WT-269 took days to find is that nothing in the logs ever
# said "I am reconnecting again".
_RECONNECT_STORM_WINDOW_S = 60.0
_RECONNECT_STORM_THRESHOLD = 5


# WT-3xx (S1) — one ingress bot per room, across replicas.
#
# meeting.track_published arrives over Redis Pub/Sub, which FANS OUT: every replica of this
# deployment (chart: replicas: 2) receives every message. The bot identity is
# "AIBot_{room_name}" with no replica discriminator, so both replicas dialled the same room
# under the same identity and LiveKit resolved it the only way it can — by evicting one of
# them. The per-room asyncio.Lock and the reuse check above are per-PROCESS and cannot see
# the other replica at all.
#
# Giving each replica a unique identity would remove the eviction and replace it with
# something worse: both bots would stay connected, both would subscribe to every human audio
# track, and every utterance would be published TWICE to audio:chunks — double STT, double
# translation, double TTS, double cost, and duplicated captions. Eviction is the symptom;
# the invariant we actually need is "exactly one ingress bot per room".
#
# So ownership is elected in Redis, which every replica already shares. A replica claims
# "livekit:ingress:room-owner:{room}" with SET NX EX before it dials, renews it while it
# holds the room, and drops it when it leaves. A replica that does not hold the claim does
# nothing but remember the room, so it can take over if the owner dies.
#
# Why a lease rather than moving the event onto a Redis Stream with a consumer group: the
# producer is the backend meeting-service (MeetingRoomService.PublishTrackPublishedAsync ->
# IRedisService.PublishEventAsync) and LiveKit's own webhook (MeetingWebhookService), both
# PUBLISH. Converting the transport means a coordinated two-repo, two-service deploy where
# either half shipping alone drops every event on the floor. And a consumer group would only
# make the MESSAGE single-delivery, while the resource that must be single-owner is the
# LiveKit room — a room survives many messages and outlives all of them, so the claim
# belongs on the room. The lease also preserves failover, which the eviction behaviour
# accidentally provided: whichever replica lost the race used to pick the room back up.
_ROOM_OWNER_KEY_PREFIX = "livekit:ingress:room-owner:"
# Three sweep intervals. Long enough that an ordinary renew miss (one slow Redis round trip)
# does not hand the room away; short enough that a replica killed mid-meeting frees the room
# inside one grace window rather than stranding it until the meeting ends.
_ROOM_LEASE_TTL_S = 45

# The storm alarm above counted connects in a per-process deque, so with replicas: 2 each
# replica saw only its own half and the threshold could be missed while the project as a
# whole was being rate-limited — the alarm could not fire for the very failure mode it was
# added to catch. The count now lives in Redis, keyed by room and by fixed window, so every
# replica's dials are added together.
_STORM_COUNTER_KEY_PREFIX = "livekit:ingress:connects:"


# WT-314 — idle-room governance.
#
# MeetingRoomService publishes a synthetic meeting.track_published on every
# JoinMeetingAsync, so every human join summons "AIBot_{room}" into the LiveKit room
# whether or not anyone ever presses Start Translation. Until now the ONLY way that bot
# ever left was _cleanup_room(), which fires only when the backend publishes an
# AUDIO_ROUTES_UPDATED carrying a terminal room status — and the backend suppressed that
# publish for a room that never had any audio routes. A meeting where nobody started
# translation therefore left the bot connected for the rest of the process's life,
# billing LiveKit connection minutes forever. Because the bot itself is a participant,
# LiveKit's own empty_timeout never fires either: the room is never empty.
#
# The fix is not to plug that one hole (see the backend half of WT-314 for that) but to
# stop depending on another service's message for the release at all. This worker now
# checks its own rooms on a timer and leaves any room that has had no human participant
# for a bounded interval. Every future missed door becomes a bounded delay instead of a
# permanent leak.
#
# 15s matches tts_worker/livekit_publisher.py's _REAP_INTERVAL_S — one reap cadence
# across the pipeline — and bounds the overshoot past the grace period to one tick.
_IDLE_SWEEP_INTERVAL_S = 15.0

# Room lifecycle states that mean "this meeting is happening right now", used by
# _rediscover_active_rooms to decide which snapshots are worth reclaiming after a restart.
#
# Matched against the `room_status` the backend writes into
# translationRoom:{id}:audio_routes. Deliberately an allow-list of live states rather than a
# deny-list of finished ones: a status this worker has never heard of should be left alone,
# not dialled into. The snapshot key outlives the meeting by its TTL, so without this a
# restart would reconnect the bot to every meeting that ended in the last twelve hours.
_LIVE_ROOM_STATUSES = frozenset({"IN_PROGRESS", "AUDIO_ROUTING_ACTIVE"})
# How long a room may hold zero HUMAN remote participants before the bot leaves.
#
# The bot legitimately sits alone twice: between its own connect and the joining human's
# SFU handshake completing, and — after a rate-limited failure — for up to
# _RATE_LIMITED_DELAY_S (120s) while that human's client is backing off before it
# reconnects. 120s clears both. It deliberately does NOT try to cover a silent lull:
# idleness is measured in *participants*, not tracks or speech, and a connected human who
# has published nothing (or nothing yet) still keeps the room alive, so an unstarted
# meeting or a long pause never trips this.
#
# Erring short is cheap and self-healing — the next JoinMeetingAsync or LiveKit
# track_published webhook summons the bot straight back. Erring long is what WT-314 is.
# Worst case leak is now ~135s of connection minutes per orphan instead of unbounded.
_IDLE_ROOM_GRACE_S = 120.0


def _is_ai_bot_identity(identity: str) -> bool:
    return identity.startswith(_AI_BOT_IDENTITY_PREFIXES)


def _is_rate_limited_error(error: BaseException) -> bool:
    """Whether LiveKit refused this connect because we are being rate-limited.

    rtc.ConnectError carries the server's text rather than a status code, so this matches
    on the message. Over-matching is safe (we merely wait longer); under-matching is what
    produced the storm.
    """
    text = f"{getattr(error, 'message', '')} {error}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _parse_track_published_event(
    envelope: dict[str, Any],
) -> tuple[str, str | None, str] | None:
    """Validate and extract the versioned meeting.track_published contract."""
    if (
        envelope.get("event_type") != "meeting.track_published"
        or envelope.get("schema_version") != 1
        or envelope.get("producer") != "meeting-service"
    ):
        return None

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        return None

    room_name = payload.get("room_name")
    participant_identity = payload.get("participant_identity")
    track_id = payload.get("track_id")
    if not isinstance(room_name, str) or not room_name:
        return None
    if participant_identity is not None and not isinstance(participant_identity, str):
        return None
    if not isinstance(track_id, str) or not track_id:
        return None
    return room_name, participant_identity, track_id


# WT-B/flash mode — the per-room switch for streaming audio during speech.
#
# Accepted spellings are generous on purpose: this key is written by the backend today and by a
# person with redis-cli during an incident tomorrow, and a switch that silently ignores "true"
# is worse than one that accepts it.
_FLASH_MODE_ON = {"on", "true", "1", "enabled", "yes"}
_FLASH_MODE_OFF = {"off", "false", "0", "disabled", "no"}
# Short enough that a toggle feels immediate, long enough that a busy room does not read Redis
# once per speaker per sentence.
_FLASH_MODE_CACHE_SECONDS = 3.0

# WHERE THE DEPLOYMENT DEFAULT IS PUBLISHED SO THE BACKEND CAN READ IT.
#
# The backend owns only the per-room OVERRIDE key, and reported "off" whenever a room had never
# set one. That was true of the override and false of the room, and it was harmless only while
# the deployment default was also off. The day the default flipped to on, every host saw a
# switch that said "off" while their room was in fact streaming — and flipping it on and back
# off would then write a real override, silently costing them the latency they had just gained.
#
# Published from here rather than mirrored into the backend's own configuration because THIS is
# where the value governs behaviour. Two settings named the same thing in two services drift,
# and the one that drifts is always the one nobody is watching.
_FLASH_MODE_DEFAULT_KEY = "warptalk:stt:flash_mode_default"

# Refreshed on every heartbeat (10s), so this only has to outlive a deploy window rather than a
# week. Redis here runs allkeys-lru and has evicted live meeting state before, so the key is not
# allowed to be immortal — and an expired key degrades to the backend saying "unknown", which is
# honest, rather than to it asserting something false.
_FLASH_MODE_DEFAULT_TTL_SECONDS = 60 * 60

# The ingress energy floor: below this, a chunk is noise rather than speech.
#
# It is the ONLY always-on audio gate here — near_field_gate_enabled defaults to False — so it is
# the one thing that can discard a real utterance before STT ever sees it.
#
# MEASURED, not guessed (tools/probe_energy_floor.py, 2026-08-18). Against speech whose own RMS is
# 0.2005 the floor leaves:
#
#     3000ms utterance  ->  -19.1 dB of headroom
#      288ms utterance  ->  -14.6 dB of headroom
#
# Same speaker, same distance, 4.6 dB apart — most of one distance doubling — purely because the
# RMS was taken over the whole chunk, and a chunk carries vad_pre_speech_ms plus
# vad_silence_hangover_ms (192 + 576 = 768ms) of padding around the speech. The shorter the
# utterance, the more of that padding is averaged in, and the stricter the gate silently becomes.
# It was therefore hardest on exactly the short acknowledgements vad_min_speech_ms was lowered
# to keep.
_ENERGY_FLOOR_RMS = 0.02


class LiveKitIngressWorker(BaseWorker):
    """Worker that joins LiveKit rooms and pushes audio to Redis Streams."""

    worker_name = "livekit_ingress"
    input_stream = "pubsub:meeting.track_published"  # Logical name
    consumer_group = ""

    # Silero VAD frame size: 512 samples = 32ms at 16kHz
    VAD_FRAME_SIZE = 512
    SAMPLE_RATE = 16000

    # Declared on the class, not only in __init__, because the tests around this worker
    # build it with `__new__` and set just the attributes they exercise. An instance-only
    # default would turn "archiving is off" into an AttributeError raised from the middle of
    # the audio path — the one place that must not raise.
    _archive: MeetingAudioArchive | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rooms: dict[str, rtc.Room] = {}
        # Keyed by (room_name, speaker_id) — one reader per HUMAN, not per track.
        #
        # This key has been widened twice. It was the bare track sid, and teardown then
        # cancelled the whole dict, so a track published in one room silently killed
        # transcription in every other live room. Adding the room fixed that but left the
        # track sid in the key, which is the bug this pass removes: LiveKit issues a NEW sid
        # every time a participant republishes their mic — a reconnect, a device change, a
        # momentary network drop — so the guard saw an unfamiliar key and started a SECOND
        # reader for a speaker who already had one, while the first stayed alive. In
        # production that reached three concurrent readers on one microphone, each with its
        # own chunk counter, and the meeting transcribed every sentence three times. Because
        # each copy travelled the whole pipeline, the dubbed voice spoke it three times too.
        #
        # The sid is kept beside the task so a genuinely new track can replace a stale reader
        # rather than being refused as a duplicate of it.
        self.audio_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self.audio_task_tracks: dict[tuple[str, str], str] = {}
        # WT-529: in-flight speaker-name writes, held so the event loop cannot collect a task
        # nobody awaits. Discarded on completion — see _remember_speaker_name.
        self._speaker_name_tasks: set[asyncio.Task[None]] = set()
        self._vad_model: Any | None = None
        # One lock per room, held across the whole handler. Events arrive as independent
        # tasks (see _consume_loop), so without this two events for the same room could
        # interleave a teardown and a rejoin, and both would dial LiveKit with the same
        # "AIBot_{room}" identity — the server resolves that by evicting one of them,
        # which is a third connect nobody asked for.
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._event_tasks: set[asyncio.Task[None]] = set()
        # Keeps the speech this worker forwards to STT, so the meeting can be transcribed
        # again after it ends. Built even when disabled — `_archive` stays None then, and the
        # audio path pays one attribute check rather than a settings lookup per chunk.
        self._archive: MeetingAudioArchive | None = (
            MeetingAudioArchive(self.settings.audio_archive_root)
            if getattr(self.settings, "audio_archive_enabled", False)
            else None
        )
        self._archive_storage = ObjectStorage(
            ObjectStorageSettings(
                bucket=getattr(self.settings, "audio_archive_bucket", ""),
                prefix=getattr(self.settings, "audio_archive_prefix", ""),
                endpoint=getattr(self.settings, "audio_archive_endpoint", ""),
                region=getattr(self.settings, "audio_archive_region", "auto"),
                access_key=getattr(self.settings, "audio_archive_access_key", ""),
                secret_key=getattr(self.settings, "audio_archive_secret_key", ""),
            )
        )
        self._connect_failures: dict[str, int] = {}
        self._connect_not_before: dict[str, float] = {}
        self._connect_history: dict[str, deque[float]] = {}
        self._connects_total = 0
        # WT-314. Monotonic timestamp of the last moment each connected room held at least
        # one human remote participant (stamped at connect time to open the grace window).
        self._room_last_occupied: dict[str, float] = {}
        self._idle_sweeper: asyncio.Task[None] | None = None
        self._idle_releases_total = 0
        # S1. Rooms another replica currently owns. We hold no connection for these, but we
        # remember them so the sweeper can retry the claim — otherwise a replica dying
        # mid-meeting would silently end audio ingestion for its rooms until the next
        # track_published happened to arrive, which in a room where everyone has already
        # published is never.
        self._deferred_rooms: set[str] = set()
        self._owned_rooms: set[str] = set()

    async def load_model(self) -> None:
        """Load Silero VAD model."""
        self._vad_model, _ = torch.hub.load(  # type: ignore[no-untyped-call]
            SILERO_VAD_REPOSITORY,
            "silero_vad",
            trust_repo=True,
        )
        self.logger.info("silero_vad_loaded")

    async def _publish_heartbeat(self) -> None:
        """Prove this worker is alive, and restate how it is configured while doing so.

        The flash-mode default rides the heartbeat rather than a one-shot at startup for two
        reasons. It is republished every beat, so an eviction on a Redis running allkeys-lru
        costs at most one interval instead of lasting until the next deploy. And it keeps
        load_model doing one thing: that method is exercised by a test that builds this worker
        with __new__ purely to assert the Silero release is pinned, and a Redis write bolted
        into it made an unrelated contract test depend on settings and a connection.
        """
        await super()._publish_heartbeat()
        await self._publish_flash_mode_default()

    async def _publish_flash_mode_default(self) -> None:
        """Tell the backend what a room with no override actually does. See _FLASH_MODE_DEFAULT_KEY.

        Best effort on purpose: this exists so a HOST'S SWITCH can describe reality, and failing
        to describe reality must never stop audio being processed. A worker that cannot write it
        simply leaves the backend reporting "unknown", which is what it reports before any worker
        has started anyway — and swallowing here is also what keeps a Redis blip from taking
        down the heartbeat that calls it.
        """
        value = "on" if self.settings.stt_streaming_enabled else "off"
        try:
            await self.redis.set_with_ttl(
                _FLASH_MODE_DEFAULT_KEY, value, _FLASH_MODE_DEFAULT_TTL_SECONDS
            )
            self.logger.info("flash_mode_default_published", default=value)
        except Exception:
            self.logger.warning("flash_mode_default_publish_failed", default=value, exc_info=True)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Not used, we override _consume_loop for Pub/Sub."""
        pass

    async def _rediscover_active_rooms(self) -> int:
        """Pick meetings back up after this process restarts mid-call.

        WHAT THIS COSTS WHEN IT IS MISSING
            15 Aug, 12:17:07 local. The cgroup OOM killer took this worker mid-meeting with
            two people talking. It restarted eight seconds later, opened its listeners, and
            then sat completely idle for FOUR MINUTES AND NINETEEN SECONDS while the meeting
            carried on without it. No transcript, no translation, no dub. It only woke up at
            12:21:44 because somebody happened to publish a new track.

            The reason is the one this codebase has now hit three times: meeting.track_published
            arrives over Redis Pub/Sub, and pub/sub has no replay. Every participant in that
            room had already published, so the event that would have summoned this worker back
            had come and gone while the process was dead. In a settled meeting it never comes
            again.

            `_deferred_rooms` exists for the neighbouring case — another replica dying — and
            the sweeper already retries the claim for everything in it. But it is an in-memory
            set, so it dies with the process that owns it: the one replica that most needs to
            recover its rooms is the one that just lost the record of them.

        The durable record already exists and needs no new publisher: the backend writes
        `translationRoom:{id}:audio_routes` on every route change, carrying `room_status`. That
        is the same key `_hydrate_room_status` and `_load_route_snapshot` already read. Seeding
        `_deferred_rooms` from it turns a restart from "silent until someone republishes" into
        "claimed on the next sweep".

        Never fatal. A worker that cannot enumerate is exactly as capable as one that never
        tried, and refusing to start would turn a recovery miss into an outage.
        """
        try:
            keys = await self.redis.scan_keys("translationRoom:*:audio_routes")
        except Exception:
            self.logger.warning("room_rediscovery_scan_failed", exc_info=True)
            return 0

        recovered = 0
        for key in keys:
            parts = key.split(":")
            if len(parts) != 3:
                continue
            room_name = parts[1]
            try:
                cached = await self.redis.get(key)
                if not cached:
                    continue
                if isinstance(cached, bytes):
                    cached = cached.decode("utf-8")
                status = json.loads(cached).get("room_status")
            except Exception:
                self.logger.warning("room_rediscovery_read_failed", room=room_name, exc_info=True)
                continue

            # Only rooms that are actually live. An ENDED meeting's snapshot outlives it by
            # the key's TTL, and claiming those would have this worker dial into finished
            # rooms on every restart.
            if not isinstance(status, str) or status not in _LIVE_ROOM_STATUSES:
                continue

            self._route_states[room_name] = status
            self._deferred_rooms.add(room_name)
            recovered += 1

        if recovered:
            # The sweeper is what turns a deferred room into a connection, and it is started
            # lazily — without this the seeded set would sit untouched until some other event
            # happened to start it, which is the very thing that was missing.
            self._ensure_idle_sweeper()
            self.logger.warning(
                "rooms_rediscovered_after_restart",
                rooms=recovered,
                scanned=len(keys),
            )
            try:
                await self._claim_deferred_rooms()
            except Exception:
                # The sweep will try again in _IDLE_SWEEP_INTERVAL_S; one failed immediate
                # attempt must not stop the listener from starting.
                self.logger.warning("room_rediscovery_claim_failed", exc_info=True)

        return recovered

    async def _consume_loop(self) -> None:
        """Override to listen to Redis Pub/Sub instead of Streams."""
        # Before the listener, not after: everything below waits on an event that, for a
        # meeting already in progress, has already been published and will not repeat.
        await self._rediscover_active_rooms()

        self.logger.info("Starting Redis Pub/Sub listener for meeting.track_published")

        while not self._shutdown_event.is_set():
            pubsub = self.redis.redis.pubsub()
            try:
                await pubsub.subscribe("meeting.track_published")
                self.logger.info("track_published_listener_started")
                while not self._shutdown_event.is_set():
                    # get_message timeout prevents blocking indefinitely
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                    if message:
                        payload = json.loads(message["data"])
                        # asyncio keeps only a weak reference to a running task, so a
                        # bare create_task() can be collected mid-await and take its
                        # exception with it. Hold it until it finishes.
                        task = asyncio.create_task(self.handle_track_published(payload))
                        self._event_tasks.add(task)
                        task.add_done_callback(self._event_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("Error processing pub/sub message")
            finally:
                try:
                    await pubsub.close()
                except Exception:
                    self.logger.warning("track_published_listener_close_failed")

            if not self._shutdown_event.is_set():
                await asyncio.sleep(1.0)

    async def handle_track_published(self, payload: dict[str, Any]) -> None:
        parsed = _parse_track_published_event(payload)
        if parsed is None:
            self.logger.warning("invalid_track_published_event")
            return
        room_name, participant_identity, track_id = parsed
        # WT-314: re-checked on every event, not just on connect. This fires on each human
        # join and each published track, so a sweeper that somehow stopped while rooms were
        # still live is picked back up by ordinary meeting traffic.
        self._ensure_idle_sweeper()
        await self._hydrate_room_status(room_name)

        self.logger.info(
            "received_track_published",
            room=room_name,
            participant=participant_identity,
            track=track_id,
        )

        async with self._room_lock(room_name):
            # S1: fan-out means the other replica is running this exact handler for this
            # exact room right now. Exactly one of us may hold "AIBot_{room_name}".
            if not await self._claim_room_ownership(room_name):
                self._deferred_rooms.add(room_name)
                self.logger.info(
                    "track_published_room_owned_by_other_replica",
                    room=room_name,
                    track=track_id,
                )
                return

            room = self.rooms.get(room_name)
            if room is not None and room.isconnected():
                # WT-269: a newly published track arrives on the connection we already
                # hold — auto-subscribe raises "track_subscribed" for it, and the sweep
                # below covers a publication that landed before this event did. Tearing
                # the room down to pick up one new track is what caused the 429 storm.
                started = self._start_pending_audio_tasks(room_name, room)
                self.logger.info(
                    "track_published_reusing_connection",
                    room=room_name,
                    track=track_id,
                    audio_tasks_started=started,
                    connected_rooms=len(self.rooms),
                )
                return

            if room is not None:
                # Genuinely gone (or still mid-reconnect and now dead) — release it
                # before dialling, so the new connection does not have to be resolved by
                # LiveKit evicting our own previous identity.
                self.logger.warning("rejoining_disconnected_room", room=room_name)
                self.rooms.pop(room_name, None)
                self._cancel_room_audio_tasks(room_name)
                await self._disconnect_quietly(room_name, room)

            await self._connect_room(room_name)

    def _room_lock(self, room_name: str) -> asyncio.Lock:
        return self._room_locks.setdefault(room_name, asyncio.Lock())

    # ------------------------------------------------------------------
    # S1 — cross-replica room ownership
    # ------------------------------------------------------------------

    @staticmethod
    def _room_owner_key(room_name: str) -> str:
        return f"{_ROOM_OWNER_KEY_PREFIX}{room_name}"

    @staticmethod
    def _as_text(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return "" if value is None else str(value)

    async def _claim_room_ownership(self, room_name: str) -> bool:
        """Whether this replica may hold this room's bot, taking the claim if it is free.

        Fails CLOSED when Redis is unreachable. Joining a room we cannot coordinate over is
        not a degraded mode that still works: every audio chunk this worker extracts is
        published to Redis, so a bot connected without Redis transcribes nothing and does
        nothing except bill LiveKit connection minutes and race the other replica for the
        identity. Declining costs a bounded delay — the next track_published, or the next
        sweep, retries once Redis is back.
        """
        key = self._room_owner_key(room_name)
        try:
            if await self.redis.set_if_absent(key, self._consumer_name, _ROOM_LEASE_TTL_S):
                self._owned_rooms.add(room_name)
                self._deferred_rooms.discard(room_name)
                self.logger.info("livekit_room_ownership_acquired", room=room_name)
                return True

            # Already claimed — ours to renew, or somebody else's to leave alone.
            if await self.redis.extend_if_value(key, self._consumer_name, _ROOM_LEASE_TTL_S):
                self._owned_rooms.add(room_name)
                self._deferred_rooms.discard(room_name)
                return True

            self._owned_rooms.discard(room_name)
            return False
        except (RedisError, OSError):
            self.logger.warning(
                "livekit_room_ownership_unavailable",
                room=room_name,
                exc_info=True,
            )
            return False

    async def _release_room_ownership(self, room_name: str) -> None:
        """Hand the room back so another replica can pick it up immediately."""
        self._owned_rooms.discard(room_name)
        try:
            await self.redis.delete_if_value(self._room_owner_key(room_name), self._consumer_name)
        except (RedisError, OSError):
            # The lease expires on its own; losing the explicit release only costs the next
            # owner a wait of at most _ROOM_LEASE_TTL_S.
            self.logger.warning(
                "livekit_room_ownership_release_failed",
                room=room_name,
                exc_info=True,
            )

    async def _connect_room(self, room_name: str) -> None:
        """Join this room's bot, honouring any backoff a previous failure imposed."""
        now = asyncio.get_running_loop().time()
        not_before = self._connect_not_before.get(room_name, 0.0)
        if now < not_before:
            self.logger.warning(
                "livekit_connect_backoff_active",
                room=room_name,
                retry_in_s=round(not_before - now, 2),
                consecutive_failures=self._connect_failures.get(room_name, 0),
            )
            return

        await self._record_connect_attempt(room_name)
        room = await self.join_room(room_name)
        if room is not None:
            self.rooms[room_name] = room
            self._connect_failures.pop(room_name, None)
            self._connect_not_before.pop(room_name, None)
            # WT-314: open this room's idle grace window from the moment we joined, and make
            # sure something is actually watching it.
            self._room_last_occupied[room_name] = self._now()
            self._ensure_idle_sweeper()

    async def _record_connect_attempt(self, room_name: str) -> None:
        """Log every LiveKit dial, and shout when one room is dialling far too often.

        The window count is kept in Redis as well as in the local deque. LiveKit rate-limits
        the PROJECT, not one process, so a threshold evaluated against one replica's own
        dials undercounts by exactly the number of replicas — the alarm added to catch
        WT-269 could stay silent through a storm made of both replicas' traffic. The deque
        remains as the fallback the alarm falls back to when Redis is unavailable, which is
        the one moment we least want to lose the signal.
        """
        now = asyncio.get_running_loop().time()
        self._connects_total += 1
        history = self._connect_history.setdefault(room_name, deque())
        history.append(now)
        while history and now - history[0] > _RECONNECT_STORM_WINDOW_S:
            history.popleft()

        connects_in_window = max(len(history), await self._fleet_connects_in_window(room_name))

        self.logger.info(
            "livekit_room_connect_attempt",
            room=room_name,
            connects_in_window=connects_in_window,
            connects_this_replica=len(history),
            connects_total=self._connects_total,
        )
        if connects_in_window >= _RECONNECT_STORM_THRESHOLD:
            self.logger.error(
                "livekit_reconnect_storm_suspected",
                room=room_name,
                connects_in_window=connects_in_window,
                connects_this_replica=len(history),
                window_s=_RECONNECT_STORM_WINDOW_S,
                connects_total=self._connects_total,
            )

    async def _fleet_connects_in_window(self, room_name: str) -> int:
        """Dials against this room by EVERY replica in the current fixed window.

        A fixed window (not the sliding deque) because it needs one shared key that any
        replica can INCR without coordination. It can under-report by at most a window
        boundary; the deque covers the local half either way.
        """
        bucket = int(time.time() // _RECONNECT_STORM_WINDOW_S)
        key = f"{_STORM_COUNTER_KEY_PREFIX}{room_name}:{bucket}"
        try:
            return int(await self.redis.incr_with_ttl(key, int(_RECONNECT_STORM_WINDOW_S) * 2))
        except (RedisError, OSError):
            self.logger.warning("livekit_storm_counter_unavailable", room=room_name)
            return 0

    def _note_connect_failure(self, room_name: str, error: BaseException) -> None:
        failures = self._connect_failures.get(room_name, 0) + 1
        self._connect_failures[room_name] = failures
        rate_limited = _is_rate_limited_error(error)
        delay = (
            _RATE_LIMITED_DELAY_S
            if rate_limited
            else min(_RECONNECT_BASE_DELAY_S * 2 ** (failures - 1), _RECONNECT_MAX_DELAY_S)
        )
        delay *= 1.0 + random.uniform(-_RECONNECT_JITTER_RATIO, _RECONNECT_JITTER_RATIO)
        self._connect_not_before[room_name] = asyncio.get_running_loop().time() + delay
        self.logger.error(
            "livekit_rate_limited" if rate_limited else "failed_to_join_room",
            room=room_name,
            consecutive_failures=failures,
            backoff_s=round(delay, 2),
            exc_info=True,
        )

    async def _disconnect_quietly(self, room_name: str, room: rtc.Room) -> None:
        try:
            await room.disconnect()
        except Exception:
            self.logger.warning("room_disconnect_failed", room=room_name, exc_info=True)

    # Long enough to outlive any real meeting — the summary is written when it ends — and short
    # enough that a finished room's names expire on their own. Matches the transcript anchor.
    _SPEAKER_NAMES_TTL_S = 6 * 60 * 60

    def _remember_speaker_name(self, room_name: str, participant: rtc.RemoteParticipant) -> None:
        """Record how to say this speaker's name out loud. WT-529.

        The summariser accumulates `(speaker_id, text, ts)` off `stt:results`, where speaker_id is
        the LiveKit identity — `speaker-019f0d00-…`. That went into the prompt verbatim, the model
        repeated the only name it was given, and meeting summaries attributed decisions to a uuid.

        The name is already here and was simply never written down: the join token carries a
        `name` claim (LiveKitTokenService), so LiveKit populates `participant.name` for us.

        A HASH, one field per participant, because this runs once per speaker as they are met —
        two ingress replicas doing read-modify-write on a JSON blob would lose whichever name
        landed second.

        Fire-and-forget and entirely best-effort: a room with no names summarises with readable
        per-meeting pseudonyms (`Speaker 1`), which is what SpeakerNamer is for. Nothing about
        reading audio may depend on this write.
        """
        name = (participant.name or "").strip()
        if not name or name == participant.identity:
            # No claim on the token, or a client that sent the identity as the name. Either way
            # there is nothing here a reader could not already see, and writing it would only
            # move the uuid from the transcript into the map.
            return

        async def write() -> None:
            try:
                key = f"meeting:{room_name}:speaker_names"
                await self.redis.hset(key, participant.identity, name)
                await self.redis.expire(key, self._SPEAKER_NAMES_TTL_S)
            except Exception:
                self.logger.warning(
                    "speaker_name_not_recorded",
                    room=room_name,
                    participant=participant.identity,
                    exc_info=True,
                )

        task = asyncio.create_task(write())
        # Held so the loop cannot garbage-collect a task nobody awaits, and discarded on
        # completion so the set does not grow for the life of the process.
        self._speaker_name_tasks.add(task)
        task.add_done_callback(self._speaker_name_tasks.discard)

    def _start_audio_task(self, room_name: str, speaker_id: str, track: rtc.Track) -> bool:
        """Start this speaker's VAD pipeline, replacing any reader left on a stale track."""
        key = (room_name, speaker_id)
        existing = self.audio_tasks.get(key)

        if existing is not None and not existing.done():
            if self.audio_task_tracks.get(key) == track.sid:
                # Same microphone, already being read. Nothing to do.
                return False
            # A new sid for a speaker who already has a live reader means they republished:
            # the old track is stale and its reader must go, or both will publish the same
            # speech under two chunk counters.
            self.logger.info(
                "replacing_stale_audio_reader",
                room=room_name,
                speaker_id=speaker_id,
                previous_track=self.audio_task_tracks.get(key),
                new_track=track.sid,
            )
            existing.cancel()

        task = asyncio.create_task(self.process_audio_track(room_name, speaker_id, track))
        self.audio_tasks[key] = task
        self.audio_task_tracks[key] = track.sid
        task.add_done_callback(lambda finished: self._forget_audio_task(key, finished))
        return True

    def _forget_audio_task(self, key: tuple[str, str], task: asyncio.Task[None]) -> None:
        # Only if it is still OURS: a cancelled reader finishes after its replacement has
        # already registered, and deleting unconditionally would drop the live one.
        if self.audio_tasks.get(key) is task:
            del self.audio_tasks[key]
            self.audio_task_tracks.pop(key, None)

    def _cancel_audio_task(self, room_name: str, speaker_id: str) -> bool:
        """Stop reading one speaker, leaving the rest of the room untouched.

        Returns whether there was a live reader to stop, so the caller can log a real state
        change rather than every repeat of an event LiveKit may fire more than once.
        """
        key = (room_name, speaker_id)
        task = self.audio_tasks.get(key)
        if task is None or task.done():
            return False
        task.cancel()
        # Dropped here rather than waiting for the done-callback: the reaper sweep runs on its
        # own clock and would see a cancelled-but-still-registered reader as live, so a speaker
        # who unmutes in that window would get no reader at all.
        self.audio_tasks.pop(key, None)
        self.audio_task_tracks.pop(key, None)
        return True

    def _start_pending_audio_tasks(self, room_name: str, room: rtc.Room) -> int:
        """Attach to every already-published, UNMUTED human audio track we are not reading yet."""
        started = 0
        for participant in room.remote_participants.values():
            if _is_ai_bot_identity(participant.identity):
                continue
            for pub in participant.track_publications.values():
                track = pub.track
                if track is None or track.kind != rtc.TrackKind.KIND_AUDIO:
                    continue
                # WT-542. Without this the reaper sweep undoes the mute: it re-attaches any
                # track with no live reader, which after on_track_muted is precisely the
                # muted one, and the hallucinations resume one sweep later.
                if pub.muted:
                    continue
                self._remember_speaker_name(room_name, participant)
                if self._start_audio_task(room_name, participant.identity, track):
                    self.logger.info(
                        "subscribing_existing_audio_track",
                        room=room_name,
                        participant=participant.identity,
                        track=track.sid,
                    )
                    started += 1
        return started

    def _cancel_room_audio_tasks(self, room_name: str) -> None:
        """Cancel only THIS room's pipelines — never another live meeting's."""
        for key, task in list(self.audio_tasks.items()):
            if key[0] != room_name:
                continue
            task.cancel()
            self.audio_tasks.pop(key, None)
            self.audio_task_tracks.pop(key, None)

    async def _finish_archive(self, room_id: str) -> None:
        """Seal one meeting's audio tracks, put them somewhere durable, free the disk.

        The upload runs in a thread. It is a network call on a path that also disconnects
        the bot and releases the room lock, and blocking the event loop on a slow bucket
        would stall every OTHER meeting this worker is carrying.

        A track that fails to upload is KEPT on local disk and named in the log. Deleting it
        would destroy the only copy of a meeting's audio to tidy up after a transient S3
        error, and this archive exists precisely because the audio is otherwise gone.
        """
        archive = self._archive
        if archive is None:
            return

        try:
            tracks = await asyncio.to_thread(archive.close_meeting, room_id)
            if not tracks:
                return

            if not self._archive_storage.settings.configured:
                # Not a failure — a development box with no bucket. Said once, with the
                # location, so nobody has to guess where the audio went.
                self.logger.info(
                    "meeting_audio_archived_locally",
                    room=room_id,
                    path=str(tracks[0].path.parent),
                    **describe(tracks),
                )
                return

            uploaded = 0
            for track in tracks:
                key = f"{track.meeting_id}/{track.speaker_id}.flac"
                uri = await asyncio.to_thread(self._archive_storage.upload, track.path, key)
                if uri is None:
                    self.logger.warning(
                        "meeting_audio_upload_failed_keeping_local",
                        room=room_id,
                        speaker_id=track.speaker_id,
                        path=str(track.path),
                    )
                    continue
                uploaded += 1
                with suppress(OSError):
                    track.path.unlink()

            self.logger.info(
                "meeting_audio_archived",
                room=room_id,
                uploaded=uploaded,
                **describe(tracks),
            )
        except Exception:
            # The meeting is already over and the bot is already going home. Losing the
            # archive is a degraded outcome; letting this escape would abandon the rest of
            # the teardown that runs alongside it.
            self.logger.warning("meeting_audio_archive_failed", room=room_id, exc_info=True)

    def _cleanup_room(self, room_id: str) -> None:
        """Release the bot when the translation room reaches a terminal state.

        Nothing used to do this, so the bot stayed connected for the rest of the process's
        life — and if the same room was restarted, the next connect had to be resolved by
        LiveKit evicting that stale identity, one more avoidable reconnect.
        """
        super()._cleanup_room(room_id)
        self._cancel_room_audio_tasks(room_id)
        self._connect_failures.pop(room_id, None)
        self._connect_not_before.pop(room_id, None)
        self._connect_history.pop(room_id, None)
        self._room_last_occupied.pop(room_id, None)
        # The room is over for everyone, so no replica should keep chasing it.
        self._deferred_rooms.discard(room_id)
        # Sealed only after _cancel_room_audio_tasks above has stopped this room's readers,
        # so no chunk can arrive once the files are closed and be dropped without a trace.
        if self._archive is not None:
            finish = asyncio.create_task(self._finish_archive(room_id))
            self._event_tasks.add(finish)
            finish.add_done_callback(self._event_tasks.discard)
        room = self.rooms.pop(room_id, None)
        # Compare-and-delete, so this is a harmless no-op on a replica that never owned it.
        release = asyncio.create_task(self._release_room_ownership(room_id))
        self._event_tasks.add(release)
        release.add_done_callback(self._event_tasks.discard)
        if room is not None:
            self.logger.info("disconnecting_finished_room", room=room_id)
            task = asyncio.create_task(self._disconnect_quietly(room_id, room))
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)

    # ------------------------------------------------------------------
    # WT-314 — idle-room sweep
    # ------------------------------------------------------------------

    def _now(self) -> float:
        """Monotonic clock, isolated so tests can drive the sweep without sleeping."""
        return time.monotonic()

    def _ensure_idle_sweeper(self) -> None:
        """Start the idle sweep once, and restart it if it ever died.

        Started lazily rather than in __init__ (which runs outside any event loop), and
        re-checked from every track_published and every connect — a sweeper that stopped is
        indistinguishable from no sweeper at all, and silently restores the exact leak this
        exists to prevent. No room can enter self.rooms without passing _connect_room, so
        there is never a live room the sweeper has not been offered a chance to watch.
        """
        if self._idle_sweeper is None or self._idle_sweeper.done():
            self._idle_sweeper = asyncio.create_task(self._idle_sweep_loop())

    async def _idle_sweep_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(_IDLE_SWEEP_INTERVAL_S)
            try:
                await self._renew_room_ownership()
                await self._sweep_idle_rooms()
                await self._claim_deferred_rooms()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad sweep must not end the loop and quietly hand the rooms back to
                # "connected until the process dies".
                self.logger.exception("livekit_idle_sweep_error")

    async def _renew_room_ownership(self) -> None:
        """Keep this replica's claim on every room it is actually holding.

        A claim that lapses while we are still connected is worse than no claim: the other
        replica takes it, dials the same "AIBot_{room}" identity, and we are back to the
        eviction S1 exists to stop. If the extend fails the claim is already gone, so we
        stand down and let the new owner have the room rather than fight for the identity.
        """
        for room_name in list(self.rooms):
            try:
                still_ours = await self.redis.extend_if_value(
                    self._room_owner_key(room_name),
                    self._consumer_name,
                    _ROOM_LEASE_TTL_S,
                )
            except (RedisError, OSError):
                self.logger.warning("livekit_room_lease_renew_failed", room=room_name)
                continue

            if still_ours:
                self._owned_rooms.add(room_name)
                continue

            self.logger.warning("livekit_room_ownership_lost", room=room_name)
            self._owned_rooms.discard(room_name)
            self._deferred_rooms.add(room_name)
            await self._yield_room(room_name)

    async def _yield_room(self, room_name: str) -> None:
        """Drop a room whose claim now belongs to another replica."""
        async with self._room_lock(room_name):
            room = self.rooms.pop(room_name, None)
            self._room_last_occupied.pop(room_name, None)
            self._cancel_room_audio_tasks(room_name)
            if room is not None:
                await self._disconnect_quietly(room_name, room)

    async def _claim_deferred_rooms(self) -> None:
        """Take over rooms whose owning replica went away.

        Without this, electing an owner would remove the accidental failover the old
        eviction behaviour provided: the loser used to reconnect and carry on. A room only
        leaves this set when it is claimed or reaches a terminal status, so a replica killed
        mid-meeting costs at most one lease TTL plus one sweep of lost audio, not the rest
        of the meeting.
        """
        for room_name in list(self._deferred_rooms):
            if room_name in self.rooms:
                self._deferred_rooms.discard(room_name)
                continue
            async with self._room_lock(room_name):
                if room_name in self.rooms:
                    continue
                if not await self._claim_room_ownership(room_name):
                    continue
                self.logger.info("livekit_room_ownership_taken_over", room=room_name)
                await self._connect_room(room_name)

    def _human_participant_count(self, room: rtc.Room) -> int:
        """Remote participants that are not one of our own bots.

        The TTS publisher's "ai-interpreter-*" bots are remote participants in this same
        room. Counting them would mean a room holding nothing but our own machinery looks
        occupied, which is precisely the state WT-314 leaks.
        """
        return sum(
            1
            for participant in room.remote_participants.values()
            if not _is_ai_bot_identity(participant.identity)
        )

    async def _sweep_idle_rooms(self) -> None:
        """Release the bot from every room that has had no human in it for too long."""
        now = self._now()
        idle_rooms: list[str] = []
        occupied = 0
        humans = 0
        reattached_readers = 0

        lost: list[str] = []

        for room_name, room in list(self.rooms.items()):
            try:
                connected = room.isconnected()
                count = self._human_participant_count(room) if connected else 0
            except Exception:
                self.logger.warning("idle_sweep_room_probe_failed", room=room_name, exc_info=True)
                continue

            # WT-395 — a connection we lost is NOT an empty room, and must not be retired by
            # the path that retires empty ones.
            #
            # This branch used to fall through to `count = 0`, on the reasoning that a handle
            # LiveKit already dropped bills nothing. True for the cost question WT-314 was
            # about, and wrong for this one: retiring is only safe if something re-joins, and
            # nothing does. `_release_idle_room` discards the room from `_deferred_rooms` too,
            # and the only other way back is a `meeting.track_published` event — which, as the
            # comment on `_deferred_rooms` says, in a room where everyone has already
            # published is never.
            #
            # So one dropped connection ended audio ingestion for the rest of the meeting.
            # Production 2026-08-14, room 01a00058: both speakers' transcripts stopped within
            # four seconds of each other at 13:01 and never resumed, while the room and the
            # translation session stayed alive until 13:11. Nothing failed loudly; the meeting
            # simply stopped being heard.
            #
            # Requeued instead. `_claim_deferred_rooms` runs immediately after this sweep in
            # the same tick and re-dials, so the cost is one sweep interval of lost audio
            # rather than the remainder of the meeting. Ownership is dropped first because the
            # claim is what that path gates on.
            if not connected:
                self.rooms.pop(room_name, None)
                self._room_last_occupied.pop(room_name, None)
                self._cancel_room_audio_tasks(room_name)
                self._deferred_rooms.add(room_name)
                lost.append(room_name)
                continue

            if count > 0:
                self._room_last_occupied[room_name] = now
                occupied += 1
                humans += count

                # WT-404 — a connected room with people in it is not the same as a room being
                # HEARD, and until now nothing checked the difference.
                #
                # `process_audio_track` ends silently when its AudioStream ends without raising:
                # the task completes, `_forget_audio_task` drops it, and the only trace is one
                # INFO line. `_start_audio_task` is called from exactly two places — the
                # `track_subscribed` event and joining a room — so if LiveKit does not fire a
                # fresh subscribe for that track, that speaker is never read again.
                #
                # Production 2026-08-14, room 01a0015d: audio reached `audio:chunks` for 76
                # seconds of an eight-minute meeting and then stopped, while this very sweep went
                # on reporting the room connected and occupied. One transcript segment was saved.
                # The listener still heard dubbed audio, so nothing looked broken from outside.
                #
                # Idempotent by construction: `_start_audio_task` returns False for a live reader
                # on the same track, so this only ever fills a genuine gap.
                reattached = self._start_pending_audio_tasks(room_name, room)
                if reattached:
                    reattached_readers += reattached
                    self.logger.warning(
                        "audio_reader_reattached",
                        room=room_name,
                        readers=reattached,
                        detail="a speaker in a live room had no audio reader",
                    )
                continue

            last_occupied = self._room_last_occupied.setdefault(room_name, now)
            if now - last_occupied >= _IDLE_ROOM_GRACE_S:
                idle_rooms.append(room_name)

        # The reason WT-314 ran undetected is that a leaked bot is completely silent. This
        # is the gauge that makes a recurrence visible in logs instead of on the invoice.
        if self.rooms or idle_rooms or lost:
            self.logger.info(
                "livekit_ingress_room_census",
                connected_rooms=len(self.rooms),
                occupied_rooms=occupied,
                human_participants=humans,
                releasing_idle_rooms=len(idle_rooms),
                requeued_lost_rooms=len(lost),
                # WT-404. The census reported a healthy room throughout a meeting nobody was
                # being heard in. This is the number that would have shown it.
                reattached_readers=reattached_readers,
                idle_releases_total=self._idle_releases_total,
            )

        for room_name in lost:
            # Warning, not info: losing the connection to a live meeting is the failure this
            # branch exists to catch, and it left no trace at all before today.
            self.logger.warning("livekit_room_connection_lost_requeued", room=room_name)
            await self._release_room_ownership(room_name)

        for room_name in idle_rooms:
            # Strong reference held for the task's whole life. asyncio keeps only a weak
            # reference to a running task, so a bare create_task() can be collected
            # mid-await — and a disconnect that never ran is a leak that reports itself as
            # fixed. (tts_worker/livekit_publisher.py:_sweep_idle_bots still has this
            # hazard; it is a separate bug, not a pattern to copy.)
            task = asyncio.create_task(self._release_idle_room(room_name))
            self._event_tasks.add(task)
            task.add_done_callback(self._event_tasks.discard)

    async def _release_idle_room(self, room_name: str) -> None:
        """Disconnect one idle room, under its own lock so a concurrent join cannot race."""
        async with self._room_lock(room_name):
            room = self.rooms.get(room_name)
            if room is None:
                return
            # Re-check under the lock: a track_published may have landed while this task was
            # waiting, and disconnecting a room somebody just joined is a real outage.
            try:
                if room.isconnected() and self._human_participant_count(room) > 0:
                    self._room_last_occupied[room_name] = self._now()
                    return
            except Exception:
                self.logger.warning("idle_sweep_room_probe_failed", room=room_name, exc_info=True)
                return

            self.rooms.pop(room_name, None)
            self._room_last_occupied.pop(room_name, None)
            self._cancel_room_audio_tasks(room_name)
            # Nobody is in this room, so nobody should be chasing it either: drop the claim
            # and forget it, rather than leaving a lease the sweeper keeps renewing for a
            # room we are no longer connected to.
            self._deferred_rooms.discard(room_name)
            await self._release_room_ownership(room_name)
            self._idle_releases_total += 1
            # Warning, not info: every sweep-driven release means some upstream path failed
            # to tell us the room was over. It should be rare enough to be worth reading.
            self.logger.warning(
                "livekit_idle_room_released",
                room=room_name,
                grace_s=_IDLE_ROOM_GRACE_S,
                translation_state=self._route_states.get(room_name),
                connected_rooms=len(self.rooms),
                idle_releases_total=self._idle_releases_total,
            )
            await self._disconnect_quietly(room_name, room)

    async def _cleanup(self) -> None:
        """Stop the sweeper and hand every room back to LiveKit before the process exits."""
        if self._idle_sweeper is not None:
            self._idle_sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await self._idle_sweeper
            self._idle_sweeper = None

        for room_name, room in list(self.rooms.items()):
            self.rooms.pop(room_name, None)
            self._room_last_occupied.pop(room_name, None)
            self._cancel_room_audio_tasks(room_name)
            # Drop the claim on the way out so a rolling restart hands each room to the
            # surviving replica on its next sweep instead of after a full lease TTL.
            await self._release_room_ownership(room_name)
            await self._disconnect_quietly(room_name, room)

    # _is_translation_active moved to BaseWorker: this worker no longer asks the question at
    # all, because transcription must not wait for translation to start. See
    # shared/base_worker.py and the comments at both former call sites above.

    async def _hydrate_room_status(self, room_name: str) -> None:
        """Restore the last persisted room lifecycle after an ingress-worker restart."""
        try:
            cached = await self.redis.get(f"translationRoom:{room_name}:audio_routes")
            if not cached:
                return
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            payload = json.loads(cached)
            status = payload.get("room_status")
            if isinstance(status, str) and status:
                self._route_states[room_name] = status
        except Exception:
            self.logger.warning(
                "room_status_hydration_failed",
                room=room_name,
                exc_info=True,
            )

    async def join_room(self, room_name: str) -> rtc.Room | None:
        """Generate a token and connect an anonymous Bot to the LiveKit room."""
        # Generate token using livekit-api
        token = (
            api.AccessToken(self.settings.livekit.api_key, self.settings.livekit.api_secret)
            .with_identity(f"AIBot_{room_name}")
            .with_name("WarpTalk AI")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

        room = rtc.Room()

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_AUDIO and not _is_ai_bot_identity(
                participant.identity
            ):
                if publication.muted:
                    # WT-542. Somebody who joined muted is not speaking, and reading them
                    # anyway is what put words in their mouth — see on_track_muted.
                    self.logger.info(
                        "audio_track_subscribed_muted",
                        participant=participant.identity,
                        track=track.sid,
                    )
                    return
                self.logger.info(
                    "audio_track_subscribed", participant=participant.identity, track=track.sid
                )
                # WT-529 — here rather than inside _start_audio_task because that method takes
                # the identity string and never sees the participant carrying the name.
                self._remember_speaker_name(room_name, participant)
                self._start_audio_task(room_name, participant.identity, track)

        @room.on("track_muted")
        def on_track_muted(
            participant: rtc.RemoteParticipant,
            publication: rtc.RemoteTrackPublication,
        ) -> None:
            """WT-542: stop reading a microphone its owner has switched off.

            A muted publication is not a silent one. LiveKit keeps the subscription alive and
            the reader keeps receiving frames — near-silence, room tone, whatever the encoder
            emits — and near-silence is exactly the input Whisper invents text from. Production
            room 01a01e3f: a participant who was muted for the whole meeting was credited with
            thirteen segments of English in a Vietnamese call, including the profanity string
            Whisper is known to hallucinate on empty audio.

            No amount of downstream filtering fixes that honestly, because there is no text a
            muted microphone could legitimately produce. The audio must not be read at all.
            """
            if publication.kind != rtc.TrackKind.KIND_AUDIO or _is_ai_bot_identity(
                participant.identity
            ):
                return
            if self._cancel_audio_task(room_name, participant.identity):
                self.logger.info(
                    "audio_reader_stopped_on_mute",
                    room=room_name,
                    participant=participant.identity,
                    track=publication.sid,
                )

        @room.on("track_unmuted")
        def on_track_unmuted(
            participant: rtc.RemoteParticipant,
            publication: rtc.RemoteTrackPublication,
        ) -> None:
            """The other half of WT-542 — and the half that must not be missed.

            Without it a participant who muted once would stay unread for the rest of the
            meeting. The reaper sweep would eventually re-attach them, but "eventually" is up
            to one sweep interval of speech lost every time somebody unmutes, so the event
            re-attaches immediately and the sweep stays the backstop it was written to be.
            """
            if publication.kind != rtc.TrackKind.KIND_AUDIO or _is_ai_bot_identity(
                participant.identity
            ):
                return
            track = publication.track
            if track is None:
                return
            if self._start_audio_task(room_name, participant.identity, track):
                self.logger.info(
                    "audio_reader_resumed_on_unmute",
                    room=room_name,
                    participant=participant.identity,
                    track=track.sid,
                )

        @room.on("disconnected")
        def on_disconnected(reason: Any = None) -> None:
            # The only signal that a rejoin is legitimately needed. Logged so a room that
            # keeps dropping is visible next to the connect attempts it causes.
            self.logger.warning("livekit_room_disconnected", room=room_name, reason=str(reason))

        try:
            await room.connect(self.settings.livekit.url, token)
            self.logger.info(
                "connected_to_room", room=room_name, connects_total=self._connects_total
            )

            # Subscribe to already-published audio tracks from existing participants
            self._start_pending_audio_tasks(room_name, room)

            return room
        except Exception as error:
            self._note_connect_failure(room_name, error)
            return None

    def _run_vad_on_window(
        self,
        pcm_f32: npt.NDArray[np.float32],
        threshold: float,
        vad_model: Any | None = None,
    ) -> float:
        """Run Silero VAD and reject isolated probability spikes.

        Silero requires 512-sample (32ms) frames. Every frame in the window is scored, and the
        window counts as speech once MIN_VAD_SPEECH_FRAMES of them clear the threshold — a
        majority, so one noisy frame cannot classify the window as speech and one weak frame
        cannot disqualify it.

        The evidence bar and the window length are separate constants on purpose; see
        MIN_VAD_SPEECH_FRAMES for what happened when they were the same one.
        """
        model = vad_model or self._require_vad_model()
        max_prob = 0.0
        speech_frames = 0
        for i in range(0, len(pcm_f32) - self.VAD_FRAME_SIZE + 1, self.VAD_FRAME_SIZE):
            frame = pcm_f32[i : i + self.VAD_FRAME_SIZE]
            tensor = torch.from_numpy(frame).unsqueeze(0)
            prob = model(tensor, self.SAMPLE_RATE).item()
            if prob > max_prob:
                max_prob = prob
            if prob >= threshold:
                speech_frames += 1
        return max_prob if speech_frames >= MIN_VAD_SPEECH_FRAMES else 0.0

    async def process_audio_track(self, room_name: str, speaker_id: str, track: rtc.Track) -> None:
        """Stream audio from LiveKit, gate with VAD, publish only speech chunks."""
        audio_stream = rtc.AudioStream(track)
        sample_rate = self.SAMPLE_RATE
        # Distinguishes a reader somebody stopped from one that stopped itself — see the finally
        # block. Without it both ended on the same INFO line and WT-404 was invisible.
        cancelled = False

        # VAD configuration from settings
        vad_threshold = self.settings.vad_threshold
        pre_speech_ms = self.settings.vad_pre_speech_ms
        silence_hangover_ms = self.settings.vad_silence_hangover_ms
        min_speech_ms = self.settings.vad_min_speech_ms
        max_chunk_ms = self.settings.chunk_duration_ms

        # Calculate sizes
        # Three exact Silero frames (~96ms) are the smallest window that can satisfy
        # MIN_VAD_SPEECH_FRAMES. The old 500ms polling window imposed up to half a second
        # of avoidable capture latency before STT could even start.
        vad_window_samples = VAD_WINDOW_SAMPLES
        pre_speech_samples = int(sample_rate * (pre_speech_ms / 1000.0))
        silence_hangover_samples = int(sample_rate * (silence_hangover_ms / 1000.0))
        silence_hangover_windows = max(
            1,
            (silence_hangover_samples + vad_window_samples - 1) // vad_window_samples,
        )
        min_speech_samples = int(sample_rate * (min_speech_ms / 1000.0))
        max_chunk_samples = int(sample_rate * (max_chunk_ms / 1000.0))

        # BOUNDARY SEEKING. Past `seek_after_samples` of actual speech, a shorter pause is
        # accepted as a cut point, so a long turn is split where the speaker drew breath
        # instead of wherever `max_chunk_samples` happened to land. See the settings for the
        # measurements behind it.
        #
        # Deliberately reuses the hangover branch below rather than adding a third one: a
        # 250ms pause IS the end of a clause, so the turn closes exactly as it would on a
        # long pause, and the next clause collects its own pre-speech ring. Everything that
        # branch already gets right — the min-speech gate, turn bookkeeping, not resetting
        # Silero's recurrent state — is got right here too, for free.
        #
        # Measured on SPEECH, not on the buffer: the buffer also holds padding and every
        # internal pause, so a hesitant speaker would otherwise trip the seek threshold
        # having said very little.
        seek_after_samples = int(sample_rate * (self.settings.vad_seek_boundary_after_ms / 1000.0))
        seek_hangover_samples = int(sample_rate * (self.settings.vad_seek_hangover_ms / 1000.0))
        seek_hangover_windows = max(
            1,
            (seek_hangover_samples + vad_window_samples - 1) // vad_window_samples,
        )

        # One near-field gate per track — see near_field_gate.py. It builds a running
        # peak-amplitude reference from this track's own earlier chunks, so it must live
        # for the whole track lifetime, not be recreated per chunk.
        #
        # WT-525: except for the external-bridge stand-in, where the concept does not apply.
        # That track is not somebody at a microphone — it is a line-level feed from Google Meet
        # mixing several people in another room. The gate is one-directional and raises its
        # baseline on every louder chunk, so one person on the far side speaking loudly would
        # lift the bar and silence the next person who speaks quietly. The symptom is
        # "translation works sometimes" with no error anywhere, which is close to undiagnosable
        # from the outside.
        #
        # The hallucination risk the gate exists to prevent is genuinely lower here: a conference
        # feed carries clean digital audio, not a far-field voice bleeding into a mic.
        bridge_speaker = is_external_bridge_speaker(speaker_id)
        near_field_gate = None if bridge_speaker else NearFieldGate(self.settings)
        if bridge_speaker:
            self.logger.info(
                "near_field_gate_disabled_for_bridge",
                room=room_name,
                speaker_id=speaker_id,
            )
        # Silero VAD carries recurrent state. Sharing one model across concurrently
        # iterated participant tracks interleaves unrelated audio histories and causes
        # missed/fragmented speech. Each track owns an independent cloned state machine.
        track_vad_model = copy.deepcopy(self._require_vad_model())

        # Streaming state. A TURN is the audio that the NEXT audio:chunks message will commit
        # — so it is rotated by a max-chunk flush as well as by silence, because both publish a
        # chunk and both therefore end a commit boundary on the STT side.
        # Decided at each speech onset (see _flash_mode_enabled), not once here: this is a
        # switch somebody flips mid-meeting and then listens for.
        streaming = False
        turn_id = ""
        frame_seq = 0
        streamed_bytes = 0

        # State
        raw_buffer = bytearray()  # Incoming raw resampled audio
        speech_buffer = bytearray()  # Audio collected during speech
        pre_speech_ring: deque[bytes] = deque(
            maxlen=max(1, pre_speech_samples * 2 // (vad_window_samples * 2))
        )  # Rolling pre-speech windows (in bytes, 2 bytes/sample)
        # Whether this turn has ALREADY sent a chunk, i.e. whether what is in the buffer now
        # is a continuation rather than a fresh utterance. The min-speech gate exists to stop a
        # cough being transcribed; a 190ms tail left over after a cap or seek cut is not a
        # cough, it is the rest of a sentence, and dropping it lost real words silently.
        published_this_turn = False
        is_speaking = False
        # Samples in speech_buffer that VAD actually called speech — excludes the pre-speech
        # padding and the hangover tail. This, not the buffer length, is what the minimum-speech
        # gate weighs; see where it is compared for what the buffer length let through.
        speech_samples = 0
        silence_counter = 0
        chunk_index = 0
        resampler = None
        first_frame_logged = False

        self.logger.info(
            "started_vad_audio_processing",
            track_sid=track.sid,
            vad_threshold=vad_threshold,
            pre_speech_ms=pre_speech_ms,
            hangover_ms=silence_hangover_ms,
        )

        # Reset VAD model state for this track
        track_vad_model.reset_states()

        try:
            async for event in audio_stream:
                if self._shutdown_event.is_set():
                    break
                if room_name in self._paused_rooms:
                    # Paused only. This used to drop frames until translation had been
                    # started, which meant a live meeting produced no transcript until
                    # somebody pressed a button that has nothing to do with transcription.
                    # A pause is different: the participants asked for the room to stop
                    # listening, so discard the buffers completely rather than letting
                    # speech from the pause leak into the first chunk after resume.
                    raw_buffer = bytearray()
                    speech_buffer = bytearray()
                    pre_speech_ring.clear()
                    is_speaking = False
                    published_this_turn = False
                    speech_samples = 0
                    silence_counter = 0
                    track_vad_model.reset_states()
                    continue

                frame = event.frame
                if not first_frame_logged:
                    self.logger.info(
                        "received_first_audio_frame",
                        track_sid=track.sid,
                        len=len(frame.data),
                        sample_rate=frame.sample_rate,
                        channels=frame.num_channels,
                    )
                    first_frame_logged = True

                if resampler is None:
                    resampler = rtc.AudioResampler(
                        input_rate=frame.sample_rate,
                        output_rate=sample_rate,
                        num_channels=frame.num_channels,
                    )

                resampled_frames = resampler.push(frame)

                for r_frame in resampled_frames:
                    data = bytes(r_frame.data)
                    if frame.num_channels > 1:
                        arr = np.frombuffer(data, dtype=np.int16)
                        arr = arr.reshape(-1, frame.num_channels)
                        data = arr.mean(axis=1).astype(np.int16).tobytes()

                    raw_buffer.extend(data)

                # Process in ~96ms windows (three exact Silero frames).
                window_bytes = vad_window_samples * 2  # 2 bytes per sample
                while len(raw_buffer) >= window_bytes:
                    window_data = bytes(raw_buffer[:window_bytes])
                    raw_buffer = raw_buffer[window_bytes:]

                    # Run VAD on this ~96ms window (VAD_WINDOW_FRAMES exact Silero frames).
                    window_pcm = np.frombuffer(window_data, dtype=np.int16)
                    window_f32 = window_pcm.astype(np.float32) / 32768.0
                    vad_prob = self._run_vad_on_window(
                        window_f32,
                        threshold=vad_threshold,
                        vad_model=track_vad_model,
                    )

                    if vad_prob >= vad_threshold:
                        # Speech detected
                        if not is_speaking:
                            # Speech onset — prepend pre-speech ring buffer
                            is_speaking = True
                            speech_buffer = bytearray()
                            speech_samples = 0
                            streaming = await self._flash_mode_enabled(room_name)
                            if streaming:
                                turn_id = uuid.uuid4().hex
                                frame_seq = 0
                                streamed_bytes = 0
                            for pre_window in pre_speech_ring:
                                speech_buffer.extend(pre_window)
                            self.logger.info(
                                "speech_start",
                                chunk_index=chunk_index,
                                vad_prob=round(vad_prob, 3),
                                pre_buffer_ms=len(speech_buffer) // 2 * 1000 // sample_rate,
                            )

                        speech_buffer.extend(window_data)
                        speech_samples += len(window_data) // 2
                        silence_counter = 0

                        # The pre-speech ring goes out with the first frame rather than being
                        # skipped: it is the word onset the ring exists to preserve, and a
                        # session that never hears it transcribes a clipped first syllable.
                        if streaming and len(speech_buffer) > streamed_bytes:
                            await self._publish_speech_frame(
                                room_name,
                                speaker_id,
                                turn_id,
                                frame_seq,
                                bytes(speech_buffer[streamed_bytes:]),
                                sample_rate,
                            )
                            frame_seq += 1
                            streamed_bytes = len(speech_buffer)

                        # Max chunk length is about the SIZE of what gets sent, so it weighs the
                        # whole buffer — padding included. Only the minimum-speech gate below asks
                        # the different question of whether anyone actually spoke.
                        if len(speech_buffer) // 2 >= max_chunk_samples:
                            await self._publish_speech_chunk(
                                room_name,
                                speaker_id,
                                speech_buffer,
                                chunk_index,
                                sample_rate,
                                near_field_gate=near_field_gate,
                                turn_id=turn_id,
                                speech_samples=speech_samples,
                            )
                            chunk_index += 1
                            published_this_turn = True
                            # The SPEAKER has not stopped, but the commit boundary has moved:
                            # STT will commit everything appended so far when it sees the chunk
                            # above, so what comes next belongs to a new turn.
                            if streaming:
                                turn_id = uuid.uuid4().hex
                                frame_seq = 0
                                streamed_bytes = 0
                            speech_buffer = bytearray()
                            # Still mid-utterance — the speaker has simply run past the maximum
                            # chunk length. The accumulated speech went out with the chunk, so the
                            # remainder starts its own count.
                            speech_samples = 0

                    else:
                        # No speech in this window
                        if is_speaking:
                            silence_counter += 1
                            speech_buffer.extend(window_data)  # Keep recording during pauses

                            # A long turn takes the next real pause; a short one still waits
                            # for the full end-of-sentence hangover. This is the whole of
                            # boundary seeking — see seek_hangover_windows above.
                            hangover_windows = (
                                seek_hangover_windows
                                if speech_samples >= seek_after_samples
                                else silence_hangover_windows
                            )

                            if silence_counter >= hangover_windows:
                                # End of speech — publish if there was enough SPEECH in it.
                                #
                                # WT-371 #7: this measured the whole buffer, which by this point
                                # also holds the pre-speech padding and the entire 576ms hangover.
                                # A 100ms cough therefore arrived at the gate as ~870ms and sailed
                                # past a 288ms minimum. What reached Whisper was a fragment of
                                # non-speech with no confidence signal of its own — the exact input
                                # it invents fluent sentences from (see test_vad_threshold_default).
                                #
                                # The padding still ships; it is just no longer counted as evidence
                                # that somebody spoke.
                                if speech_samples >= min_speech_samples or published_this_turn:
                                    await self._publish_speech_chunk(
                                        room_name,
                                        speaker_id,
                                        speech_buffer,
                                        chunk_index,
                                        sample_rate,
                                        near_field_gate=near_field_gate,
                                        turn_id=turn_id,
                                        speech_samples=speech_samples,
                                    )
                                    chunk_index += 1
                                else:
                                    self.logger.debug(
                                        "speech_too_short",
                                        samples=speech_samples,
                                        min_required=min_speech_samples,
                                    )
                                    # No chunk is coming for this turn, so nothing will commit
                                    # the frames already appended for it. Nothing is published to
                                    # say so either: the STT side notices that the next turn_id
                                    # arrived without the previous one ever being committed and
                                    # clears the buffer itself. That one rule also covers a lost
                                    # frame and an ingress that died mid-turn, which a marker
                                    # message would not.

                                # Closed either way — a published chunk commits this turn, and an
                                # unpublished one is discarded by the rule above. Both end it.
                                turn_id = ""
                                frame_seq = 0
                                streamed_bytes = 0
                                is_speaking = False
                                published_this_turn = False
                                speech_buffer = bytearray()
                                speech_samples = 0
                                silence_counter = 0
                                # WT-371 #7: the VAD state is NOT reset here any more.
                                #
                                # Silero is recurrent. Resetting it discards everything it has
                                # learned about this microphone and this room, and its first frames
                                # after a reset are its least reliable — which is precisely the
                                # moment the next utterance begins. Doing it after EVERY utterance
                                # meant every sentence in a conversation was judged by a cold model,
                                # so the first word registered late or not at all, and registered
                                # better when background noise kept the probabilities up. That is
                                # the reported symptom, and it is self-inflicted.
                                #
                                # A reset is right where the audio genuinely discontinues: a new
                                # track (above) and a pause/resume (which discards the buffers for
                                # the same reason). A pause between two sentences is not a
                                # discontinuity — it is the signal Silero is built to model.
                        else:
                            # Store in pre-speech ring for next onset
                            pre_speech_ring.append(window_data)

        except asyncio.CancelledError:
            # Somebody decided this reader should stop — the speaker left, the room was
            # released, or a republished track replaced it. Expected, and told apart from the
            # case below on purpose (WT-404).
            cancelled = True
            raise
        except Exception:
            self.logger.exception("process_audio_track_error", track_sid=track.sid)
        finally:
            # Publish any remaining speech buffer. Gated on the speech in it, for the same reason
            # the end-of-utterance path is: the track can end on a hangover tail, and buffer
            # length would count that padding as somebody having spoken.
            if speech_buffer and speech_samples >= min_speech_samples:
                await self._publish_speech_chunk(
                    room_name,
                    speaker_id,
                    speech_buffer,
                    chunk_index,
                    sample_rate,
                    near_field_gate=near_field_gate,
                    speech_samples=speech_samples,
                )
            if cancelled:
                self.logger.info("stopped_audio_stream_processing", track_sid=track.sid)
            else:
                # THE LINE THAT WAS MISSING. An AudioStream that ends on its own is how a
                # speaker stops being heard while the room stays connected and the census keeps
                # reporting it healthy — the whole of WT-404. It was logged at INFO, in the same
                # words as an ordinary cancellation, so it could happen mid-meeting and read as
                # routine teardown.
                #
                # Not an error: a track really does end when somebody leaves, and the sweep will
                # simply find nothing to re-attach. It is a warning because in a room that is
                # still live it means somebody has gone silent.
                self.logger.warning(
                    "audio_stream_ended_on_its_own",
                    room=room_name,
                    speaker_id=speaker_id,
                    track_sid=track.sid,
                    detail="the reader stopped without being cancelled; "
                    "the idle sweep re-attaches if the track is still published",
                )

    def _require_vad_model(self) -> Any:
        if self._vad_model is None:
            raise RuntimeError("Silero VAD model is not loaded")
        return self._vad_model

    async def _speaker_language(self, room_name: str, speaker_id: str) -> str:
        """This speaker's own chosen speak-language, or "auto".

        TranslationRoomHub.JoinTranslationRoom persists it (see NormalizeLanguageCode there)
        keyed by userId, which is the same value LiveKit uses as participant.identity and
        therefore as speaker_id here. Falls back to "auto" — STT's own guess — only if the
        speaker somehow is not registered yet.

        Extracted so the streamed frames and the closed utterance cannot disagree about it: a
        frame appended under one language and committed under another is a session pinned to the
        wrong language for that turn.
        """
        try:
            raw_language = await self.redis.hget(
                f"translationRoom:{room_name}:speak_languages", speaker_id
            )
        except RedisError:
            self.logger.warning(
                "speak_language_lookup_failed",
                room=room_name,
                speaker_id=speaker_id,
                exc_info=True,
            )
            return "auto"
        if not raw_language:
            return "auto"
        return raw_language.decode() if isinstance(raw_language, bytes) else raw_language

    async def _flash_mode_enabled(self, room_name: str) -> bool:
        """Whether THIS room streams audio while the speaker is still talking (flash mode).

        WHY PER ROOM, AND WHY IT IS READ HERE RATHER THAN ONCE PER TRACK
            One environment variable makes this an all-or-nothing choice for the whole platform,
            which is the same trap WT-427 documented for denoising: whichever way it is set, half
            the estate is configured for the other half. It is also the only way to A/B the thing
            in a real meeting — one room on, one room off, same build.

            Read at SPEECH ONSET, not once when the track opens. A person toggling it mid-meeting
            expects the next thing they say to be affected; captured at track open it would do
            nothing until they rejoined, which reads as a dead switch.

        WHY THE VALUE IS HELD FOR THE WHOLE UTTERANCE
            A turn that starts streaming must finish streaming. Flipping mid-utterance would send
            the STT side a turn whose frames stop partway, which its own gap check would then have
            to throw away — so a toggle takes effect on the next thing the speaker says, at most a
            sentence later.

        Falls back to the deployment default on anything unexpected, which is what every room did
        before this key existed.
        """
        cache: dict[str, tuple[float, bool]] | None = getattr(self, "_flash_mode_cache", None)
        if cache is None:
            cache = {}
            self._flash_mode_cache = cache

        now = time.monotonic()
        cached = cache.get(room_name)
        # Short, unlike WT-427's permanent per-room cache, because this one is behind a switch a
        # person flips DURING a meeting and then listens for. Three seconds keeps a toggle feeling
        # immediate while still collapsing the per-turn reads of a room full of people.
        if cached is not None and now - cached[0] < _FLASH_MODE_CACHE_SECONDS:
            return cached[1]

        value = self.settings.stt_streaming_enabled
        try:
            raw = await self.redis.get(f"translationRoom:{room_name}:flash_mode")
            if raw:
                decoded = (raw.decode() if isinstance(raw, bytes) else raw).strip().lower()
                if decoded in _FLASH_MODE_ON:
                    value = True
                elif decoded in _FLASH_MODE_OFF:
                    value = False
                else:
                    self.logger.warning(
                        "flash_mode_unrecognised", room_name=room_name, value=decoded
                    )
        except Exception:
            # A room that cannot be read falls back to the deployment default. Never let a
            # settings lookup stop audio from being processed.
            self.logger.warning("flash_mode_unavailable", room_name=room_name)

        cache[room_name] = (now, value)
        return value

    async def _publish_speech_frame(
        self,
        room_name: str,
        speaker_id: str,
        turn_id: str,
        seq: int,
        pcm: bytes,
        sample_rate: int,
    ) -> None:
        """Hand STT one VAD window WHILE the speaker is still producing the turn.

        Best effort in the strongest sense: a frame that cannot be published is simply not
        appended, and the closed utterance on `audio:chunks` still carries the whole turn's
        audio. So the worst case of this whole feature failing is the latency the pipeline had
        before it existed — never a lost sentence. That is why nothing here raises.

        NO ENERGY OR NEAR-FIELD GATE, unlike _publish_speech_chunk. Those gates judge a WHOLE
        utterance and reject it as noise; a single 96ms window has no such verdict to give, and
        applying a per-utterance threshold to a frame would punch holes in the middle of real
        speech. The utterance-level judgement still happens — on `audio:chunks`, where it always
        did — and STT only commits what that message tells it to.
        """
        if room_name in self._paused_rooms:
            return
        try:
            frame = AudioFrameMessage(
                meeting_id=room_name,
                speaker_id=speaker_id,
                turn_id=turn_id,
                seq=seq,
                audio_data=pcm,
                sample_rate=sample_rate,
                language=await self._speaker_language(room_name, speaker_id),
            )
            await self.redis.publish_ephemeral(
                STT_FRAME_STREAM, frame.to_redis(), STT_FRAME_STREAM_MAXLEN
            )
        except Exception:
            # Deliberately quiet at debug: this fires per 96ms window per speaker, so a warning
            # here would bury the log the moment Redis hiccuped — and the fallback is silent and
            # complete.
            self.logger.debug("speech_frame_publish_failed", room=room_name, exc_info=True)

    async def _publish_speech_chunk(
        self,
        room_name: str,
        speaker_id: str,
        speech_buffer: bytearray,
        chunk_index: int,
        sample_rate: int,
        near_field_gate: NearFieldGate | None = None,
        turn_id: str = "",
        speech_samples: int | None = None,
    ) -> None:
        # Transcription is NOT translation, and this gate used to conflate them.
        #
        # It discarded every chunk until the room reported IN_PROGRESS/AUDIO_ROUTING_ACTIVE,
        # a state only reached once someone started translation. So a live meeting where
        # nobody had pressed Start produced no transcript at all — the microphone was read,
        # VAD cut clean utterances, and each one was dropped here. It logged at debug, which
        # production does not emit, so the whole pipeline failed in complete silence.
        #
        # The bot is only in this room because a participant published a microphone, so
        # being here IS the signal that a meeting is live. Transcribe on that basis, and
        # leave "has translation been started" to the translation worker, which is the
        # stage that actually costs a translation.
        if room_name in self._paused_rooms:
            self.logger.info(
                "speech_chunk_discarded_room_paused",
                room=room_name,
                speaker_id=speaker_id,
            )
            return

        """Normalize and publish a speech chunk to Redis."""
        raw_bytes = bytes(speech_buffer)
        pcm = np.frombuffer(raw_bytes, dtype=np.int16)
        raw_rms = np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))
        raw_peak = np.max(np.abs(pcm.astype(np.float32) / 32768.0))
        duration_ms = len(pcm) * 1000 // sample_rate

        # Energy gate: skip chunks that are too quiet (noise, not speech).
        #
        # WEIGHED AGAINST THE SPEECH, NOT AGAINST THE PADDING AROUND IT. raw_rms averages the whole
        # chunk, and VAD deliberately wraps every utterance in pre-speech and hangover padding, so
        # a comparison against a fixed floor asked a shorter utterance to be LOUDER than a long one
        # to survive — see _ENERGY_FLOOR_RMS for the measurement. Scaling the floor by the square
        # root of the speech share undoes exactly that dilution, which makes the gate judge what it
        # already says it judges. A long utterance barely moves (80% share -> 0.0178); a
        # minimum-length one stops being 4.6 dB stricter for no reason (27% share -> 0.0104).
        #
        # Not a loosening dressed up as a fix: the threshold on SPEECH loudness is now the same
        # 0.02 for every utterance length, which is what one absolute floor was always meant to be.
        # Marginal audio that gets through still faces min_avg_logprob downstream, which this model
        # does populate (see STTSettings for the production values that calibrated it).
        total_samples = len(pcm)
        floor = _ENERGY_FLOOR_RMS
        if speech_samples is not None and 0 < speech_samples < total_samples:
            floor *= float(np.sqrt(speech_samples / total_samples))
        if raw_rms < floor:
            self.logger.debug(
                "skipped_low_energy_chunk",
                chunk_index=chunk_index,
                raw_rms=round(float(raw_rms), 6),
                floor=round(float(floor), 6),
                speech_share=(
                    round(speech_samples / total_samples, 3)
                    if speech_samples is not None and total_samples
                    else None
                ),
            )
            return

        # Near-field gate: reject a chunk that's much quieter than this track's own
        # established near-field peak — a far-away/muffled voice, not the primary
        # speaker. See near_field_gate.py — fails open by design.
        if near_field_gate is not None and not near_field_gate.accept(float(raw_peak)):
            self.logger.info(
                "skipped_far_field_chunk",
                speaker_id=speaker_id,
                chunk_index=chunk_index,
            )
            return

        # Send raw audio (no normalization — Whisper handles natural levels better)
        # Peak normalization was amplifying noise to speech levels, causing hallucinations

        language = await self._speaker_language(room_name, speaker_id)

        msg = AudioChunkMessage(
            meeting_id=room_name,
            speaker_id=speaker_id,
            chunk_index=chunk_index,
            audio_data=bytes(pcm),
            sample_rate=sample_rate,
            language=language,
            # The turn whose streamed frames this message commits. Empty when streaming is off,
            # which is also exactly what an older ingress sends through a rolling deploy — so
            # the STT side reads "empty" as "the audio is in this message" and behaves as it
            # always did.
            turn_id=turn_id,
            timestamp_ms=int(time.time() * 1000),
        )

        # Tapped here, from the bytes this message carries, so a second pass is handed
        # exactly what the first pass was handed. Archiving from anywhere else would make a
        # pass-1-vs-pass-2 accuracy comparison a comparison of two audio paths instead.
        if self._archive is not None:
            self._archive.append(room_name, speaker_id, msg.audio_data, sample_rate)

        payload = msg.to_redis()
        for attempt in range(1, 4):
            try:
                await self.publish("audio:chunks", room_name, payload)
                break
            except RedisError:
                if attempt == 3:
                    self.logger.error(
                        "speech_chunk_publish_failed",
                        room=room_name,
                        speaker_id=speaker_id,
                        chunk_index=chunk_index,
                        exc_info=True,
                    )
                    return
                self.logger.warning(
                    "speech_chunk_publish_retry",
                    room=room_name,
                    speaker_id=speaker_id,
                    chunk_index=chunk_index,
                    attempt=attempt,
                    exc_info=True,
                )
                await asyncio.sleep(0.05 * attempt)

        self.logger.info(
            "speech_chunk_published",
            room=room_name,
            speaker_id=speaker_id,
            chunk_index=chunk_index,
            duration_ms=duration_ms,
            raw_rms=round(float(raw_rms), 6),
            raw_peak=round(float(raw_peak), 6),
        )
