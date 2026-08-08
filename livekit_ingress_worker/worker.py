"""LiveKit Ingress Worker — Connects to LiveKit and extracts audio chunks.

Uses Silero VAD as a standalone gatekeeper to only publish audio chunks
that contain actual speech, preventing Whisper from seeing silence/noise.
"""

import asyncio
import copy
import json
import random
import time
from collections import deque
from contextlib import suppress
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from livekit import api, rtc
from redis.exceptions import RedisError

from livekit_ingress_worker.near_field_gate import NearFieldGate
from shared.base_worker import BaseWorker
from shared.schemas import AudioChunkMessage

SILERO_VAD_REPOSITORY = "snakers4/silero-vad:v6.2.1"
MIN_VAD_SPEECH_FRAMES = 3
VAD_WINDOW_SAMPLES = 512 * MIN_VAD_SPEECH_FRAMES
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


class LiveKitIngressWorker(BaseWorker):
    """Worker that joins LiveKit rooms and pushes audio to Redis Streams."""

    worker_name = "livekit_ingress"
    input_stream = "pubsub:meeting.track_published"  # Logical name
    consumer_group = ""

    # Silero VAD frame size: 512 samples = 32ms at 16kHz
    VAD_FRAME_SIZE = 512
    SAMPLE_RATE = 16000

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
        self._vad_model: Any | None = None
        # One lock per room, held across the whole handler. Events arrive as independent
        # tasks (see _consume_loop), so without this two events for the same room could
        # interleave a teardown and a rejoin, and both would dial LiveKit with the same
        # "AIBot_{room}" identity — the server resolves that by evicting one of them,
        # which is a third connect nobody asked for.
        self._room_locks: dict[str, asyncio.Lock] = {}
        self._event_tasks: set[asyncio.Task[None]] = set()
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

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Not used, we override _consume_loop for Pub/Sub."""
        pass

    async def _consume_loop(self) -> None:
        """Override to listen to Redis Pub/Sub instead of Streams."""
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

    def _start_pending_audio_tasks(self, room_name: str, room: rtc.Room) -> int:
        """Attach to every already-published human audio track we are not reading yet."""
        started = 0
        for participant in room.remote_participants.values():
            if _is_ai_bot_identity(participant.identity):
                continue
            for pub in participant.track_publications.values():
                track = pub.track
                if track is None or track.kind != rtc.TrackKind.KIND_AUDIO:
                    continue
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

        for room_name, room in list(self.rooms.items()):
            try:
                # A handle we hold on a room LiveKit already dropped bills nothing, but it
                # is not occupied either — let the same grace window retire it.
                count = self._human_participant_count(room) if room.isconnected() else 0
            except Exception:
                self.logger.warning("idle_sweep_room_probe_failed", room=room_name, exc_info=True)
                continue

            if count > 0:
                self._room_last_occupied[room_name] = now
                occupied += 1
                humans += count
                continue

            last_occupied = self._room_last_occupied.setdefault(room_name, now)
            if now - last_occupied >= _IDLE_ROOM_GRACE_S:
                idle_rooms.append(room_name)

        # The reason WT-314 ran undetected is that a leaked bot is completely silent. This
        # is the gauge that makes a recurrence visible in logs instead of on the invoice.
        if self.rooms or idle_rooms:
            self.logger.info(
                "livekit_ingress_room_census",
                connected_rooms=len(self.rooms),
                occupied_rooms=occupied,
                human_participants=humans,
                releasing_idle_rooms=len(idle_rooms),
                idle_releases_total=self._idle_releases_total,
            )

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

    def _is_translation_active(self, room_name: str) -> bool:
        return self._route_states.get(room_name) in {
            "IN_PROGRESS",
            "AUDIO_ROUTING_ACTIVE",
        }

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
                self.logger.info(
                    "audio_track_subscribed", participant=participant.identity, track=track.sid
                )
                self._start_audio_task(room_name, participant.identity, track)

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

        Silero requires 512-sample (32ms) frames. We process all frames
        in the window, but require roughly 96ms of speech evidence before returning
        the strongest probability. A single noisy frame must not classify the window
        as speech.
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

        # One near-field gate per track — see near_field_gate.py. It builds a running
        # peak-amplitude reference from this track's own earlier chunks, so it must live
        # for the whole track lifetime, not be recreated per chunk.
        near_field_gate = NearFieldGate(self.settings)
        # Silero VAD carries recurrent state. Sharing one model across concurrently
        # iterated participant tracks interleaves unrelated audio histories and causes
        # missed/fragmented speech. Each track owns an independent cloned state machine.
        track_vad_model = copy.deepcopy(self._require_vad_model())

        # State
        raw_buffer = bytearray()  # Incoming raw resampled audio
        speech_buffer = bytearray()  # Audio collected during speech
        pre_speech_ring: deque[bytes] = deque(
            maxlen=max(1, pre_speech_samples * 2 // (vad_window_samples * 2))
        )  # Rolling pre-speech windows (in bytes, 2 bytes/sample)
        is_speaking = False
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
                if not self._is_translation_active(room_name):
                    # The meeting microphone is allowed to remain published before the host
                    # starts translation and while translation is paused. Discard those frames
                    # completely so pre-start speech cannot leak into the first active chunk.
                    raw_buffer = bytearray()
                    speech_buffer = bytearray()
                    pre_speech_ring.clear()
                    is_speaking = False
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

                    # Run VAD on this 500ms window
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
                            for pre_window in pre_speech_ring:
                                speech_buffer.extend(pre_window)
                            self.logger.info(
                                "speech_start",
                                chunk_index=chunk_index,
                                vad_prob=round(vad_prob, 3),
                                pre_buffer_ms=len(speech_buffer) // 2 * 1000 // sample_rate,
                            )

                        speech_buffer.extend(window_data)
                        silence_counter = 0

                        # Check max chunk length — publish if exceeded
                        speech_samples = len(speech_buffer) // 2
                        if speech_samples >= max_chunk_samples:
                            await self._publish_speech_chunk(
                                room_name,
                                speaker_id,
                                speech_buffer,
                                chunk_index,
                                sample_rate,
                                near_field_gate=near_field_gate,
                            )
                            chunk_index += 1
                            speech_buffer = bytearray()

                    else:
                        # No speech in this window
                        if is_speaking:
                            silence_counter += 1
                            speech_buffer.extend(window_data)  # Keep recording during pauses

                            if silence_counter >= silence_hangover_windows:
                                # End of speech — publish if long enough
                                speech_samples = len(speech_buffer) // 2
                                if speech_samples >= min_speech_samples:
                                    await self._publish_speech_chunk(
                                        room_name,
                                        speaker_id,
                                        speech_buffer,
                                        chunk_index,
                                        sample_rate,
                                        near_field_gate=near_field_gate,
                                    )
                                    chunk_index += 1
                                else:
                                    self.logger.debug(
                                        "speech_too_short",
                                        samples=speech_samples,
                                        min_required=min_speech_samples,
                                    )

                                is_speaking = False
                                speech_buffer = bytearray()
                                silence_counter = 0
                                track_vad_model.reset_states()
                        else:
                            # Store in pre-speech ring for next onset
                            pre_speech_ring.append(window_data)

        except Exception:
            self.logger.exception("process_audio_track_error", track_sid=track.sid)
        finally:
            # Publish any remaining speech buffer
            if speech_buffer and len(speech_buffer) // 2 >= min_speech_samples:
                await self._publish_speech_chunk(
                    room_name,
                    speaker_id,
                    speech_buffer,
                    chunk_index,
                    sample_rate,
                    near_field_gate=near_field_gate,
                )
            self.logger.info("stopped_audio_stream_processing", track_sid=track.sid)

    def _require_vad_model(self) -> Any:
        if self._vad_model is None:
            raise RuntimeError("Silero VAD model is not loaded")
        return self._vad_model

    async def _publish_speech_chunk(
        self,
        room_name: str,
        speaker_id: str,
        speech_buffer: bytearray,
        chunk_index: int,
        sample_rate: int,
        near_field_gate: NearFieldGate | None = None,
    ) -> None:
        if not self._is_translation_active(room_name):
            self.logger.debug(
                "speech_chunk_discarded_translation_inactive",
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

        # Energy gate: skip chunks that are too quiet (noise, not speech)
        if raw_rms < 0.02:
            self.logger.debug(
                "skipped_low_energy_chunk",
                chunk_index=chunk_index,
                raw_rms=round(float(raw_rms), 6),
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

        # This speaker's own chosen speak-language — TranslationRoomHub.JoinTranslationRoom
        # persists it (see NormalizeLanguageCode there) keyed by userId, which is the same
        # value LiveKit uses as participant.identity/speaker_id here. Falls back to "auto"
        # (STT's own guess) only if the speaker somehow isn't registered yet.
        language = "auto"
        try:
            raw_language = await self.redis.hget(
                f"translationRoom:{room_name}:speak_languages", speaker_id
            )
        except RedisError:
            raw_language = None
            self.logger.warning(
                "speak_language_lookup_failed",
                room=room_name,
                speaker_id=speaker_id,
                exc_info=True,
            )
        if raw_language:
            language = raw_language.decode() if isinstance(raw_language, bytes) else raw_language

        msg = AudioChunkMessage(
            meeting_id=room_name,
            speaker_id=speaker_id,
            chunk_index=chunk_index,
            audio_data=bytes(pcm),
            sample_rate=sample_rate,
            language=language,
            timestamp_ms=int(time.time() * 1000),
        )

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
