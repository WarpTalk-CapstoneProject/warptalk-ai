"""LiveKit Ingress Worker — Connects to LiveKit and extracts audio chunks.

Uses Silero VAD as a standalone gatekeeper to only publish audio chunks
that contain actual speech, preventing Whisper from seeing silence/noise.
"""

import asyncio
import copy
import json
import time
from collections import deque
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


def _is_ai_bot_identity(identity: str) -> bool:
    return identity.startswith(_AI_BOT_IDENTITY_PREFIXES)


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
        self.audio_tasks: dict[str, asyncio.Task[None]] = {}
        self._vad_model: Any | None = None

    async def load_model(self) -> None:
        """Load Silero VAD model."""
        self._vad_model, _ = torch.hub.load(
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
                        asyncio.create_task(self.handle_track_published(payload))
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
        await self._hydrate_room_status(room_name)

        self.logger.info(
            "received_track_published",
            room=room_name,
            participant=participant_identity,
            track=track_id,
        )

        # Always disconnect old room and rejoin fresh
        old_room = self.rooms.pop(room_name, None)
        if old_room:
            self.logger.info("disconnecting_stale_room", room=room_name)
            # Cancel all audio processing tasks for this room
            for tid, task in list(self.audio_tasks.items()):
                task.cancel()
            self.audio_tasks.clear()
            await old_room.disconnect()

        room = await self.join_room(room_name)
        if room:
            self.rooms[room_name] = room

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
                task = asyncio.create_task(
                    self.process_audio_track(room_name, participant.identity, track)
                )
                self.audio_tasks[track.sid] = task

        try:
            await room.connect(self.settings.livekit.url, token)
            self.logger.info("connected_to_room", room=room_name)

            # Subscribe to already-published audio tracks from existing participants
            for participant in room.remote_participants.values():
                if _is_ai_bot_identity(participant.identity):
                    continue
                for pub in participant.track_publications.values():
                    if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                        self.logger.info(
                            "subscribing_existing_audio_track",
                            participant=participant.identity,
                            track=pub.track.sid,
                        )
                        task = asyncio.create_task(
                            self.process_audio_track(room_name, participant.identity, pub.track)
                        )
                        self.audio_tasks[pub.track.sid] = task

            return room
        except Exception:
            self.logger.exception("failed_to_join_room", room=room_name)
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
