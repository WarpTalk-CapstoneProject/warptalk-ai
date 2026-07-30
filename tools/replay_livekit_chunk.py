"""Replay one retained Redis audio chunk through a real LiveKit microphone track.

This is a local QA utility for measuring the complete
LiveKit -> ingress/VAD -> STT -> translation path with deterministic PCM.
It never prints credentials or audio payloads.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from livekit import api, rtc
from redis.asyncio import Redis

from shared.config import WorkerSettings
from shared.schemas import AudioChunkMessage

FRAME_MS = 20
SUBSCRIBER_SETTLE_SECONDS = 2.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--stream", default="audio:chunks")
    return parser.parse_args()


async def replay(room_name: str, stream: str, stream_id: str) -> dict[str, int | str]:
    """Publish a retained chunk as a non-bot microphone participant."""
    settings = WorkerSettings()

    redis = Redis.from_url(
        settings.redis.url,
        password=settings.redis.password or None,
        decode_responses=False,
    )
    try:
        entries = await redis.xrange(stream, min=stream_id, max=stream_id, count=1)
    finally:
        await redis.aclose()
    if not entries:
        raise RuntimeError(f"Redis stream entry not found: {stream} {stream_id}")

    _, fields = entries[0]
    if fields is None:
        raise RuntimeError(f"Redis stream entry has no fields: {stream} {stream_id}")
    chunk = AudioChunkMessage.from_redis(fields)
    sample_rate = chunk.sample_rate
    identity = f"qa-replay-{int(time.time())}"
    token = (
        api.AccessToken(settings.livekit.api_key, settings.livekit.api_secret)
        .with_identity(identity)
        .with_name("WarpTalk deterministic replay")
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    livekit_room = rtc.Room()
    await livekit_room.connect(settings.livekit.url, token)
    source = rtc.AudioSource(sample_rate=sample_rate, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track("qa-replay-microphone", source)
    await livekit_room.local_participant.publish_track(
        track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    # Publishing the track and receiving the publish ACK does not mean remote
    # subscribers have installed their AudioStream yet. Give ingress enough
    # time to subscribe so the deterministic fixture is not truncated at the
    # beginning.
    await asyncio.sleep(SUBSCRIBER_SETTLE_SECONDS)

    frame_bytes = sample_rate * FRAME_MS // 1000 * 2
    usable = len(chunk.audio_data) - len(chunk.audio_data) % frame_bytes
    started_ms = int(time.time() * 1000)
    for offset in range(0, usable, frame_bytes):
        pcm = chunk.audio_data[offset : offset + frame_bytes]
        await source.capture_frame(
            rtc.AudioFrame(
                data=pcm,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )
        )

    duration_ms = usable // 2 * 1000 // sample_rate
    await asyncio.sleep(duration_ms / 1000 + 1.0)
    await livekit_room.disconnect()
    return {
        "identity": identity,
        "sample_rate": sample_rate,
        "duration_ms": duration_ms,
        "started_ms": started_ms,
    }


if __name__ == "__main__":
    parsed = _parse_args()
    print(asyncio.run(replay(parsed.room, parsed.stream, parsed.stream_id)))
