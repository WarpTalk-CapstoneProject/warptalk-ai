"""A sentence the pipeline synthesized must reach the listener, not a track nobody speaks on.

Production 30 Sep, room 01a0f21a: one vi listener had picked a voice, so until the speaker's live
clone was ready every sentence was also rendered on `ai-interpreter-vi-voice-935a9060-{speaker}`.
The clone was cached at 18:42:33. From then on the worker rendered only the default (cloned)
track, but the preference bot stayed in the room until its idle timeout at 18:43:30 — and the web
client, which prefers a listener's own voice track whenever one exists for that speaker, kept
ignoring the default track. The 7 sentences in that minute were synthesized, published, billed,
and heard by nobody. Only the later ones came through, in the cloned voice.

Also here: a WAV header with no samples (what a retired Cartesia context returned before WT-874)
must not be cached and replayed as silence.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import LiveKitSettings, TTSSettings, WorkerSettings
from tests.test_tts_worker import _make_msg, _make_worker
from tts_worker.livekit_publisher import LiveKitTTSPublisher

PCM = b"\x00\x01" * 320
PREFERENCE = "voice-935a9060"


@pytest.fixture(autouse=True)
def mock_livekit_sdk():
    with (
        patch("tts_worker.livekit_publisher.rtc") as mock_rtc,
        patch("tts_worker.livekit_publisher.api") as mock_api,
    ):
        rooms: list[MagicMock] = []

        def _new_room() -> MagicMock:
            room = MagicMock()
            room.connect = AsyncMock()
            room.disconnect = AsyncMock()
            room.local_participant.publish_track = AsyncMock()
            rooms.append(room)
            return room

        mock_rtc.Room.side_effect = _new_room
        source = MagicMock()
        source.capture_frame = AsyncMock()
        mock_rtc.AudioSource.return_value = source

        token_builder = MagicMock()
        token_builder.with_identity.return_value = token_builder
        token_builder.with_name.return_value = token_builder
        token_builder.with_grants.return_value = token_builder
        token_builder.to_jwt.return_value = "fake-jwt"
        mock_api.AccessToken.return_value = token_builder
        yield rooms


def _publisher() -> LiveKitTTSPublisher:
    return LiveKitTTSPublisher(LiveKitSettings(url="wss://x", api_key="k", api_secret="s"))


async def test_a_variant_no_longer_rendered_leaves_the_room_at_once(mock_livekit_sdk) -> None:
    publisher = _publisher()
    await publisher.publish_pcm("m1", "s1", "vi", PCM, 16000)
    await publisher.publish_pcm("m1", "s1", "vi", PCM, 16000, voice_key=PREFERENCE)
    default_room, preference_room = mock_livekit_sdk

    retired = publisher.retire_voice_variants("m1", "s1", "vi", keep={""})
    await asyncio.sleep(0)

    assert retired == [PREFERENCE]
    assert ("m1", "s1", "vi", PREFERENCE) not in publisher._bots
    preference_room.disconnect.assert_awaited_once()
    # The track the listener must now fall back to is untouched.
    assert ("m1", "s1", "vi", "") in publisher._bots
    default_room.disconnect.assert_not_awaited()


async def test_other_speakers_and_languages_keep_their_variants(mock_livekit_sdk) -> None:
    publisher = _publisher()
    await publisher.publish_pcm("m1", "s2", "vi", PCM, 16000, voice_key=PREFERENCE)
    await publisher.publish_pcm("m1", "s1", "ja", PCM, 16000, voice_key=PREFERENCE)
    await publisher.publish_pcm("m2", "s1", "vi", PCM, 16000, voice_key=PREFERENCE)

    assert publisher.retire_voice_variants("m1", "s1", "vi", keep={""}) == []
    assert len(publisher._bots) == 3


async def test_a_variant_still_rendered_is_kept(mock_livekit_sdk) -> None:
    publisher = _publisher()
    await publisher.publish_pcm("m1", "s1", "vi", PCM, 16000, voice_key=PREFERENCE)

    assert publisher.retire_voice_variants("m1", "s1", "vi", keep={"", PREFERENCE}) == []
    assert ("m1", "s1", "vi", PREFERENCE) in publisher._bots


async def test_a_bot_mid_sentence_is_not_cut_off(mock_livekit_sdk) -> None:
    publisher = _publisher()
    await publisher.publish_pcm("m1", "s1", "vi", PCM, 16000, voice_key=PREFERENCE)
    key = ("m1", "s1", "vi", PREFERENCE)

    async with publisher._locks.setdefault(key, asyncio.Lock()):
        assert publisher.retire_voice_variants("m1", "s1", "vi", keep={""}) == []
    assert key in publisher._bots


async def test_a_speaker_who_gets_a_clone_stops_being_heard_on_the_preference_track(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    """The worker retires the variant BEFORE it synthesizes the first cloned sentence, so the
    listener's client has already switched to the default track when that sentence plays."""
    worker = _make_worker(mock_redis_client, worker_settings)
    order: list[str] = []
    publisher = MagicMock()
    publisher.retire_voice_variants = MagicMock(
        side_effect=lambda *a, **k: order.append("retire") or [PREFERENCE]
    )
    publisher.publish_pcm = AsyncMock()
    worker.livekit_publisher = publisher

    async def _synthesize(**_kwargs: object) -> tuple[bytes, int, str]:
        order.append("synthesize")
        return b"\x00" * 44 + PCM, 20, "cloned-voice"

    worker.cartesia.synthesize = _synthesize
    worker._get_voice_id = AsyncMock(return_value="cloned-voice")  # type: ignore[method-assign]
    mock_redis_client._redis.get.return_value = None

    await worker.process(b"msg-1", _make_msg().to_redis())

    assert order == ["retire", "synthesize"]
    publisher.retire_voice_variants.assert_called_once_with("m1", "s1", "vi", keep={""})


async def test_an_empty_cached_wav_is_a_miss(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    worker = _make_worker(mock_redis_client, worker_settings)
    mock_redis_client._redis.get.return_value = b"\x00" * 44

    await worker.process(b"msg-1", _make_msg().to_redis())

    worker.cartesia.synthesize.assert_awaited_once()


async def test_an_empty_rendering_is_not_cached_or_reported_as_synthesized(
    mock_redis_client, worker_settings: WorkerSettings
) -> None:
    worker = _make_worker(
        mock_redis_client, worker_settings, tts_settings=TTSSettings(prosody_continuity=False)
    )
    worker.cartesia.synthesize = AsyncMock(return_value=(b"\x00" * 44, 0, "voice"))
    mock_redis_client._redis.get.return_value = None
    mock_redis_client.set_with_ttl = AsyncMock()  # type: ignore[method-assign]

    await worker.process(b"msg-1", _make_msg().to_redis())

    mock_redis_client.set_with_ttl.assert_not_awaited()
    synthesized = [
        c.kwargs
        for c in worker.logger.info.call_args_list
        if c.args and c.args[0] == "audio_synthesized"
    ]
    assert synthesized and synthesized[0]["synthesized"] is False
