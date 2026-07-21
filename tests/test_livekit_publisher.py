"""Tests for LiveKitTTSPublisher — publishing synthesized TTS audio as a LiveKit
audio track (bot participant per meeting_id/target_lang), instead of relying on a
frontend SignalR audio-playback path that turned out not to exist.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import LiveKitSettings
from tts_worker.livekit_publisher import LiveKitTTSPublisher


def _settings() -> LiveKitSettings:
    return LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")


@pytest.fixture(autouse=True)
def mock_livekit_sdk():
    with patch("tts_worker.livekit_publisher.rtc") as mock_rtc, \
         patch("tts_worker.livekit_publisher.api") as mock_api:
        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.local_participant.publish_track = AsyncMock()
        mock_rtc.Room.return_value = mock_room

        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        mock_rtc.AudioSource.return_value = mock_source

        mock_track = MagicMock()
        mock_rtc.LocalAudioTrack.create_audio_track.return_value = mock_track

        token_builder = MagicMock()
        token_builder.with_identity.return_value = token_builder
        token_builder.with_name.return_value = token_builder
        token_builder.with_grants.return_value = token_builder
        token_builder.to_jwt.return_value = "fake-jwt"
        mock_api.AccessToken.return_value = token_builder

        yield {"rtc": mock_rtc, "api": mock_api, "room": mock_room, "source": mock_source, "token_builder": token_builder}


class TestLiveKitTTSPublisher:
    async def test_publish_pcm_connects_and_publishes_track(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        # 20ms @ 16kHz mono 16-bit = 640 bytes/frame; send exactly 2 frames.
        pcm = b"\x00\x01" * 640

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        mock_livekit_sdk["room"].connect.assert_awaited_once_with("ws://livekit:7880", "fake-jwt")
        mock_livekit_sdk["room"].local_participant.publish_track.assert_awaited_once()
        assert mock_livekit_sdk["source"].capture_frame.await_count == 2

    async def test_bot_identity_and_room_grant_match_speaker_and_lang(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320  # one 20ms frame

        await publisher.publish_pcm(
            "019f6a39-a32c-7745-886e-1fe622c1f747", "spk-42", "ja", pcm, sample_rate=16000
        )

        # Identity is language-first, speaker GUID last: ai-interpreter-{lang}-{speaker}.
        mock_livekit_sdk["token_builder"].with_identity.assert_called_once_with(
            "ai-interpreter-ja-spk-42"
        )
        mock_livekit_sdk["api"].VideoGrants.assert_called_once_with(
            room_join=True, room="019f6a39-a32c-7745-886e-1fe622c1f747"
        )

    async def test_reuses_bot_across_calls_for_same_speaker_and_lang(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        mock_livekit_sdk["room"].connect.assert_awaited_once()
        assert mock_livekit_sdk["source"].capture_frame.await_count == 2

    async def test_different_target_lang_gets_its_own_bot(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm("room-1", "s1", "ja", pcm, sample_rate=16000)

        assert mock_livekit_sdk["room"].connect.await_count == 2

    async def test_different_speaker_same_lang_gets_its_own_bot(self, mock_livekit_sdk) -> None:
        """The core of per-speaker routing: two speakers dubbed into the SAME language
        must get separate tracks so their audio plays in parallel, not serialized onto
        one shared interpreter track."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm("room-1", "s2", "vi", pcm, sample_rate=16000)

        assert mock_livekit_sdk["room"].connect.await_count == 2

    async def test_empty_pcm_is_a_noop(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())

        await publisher.publish_pcm("room-1", "s1", "vi", b"", sample_rate=16000)

        mock_livekit_sdk["room"].connect.assert_not_called()

    async def test_connect_error_is_caught_not_raised(self, mock_livekit_sdk) -> None:
        mock_livekit_sdk["room"].connect.side_effect = Exception("livekit unreachable")
        publisher = LiveKitTTSPublisher(_settings())

        # Must not raise — a LiveKit outage must never break the tts:results publish
        # path this sits alongside (billing/transcript persistence depend on it).
        await publisher.publish_pcm("room-1", "s1", "vi", b"\x00\x01" * 320, sample_rate=16000)

    async def test_partial_trailing_frame_is_dropped_not_corrupted(self, mock_livekit_sdk) -> None:
        """PCM not landing on an exact frame boundary must drop the remainder rather
        than send a truncated/misaligned AudioFrame to LiveKit."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = (b"\x00\x01" * 320) + b"\x02"  # one full frame + 1 stray byte

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        assert mock_livekit_sdk["source"].capture_frame.await_count == 1

    async def test_capture_frame_failure_evicts_bot_and_retries_once_on_fresh_connection(
        self, mock_livekit_sdk
    ) -> None:
        """capture_frame fails sporadically with InvalidState (known upstream LiveKit
        issue, not deterministically reproducible) — the mitigation is: evict and
        reconnect once before giving up on this sentence.
        """
        mock_livekit_sdk["source"].capture_frame = AsyncMock(
            side_effect=[Exception("InvalidState - failed to capture frame"), None]
        )
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        assert mock_livekit_sdk["room"].connect.await_count == 2
        assert ("room-1", "s1", "vi") in publisher._bots

    async def test_capture_frame_fails_twice_gives_up_without_raising(self, mock_livekit_sdk) -> None:
        mock_livekit_sdk["source"].capture_frame = AsyncMock(
            side_effect=Exception("InvalidState - failed to capture frame")
        )
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        assert mock_livekit_sdk["room"].connect.await_count == 2
        assert ("room-1", "s1", "vi") not in publisher._bots
