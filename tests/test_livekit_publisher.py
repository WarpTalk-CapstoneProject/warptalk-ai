"""Tests for LiveKitTTSPublisher — publishing synthesized TTS audio as a LiveKit
audio track (bot participant per meeting_id/target_lang), instead of relying on a
frontend SignalR audio-playback path that turned out not to exist.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import LiveKitSettings
from tts_worker.livekit_publisher import SESSION_IDLE_TIMEOUT_S, LiveKitTTSPublisher


def _settings() -> LiveKitSettings:
    return LiveKitSettings(url="ws://livekit:7880", api_key="key", api_secret="secret")


def test_tts_bot_idle_timeout_limits_cloud_participant_minutes() -> None:
    assert SESSION_IDLE_TIMEOUT_S == 60.0


@pytest.fixture(autouse=True)
def mock_livekit_sdk():
    with (
        patch("tts_worker.livekit_publisher.rtc") as mock_rtc,
        patch("tts_worker.livekit_publisher.api") as mock_api,
    ):
        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
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

        yield {
            "rtc": mock_rtc,
            "api": mock_api,
            "room": mock_room,
            "source": mock_source,
            "token_builder": token_builder,
        }


class TestLiveKitTTSPublisher:
    async def test_publish_pcm_connects_and_publishes_track(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        # 20ms @ 16kHz mono 16-bit = 640 bytes/frame; send exactly 2 frames.
        pcm = b"\x00\x01" * 640

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        mock_livekit_sdk["room"].connect.assert_awaited_once_with("ws://livekit:7880", "fake-jwt")
        mock_livekit_sdk["room"].local_participant.publish_track.assert_awaited_once()
        assert mock_livekit_sdk["source"].capture_frame.await_count == 2

    async def test_bot_identity_and_room_grant_match_speaker_and_lang(
        self, mock_livekit_sdk
    ) -> None:
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

    async def test_reuses_bot_across_calls_for_same_speaker_and_lang(
        self, mock_livekit_sdk
    ) -> None:
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
        assert ("room-1", "s1", "vi", "") in publisher._bots

    async def test_capture_frame_fails_twice_gives_up_without_raising(
        self, mock_livekit_sdk
    ) -> None:
        mock_livekit_sdk["source"].capture_frame = AsyncMock(
            side_effect=Exception("InvalidState - failed to capture frame")
        )
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)

        assert mock_livekit_sdk["room"].connect.await_count == 2
        assert ("room-1", "s1", "vi", "") not in publisher._bots

    async def test_idle_bot_is_swept_and_disconnected(self, mock_livekit_sdk) -> None:
        """A bot nobody has synthesized through in SESSION_IDLE_TIMEOUT_S — e.g. every
        listener switched away from its target_lang via SetListenLanguage — must
        disconnect instead of leaking its LiveKit room connection forever."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        key = ("room-1", "s1", "vi", "")
        assert key in publisher._bots

        # Simulate the idle timeout elapsing without touching real time.
        publisher._bots[key]["last_used"] -= publisher.settings.tts_bot_idle_timeout_s + 1

        # _get_or_create_bot() sweeps opportunistically on every call — any subsequent
        # publish (even for an unrelated key) triggers it.
        await publisher.publish_pcm("room-1", "s2", "ja", pcm, sample_rate=16000)
        await asyncio.sleep(0)  # let the fire-and-forget disconnect task run

        assert key not in publisher._bots
        mock_livekit_sdk["room"].disconnect.assert_awaited()

    async def test_a_minute_of_silence_keeps_the_bot_warm(self, mock_livekit_sdk) -> None:
        """Prod 3 Oct: a 60s timeout made every sentence after a pause pay a 2.5-3.6s join."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        key = ("room-1", "s1", "vi", "")
        publisher._bots[key]["last_used"] -= SESSION_IDLE_TIMEOUT_S + 1

        await publisher.publish_pcm("room-1", "s2", "ja", pcm, sample_rate=16000)
        await asyncio.sleep(0)

        assert key in publisher._bots

    async def test_retire_meeting_releases_only_that_meetings_bots(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320
        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm("room-2", "s1", "vi", pcm, sample_rate=16000)

        retired = publisher.retire_meeting("room-1", reason="paused")
        await asyncio.sleep(0)

        assert retired == 1
        assert ("room-1", "s1", "vi", "") not in publisher._bots
        assert ("room-2", "s1", "vi", "") in publisher._bots

    async def test_retire_meeting_lets_a_sentence_in_progress_finish(
        self, mock_livekit_sdk
    ) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320
        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        key = ("room-1", "s1", "vi", "")
        lock = publisher._locks.setdefault(key, asyncio.Lock())

        async with lock:
            assert publisher.retire_meeting("room-1", reason="paused") == 0
            assert key in publisher._bots
        # Expired rather than closed: the next sweep takes it once the sentence is done.
        publisher._sweep_idle_bots()
        await asyncio.sleep(0)
        assert key not in publisher._bots

    async def test_active_bot_survives_sweep(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm("room-1", "s2", "ja", pcm, sample_rate=16000)

        assert ("room-1", "s1", "vi", "") in publisher._bots
        assert ("room-1", "s2", "ja", "") in publisher._bots

    async def test_voice_key_gets_its_own_bot_with_suffixed_identity(
        self, mock_livekit_sdk
    ) -> None:
        """A listener's explicit voice preference (see TTSWorker._resolve_voice_variants)
        must land on its own dedicated track, distinct from the shared default track for
        the same speaker+language — this is what lets that one listener hear a different
        voice than everyone else without affecting them."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm(
            "room-1", "s1", "vi", pcm, sample_rate=16000, voice_key="voice-abc12345"
        )

        mock_livekit_sdk["token_builder"].with_identity.assert_called_once_with(
            "ai-interpreter-vi-voice-abc12345-s1"
        )
        assert ("room-1", "s1", "vi", "voice-abc12345") in publisher._bots

    async def test_default_and_voice_override_bots_coexist_independently(
        self, mock_livekit_sdk
    ) -> None:
        """The default track (voice_key="") and a preference-override track for the
        SAME speaker+language are separate bots — publishing to one must not disturb
        or replace the other."""
        publisher = LiveKitTTSPublisher(_settings())
        pcm = b"\x00\x01" * 320

        await publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000)
        await publisher.publish_pcm(
            "room-1", "s1", "vi", pcm, sample_rate=16000, voice_key="voice-abc12345"
        )

        assert mock_livekit_sdk["room"].connect.await_count == 2
        assert ("room-1", "s1", "vi", "") in publisher._bots
        assert ("room-1", "s1", "vi", "voice-abc12345") in publisher._bots

    async def test_concurrent_calls_for_same_key_never_overlap(self, mock_livekit_sdk) -> None:
        """tts_worker now dispatches translate:results messages concurrently (see
        TTSWorker._consume_loop) — two publish_pcm() calls for the SAME key running
        at the same time must never have their capture_frame() calls in flight
        simultaneously, or sentence 2's frames could interleave into sentence 1's on
        the shared AudioSource."""
        publisher = LiveKitTTSPublisher(_settings())
        in_flight = 0
        max_in_flight = 0

        async def fake_capture_frame(frame):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1

        mock_livekit_sdk["source"].capture_frame = AsyncMock(side_effect=fake_capture_frame)
        pcm = b"\x00\x01" * 640  # 2 frames per call

        await asyncio.gather(
            publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000),
            publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000),
        )

        assert max_in_flight == 1

    async def test_concurrent_calls_for_different_keys_run_in_parallel(
        self, mock_livekit_sdk
    ) -> None:
        """A DIFFERENT key (another speaker, or the same speaker's other target
        language) must NOT be blocked by an in-flight call for an unrelated key —
        the lock is per-key, not global."""
        publisher = LiveKitTTSPublisher(_settings())
        in_flight = 0
        max_in_flight = 0

        async def fake_capture_frame(frame):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.05)
            in_flight -= 1

        mock_livekit_sdk["source"].capture_frame = AsyncMock(side_effect=fake_capture_frame)
        pcm = b"\x00\x01" * 320  # 1 frame per call

        await asyncio.gather(
            publisher.publish_pcm("room-1", "s1", "vi", pcm, sample_rate=16000),
            publisher.publish_pcm("room-1", "s2", "ja", pcm, sample_rate=16000),
        )

        assert max_in_flight == 2


class TestStreamJoinsBeforeTheFirstChunk:
    """3 Oct 2026: a cold key paid Cartesia's first byte and THEN the LiveKit join, in series."""

    async def test_join_starts_when_the_stream_opens_not_at_the_first_chunk(
        self, mock_livekit_sdk
    ) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        async with publisher.stream("room-1", "s1", "vi", 16000):
            await asyncio.sleep(0)  # let the join task run, with no audio fed at all
            mock_livekit_sdk["room"].connect.assert_awaited_once()

    async def test_one_join_per_sentence_when_audio_follows(self, mock_livekit_sdk) -> None:
        publisher = LiveKitTTSPublisher(_settings())
        async with publisher.stream("room-1", "s1", "vi", 16000) as track:
            await track.feed(b"\x00\x01" * 320 * 10)

        assert mock_livekit_sdk["room"].connect.await_count == 1
        assert mock_livekit_sdk["source"].capture_frame.await_count > 0
        assert track.spoken_bytes > 0

    async def test_close_waits_for_a_join_still_in_flight(self, mock_livekit_sdk) -> None:
        # Released mid-join, the key's lock would let publish_pcm's fallback start a second bot
        # with the same identity.
        gate = asyncio.Event()

        async def slow_connect(*_args) -> None:
            await gate.wait()

        mock_livekit_sdk["room"].connect = AsyncMock(side_effect=slow_connect)
        publisher = LiveKitTTSPublisher(_settings())

        async def sentence_without_audio() -> None:
            async with publisher.stream("room-1", "s1", "vi", 16000):
                pass

        closing = asyncio.create_task(sentence_without_audio())
        await asyncio.sleep(0.05)
        assert not closing.done()
        gate.set()
        await asyncio.wait_for(closing, timeout=2)

        await publisher.publish_pcm("room-1", "s1", "vi", b"\x00\x01" * 320, sample_rate=16000)
        assert mock_livekit_sdk["room"].connect.await_count == 1

    async def test_a_failed_join_breaks_only_this_sentence(self, mock_livekit_sdk) -> None:
        mock_livekit_sdk["room"].connect = AsyncMock(side_effect=ConnectionError("down"))
        publisher = LiveKitTTSPublisher(_settings())
        async with publisher.stream("room-1", "s1", "vi", 16000) as track:
            await track.feed(b"\x00\x01" * 320 * 10)
        assert track.spoken_bytes == 0
