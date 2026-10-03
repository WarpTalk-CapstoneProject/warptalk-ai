"""A Meet-side sentence's dub: what decides it, and that every decision is now on the record.

WHAT WAS REPORTED (prod 2026-10-03, Google Meet bridge rooms 01a103e0 and 01a103ef)
    "I speak, they hear the cloned voice. When they speak, I hear no dub." The Meet-side lines
    were on the host's screen WITH their English translation, so capture, STT and translation had
    worked. Whether the dub was then synthesized, into which language, and under which LiveKit
    identity could not be read back from this side's logs: two exits said nothing, and no line
    named the identity the listener's client has to match.

WHAT THESE PIN
    translation_worker — the stand-in's targets are the OTHER participants' listen languages,
        minus the language the Meet side spoke; with nobody registered it is the "en" fallback.
        Each of those is one `far_side_dub_targets` line. A native speaker gets none.
    tts_worker — a translated stand-in sentence is spoken on `ai-interpreter-{target}-{stand-in}`
        whatever its caption name looks like (long, a whole sentence, different every line),
        the host's text-only mode never reaches it, and every outcome is one
        `far_side_dub_decision` line that never carries the name. A native speaker gets none.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TranslationSettings, TTSSettings, WorkerSettings
from shared.control_markers import EXTERNAL_BRIDGE_SPEAKER_ID
from shared.schemas import TranslationResultMessage
from translation_worker.worker import TranslationWorker
from tts_worker.livekit_publisher import interpreter_identity
from tts_worker.worker import TTSWorker, far_speaker_voice_field, far_speaker_voice_key

MEETING = "m1"
STAND_IN = EXTERNAL_BRIDGE_SPEAKER_ID
HOST = "019f0d00-0de0-7000-9000-000000000001"
STAND_IN_DUB = f"ai-interpreter-en-{STAND_IN}"

# A caption sentence a browser extension made look like a participant name.
GARBAGE_NAMES = [
    "Xin chào Hạnh Nhi, hôm nay chúng ta bắt đầu cuộc họp nhé",
    "Mình sẽ nói về kế hoạch của tuần này và những việc còn dang dở",
    "Sau đó mỗi người cập nhật phần việc của mình trong vài phút",
]

# 44 bytes of WAV header and one 20 ms frame of PCM: anything shorter is "nothing to publish".
_WAV = b"\x00" * 44 + b"\x01\x02" * 480


def _events(worker: Any, name: str) -> list[dict[str, Any]]:
    return [call.kwargs for call in worker.logger.info.call_args_list if call.args[:1] == (name,)]


def _everything_logged(worker: Any) -> str:
    return " ".join(
        str(call)
        for level in ("debug", "info", "warning", "error", "exception")
        for call in getattr(worker.logger, level).call_args_list
    )


# ── translation_worker: which languages a Meet-side sentence is dubbed into ─────────────────────


def _translation_worker(mock_redis_client: Any, worker_settings: WorkerSettings) -> Any:
    worker = TranslationWorker.__new__(TranslationWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    worker.translation_settings = TranslationSettings()
    worker.worker_name = "translation"
    return worker


class TestWhichLanguagesTheMeetSideIsDubbedInto:
    async def test_the_listeners_language_and_the_line_says_who_listens(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        """The reported setup: the host listens in English, the Meet side spoke Vietnamese."""
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {
            HOST.encode(): b"en",
            STAND_IN.encode(): b"vi",
        }

        assert await worker._get_target_languages(MEETING, STAND_IN, "vi") == {"en"}

        assert _events(worker, "far_side_dub_targets") == [
            {
                "meeting_id": MEETING,
                "source_lang": "vi",
                "targets": ["en"],
                "listeners": {"en": 1},
                "fallback_english": False,
            }
        ]

    async def test_a_listener_in_the_language_spoken_is_no_dub_and_that_is_by_design(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {
            HOST.encode(): b"vi",
            STAND_IN.encode(): b"en",
        }

        assert await worker._get_target_languages(MEETING, STAND_IN, "vi") == set()

        (line,) = _events(worker, "far_side_dub_targets")
        assert line["targets"] == []
        assert line["listeners"] == {"vi": 1}
        assert line["fallback_english"] is False

    async def test_a_listener_missing_from_the_hash_is_the_english_fallback_and_is_told_apart(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        """The host's hub socket is down, so only the stand-in's own seat is in the hash. The
        sentence is still dubbed into English — and a host listening in anything else is not
        tuned to that track. Before this line it read exactly like a real English listener."""
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {STAND_IN.encode(): b"vi"}

        assert await worker._get_target_languages(MEETING, STAND_IN, "vi") == {"en"}

        (line,) = _events(worker, "far_side_dub_targets")
        assert line["targets"] == ["en"]
        assert line["listeners"] == {}
        assert line["fallback_english"] is True

    async def test_a_native_speaker_is_not_a_bridge_line(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _translation_worker(mock_redis_client, worker_settings)
        mock_redis_client._redis.hgetall.return_value = {
            HOST.encode(): b"en",
            STAND_IN.encode(): b"vi",
        }

        assert await worker._get_target_languages(MEETING, HOST, "en") == {"vi"}

        assert _events(worker, "far_side_dub_targets") == []


# ── tts_worker: what becomes of the translated sentence ─────────────────────────────────────────


def _tts_worker(mock_redis_client: Any, worker_settings: WorkerSettings) -> Any:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = worker_settings
    worker.redis = mock_redis_client
    worker.logger = MagicMock()
    # The one-shot path: these are about whether and where a sentence is spoken, not transport.
    # The far-speaker clone flag is left at its default (on), as in production.
    worker.tts_settings = TTSSettings(prosody_continuity=False)
    worker._route_states = {}
    worker._room_routes = {}
    worker._consumer_name = "test-consumer"
    worker.worker_name = "tts"
    worker.cartesia = MagicMock()
    worker.cartesia.generation_slot = MagicMock(return_value=asyncio.Semaphore(64))
    worker.cartesia.synthesize = AsyncMock(return_value=(_WAV, 20, "resolved-voice-id"))
    worker.cartesia.list_voices = AsyncMock(
        return_value=[
            {"id": f"en-{n}", "name": str(n), "gender": "feminine" if n % 2 else "masculine"}
            for n in range(8)
        ]
    )
    worker.livekit_publisher = MagicMock()
    worker.livekit_publisher.publish_pcm = AsyncMock()
    worker.livekit_publisher.set_voice_kind = AsyncMock()
    mock_redis_client._redis.hget.return_value = None
    mock_redis_client._redis.get.return_value = None
    return worker


def _sentence(
    *,
    speaker: str = STAND_IN,
    text: str = "Hello Hạnh Nhi.",
    source: str = "vi",
    target: str = "en",
    segment: str = "seg-1",
    name: str | None = None,
    confidence: float | None = None,
) -> dict[str, str]:
    return TranslationResultMessage(
        segment_id=segment,
        meeting_id=MEETING,
        speaker_id=speaker,
        original_text="Xin chào Hạnh Nhi.",
        translated_text=text,
        source_lang=source,
        target_lang=target,
        far_speaker_name=name,
        far_speaker_confidence=confidence,
    ).to_redis()


def _published_to(worker: Any) -> list[tuple[str, str, str, str]]:
    """(meeting, speaker, language, voice_key) of every push onto a LiveKit dub track."""
    return [
        (*call.args[:3], call.kwargs.get("voice_key", ""))
        for call in worker.livekit_publisher.publish_pcm.await_args_list
    ]


def test_the_identity_is_the_one_the_listeners_client_matches() -> None:
    assert interpreter_identity("en", STAND_IN) == STAND_IN_DUB
    assert (
        interpreter_identity("en", STAND_IN, "voice-abc12345")
        == f"ai-interpreter-en-voice-abc12345-{STAND_IN}"
    )


class TestAMeetSideSentenceIsSpoken:
    async def test_on_the_stand_ins_own_track_in_the_listeners_language(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _sentence())

        assert _published_to(worker) == [(MEETING, STAND_IN, "en", "")]
        (line,) = _events(worker, "far_side_dub_decision")
        assert line["decision"] == "speak"
        assert line["identities"] == [STAND_IN_DUB]
        assert line["voice_type"] == "default"
        assert (line["meeting_id"], line["segment_id"]) == (MEETING, "seg-1")
        assert (line["source_lang"], line["target_lang"]) == ("vi", "en")
        # Nobody named it: the shared stand-in voice.
        assert line["far_speaker_hash"] is None
        assert line["far_speaker_name_chars"] == 0

    async def test_whatever_the_caption_name_looks_like(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        """A caption sentence mistaken for a participant name, certain, and different on every
        line: each sentence is still spoken, at once, on the same track. Only the stock voice
        differs, and the log says so without the name."""
        worker = _tts_worker(mock_redis_client, worker_settings)

        for index, name in enumerate(GARBAGE_NAMES):
            await worker.process(
                f"msg-{index}".encode(),
                _sentence(segment=f"seg-{index}", name=name, confidence=1.0),
            )

        assert _published_to(worker) == [(MEETING, STAND_IN, "en", "")] * len(GARBAGE_NAMES)
        assert worker.cartesia.synthesize.await_count == len(GARBAGE_NAMES)
        lines = _events(worker, "far_side_dub_decision")
        assert [line["decision"] for line in lines] == ["speak"] * len(GARBAGE_NAMES)
        assert [line["identities"] for line in lines] == [[STAND_IN_DUB]] * len(GARBAGE_NAMES)
        assert [line["far_speaker_name_chars"] for line in lines] == [
            len(name) for name in GARBAGE_NAMES
        ]
        # The hash far_speaker_voice_assigned carries, one per "name".
        assert [line["far_speaker_hash"] for line in lines] == [
            far_speaker_voice_field(far_speaker_voice_key(STAND_IN, name, 1.0) or "")
            for name in GARBAGE_NAMES
        ]
        assert len({line["far_speaker_hash"] for line in lines}) == len(GARBAGE_NAMES)
        logged = _everything_logged(worker)
        for name in GARBAGE_NAMES:
            assert name not in logged
            assert name.casefold() not in logged

    async def test_an_unsure_name_is_the_shared_voice_and_still_spoken(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _sentence(name="Lan Nguyen", confidence=0.5))

        assert _published_to(worker) == [(MEETING, STAND_IN, "en", "")]
        (line,) = _events(worker, "far_side_dub_decision")
        assert line["decision"] == "speak"
        assert line["far_speaker_hash"] is None
        assert line["far_speaker_confidence"] == pytest.approx(0.5)
        assert "Lan Nguyen" not in _everything_logged(worker)

    async def test_the_hosts_text_only_mode_never_reaches_the_meet_sides_dub(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        """Text-only is the OUTBOUND direction: the backend marks only host -> stand-in routes
        (TranslationRoomAudioRouteMapper.IsTextOnlyBridgeOutbound). The host's own dub into the
        Meet side's language is skipped; the Meet side's dub for the host is not."""
        worker = _tts_worker(mock_redis_client, worker_settings)
        worker._room_routes = {
            MEETING: [
                {"SourceUserId": HOST, "TargetLanguage": "vi", "TextOnly": True},
                {"SourceUserId": STAND_IN, "TargetLanguage": "en", "TextOnly": False},
            ]
        }
        # The durable snapshot agrees, so the skip below is confirmed rather than overturned.
        worker._load_route_snapshot = AsyncMock(return_value=True)

        await worker.process(b"msg-1", _sentence())
        await worker.process(
            b"msg-2",
            _sentence(speaker=HOST, text="Xin chào.", source="en", target="vi", segment="seg-2"),
        )

        assert _published_to(worker) == [(MEETING, STAND_IN, "en", "")]
        assert [line["decision"] for line in _events(worker, "far_side_dub_decision")] == ["speak"]

    async def test_a_room_that_never_chose_text_only_dubs(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        """Routes from a backend that does not send the field at all: the default is the dub."""
        worker = _tts_worker(mock_redis_client, worker_settings)
        worker._room_routes = {MEETING: [{"SourceUserId": STAND_IN, "TargetLanguage": "en"}]}

        await worker.process(b"msg-1", _sentence())

        assert _published_to(worker) == [(MEETING, STAND_IN, "en", "")]


class TestASkippedMeetSideSentenceSaysWhy:
    @pytest.mark.parametrize(
        ("room_status", "reason"),
        [("PAUSED", "room_paused"), ("TEXT_ONLY_MODE", "room_text_only_mode")],
    )
    async def test_the_two_exits_that_used_to_say_nothing(
        self,
        mock_redis_client: Any,
        worker_settings: WorkerSettings,
        room_status: str,
        reason: str,
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)
        worker._route_states = {MEETING: room_status}

        await worker.process(b"msg-1", _sentence())

        worker.cartesia.synthesize.assert_not_called()
        (line,) = _events(worker, "far_side_dub_decision")
        assert (line["decision"], line["reason"]) == ("skip", reason)

    async def test_the_language_the_meet_side_already_spoke(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _sentence(source="vi", target="vi-VN"))

        worker.cartesia.synthesize.assert_not_called()
        (line,) = _events(worker, "far_side_dub_decision")
        assert (line["decision"], line["reason"]) == ("skip", "same_language")

    async def test_the_empty_turn_closing_marker_is_not_a_sentence(
        self, mock_redis_client: Any, worker_settings: WorkerSettings
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)

        await worker.process(b"msg-1", _sentence(text=""))

        worker.cartesia.synthesize.assert_not_called()
        assert _events(worker, "far_side_dub_decision") == []


class TestNativeSpeakersAreUntouched:
    @pytest.mark.parametrize("room_status", ["IN_PROGRESS", "PAUSED", "TEXT_ONLY_MODE"])
    async def test_no_bridge_line_whatever_happens_to_the_sentence(
        self, mock_redis_client: Any, worker_settings: WorkerSettings, room_status: str
    ) -> None:
        worker = _tts_worker(mock_redis_client, worker_settings)
        worker._route_states = {MEETING: room_status}

        # A native speaker carrying far_speaker fields is still a native speaker.
        await worker.process(
            b"msg-1",
            _sentence(speaker=HOST, source="en", target="vi", name="Lan Nguyen", confidence=1.0),
        )

        assert _events(worker, "far_side_dub_decision") == []
        expected = [(MEETING, HOST, "vi", "")] if room_status == "IN_PROGRESS" else []
        assert _published_to(worker) == expected
