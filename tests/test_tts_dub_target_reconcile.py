"""A dub track whose (speaker, target language) stopped being a target leaves promptly.

Mid-meeting language changes (SetSpeakLanguage / SetListenLanguage / SetExternalMeetingLanguage)
reach tts_worker only as messages that stop arriving. The track used to stay for the idle
timeout, and while it is in the room the web client mutes the speaker's microphone for that
language's listeners — the bridge host who switched to the Meet side's language went silent there
for over a minute.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

from tts_worker.livekit_publisher import LiveKitTTSPublisher
from tts_worker.worker import TTSWorker

STANDIN = "00000000-0000-0000-0000-00000000b21d"


def _publisher_with(bots: dict[tuple[str, str, str, str], float]) -> LiveKitTTSPublisher:
    publisher = LiveKitTTSPublisher.__new__(LiveKitTTSPublisher)
    publisher._bots = {}
    publisher._locks = {}
    publisher._reaper = None
    for key, last_used in bots.items():
        room = MagicMock()
        room.disconnect = AsyncMock()
        publisher._bots[key] = {"room": room, "last_used": last_used}
    return publisher


def _worker(publisher: LiveKitTTSPublisher, listen: dict, speak: dict) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.livekit_publisher = publisher
    worker.logger = MagicMock()
    worker.redis = MagicMock()

    async def hgetall(key: str) -> dict:
        if key.endswith(":speak_languages"):
            return dict(speak)
        if key.endswith(":languages"):
            return dict(listen)
        raise AssertionError(key)

    worker.redis.hgetall = AsyncMock(side_effect=hgetall)
    return worker


class TestPublisherRetireTargetLanguage:
    async def test_retires_default_and_variant_tracks_of_that_language_only(self) -> None:
        now = time.monotonic()
        publisher = _publisher_with(
            {
                ("m", "host", "en", ""): now,
                ("m", "host", "en", "voice-1"): now,
                ("m", "host", "ja", ""): now,
                ("m", "other", "en", ""): now,
            }
        )
        assert publisher.retire_target_language("m", "host", "en", "test") == 2
        await asyncio.sleep(0)
        assert set(publisher._bots) == {("m", "host", "ja", ""), ("m", "other", "en", "")}

    async def test_a_bot_mid_sentence_is_left_for_the_reaper(self) -> None:
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic()})
        lock = asyncio.Lock()
        await lock.acquire()
        publisher._locks[("m", "host", "en", "")] = lock
        assert publisher.retire_target_language("m", "host", "en", "test") == 0
        assert ("m", "host", "en", "") in publisher._bots

    def test_dub_targets_reports_latest_use_per_language(self) -> None:
        publisher = _publisher_with({("m", "s", "en", ""): 10.0, ("m", "s", "en", "v"): 20.0})
        assert publisher.dub_targets() == {("m", "s"): {"en": 20.0}}


class TestReconcileDubTargets:
    async def test_last_listener_leaving_a_language_retires_its_track(self) -> None:
        publisher = _publisher_with(
            {("m", "host", "ja", ""): time.monotonic(), ("m", "host", "en", ""): time.monotonic()}
        )
        # The only ja listener switched to en.
        worker = _worker(publisher, listen={"host": "vi", "guest": "en"}, speak={"host": "vi"})
        assert await worker._reconcile_dub_targets() == 1
        assert set(publisher._bots) == {("m", "host", "en", "")}

    async def test_bridge_host_switching_to_the_meet_language_frees_their_microphone(
        self,
    ) -> None:
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic() - 10})
        speak = {"host": "vi", STANDIN: "en"}
        listen = {"host": "vi", STANDIN: "en"}
        worker = _worker(publisher, listen=listen, speak=speak)
        # First look: nothing is a change yet, and en is wanted.
        assert await worker._reconcile_dub_targets() == 0
        # SetSpeakLanguage(en) for the host.
        speak["host"] = "en"
        assert await worker._reconcile_dub_targets() == 1
        assert publisher._bots == {}

    async def test_a_track_translation_still_feeds_is_kept_after_the_change(self) -> None:
        # Declared en now, but still audibly speaking vi: translation keeps producing en dubs
        # (segment language vi), so the en track keeps publishing after the change.
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic()})
        speak = {"host": "vi"}
        worker = _worker(publisher, listen={"host": "vi", "guest": "en"}, speak=speak)
        await worker._reconcile_dub_targets()
        speak["host"] = "en"
        await worker._reconcile_dub_targets()  # retires the bot that predates the change
        publisher._bots.clear()
        changed_at = worker._dub_declared_speak[("m", "host")][1]
        assert changed_at is not None
        room = MagicMock()
        room.disconnect = AsyncMock()
        publisher._bots[("m", "host", "en", "")] = {"room": room, "last_used": changed_at + 60}
        assert await worker._reconcile_dub_targets() == 0
        assert ("m", "host", "en", "") in publisher._bots

    async def test_a_declaration_that_never_changed_never_retires_a_wanted_track(self) -> None:
        # Declared en from the start, speaking vi: the en track is live and must stay.
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic() - 30})
        worker = _worker(publisher, listen={"host": "vi", "guest": "en"}, speak={"host": "en"})
        assert await worker._reconcile_dub_targets() == 0
        assert await worker._reconcile_dub_targets() == 0
        assert ("m", "host", "en", "") in publisher._bots

    async def test_far_side_language_change_moves_the_stand_in_dub(self) -> None:
        # SetExternalMeetingLanguage(en -> fr): the host's en dub into Meet has no listener.
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic()})
        worker = _worker(
            publisher,
            listen={"host": "vi", STANDIN: "fr"},
            speak={"host": "vi", STANDIN: "fr"},
        )
        assert await worker._reconcile_dub_targets() == 1

    async def test_a_multi_language_far_side_is_never_treated_as_speaking_a_target(
        self,
    ) -> None:
        publisher = _publisher_with({("m", STANDIN, "vi", ""): time.monotonic() - 30})
        speak = {"host": "vi", STANDIN: "en"}
        worker = _worker(publisher, listen={"host": "vi", STANDIN: "en"}, speak=speak)
        await worker._reconcile_dub_targets()
        speak[STANDIN] = "auto"
        assert await worker._reconcile_dub_targets() == 0

    async def test_nobody_else_registered_mirrors_translations_english_fallback(self) -> None:
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic()})
        worker = _worker(publisher, listen={"host": "vi"}, speak={"host": "vi"})
        assert await worker._reconcile_dub_targets() == 0

    async def test_an_empty_or_unreadable_room_is_left_alone(self) -> None:
        publisher = _publisher_with({("m", "host", "en", ""): time.monotonic()})
        worker = _worker(publisher, listen={}, speak={})
        assert await worker._reconcile_dub_targets() == 0

        worker.redis.hgetall = AsyncMock(side_effect=ConnectionError("down"))
        assert await worker._reconcile_dub_targets() == 0
        assert ("m", "host", "en", "") in publisher._bots

    async def test_no_publisher_is_a_no_op(self) -> None:
        worker = TTSWorker.__new__(TTSWorker)
        worker.livekit_publisher = None
        assert await worker._reconcile_dub_targets() == 0
