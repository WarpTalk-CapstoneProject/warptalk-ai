"""How the worker decides between a continuous turn and an independent generation.

The sentence-boundary protocol itself is covered in test_prosody_context.py. This covers the
wiring around it: when a context is opened, when it is reused, when it is closed, and — the part
that matters most in a live meeting — that every failure still produces audio.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared import isochrony
from shared.config import TTSSettings, WorkerSettings
from shared.schemas import ProsodyEnvelope, TranslationResultMessage
from tts_worker.worker import TTSWorker


class _FakeTurn:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, Any]] = []
        self.closed = False
        self.abandoned = False
        self.fail = False
        self.fail_after_streaming = False
        #: Retire the context on the way out of the NEXT speak(), without raising — the real
        #: ProsodyContext does this when Cartesia sends `done` (prosody_context.py `_collect`).
        self.retire_after_speaking = False

    @property
    def is_closed(self) -> bool:
        return self.closed

    async def speak(
        self,
        text: str,
        generation_config: Any = None,
        on_pcm: Any = None,
    ) -> tuple[bytes, int]:
        if self.fail:
            raise RuntimeError("socket died")
        if self.closed:
            # Exactly what the real one does, and the symptom seen in production.
            raise RuntimeError("ProsodyContext is closed")
        self.spoken.append((text, generation_config))
        pcm = b"\x01\x02" * 100
        if on_pcm is not None:
            await on_pcm(pcm)
        if self.fail_after_streaming:
            # The socket dies with part of the sentence already on the track — the one case
            # the fallback cannot simply be played into.
            raise RuntimeError("socket died mid-sentence")
        if self.retire_after_speaking:
            # Cartesia's `done`: this sentence is fine, the NEXT one cannot use this context.
            self.closed = True
        return b"\x00" * 44 + pcm, 12

    async def aclose(self) -> None:
        self.closed = True

    async def abandon(self) -> None:
        self.abandoned = True


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeSynthesizer:
    """Records which path was taken."""

    def __init__(self) -> None:
        self.one_shot_calls: list[str] = []
        self.opened: list[str] = []
        self.turns: list[_FakeTurn] = []
        self.connections: list[_FakeConnection] = []

    async def synthesize(
        self, *, text: str, language: str, voice_id: str | None, generation_config: Any
    ) -> tuple[bytes, int, str]:
        self.one_shot_calls.append(text)
        return b"\x00" * 44 + b"\xaa\xbb" * 10, 5, voice_id or "default-voice"

    async def open_prosody_context(
        self, *, context_id: str, language: str, voice_id: str | None
    ) -> tuple[_FakeTurn, _FakeConnection]:
        self.opened.append(context_id)
        turn, connection = _FakeTurn(), _FakeConnection()
        self.turns.append(turn)
        self.connections.append(connection)
        return turn, connection


class _FakeTrack:
    """Stands in for TrackStream — records what was streamed, and how much was heard."""

    def __init__(self, deaf: bool = False) -> None:
        self.fed: list[bytes] = []
        self.closed = False
        self.first_audio_at: float | None = None
        self._deaf = deaf

    async def feed(self, pcm: bytes) -> None:
        self.fed.append(pcm)
        if not self._deaf and self.first_audio_at is None:
            self.first_audio_at = time.monotonic()

    @property
    def spoken_bytes(self) -> int:
        if not self.closed:
            # The real pump is still draining when speak() returns, so a count taken before
            # close() undercounts — modelled here so a caller that asks too early fails.
            return 0
        return 0 if self._deaf else sum(len(c) for c in self.fed)


class _FakePublisher:
    """Only the streaming seam — the rest of LiveKitTTSPublisher is covered elsewhere."""

    def __init__(self, deaf: bool = False) -> None:
        self.tracks: list[_FakeTrack] = []
        self.published: list[bytes] = []
        self._deaf = deaf

    @asynccontextmanager
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[_FakeTrack]:
        track = _FakeTrack(deaf=self._deaf)
        self.tracks.append(track)
        try:
            yield track
        finally:
            track.closed = True

    async def publish_pcm(self, *args: Any, **kwargs: Any) -> None:
        self.published.append(args[3])


def _worker(
    *, continuity: bool, publisher: _FakePublisher | None = None
) -> tuple[TTSWorker, _FakeSynthesizer]:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(prosody_continuity=continuity)
    worker.logger = MagicMock()
    worker._turns = {}
    worker._turn_connections = {}
    worker.livekit_publisher = publisher  # type: ignore[assignment]
    synthesizer = _FakeSynthesizer()
    worker.cartesia = synthesizer  # type: ignore[assignment]
    return worker, synthesizer


def _msg(text: str, *, final: bool = False, chunk: int = 0) -> TranslationResultMessage:
    return TranslationResultMessage(
        segment_id="seg-1",
        meeting_id="m1",
        speaker_id="s1",
        original_text="src",
        translated_text=text,
        source_lang="en",
        target_lang="vi",
        chunk_index=chunk,
        is_final_chunk=final,
    )


async def _say(worker: TTSWorker, message: TranslationResultMessage, voice: str = "v1") -> Any:
    return await worker._synthesize_sentence(
        translation=message,
        text=message.translated_text,
        voice_id=voice,
        voice_key="",
        generation_config={"speed": 1.0, "volume": 1.0},
    )


@pytest.mark.asyncio
async def test_turning_it_off_falls_back_to_the_proven_one_shot_path() -> None:
    """The escape hatch has to keep working now that the default is ON.

    It shipped dark in v79 and was turned on for v81 once the wedged-socket hang was closed —
    but the path still has not been LISTENED to, so `TTS_PROSODY_CONTINUITY=false` must remain
    a real off switch that needs no rebuild.
    """
    assert TTSSettings().prosody_continuity is True, "the default was flipped on for v81"

    worker, synth = _worker(continuity=False)
    await _say(worker, _msg("Một."))

    assert synth.one_shot_calls == ["Một."]
    assert synth.opened == []


@pytest.mark.asyncio
async def test_the_sentences_of_one_turn_share_a_context() -> None:
    """The whole point. Two sentences of the same turn must be one prosodic thread, not two
    independent generations."""
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0))
    await _say(worker, _msg("Câu hai.", chunk=1))

    assert len(synth.opened) == 1, "the second sentence opened a second context"
    assert [text for text, _ in synth.turns[0].spoken] == ["Câu một.", "Câu hai."]
    assert synth.one_shot_calls == []


@pytest.mark.asyncio
async def test_the_turn_ends_where_the_speaker_stopped() -> None:
    # is_final_chunk is the only signal that carries where the SPEAKER stopped, as opposed to
    # where a chunk boundary happened to fall.
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0))
    assert synth.turns[0].closed is False

    await _say(worker, _msg("Câu hai.", chunk=1, final=True))

    assert synth.turns[0].closed is True
    assert synth.connections[0].closed is True, "the socket outlived its context"
    assert worker._turns == {}, "a finished turn must not be reused by the next one"


@pytest.mark.asyncio
async def test_a_new_turn_after_the_last_one_closed_opens_a_fresh_context() -> None:
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Lượt một.", final=True))
    await _say(worker, _msg("Lượt hai.", final=True))

    assert len(synth.opened) == 2


@pytest.mark.asyncio
async def test_a_broken_context_still_produces_audio() -> None:
    """The failure that matters. A dead socket must not become silence in the meeting — this
    sentence falls back to the one-shot path, and the dead turn is discarded rather than
    retried into."""
    worker, synth = _worker(continuity=True)
    await _say(worker, _msg("Câu một.", chunk=0))
    synth.turns[0].fail = True

    sentence = await _say(worker, _msg("Câu hai.", chunk=1))

    assert synth.one_shot_calls == ["Câu hai."]
    assert len(sentence.audio) > 44
    assert worker._turns == {}, "the failed turn was kept and would fail again"
    assert sentence.already_spoken is False, (
        "nothing was streamed, so the fallback must still be played"
    )


@pytest.mark.asyncio
async def test_a_voice_change_mid_meeting_does_not_continue_the_old_voice() -> None:
    # voice_clone_max_upgrades replaces a speaker's voice mid-meeting. Continuing a turn into a
    # different voice would be a worse seam than the one this removes.
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Câu một.", chunk=0), voice="voice-a")
    await _say(worker, _msg("Câu hai.", chunk=1), voice="voice-b")

    assert len(synth.opened) == 2


@pytest.mark.asyncio
async def test_ending_a_room_abandons_its_turns_without_waiting() -> None:
    worker, synth = _worker(continuity=True)
    # BaseWorker._cleanup_room touches state that __init__ normally creates; these workers are
    # built with __new__, so the fields it clears have to exist.
    worker._key_locks = {}
    worker._route_states = {}
    worker._room_routes = {}
    worker._translation_active = {}
    worker._paused_rooms = set()
    await _say(worker, _msg("Câu một.", chunk=0))

    worker._cleanup_room("m1")

    assert worker._turns == {}


# ── WT-397: who has already heard what ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_streamed_sentence_is_reported_as_already_spoken() -> None:
    """The caller publishes what synthesis returns. Once the sentence has gone out chunk by
    chunk, publishing it again would speak the whole line a second time."""
    publisher = _FakePublisher()
    worker, synth = _worker(continuity=True, publisher=publisher)

    sentence = await _say(worker, _msg("Câu một."))

    assert publisher.tracks[0].fed == [b"\x01\x02" * 100], "nothing reached the track"
    assert publisher.tracks[0].closed is True, "the track was left open past the sentence"
    assert sentence.already_spoken is True
    assert synth.one_shot_calls == []


@pytest.mark.asyncio
async def test_a_context_that_dies_mid_sentence_does_not_speak_the_opening_twice() -> None:
    """The trap this feature turns on.

    The fallback re-synthesizes the WHOLE sentence. If part of it is already on the track,
    playing the fallback repeats the opening words — a dub that stutters reads as the system
    being broken, which is worse than the truncation it would be repairing. The bytes are still
    returned in full, because billing and the transcript read them and neither is audible.
    """
    publisher = _FakePublisher()
    worker, synth = _worker(continuity=True, publisher=publisher)
    await _say(worker, _msg("Câu một.", chunk=0))
    synth.turns[0].fail_after_streaming = True

    sentence = await _say(worker, _msg("Câu hai.", chunk=1))

    assert sentence.already_spoken is True, (
        "the fallback would be played on top of a half-spoken line"
    )
    assert synth.one_shot_calls == ["Câu hai."], "the sentence was not re-synthesized at all"
    assert len(sentence.audio) > 44, "the transcript and billing were handed an empty sentence"


@pytest.mark.asyncio
async def test_a_context_that_dies_before_the_first_chunk_still_gets_its_fallback() -> None:
    # Nothing was heard, so there is nothing to collide with — the proven one-shot path must
    # still reach the room, or a dead socket becomes silence.
    publisher = _FakePublisher()
    worker, synth = _worker(continuity=True, publisher=publisher)
    await _say(worker, _msg("Câu một.", chunk=0))
    synth.turns[0].fail = True

    sentence = await _say(worker, _msg("Câu hai.", chunk=1))

    assert sentence.already_spoken is False
    assert synth.one_shot_calls == ["Câu hai."]


@pytest.mark.asyncio
async def test_a_track_that_swallowed_everything_is_not_counted_as_spoken() -> None:
    """`spoken_bytes`, not "we called feed".

    A track that never connected accepts every chunk and plays none of them. Treating the
    attempt as success would suppress the fallback and leave the room silent.
    """
    publisher = _FakePublisher(deaf=True)
    worker, synth = _worker(continuity=True, publisher=publisher)
    synth_turn_msg = _msg("Câu một.", chunk=0)
    await _say(worker, synth_turn_msg)
    synth.turns[0].fail_after_streaming = True

    sentence = await _say(worker, _msg("Câu hai.", chunk=1))

    assert sentence.already_spoken is False, "a silent track was reported as having spoken"


@pytest.mark.asyncio
async def test_the_kill_switch_stops_streaming_without_stopping_the_dub() -> None:
    # TTS_STREAM_TO_LIVEKIT=false must leave the pre-WT-397 behaviour exactly in place: the
    # sentence is still synthesized through the shared context, and the caller still publishes.
    assert TTSSettings().stream_to_livekit is True, "streaming ships on"

    publisher = _FakePublisher()
    worker, synth = _worker(continuity=True, publisher=publisher)
    worker.tts_settings = TTSSettings(prosody_continuity=True, stream_to_livekit=False)

    sentence = await _say(worker, _msg("Câu một."))

    assert publisher.tracks == [], "the track was opened with streaming switched off"
    assert sentence.already_spoken is False
    assert synth.one_shot_calls == [], "the kill switch also disabled prosody continuity"


@pytest.mark.asyncio
async def test_billing_and_the_transcript_still_see_a_streamed_sentence() -> None:
    """tts:results drives billing_worker and TranscriptRedisConsumerService. Suppressing the
    LiveKit push must not suppress that message, or a streamed sentence would be free and
    would never appear in the transcript."""
    publisher = _FakePublisher()
    worker, _synth = _worker(continuity=True, publisher=publisher)
    worker.redis = AsyncMock()
    worker.redis.get = AsyncMock(return_value=None)
    worker.publish = AsyncMock()
    worker._publish_livekit_only = AsyncMock()
    worker._dub_fits = {}
    worker._turn_dub_ms = {}

    await worker._synthesize_and_publish(
        _msg("Câu một.", final=True), "Câu một.", "v1", "default", ""
    )

    assert worker.publish.await_count == 1
    assert worker.publish.await_args.args[0] == "tts:results"
    worker._publish_livekit_only.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_number_this_feature_moves_is_actually_recorded() -> None:
    """Without `tts_first_audio` there is no evidence WT-397 did anything.

    `tts_synthesis` covers the whole call and now includes handing the audio to the track,
    which back-pressures to real time — so it RISES with streaming on. Shipping only that
    would make the dashboard read as a regression while listeners waited less. This is the
    number that answers the complaint, and metrics_exporter picks up any stage key by glob, so
    recording it here is all the wiring there is.
    """
    publisher = _FakePublisher()
    worker, _synth = _worker(continuity=True, publisher=publisher)
    worker.redis = AsyncMock()
    worker.redis.get = AsyncMock(return_value=None)
    worker.publish = AsyncMock()
    worker._publish_livekit_only = AsyncMock()
    worker._dub_fits = {}
    worker._turn_dub_ms = {}

    await worker._synthesize_and_publish(
        _msg("Câu một.", final=True), "Câu một.", "v1", "default", ""
    )

    stages = [call.args[0] for call in worker.redis.record_latency.await_args_list]
    assert "tts_first_audio" in stages, "the only measure of this feature was never recorded"
    assert "tts_synthesis" in stages, "the existing stage measurement was dropped"


# ── Isochrony wiring ────────────────────────────────────────────────────────────────────────


def _iso_worker() -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(prosody_enabled=True)
    worker.logger = MagicMock()
    worker._dub_fits = {}
    worker._turn_dub_ms = {}
    return worker


def _turn(worker: TTSWorker, *, source_ms: int, sentence_ms: list[int]) -> None:
    """One spoken turn, delivered as len(sentence_ms) sentences."""
    for index, dub_ms in enumerate(sentence_ms):
        message = _msg(
            f"câu {index}",
            chunk=index,
            final=index == len(sentence_ms) - 1,
        )
        message = message.model_copy(update={"start_ms": 1000, "end_ms": 1000 + source_ms})
        worker._observe_dub_fit(message, dub_ms)


@pytest.mark.asyncio
async def test_a_turn_is_measured_whole_not_sentence_by_sentence() -> None:
    """`start_ms`/`end_ms` describe the WHOLE turn.

    Weighing one sentence's dub against them reports a fit of roughly 1/N for an N-sentence
    turn, and the controller then drives everybody faster and faster. The sentences are summed
    and the total is what gets folded in.
    """
    worker = _iso_worker()

    # Four sentences totalling 4000ms against a 4000ms source: a perfect fit.
    for _ in range(6):
        _turn(worker, source_ms=4000, sentence_ms=[1000, 1000, 1000, 1000])

    fit = worker._dub_fits[("m1", "s1", "vi")]
    assert fit.ratio == pytest.approx(1.0, abs=0.05), (
        f"a turn that fits exactly measured as {fit.ratio:.2f} — the sentences were not summed"
    )


@pytest.mark.asyncio
async def test_an_overrunning_turn_speeds_the_next_one_up() -> None:
    worker = _iso_worker()

    for _ in range(6):
        _turn(worker, source_ms=4000, sentence_ms=[2600, 2600])

    message = _msg("tiếp")
    message = message.model_copy(update={"start_ms": 0, "end_ms": 4000})
    assert isochrony.speed_center(worker._dub_fit(message)) > 1.0


@pytest.mark.asyncio
async def test_an_unfinished_turn_does_not_leak_into_the_next() -> None:
    # A turn whose final chunk never arrives must not have its partial total added to whatever
    # the speaker says next.
    worker = _iso_worker()
    _turn(worker, source_ms=4000, sentence_ms=[1500, 1500])
    assert worker._turn_dub_ms == {}


@pytest.mark.asyncio
async def test_prosody_off_records_nothing() -> None:
    worker = _iso_worker()
    worker.tts_settings = TTSSettings(prosody_enabled=False)

    _turn(worker, source_ms=4000, sentence_ms=[6000])

    assert worker._dub_fits == {}


@pytest.mark.asyncio
async def test_the_learned_fit_actually_reaches_cartesias_controls() -> None:
    """The wiring, not the arithmetic.

    Measuring the fit and never sending it would leave the whole controller inert while every
    unit test above still passed — this asserts on what `_generation_config` really produces.
    """
    worker = _iso_worker()
    envelope = ProsodyEnvelope(rate_ratio=1.0, arousal="neutral")

    def _measured(final: bool = True) -> TranslationResultMessage:
        message = _msg("câu", final=final)
        return message.model_copy(update={"start_ms": 0, "end_ms": 4000, "prosody": envelope})

    before = worker._generation_config(_measured())
    assert before is not None
    baseline_speed = float(before["speed"])

    # Six turns that each overran by 30%.
    for _ in range(6):
        _turn(worker, source_ms=4000, sentence_ms=[5200])

    after = worker._generation_config(_measured())
    assert after is not None
    assert float(after["speed"]) > baseline_speed, (
        "the fit was learned but never reached generation_config — the controller is inert"
    )


@pytest.mark.asyncio
async def test_a_context_cartesia_retired_is_replaced_not_reused() -> None:
    """WT-405. A context can die WITHOUT raising, and the old code could not tell.

    `_collect` treats Cartesia's `done` as an ordinary end of stream: it marks the context
    closed, breaks, and returns the audio it has. So `speak()` SUCCEEDS — the caller never
    reaches the except branch that calls `_end_turn`, and the spent context stays in `_turns`.
    The next sentence for the same key found it not-None, called `speak()`, and got
    "ProsodyContext is closed": one wasted sentence per `done`, re-synthesized one-shot with no
    streaming at all.

    Production, 15 Aug, meeting 01a0033f: 12 of 47 sentences, up to 10.2s each. Cartesia retires
    an idle context, and with two people in a conversation each speaker's context idles while
    the other talks — so the more natural the exchange, the more often it fired. That is exactly
    the report: "nói liên tục 2 người thì ... vẫn đang còn bị chậm".
    """
    worker, synth = _worker(continuity=True)

    first = await _say(worker, _msg("Một."))
    synth.turns[0].retire_after_speaking = True
    # This sentence still succeeds — it is the one that receives `done`.
    await _say(worker, _msg("Hai."))
    assert synth.turns[0].is_closed, "the fake must model a context retired by the server"

    second = await _say(worker, _msg("Ba."))

    assert synth.one_shot_calls == [], (
        "The sentence after a retired context fell back to one-shot synthesis — no streaming, "
        "full re-synthesis, and the listener waits for all of it. That is the p95 tail."
    )
    assert len(synth.opened) == 2, (
        f"a retired context must be replaced with a fresh one; opened={synth.opened}"
    )
    assert synth.turns[1].spoken, "the third sentence should have been spoken on the NEW context"
    assert first is not None and second is not None


@pytest.mark.asyncio
async def test_replacing_a_retired_context_releases_its_connection() -> None:
    """The old socket has to go with it, or every `done` leaks a Cartesia connection for the
    lifetime of the meeting."""
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Một."))
    synth.turns[0].retire_after_speaking = True
    await _say(worker, _msg("Hai."))
    await _say(worker, _msg("Ba."))

    assert len(synth.connections) == 2, (
        "A retired context was never replaced, so there is no second connection to check — "
        f"the spent one was reused instead; connections={len(synth.connections)}"
    )
    assert synth.connections[0].closed, "the retired context's connection was left open"
    assert not synth.connections[1].closed, "the replacement connection must still be live"


@pytest.mark.asyncio
async def test_a_healthy_context_is_still_reused() -> None:
    """The guard must not become 'open a new context every sentence' — that would silently
    undo prosodic continuity while every test still passed."""
    worker, synth = _worker(continuity=True)

    await _say(worker, _msg("Một."))
    await _say(worker, _msg("Hai."))
    await _say(worker, _msg("Ba."))

    assert len(synth.opened) == 1, f"one turn, one context; opened={synth.opened}"
    assert len(synth.turns[0].spoken) == 3
