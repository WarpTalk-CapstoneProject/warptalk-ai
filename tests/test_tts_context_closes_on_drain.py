"""A Cartesia context must never sit open and idle, and the gate must count what Cartesia counts.

Measured 2026-10-01 with the production key, from the tts-worker pod. The plan (Pro) allows three
concurrent generations and QUEUES the excess silently — no 429, no log, just latency:

    3 parallel /tts/bytes, nothing else open                        TTFB ~0.7s
    same, 3 contexts left open and idle after their flush_done      TTFB ~4.0s
    same, those 3 contexts ended right after flush_done             TTFB ~0.7s

The idle contexts were retired by the server itself ~4.7-5.2s after their last flush_done, and the
queued requests started right after; ending them explicitly got `done` back within ~0.2s. So an
open context occupies a vendor slot from its first push until its `done`, idle or not.

The worker used to keep a turn's context open between sentences and give the slot back at each
flush_done, on the stated belief that an idle context "was never inside the slot". With three or
more active (speaker, language) keys the gate then handed out slots Cartesia was already using,
and Cartesia queued the difference where no log could see it.

`_Vendor` below is Cartesia as measured: a context counts from its first push until it sends
`done` (or is cancelled), and anything counted beyond the plan is recorded as vendor-queued. Its
semaphore audits every release against that count, so "gate == vendor truth" is checked on every
test, not asserted once.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config import TTSSettings, WorkerSettings
from shared.schemas import TranslationResultMessage
from tts_worker import prosody_context
from tts_worker.prosody_context import ProsodyContext
from tts_worker.synthesizer import GenerationLease
from tts_worker.worker import TTSWorker

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Cartesia, as measured
# ---------------------------------------------------------------------------


@dataclass
class _Chunk:
    audio: bytes
    flush_id: int
    type: str = "chunk"


@dataclass
class _FlushDone:
    flush_id: int
    type: str = "flush_done"


@dataclass
class _Done:
    type: str = "done"


class _AuditedSlots(asyncio.Semaphore):
    """The worker's gate, checked against the vendor every time a slot is given back."""

    def __init__(self, vendor: _Vendor, size: int) -> None:
        super().__init__(size)
        self._vendor = vendor
        self.size = size

    @property
    def held(self) -> int:
        return self.size - self._value

    def release(self) -> None:
        super().release()
        self._vendor.audit("slot released")


class _VendorContext:
    """One websocket context, as the server treats it."""

    def __init__(self, vendor: _Vendor, context_id: str) -> None:
        self._vendor = vendor
        self.id = context_id
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.pushes: list[dict[str, Any]] = []
        self.counted = False
        self.ended = False
        self.close_sent = False
        self._flushes = 0

    async def push(self, transcript: str, *, continue_: bool = True, **kwargs: Any) -> None:
        self.pushes.append({"transcript": transcript, "continue_": continue_, **kwargs})
        if not self.counted:
            self.counted = True
            self._vendor.count(+1, self.id)
        flush_id = self._flushes
        self._flushes += 1
        self._vendor.spawn(self._generate(flush_id))

    async def _generate(self, flush_id: int) -> None:
        # Generation takes time, so a sentence dispatched meanwhile is already queued behind
        # this one by its flush_done — which is what "back-to-back" means in production.
        await asyncio.sleep(self._vendor.generation_seconds)
        self.queue.put_nowait(_Chunk(self._vendor.sentence_audio, flush_id))
        self.queue.put_nowait(_FlushDone(flush_id))

    def receive(self) -> Any:
        async def _events() -> AsyncIterator[Any]:
            while True:
                event = await self.queue.get()
                yield event
                if event.type in ("done", "error"):
                    return

        return _events()

    async def no_more_inputs(self) -> None:
        # Empty transcript, continue=false. The server answers `done`, and stops counting the
        # context only then.
        self.close_sent = True
        self._vendor.timeline.append(f"close:{self.id}")
        self._vendor.spawn(self._answer_close())

    async def _answer_close(self) -> None:
        await self._vendor.close_ack.wait()
        self._end()
        self.queue.put_nowait(_Done())

    async def cancel(self) -> None:
        self._end()

    def _end(self) -> None:
        if self.counted and not self.ended:
            self.ended = True
            self._vendor.count(-1, self.id)


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _Vendor:
    """Stands in for CartesiaSynthesizer: the gate, and contexts that count like Cartesia's."""

    def __init__(self, plan_limit: int, *, sentence_ms: int = 25) -> None:
        self.plan_limit = plan_limit
        self.slots = _AuditedSlots(self, plan_limit)
        self.contexts: list[_VendorContext] = []
        self.connections: list[_Connection] = []
        self.counted = 0
        #: Contexts Cartesia would have QUEUED: counted beyond the plan at their first push.
        self.vendor_queued: list[str] = []
        #: Gate releases that left the vendor counting more contexts than the gate holds.
        self.violations: list[str] = []
        self.timeline: list[str] = []
        self.generation_seconds = 0.02
        self.sentence_audio = b"\x01\x02" * int(SAMPLE_RATE * sentence_ms / 1000)
        #: Cleared to hold back the `done` that answers a close.
        self.close_ack = asyncio.Event()
        self.close_ack.set()
        self._tasks: set[asyncio.Task[None]] = set()

    # -- the slice of CartesiaSynthesizer the worker uses --------------------------------------

    def generation_slot(self) -> asyncio.Semaphore:
        return self.slots

    async def open_prosody_context(
        self, *, context_id: str, language: str, voice_id: str | None
    ) -> tuple[ProsodyContext, _Connection]:
        transport = _VendorContext(self, context_id)
        connection = _Connection()
        self.contexts.append(transport)
        self.connections.append(connection)
        return ProsodyContext(transport, SAMPLE_RATE), connection

    async def synthesize(self, **kwargs: Any) -> tuple[bytes, int, str]:
        raise AssertionError("no sentence in these tests should fall back to one-shot")

    # -- bookkeeping ----------------------------------------------------------------------------

    def count(self, delta: int, context_id: str) -> None:
        self.counted += delta
        if delta > 0:
            if self.counted > self.plan_limit:
                self.vendor_queued.append(context_id)
            self.audit(f"first push of {context_id}")

    def audit(self, moment: str) -> None:
        if self.counted > self.slots.held:
            self.violations.append(
                f"{moment}: Cartesia counts {self.counted} contexts, gate holds {self.slots.held}"
            )

    def spawn(self, coroutine: Any) -> None:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


# ---------------------------------------------------------------------------
# A track that plays in real time — leaving `stream()` waits until the test lets it finish
# ---------------------------------------------------------------------------


class _Track:
    def __init__(self) -> None:
        self.fed = 0
        self.first_audio_at: float | None = None

    async def feed(self, pcm: bytes) -> None:
        self.fed += len(pcm)
        if self.first_audio_at is None:
            self.first_audio_at = time.monotonic()

    @property
    def spoken_bytes(self) -> int:
        return self.fed


class _Playout:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.finish = asyncio.Event()

    @asynccontextmanager
    async def stream(
        self, meeting_id: str, speaker_id: str, target_lang: str, *args: Any, **kwargs: Any
    ) -> AsyncIterator[_Track]:
        track = _Track()
        try:
            yield track
        finally:
            self.log.append(f"playing:{speaker_id}:{target_lang}")
            await self.finish.wait()
            self.log.append(f"played:{speaker_id}:{target_lang}")

    def playing(self, speaker: str, lang: str = "vi") -> int:
        return self.log.count(f"playing:{speaker}:{lang}")


# ---------------------------------------------------------------------------
# The worker, driven through the real per-key dispatch
# ---------------------------------------------------------------------------


def _worker(vendor: _Vendor, playout: _Playout) -> TTSWorker:
    worker = TTSWorker.__new__(TTSWorker)
    worker.settings = WorkerSettings()
    worker.tts_settings = TTSSettings(
        prosody_continuity=True,
        stream_to_livekit=True,
        cache_enabled=False,
        sample_rate=SAMPLE_RATE,
    )
    worker.logger = MagicMock()
    worker.redis = AsyncMock()
    worker.publish = AsyncMock()  # type: ignore[method-assign]
    worker._publish_livekit_only = AsyncMock()  # type: ignore[method-assign]
    worker._key_locks = {}
    worker._dub_fits = {}
    worker._turn_dub_ms = {}
    worker.livekit_publisher = playout  # type: ignore[assignment]
    worker.cartesia = vendor  # type: ignore[assignment]

    async def handle(message_id: bytes, data: dict[bytes, bytes]) -> None:
        # What process() does for a plain default-voice sentence, minus the room plumbing.
        message = TranslationResultMessage.from_redis(data)
        await worker._synthesize_and_publish(
            message, message.translated_text, "voice-1", "default", ""
        )

    worker._process_and_log_errors = handle  # type: ignore[method-assign]
    return worker


def _msg(speaker: str, text: str, *, chunk: int = 0, lang: str = "vi") -> Any:
    return TranslationResultMessage(
        segment_id=f"seg-{speaker}",
        meeting_id="m1",
        speaker_id=speaker,
        original_text="src",
        translated_text=text,
        source_lang="en",
        target_lang=lang,
        chunk_index=chunk,
    ).to_redis()


def _dispatch(worker: TTSWorker, message_id: bytes, data: Any) -> asyncio.Task[None]:
    """What consume_pipelined does with a message it has read."""
    return asyncio.create_task(worker._run_in_key_order(message_id, data))


async def _until(predicate: Any, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _slot_waits(worker: TTSWorker) -> list[int]:
    return [
        call.args[1]
        for call in worker.redis.record_latency.await_args_list
        if call.args[0] == "tts_slot_wait"
    ]


def _logged(worker: TTSWorker, event: str) -> list[dict[str, Any]]:
    calls = [*worker.logger.info.call_args_list, *worker.logger.warning.call_args_list]
    return [call.kwargs for call in calls if call.args and call.args[0] == event]


# ---------------------------------------------------------------------------
# Close on drain
# ---------------------------------------------------------------------------


async def test_a_lone_sentence_closes_its_context_right_after_flush_done() -> None:
    """Nothing queued behind it, so nothing can continue it: the context is ended as soon as
    the sentence's audio is in — while the sentence is still PLAYING — and its slot follows."""
    vendor, playout = _Vendor(plan_limit=3), _Playout()
    worker = _worker(vendor, playout)

    task = _dispatch(worker, b"1-0", _msg("s1", "Một."))
    await _until(lambda: playout.playing("s1") == 1)
    context = vendor.contexts[0]
    # The last step of ending a context: close, `done`, slot back, then the socket.
    await _until(lambda: vendor.connections[0].closed)

    assert context.close_sent, "ended by our close, not left for the server to retire"
    assert [push["continue_"] for push in context.pushes] == [True]
    assert vendor.counted == 0
    assert vendor.slots.held == 0, "the slot outlived the context"
    assert "played:s1:vi" not in playout.log, "closing waited for the playout"
    assert vendor.connections[0].closed
    assert worker._open_contexts() == {}

    playout.finish.set()
    await task
    assert vendor.violations == []


async def test_back_to_back_sentences_share_one_context_and_then_close() -> None:
    """The continuation that is kept: the next sentence was already queued at flush_done."""
    vendor, playout = _Vendor(plan_limit=3), _Playout()
    worker = _worker(vendor, playout)

    first = _dispatch(worker, b"1-0", _msg("s1", "Một.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    # Read from the stream while the first is still generating: queued behind the key's lock.
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)
    context = vendor.contexts[0]
    await asyncio.sleep(0.02)
    assert not context.close_sent, "the context was closed with its next sentence in hand"

    playout.finish.set()
    await asyncio.gather(first, second)
    await asyncio.gather(*worker._retiring_contexts())

    assert len(vendor.contexts) == 1, "the queued sentence opened a second context"
    assert [push["transcript"] for push in context.pushes] == ["Một.", "Hai."]
    assert context.close_sent and context.ended, "the context outlived its last sentence"
    assert vendor.counted == 0 and vendor.slots.held == 0
    assert worker._open_contexts() == {}
    continued = [kwargs["continued"] for kwargs in _logged(worker, "audio_synthesized")]
    assert continued == [False, True]
    assert vendor.violations == []


async def test_the_slot_is_held_for_the_whole_life_of_an_open_context() -> None:
    """Gate == vendor truth. A context kept open for the next sentence is idle through the first
    one's playout and still counted by Cartesia, so its slot stays held across that gap; and a
    closed context is counted until its `done`, so the slot waits for that too."""
    vendor, playout = _Vendor(plan_limit=3), _Playout()
    worker = _worker(vendor, playout)
    vendor.close_ack.clear()  # Cartesia has not answered the close yet

    first = _dispatch(worker, b"1-0", _msg("s1", "Một.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)
    context = vendor.contexts[0]

    # Between the two sentences: idle, open, counted — and held.
    assert vendor.counted == 1 and vendor.slots.held == 1

    playout.finish.set()
    await _until(lambda: context.close_sent)
    await asyncio.sleep(0.02)
    # Close sent, `done` not back: Cartesia still counts it, so the gate still holds it.
    assert vendor.counted == 1 and vendor.slots.held == 1, "the slot was given back before `done`"

    vendor.close_ack.set()
    await _until(lambda: vendor.slots.held == 0)
    await asyncio.gather(first, second)
    assert vendor.violations == []


async def test_a_third_key_does_not_wait_on_an_idle_context() -> None:
    """The production bug, at the plan's limit. Two keys have spoken and are still playing;
    their contexts must not still be counted by Cartesia when a third key speaks — not by our
    gate (no wait), and not by the vendor (no silent queue)."""
    vendor, playout = _Vendor(plan_limit=2), _Playout()
    worker = _worker(vendor, playout)

    a = _dispatch(worker, b"1-0", _msg("s1", "A."))
    b = _dispatch(worker, b"2-0", _msg("s2", "B."))
    await _until(lambda: playout.playing("s1") == 1 and playout.playing("s2") == 1)
    # Our gate is free. Under the old design it was free here too — while Cartesia still
    # counted both idle contexts, which is what `vendor_queued` below would catch.
    await _until(lambda: vendor.slots.held == 0)

    c = _dispatch(worker, b"3-0", _msg("s3", "C."))
    await _until(lambda: playout.playing("s3") == 1)
    assert "played:s1:vi" not in playout.log and "played:s2:vi" not in playout.log

    playout.finish.set()
    await asyncio.gather(a, b, c)
    assert vendor.vendor_queued == [], "Cartesia would have queued the third key, silently"
    assert _slot_waits(worker) == [0, 0, 0]
    assert _logged(worker, "cartesia_slot_waited") == []
    assert vendor.violations == []


async def test_waiting_on_a_context_that_is_genuinely_open_is_ours_and_visible() -> None:
    """When a slot IS taken — a context kept open for a queued sentence — a newcomer waits at
    our gate, where it is logged and measured, instead of being queued by Cartesia unseen."""
    vendor, playout = _Vendor(plan_limit=1), _Playout()
    worker = _worker(vendor, playout)

    first = _dispatch(worker, b"1-0", _msg("s1", "Một.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)
    other = _dispatch(worker, b"3-0", _msg("s2", "Khác."))
    await asyncio.sleep(0.05)
    pushed = [context for context in vendor.contexts if context.pushes]
    assert pushed == [vendor.contexts[0]], "the newcomer reached Cartesia past a full gate"

    playout.finish.set()
    await asyncio.gather(first, second, other)

    assert vendor.vendor_queued == []
    [waited] = _logged(worker, "cartesia_slot_waited")
    assert waited["speaker_id"] == "s2" and waited["waited_ms"] >= 40
    assert max(_slot_waits(worker)) >= 40
    [s2_line] = [kw for kw in _logged(worker, "audio_synthesized") if kw["speaker_id"] == "s2"]
    assert s2_line["slot_wait_ms"] >= 40
    assert vendor.violations == []


# ---------------------------------------------------------------------------
# A context kept open must not outlive its reason
# ---------------------------------------------------------------------------


async def test_a_context_kept_for_a_queued_sentence_is_ended_at_its_idle_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queued sentence cannot push until this one has played. If that takes past the idle
    expiry, we end the context ourselves — before the server would retire it (~4.7s measured),
    so no sentence is ever pushed onto a server-retired context (the WT-874 stale `done`) — and
    the slot goes back then, not when the next sentence happens to look."""
    monkeypatch.setattr(prosody_context, "CONTEXT_IDLE_EXPIRY_SECONDS", 0.05)
    vendor, playout = _Vendor(plan_limit=3), _Playout()
    worker = _worker(vendor, playout)

    first = _dispatch(worker, b"1-0", _msg("s1", "Một.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)

    await _until(lambda: vendor.contexts[0].ended and vendor.slots.held == 0)
    assert "played:s1:vi" not in playout.log
    assert _logged(worker, "prosody_context_idle_expired")

    playout.finish.set()
    await asyncio.gather(first, second)
    assert len(vendor.contexts) == 2, "the second sentence needed a fresh context"
    assert [push["transcript"] for push in vendor.contexts[1].pushes] == ["Hai."]
    assert vendor.violations == []


async def test_a_continuation_that_cannot_arrive_in_time_is_not_kept() -> None:
    """A sentence that takes longer to hand over than the idle expiry cannot be followed onto
    the same context — the next one cannot push until it has played. Keeping the context would
    only hold the slot for the expiry to release later, so it is ended at flush_done."""
    sentence_ms = int((prosody_context.CONTEXT_IDLE_EXPIRY_SECONDS + 2.0) * 1000)
    vendor, playout = _Vendor(plan_limit=3, sentence_ms=sentence_ms), _Playout()
    worker = _worker(vendor, playout)

    first = _dispatch(worker, b"1-0", _msg("s1", "Một câu rất dài.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)

    await _until(lambda: vendor.contexts[0].ended and vendor.slots.held == 0, timeout=0.5)
    assert "played:s1:vi" not in playout.log

    playout.finish.set()
    await asyncio.gather(first, second)
    assert vendor.violations == []


async def test_a_context_kept_for_a_message_that_never_spoke_on_it_is_ended_with_it() -> None:
    """The queued message turned out not to need the context — a cache hit, a skip, a voice
    variant that changed. The context is ended when that message finishes, not left holding a
    slot until its idle expiry."""
    vendor, playout = _Vendor(plan_limit=3), _Playout()
    worker = _worker(vendor, playout)
    speak = worker._process_and_log_errors

    async def first_speaks_second_skips(message_id: bytes, data: dict[bytes, bytes]) -> None:
        if message_id == b"1-0":
            await speak(message_id, data)

    worker._process_and_log_errors = first_speaks_second_skips  # type: ignore[method-assign]

    first = _dispatch(worker, b"1-0", _msg("s1", "Một.", chunk=0))
    await _until(lambda: vendor.contexts and vendor.contexts[0].pushes)
    second = _dispatch(worker, b"2-0", _msg("s1", "Hai.", chunk=1))
    await _until(lambda: playout.playing("s1") == 1)
    assert not vendor.contexts[0].close_sent

    playout.finish.set()
    await asyncio.gather(first, second)
    await _until(lambda: vendor.contexts[0].ended and vendor.slots.held == 0, timeout=0.5)

    assert _logged(worker, "prosody_context_idle_expired") == [], "left for the expiry"
    assert vendor.violations == []


# ---------------------------------------------------------------------------
# The lease
# ---------------------------------------------------------------------------


async def test_the_lease_measures_how_long_it_waited() -> None:
    slot = asyncio.Semaphore(1)
    holder, waiter = GenerationLease(slot), GenerationLease(slot)
    await holder.acquire()

    acquiring = asyncio.create_task(waiter.acquire())
    await asyncio.sleep(0.05)
    holder.release()
    await acquiring

    assert holder.waited_ms < 20
    assert waiter.waited_ms >= 40
    waiter.release()
