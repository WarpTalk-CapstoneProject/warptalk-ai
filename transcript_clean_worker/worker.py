"""Transcript Clean Worker — turns raw STT segments into clean sentences (WT-716).

Pipeline:
    Redis Stream (stt:results)  — its OWN consumer group, parallel to translation
    → deterministic prepass per segment (shared.disfluency, or the value stt_worker already put
      on the message)
    → sentence assembly per speaker (segmenter.py)
    → publish revision 0 immediately          (transcript:clean, source="prepass")
    → LLM deletion-index pass, bounded        (llm_cleaner.py)
    → publish revision 1 when it verifies     (transcript:clean, source="llm")

WHY TWO REVISIONS AND NOT ONE
    The reader should not wait on a model for a line the rule tier has already cleaned. Revision
    0 goes out the moment the sentence closes; revision 1 replaces it in place when — and only
    when — the model's answer passes the invariants. A consumer keeps the highest revision it has
    seen per `sentence_id`, so a lost, late or never-sent revision 1 is not a broken line, it is
    just a slightly less polished one.

WHY IT IS A SEPARATE WORKER AND NOT PART OF stt_worker OR translation_worker
    It buffers across segments and it is allowed to be slow: a sentence is only complete once the
    NEXT segment (or a silence) says so. Neither of those belongs in a stage the dub waits on.
    Reading `stt:results` under its own consumer group is what keeps a slow clean pass from ever
    delaying a caption — the same shape suggestion_worker uses.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from shared.base_worker import TERMINAL_ROOM_STATUSES, BaseWorker
from shared.config import resolve_openai_api_key
from shared.control_markers import is_control_marker, is_system_speaker
from shared.disfluency import FLAG_ESCALATE, normalize_terminal_punctuation, prepass
from shared.schemas import TRANSCRIPT_CLEAN_STREAM, CleanSentenceMessage, STTResultMessage
from transcript_clean_worker.config import TranscriptCleanSettings
from transcript_clean_worker.llm_cleaner import LLMCleaner
from transcript_clean_worker.segmenter import CleanSegment, CleanSentence, SentenceSegmenter

# The wire flag vocabulary (see CleanSentenceMessage). Anything else the prepass reports stays
# internal: the backend renders these three and nothing else.
FLAG_SELF_REPAIR = "self_repair"
FLAG_FALLBACK_RAW = "fallback_raw"
_WIRE_FLAGS = frozenset({FLAG_SELF_REPAIR, FLAG_FALLBACK_RAW, FLAG_ESCALATE})


def _is_guid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return False
    return True


class TranscriptCleanWorker(BaseWorker):
    """Clean transcript assembly for every live meeting this replica sees."""

    worker_name = "transcript-clean"
    input_stream = "stt:results"
    consumer_group = "transcript-clean-workers"

    # Deliberately 1 (BaseWorker's default, restated because it is load-bearing here): sentences
    # are assembled from segments IN ORDER, and processing two segments of the same meeting
    # concurrently would let the second one reach the buffer first. The slow part — the LLM
    # call — is not in this path at all; it runs as a detached task.
    concurrency = 1

    _IDLE_TICK_SECONDS = 1.0

    # How long a meeting may say nothing before this worker forgets it. The room's terminal
    # status and the `__MEETING_END__` sentinel are the normal way state is dropped; this is the
    # backstop for the meeting that ends without either reaching this replica, which would
    # otherwise keep a (now empty) segmenter and its last line for the life of the process.
    # Rebuilding the state costs nothing — the next segment creates it — and a meeting silent for
    # fifteen minutes has no question for the next turn to be answering anyway.
    _FORGET_AFTER_MS = 15 * 60 * 1000

    def __init__(
        self,
        clean_settings: TranscriptCleanSettings | None = None,
        cleaner: LLMCleaner | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.clean_settings = clean_settings or TranscriptCleanSettings()
        self.cleaner = cleaner
        self._segmenters: dict[str, SentenceSegmenter] = {}
        # The last clean line published for a meeting. Two uses, both about context: the prepass
        # keeps a turn-initial "hmm"/"ừ"/"うん" when the previous turn was a question, and the LLM
        # is shown one line of it for the same reason.
        self._last_line: dict[str, str] = {}
        self._last_seen_ms: dict[str, int] = {}
        self._refine_tasks: set[asyncio.Task[None]] = set()
        self._idle_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load_model(self) -> None:
        if self.cleaner is None:
            self.cleaner = LLMCleaner(
                api_key=resolve_openai_api_key(self.clean_settings.api_key),
                model=self.clean_settings.model,
                timeout_s=self.clean_settings.llm_timeout_s,
                max_delete_ratio=self.clean_settings.max_delete_ratio,
                concurrency=self.clean_settings.llm_concurrency,
            )
            await self.cleaner.load()
        self._idle_task = asyncio.create_task(self._idle_loop())
        self.logger.info(
            "transcript_clean_ready",
            enabled=self.clean_settings.enabled,
            llm=self.cleaner.is_available,
            model=self.clean_settings.model,
        )

    async def _cleanup(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except asyncio.CancelledError:
                pass
        # A meeting whose last sentence is still open loses it otherwise — and on a rolling
        # deploy that is the last line of every meeting in progress.
        for meeting_id in list(self._segmenters):
            await self._flush_meeting(meeting_id, reason="shutdown")
        pending = [task for task in self._refine_tasks if not task.done()]
        if pending:
            # The LLM pass is a polish on an already-published line, so it gets the time it has
            # left and no more; revision 0 stands for anything that does not finish.
            await asyncio.wait(pending, timeout=self.clean_settings.llm_timeout_s)
            for task in pending:
                task.cancel()
        if self.cleaner is not None:
            self.logger.info("transcript_clean_llm_rejections", **self.cleaner.rejections)
            await self.cleaner.close()

    async def _idle_loop(self) -> None:
        """Close sentences whose speaker simply stopped talking, and forget dead meetings."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(self._IDLE_TICK_SECONDS)
            try:
                now_ms = self._now_ms()
                for meeting_id, segmenter in list(self._segmenters.items()):
                    for sentence in segmenter.flush_idle(now_ms):
                        await self._publish(meeting_id, sentence)
                    silent_for = now_ms - self._last_seen_ms.get(meeting_id, now_ms)
                    if segmenter.is_empty and silent_for >= self._FORGET_AFTER_MS:
                        self._forget(meeting_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("transcript_clean_idle_flush_failed")

    @staticmethod
    def _now_ms() -> int:
        # Monotonic: the idle timer measures how long ago a segment ARRIVED here, which must not
        # move when the system clock does, and is a different quantity from the meeting-relative
        # start_ms/end_ms the merge gap is measured on.
        return int(time.monotonic() * 1000)

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        if not self.clean_settings.enabled:
            return

        stt_result = STTResultMessage.from_redis(data)
        meeting_id = stt_result.meeting_id

        # `__MEETING_END__` is addressed to ai_assistant_worker, but it is the one signal that
        # says a meeting is over — and an open sentence at that moment is somebody's last
        # sentence. Flushed, then dropped; never treated as speech (see shared/control_markers).
        if is_control_marker(stt_result.text):
            await self._flush_meeting(meeting_id, reason="meeting_end")
            self._forget(meeting_id)
            return
        if is_system_speaker(stt_result.speaker_id) or not stt_result.text.strip():
            return

        # The host took this stretch off the record (WT-605). The raw segment is not stored, so a
        # clean line built from it would be a transcript row for speech the host hid — the same
        # gate, for the same reason, as suggestion_worker's.
        if await self.is_transcript_paused(meeting_id):
            return

        segmenter = self._segmenters.get(meeting_id)
        if segmenter is None:
            segmenter = SentenceSegmenter(
                merge_gap_ms=self.clean_settings.merge_gap_ms,
                max_sentence_ms=self.clean_settings.max_sentence_ms,
                idle_flush_ms=self.clean_settings.idle_flush_ms,
            )
            self._segmenters[meeting_id] = segmenter
        self._last_seen_ms[meeting_id] = self._now_ms()

        for sentence in segmenter.add(self._to_segment(stt_result)):
            await self._publish(meeting_id, sentence)

    def _to_segment(self, stt_result: STTResultMessage) -> CleanSegment:
        """One STT message as the segmenter wants it, with the prepass filled in if it is missing.

        `clean_text` is populated by stt_worker, but an older producer — or a rolling deploy, or
        a replay of a stream published before WT-716 — sends None, and this stage must not depend
        on being upstream of that change. Re-running the prepass here is cheap and deterministic:
        the same input gives the same answer, so a segment cleaned twice is cleaned identically.

        Early segments (`is_early`) are consumed like any other. They are not duplicates — the
        completed segment carries only the trailing fragment of its chunk — so skipping them
        would drop the first sentences of every long turn.
        """
        language = stt_result.language
        raw = stt_result.text
        flags = set(stt_result.clean_flags)
        fallback_raw = False

        if stt_result.clean_text is None:
            result = prepass(
                raw,
                language,
                prev_turn_is_question=self._previous_turn_was_a_question(stt_result.meeting_id),
            )
            clean_text = result.clean_text
            flags = set(result.flags)
            fallback_raw = _is_invariant_fallback(result.escalate_reasons)
        else:
            clean_text = stt_result.clean_text
            if FLAG_ESCALATE in flags and clean_text == raw:
                # The producer escalated AND returned the raw text, which is what the prepass
                # does when its own output failed the invariants — but the wire carries flags,
                # not reasons, so the two indistinguishable cases (a discourse marker it left
                # alone vs. a rule that misfired) are told apart by re-deriving the reasons.
                fallback_raw = _is_invariant_fallback(prepass(raw, language).escalate_reasons)

        if fallback_raw:
            flags.add(FLAG_FALLBACK_RAW)

        return CleanSegment(
            segment_id=stt_result.segment_id,
            speaker_id=stt_result.speaker_id,
            language=language,
            raw_text=raw,
            clean_text=clean_text,
            flags=frozenset(flags),
            start_ms=stt_result.start_ms,
            end_ms=stt_result.end_ms,
            arrived_at_ms=self._now_ms(),
        )

    def _previous_turn_was_a_question(self, meeting_id: str) -> bool:
        previous = self._last_line.get(meeting_id, "")
        return previous.rstrip().endswith(("?", "？"))

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def _publish(self, meeting_id: str, sentence: CleanSentence) -> None:
        """Publish revision 0, then start the LLM pass that may replace it with revision 1."""
        if sentence.is_empty:
            # Nothing but fillers ("Ummm"). The raw segments keep it; the clean view shows no
            # line at all, which is the whole point of the filler_only answer.
            return

        segment_ids = [sid for sid in sentence.segment_ids if _is_guid(sid)]
        if len(segment_ids) != len(sentence.segments):
            # The backend deserialises this field straight into a list of Guid. One bad id would
            # fail the whole row, so it is dropped here and said out loud.
            self.logger.warning(
                "transcript_clean_non_guid_segment_id",
                meeting_id=meeting_id,
                segment_ids=sentence.segment_ids,
            )

        text = normalize_terminal_punctuation(sentence.prepass_text, sentence.language)
        message = CleanSentenceMessage(
            meeting_id=meeting_id,
            speaker_id=sentence.speaker_id,
            segment_ids=segment_ids,
            clean_text=text,
            language=sentence.language,
            flags=sorted(sentence.flags & _WIRE_FLAGS),
            source="prepass",
            revision=0,
        )
        await self.publish(TRANSCRIPT_CLEAN_STREAM, meeting_id, message.to_redis())
        self.logger.debug(
            "transcript_clean_published",
            meeting_id=meeting_id,
            sentence_id=message.sentence_id,
            revision=0,
            reason=sentence.reason,
            segments=len(segment_ids),
        )

        previous_line = self._last_line.get(meeting_id, "")
        self._last_line[meeting_id] = text

        cleaner = self.cleaner
        if cleaner is None or not cleaner.is_available:
            return
        task = asyncio.create_task(
            self._refine(meeting_id, message, sentence, previous_line),
        )
        self._refine_tasks.add(task)
        task.add_done_callback(self._refine_tasks.discard)

    async def _refine(
        self,
        meeting_id: str,
        published: CleanSentenceMessage,
        sentence: CleanSentence,
        previous_line: str,
    ) -> None:
        """Ask the model, verify the answer, and publish revision 1 if it survives.

        Detached from `process` on purpose: nothing waits on it, a failure of any kind leaves
        revision 0 as the line, and the call is bounded by its own timeout and semaphore.
        """
        cleaner = self.cleaner
        if cleaner is None:
            return
        try:
            result = await cleaner.clean(
                sentence.raw_text,
                sentence.language,
                prepass_text=sentence.prepass_text,
                previous_line=previous_line,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("transcript_clean_llm_failed", meeting_id=meeting_id)
            return
        if result is None:
            return

        # `escalate` and `fallback_raw` described revision 0 — the rule tier was unsure, or its
        # answer had to be thrown away. Revision 1 is a different answer by a different tier, so
        # it carries only what is true of ITSELF.
        flags = [FLAG_SELF_REPAIR] if result.self_repair else []
        revised = CleanSentenceMessage(
            meeting_id=meeting_id,
            sentence_id=published.sentence_id,
            revision=published.revision + 1,
            speaker_id=published.speaker_id,
            segment_ids=list(published.segment_ids),
            clean_text=result.text,
            language=published.language,
            flags=flags,
            source="llm",
        )
        await self.publish(TRANSCRIPT_CLEAN_STREAM, meeting_id, revised.to_redis())
        if self._last_line.get(meeting_id) == published.clean_text:
            self._last_line[meeting_id] = result.text
        self.logger.info(
            "transcript_clean_revised",
            meeting_id=meeting_id,
            sentence_id=revised.sentence_id,
            revision=revised.revision,
            self_repair=result.self_repair,
        )

    # ------------------------------------------------------------------
    # Meeting lifecycle
    # ------------------------------------------------------------------

    async def _flush_meeting(self, meeting_id: str, *, reason: str) -> None:
        segmenter = self._segmenters.get(meeting_id)
        if segmenter is None:
            return
        for sentence in segmenter.flush(reason=reason):
            await self._publish(meeting_id, sentence)

    def _forget(self, meeting_id: str) -> None:
        self._segmenters.pop(meeting_id, None)
        self._last_line.pop(meeting_id, None)
        self._last_seen_ms.pop(meeting_id, None)

    async def _on_route_status_changed(self, room_id: str, new_status: str) -> None:
        if new_status in TERMINAL_ROOM_STATUSES:
            # The room ended without the assistant's sentinel reaching this worker (it is a
            # different stream entry and can be lost to a consumer-group reset). Same duty:
            # publish what is open before the state is dropped by `_cleanup_room` below.
            await self._flush_meeting(room_id, reason="meeting_end")

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        self._forget(room_id)


def _is_invariant_fallback(reasons: list[str]) -> bool:
    """Whether the prepass gave up and returned the RAW text because its output broke a rule.

    That is the `fallback_raw` case on the wire: the line the reader sees is the unedited
    recogniser output, and saying so is the difference between "nothing needed cleaning" and
    "cleaning was attempted and refused".
    """
    return any(reason.startswith(("I1_", "I2_")) for reason in reasons)
