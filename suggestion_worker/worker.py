"""Suggestion Worker — decides, unprompted, whether a live transcript segment deserves
a short inline hint, and publishes it for the gateway to push at the meeting.

Pipeline:
    Redis Stream (stt:results) — its OWN consumer group, parallel to translation
    → stage 0: local heuristics (no tokens spent)
    → cross-replica cooldown / per-meeting budget (Redis)
    → stage 1: decide (cheap model)
    → stage 2: generate (full model + meeting context)
    → ai_assistant:results with type="suggestion"

This runs beside the realtime STT → Translation → TTS path, never inside it: it reads the
same stream under a separate group, so a slow or failing suggestion never delays a
caption. Nothing here is persisted — suggestions are ephemeral by design.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from shared.base_worker import BaseWorker
from shared.config import SuggestionSettings
from shared.schemas import (
    STT_UNKNOWN_CONFIDENCE,
    STTResultMessage,
    SuggestionResultMessage,
)
from suggestion_worker.suggester import (
    SUGGESTION_CATEGORIES,
    NullSuggester,
    Suggester,
    TranscriptTurn,
)

# Route states in which a room is not actively translating. Segments can still arrive
# briefly after a pause/end (already-published stream entries), and suggesting into a
# meeting nobody is watching is pure waste.
_INACTIVE_ROUTE_STATES = frozenset({"PAUSED", "ENDED", "FAILED", "CANCELLED", "TIMEOUT"})

# How long a room's consent decision is trusted before re-reading it from Redis. Short
# enough that revoking external-LLM consent takes effect within a meeting, long enough that
# the common path is not one extra Redis round trip per transcript segment.
_POLICY_REFRESH_SECONDS = 60.0

# Interrogatives that open or close a question without a question mark. Vietnamese marks
# questions with particles rather than word order, so "gì", "sao", "à" carry the same signal an
# English "what" or "how" does. Kept deliberately small: this only decides which WORD-COUNT FLOOR
# a segment is measured against, never whether a suggestion is made — the decide model still has
# the final say, so a false positive here costs one cheap call and nothing else.
_QUESTION_MARKS = ("?", "？")
_QUESTION_WORDS = frozenset(
    {
        # Vietnamese
        "gì",
        "sao",
        "nào",
        "đâu",
        "ai",
        "bao",
        "mấy",
        "hả",
        "à",
        "chưa",
        "không",
        "thế",
        "vậy",
        "tại",
        # English
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "whose",
        "is",
        "are",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
    }
)


def _looks_like_question(text: str) -> bool:
    """Cheap, local, and deliberately generous.

    Punctuation first: production STT does emit question marks (665 of 4,622 stored segments end
    in one), so it is the strongest signal available for free. The word list is the fallback for
    the recogniser dropping the mark, which it does on short utterances — exactly the ones this
    exists to rescue.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(_QUESTION_MARKS):
        return True

    words = [word.strip("¿¡.,;:!\"'()[]").casefold() for word in stripped.split()]
    words = [word for word in words if word]
    if not words:
        return False
    # First or last word: English fronts its interrogatives, Vietnamese ends with a particle.
    return words[0] in _QUESTION_WORDS or words[-1] in _QUESTION_WORDS


def _sources_json(names: Sequence[str]) -> str:
    """Document names as the chip array every AI surface already speaks.

    Markers are issued here rather than omitted: the shape is shared with the chat assistant,
    where a marker is load-bearing, and a second shape differing in one optional field is how two
    renderers end up existing.
    """
    if not names:
        return ""
    return json.dumps(
        [
            {"marker": f"S{index}", "kind": "document", "title": name}
            for index, name in enumerate(names, start=1)
        ],
        ensure_ascii=False,
    )


class SuggestionWorker(BaseWorker):
    """Inline transcript suggestions — non-blocking, budget-capped, silent by default."""

    worker_name = "suggestion"
    input_stream = "stt:results"
    consumer_group = "suggestion-workers"  # Separate from translate/assistant/billing

    def __init__(
        self,
        suggestion_settings: SuggestionSettings | None = None,
        suggester: Suggester | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.suggestion_settings = suggestion_settings or SuggestionSettings()
        self.suggester: Suggester = suggester or NullSuggester()
        # Rolling per-room context handed to the decide stage. Bounded by window_size, so
        # this cannot grow with meeting length — only with the number of live rooms, and
        # _cleanup_room drops each one when its room ends.
        self._windows: dict[str, deque[TranscriptTurn]] = {}
        # room_id -> (allow_external_llm, fetched_at_monotonic). Re-read periodically so a
        # workspace that revokes external-LLM consent mid-meeting takes effect without a
        # restart.
        self._policies: dict[str, tuple[bool, float]] = {}

    async def load_model(self) -> None:
        await self.suggester.load()

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        # Without this, every room ever seen keeps its window alive for the life of the
        # process — the same unbounded-dict leak stt_worker._cleanup_room exists to avoid.
        # The Redis cooldown/budget keys are left to expire on their own TTL: a room can
        # emit late segments after its end event, and re-granting it a fresh budget here
        # would be worse than letting the keys lapse.
        self._windows.pop(room_id, None)
        self._policies.pop(room_id, None)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        if not self.suggestion_settings.enabled:
            return

        stt_result = STTResultMessage.from_redis(data)
        room_id = stt_result.meeting_id
        turn = TranscriptTurn(
            speaker_id=stt_result.speaker_id,
            text=" ".join(stt_result.text.split()),
            language=stt_result.language,
        )

        if not self._is_room_active(room_id):
            return

        # Context first, gating second: a segment too short to be worth suggesting on is
        # still worth remembering, because it may be what makes the NEXT one meaningful.
        window = self._windows.setdefault(
            room_id, deque(maxlen=self.suggestion_settings.window_size)
        )
        if turn.text:
            window.append(turn)

        if not self._passes_local_heuristics(turn, stt_result.confidence):
            # Stage 0 spends no tokens, so until now it wrote nothing at all — and it is the
            # gate that rejects roughly half of everything spoken. That is precisely why the
            # feature could look dead rather than starved: every other stage leaves a trace,
            # this one left none. DEBUG rather than INFO because it fires ~75x per meeting.
            self.logger.debug(
                "suggestion_stage0_rejected",
                meeting_id=room_id,
                words=len(turn.text.split()),
                question=_looks_like_question(turn.text),
                stt_confidence=stt_result.confidence,
            )
            return

        # Consent gate, before any transcript text leaves this process.
        if not await self._external_llm_allowed(room_id):
            self.logger.debug("suggestion_external_llm_denied", meeting_id=room_id)
            return

        # Read-only cooldown probe. The authoritative claim happens after the decide
        # stage — claiming here would burn a 45s slot on a segment the model then
        # declines, silencing the room for no reason.
        if await self._cooldown_active(room_id):
            self.logger.debug("suggestion_cooldown_active", meeting_id=room_id)
            return

        # The current segment is the subject, not part of its own context.
        context = [entry for entry in window if entry is not turn]
        decision = await self.suggester.decide(context, turn)

        if not decision.should_suggest:
            # The decide stage's own words for why. Paired with suggestion_below_confidence_gate
            # below, these two lines are what separate "the model is seeing this room and saying
            # no" from "nothing is reaching the model at all" — a distinction that previously
            # cost a production reproduction to make.
            self.logger.debug(
                "suggestion_declined",
                meeting_id=room_id,
                reason=decision.reason,
            )
            return
        if decision.confidence < self.suggestion_settings.min_confidence:
            self.logger.debug(
                "suggestion_below_confidence_gate",
                meeting_id=room_id,
                confidence=decision.confidence,
            )
            return
        if decision.category not in SUGGESTION_CATEGORIES:
            self.logger.warning(
                "suggestion_unknown_category",
                meeting_id=room_id,
                category=decision.category,
            )
            return

        # Read before the claim, so a hint that cannot possibly be produced does not burn a
        # cooldown slot and silence the room for the next 45 seconds.
        context_snapshot = await self._context_snapshot(room_id)

        # WT-371 Bug 6: "fact" means "a figure or reference the meeting's OWN DOCUMENTS cover".
        # With no documents attached there is nothing to ground it in, and the only thing the
        # model can do is recall a plausible number — which reaches the participants looking
        # exactly like a sourced one. Declining is the correct answer, not a weaker prompt.
        if decision.category == "fact" and not context_snapshot:
            self.logger.debug(
                "suggestion_fact_without_documents",
                meeting_id=room_id,
            )
            return

        if not await self._claim_slot(room_id):
            return

        suggestion = await self.suggester.generate(
            context,
            turn,
            decision,
            context_snapshot=context_snapshot,
        )
        if suggestion is None or not suggestion.content.strip():
            return

        await self._publish(stt_result, decision, suggestion)

    # ------------------------------------------------------------------
    # Stage 0 — local, no I/O and no tokens
    # ------------------------------------------------------------------

    def _is_room_active(self, room_id: str) -> bool:
        if room_id in self._paused_rooms:
            return False
        return self._route_states.get(room_id, "") not in _INACTIVE_ROUTE_STATES

    def _passes_local_heuristics(self, turn: TranscriptTurn, confidence: float) -> bool:
        """Reject before spending a token.

        Note what is NOT checked here: `is_final_chunk`. On this stream that flag marks
        stt_worker's trailing end-of-audio-chunk marker, which carries text="" — the
        segments with actual content have it set to false. Gating on it would discard
        every real segment and keep only the empty markers. The empty-text check below is
        what filters those markers out.

        A QUESTION IS MEASURED BY A DIFFERENT FLOOR, and that is the whole of this change.
        The single `min_words: 5` gate rejected 48% of real production segments and 39% of
        every segment that ends in a question mark — "JavaScript là gì?" is three words. It
        did so silently, because stage 0 spends no tokens and therefore logs nothing, which
        is why the badge looked dead rather than starved.
        """
        if not turn.text:
            return False
        floor = (
            self.suggestion_settings.min_question_words
            if _looks_like_question(turn.text)
            else self.suggestion_settings.min_words
        )
        if len(turn.text.split()) < floor:
            return False

        # WT-543 — AN ABSENT MEASUREMENT IS NOT A BAD ONE.
        #
        # `confidence` is an average token logprob, and STT_UNKNOWN_CONFIDENCE (-1.0) is the
        # sentinel stt_worker publishes when the provider exposed no logprobs at all. Compared as
        # if it were a score it is below every floor, so this line rejected the segment — and
        # stage 0 spends no tokens and writes no log line, so it rejected it in complete silence.
        #
        # That stopped being hypothetical on 2026-08-15, when the realtime STT path stopped
        # reporting logprobs: transcript segments carrying a confidence fell from ~85% to ~1% of
        # the day's rows, and the last suggestion this worker ever produced was 2026-08-14 11:10.
        # Six days of meetings, the worker healthy, its consumer group reading all 5,800 entries
        # with zero lag, and not one suggestion — which is precisely how WT-543 was reported.
        #
        # The same guard stt_worker/model.py already applies to this field (see the language
        # floor there): test the sentinel for equality first, and let unknown through. The decide
        # model is the real gate; this one exists to skip the obviously worthless, and "we do not
        # know how confident the recogniser was" is not evidence of worthlessness.
        if (
            confidence != STT_UNKNOWN_CONFIDENCE
            and confidence < self.suggestion_settings.min_stt_confidence
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Workspace consent
    # ------------------------------------------------------------------

    async def _external_llm_allowed(self, room_id: str) -> bool:
        """Whether this room's workspace permits sending transcript text to an external LLM.

        The policy is projected into Redis by the gateway
        (AiResultConsumerService.ResolveRoomPolicyAsync), which already resolves the same
        workspace settings for profanity masking — this worker has no gRPC client or
        service credentials to ask WorkspaceService itself.

        FAIL CLOSED. A missing or unreadable key means "we do not know whether this
        workspace consented", and the cost of guessing wrong is sending meeting speech to a
        third-party provider a workspace explicitly opted out of. The gateway writes the key
        on a room's very first AI result, well before enough transcript has accumulated for
        a suggestion to be worth making, so this costs nothing in practice.
        """
        cached = self._policies.get(room_id)
        if cached is not None and time.monotonic() - cached[1] < _POLICY_REFRESH_SECONDS:
            return cached[0]

        allowed = False
        try:
            raw = await self.redis.get(f"translationRoom:{room_id}:ai_policy")
            if raw is not None:
                payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                allowed = bool(payload.get("allow_external_llm"))
            else:
                self.logger.debug("suggestion_policy_not_published_yet", meeting_id=room_id)
        except Exception:
            self.logger.warning("suggestion_policy_unreadable", meeting_id=room_id)

        self._policies[room_id] = (allowed, time.monotonic())
        return allowed

    # ------------------------------------------------------------------
    # Rate limiting — Redis, so it holds across replicas
    # ------------------------------------------------------------------

    @staticmethod
    def _cooldown_key(room_id: str) -> str:
        return f"suggest:cd:{room_id}"

    @staticmethod
    def _budget_key(room_id: str) -> str:
        return f"suggest:n:{room_id}"

    async def _cooldown_active(self, room_id: str) -> bool:
        return await self.redis.get(self._cooldown_key(room_id)) is not None

    async def _claim_slot(self, room_id: str) -> bool:
        """Atomically take this room's next suggestion slot. False means don't suggest.

        Both limits live in Redis rather than worker memory because the production chart
        runs these workers at replicas >= 2, and the consumer group hands one room's
        segments to whichever replica is free — per-process counters would multiply the
        real budget by the replica count and make the cooldown ineffective.
        """
        claimed = await self.redis.set_if_absent(
            self._cooldown_key(room_id),
            "1",
            self.suggestion_settings.cooldown_seconds,
        )
        if not claimed:
            return False

        used = await self.redis.incr_with_ttl(
            self._budget_key(room_id),
            self.suggestion_settings.state_ttl_seconds,
        )
        if used > self.suggestion_settings.max_per_meeting:
            self.logger.info("suggestion_budget_exhausted", meeting_id=room_id, used=used)
            return False
        return True

    # ------------------------------------------------------------------
    # Stage 2 support + publish
    # ------------------------------------------------------------------

    async def _context_snapshot(self, room_id: str) -> str:
        """RAG document text the meeting was started with, if any.

        Written by the same key ai_assistant_worker reads for summaries. Only fetched
        once a suggestion has already cleared every gate, so the cost is per-suggestion,
        not per-segment.
        """
        raw = await self.redis.get(f"meeting:{room_id}:context_snapshot")
        if raw is None:
            return ""
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw

    async def _publish(
        self,
        stt_result: STTResultMessage,
        decision: Any,
        suggestion: Any,
    ) -> None:
        content = " ".join(suggestion.content.split())
        limit = self.suggestion_settings.max_suggestion_chars
        if len(content) > limit:
            content = content[:limit].rstrip()

        message = SuggestionResultMessage(
            meeting_id=stt_result.meeting_id,
            segment_id=stt_result.segment_id,
            category=suggestion.category or decision.category,
            content=content,
            detail=suggestion.detail,
            confidence=decision.confidence,
            language=stt_result.language,
            token_count=decision.token_count + suggestion.token_count,
            # The same array shape ai_assistant_worker publishes, so the client renders one
            # component on every AI surface rather than one per surface. A hint's source is
            # always a document: the transcript is what the reader is already looking at, and a
            # "Transcript" chip on a badge pinned to a transcript line says nothing.
            sources_json=_sources_json(getattr(suggestion, "sources", ())),
        )

        await self.publish("ai_assistant:results", stt_result.meeting_id, message.to_redis())

        self.logger.info(
            "suggestion_published",
            meeting_id=stt_result.meeting_id,
            segment_id=stt_result.segment_id,
            category=message.category,
            confidence=decision.confidence,
            tokens=message.token_count,
        )
