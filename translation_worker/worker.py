"""Translation Worker — Consumes STT results, produces translated text.

Pipeline:
    Redis Stream (stt:results:{meetingId})
    → OpenAI gpt-4.1-mini
    → Redis Stream (translate:results:{meetingId})

Passthrough: if source_lang == target_lang, forward without translation.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from typing import Any, cast

from shared.base_worker import BaseWorker
from shared.config import TranslationSettings, resolve_openai_api_key
from shared.control_markers import is_control_marker, is_system_speaker
from shared.lang import is_same_language
from shared.schemas import (
    ProsodyEnvelope,
    STTResultMessage,
    TranslationResultMessage,
    optional_confidence,
)
from shared.text_utils import split_into_sentences
from translation_worker.transcript_guardian import choose_transcript
from translation_worker.translator import OUT_OF_MEETING_SCOPE, OpenAITranslator
from translation_worker.valence import Valence

# What a translation task yields: the text, and the sentiment of what was said — or None when
# the model said nothing trustworthy about it. Carried together so a speculative hit keeps its
# valence instead of silently losing it, which is how a wired feature becomes an unwired one.
_Translated = tuple[str, "Valence | None"]


def _with_valence(
    envelope: ProsodyEnvelope | None, valence: Valence | None
) -> ProsodyEnvelope | None:
    """The measured delivery, plus the sentiment of what was said.

    Returns the envelope UNCHANGED when there is no valence — including when there is no
    envelope at all. An absent envelope means the STT worker could not honestly measure this
    speaker yet, and inventing one here just to carry a sentiment would tell the TTS worker that
    a delivery was measured when none was: every ratio would be a default 1.0 presented as a
    reading. Valence rides along with a measurement; it does not manufacture one.
    """
    if envelope is None or valence is None:
        return envelope
    return envelope.model_copy(update={"valence": valence})


_CONTEXT_STOPWORDS = {
    "and",
    "context",
    "from",
    "meeting",
    "the",
    "this",
    "topic",
    "và",
    "các",
    "cho",
    "chúng",
    "của",
    "là",
    "một",
    "này",
    "ta",
    "trong",
}
_CONTEXT_FREE_SHORT_TURNS = {
    "cảm ơn",
    "đồng ý",
    "được",
    "được rồi",
    "không",
    "ok",
    "okay",
    "thanks",
    "tiếp đi",
    "ừ",
    "vâng",
    "yes",
}


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        if len(token) >= 3 and token not in _CONTEXT_STOPWORDS
    }


def _select_relevance_context(text: str, meeting_context: list[str] | None) -> list[str]:
    """Use expensive semantic relevance context only when local evidence is ambiguous.

    Known acknowledgements are always legitimate meeting turns. Utterances that share
    a meaningful topic/recent-utterance token are clearly in scope and can take the
    fast translation path. Ambiguous short commands/pronouns still keep context so
    translation can resolve their meaning; unrelated speech keeps it for suppression.
    """
    if not meeting_context:
        return []
    normalized = " ".join(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))
    if normalized in _CONTEXT_FREE_SHORT_TURNS:
        return []
    utterance_tokens = _content_tokens(text)
    context_tokens = _content_tokens("\n".join(meeting_context))
    if utterance_tokens & context_tokens:
        return []
    return meeting_context


class TranslationWorker(BaseWorker):
    """Translation worker using OpenAI gpt-4.1-mini."""

    worker_name = "translation"
    input_stream = "stt:results"
    consumer_group = "translate-workers"

    # Bounds concurrent process() dispatch in _consume_loop — see its docstring.
    _CONCURRENCY_LIMIT = 8
    _CONTEXT_MIN_CONFIDENCE = -0.35
    _CONTEXT_SEGMENTS = 4
    _SPECULATIVE_TTL_SECONDS = 15.0
    # Warm Realtime responses occasionally land just above one second (1.015s in the
    # deterministic LiveKit replay). Keep one bounded speculative slot alive long
    # enough to reuse that result instead of cancelling it and paying a second call.
    _SPECULATIVE_TIMEOUT_SECONDS = 1.5
    # The same-language tidy-up's whole latency budget. Passthrough used to publish with no
    # network call at all, so this is the amount of delay a nicer-looking transcript is worth;
    # past it the raw recogniser text goes out unchanged.
    _POLISH_TIMEOUT_SECONDS = 1.5

    def __init__(
        self,
        translation_settings: TranslationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.translation_settings = translation_settings or TranslationSettings()
        self.translator: OpenAITranslator | None = None
        # meeting_id -> this workspace's glossary as [{"source": ..., "target": ...}, ...],
        # published to `translationRoom:{meeting_id}:mt_glossary` by
        # GlossaryStartedEventConsumer (TranscriptService) when the room starts. Cached for
        # the room's lifetime — same "don't hit Redis on every chunk" reasoning as
        # stt_worker's _stt_prompts (see docs/code-switching-research.md).
        self._mt_glossaries: dict[str, list[dict[str, str]]] = {}
        self._meeting_contexts: dict[str, list[str]] = {}
        self._recent_source_contexts: dict[str, deque[str]] = {}
        self._speculative_translations: dict[
            tuple[str, str, str, str],
            tuple[asyncio.Task[_Translated], float],
        ] = {}
        self._speculative_semaphore = asyncio.Semaphore(1)
        self._speculative_listener_task: asyncio.Task[None] | None = None

    async def load_model(self) -> None:
        """Initialize OpenAI translation client."""
        self.translator = OpenAITranslator(
            api_key=resolve_openai_api_key(self.translation_settings.api_key),
            model=self.translation_settings.model,
            realtime_model=self.translation_settings.realtime_model,
            realtime_reasoning_effort=self.translation_settings.realtime_reasoning_effort,
            realtime_pool_size=self.translation_settings.realtime_pool_size,
            realtime_timeout_seconds=self.translation_settings.realtime_timeout_seconds,
            realtime_max_output_tokens=self.translation_settings.realtime_max_output_tokens,
            max_tokens=self.translation_settings.max_tokens,
            temperature=self.translation_settings.temperature,
        )
        await self.translator.load()
        await self.translator.warm_up()
        self._speculative_listener_task = asyncio.create_task(
            self._listen_for_speculative_transcripts()
        )

    async def _listen_for_speculative_transcripts(self) -> None:
        """Warm translation from STT deltas without publishing unvalidated text."""
        pubsub = self.redis.redis.pubsub()
        try:
            await pubsub.subscribe("stt:speculative")
            self.logger.info("speculative_translation_listener_started")
            while not self._shutdown_event.is_set():
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if not message:
                    continue
                try:
                    payload = json.loads(message["data"])
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                    continue
                if isinstance(payload, dict):
                    asyncio.create_task(self._prefetch_from_event(payload))
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.exception("speculative_translation_listener_failed")
        finally:
            try:
                await pubsub.close()
            except Exception:
                self.logger.warning("speculative_translation_listener_close_failed")

    @staticmethod
    def _speculative_text_key(text: str) -> str:
        return " ".join(text.casefold().split()).rstrip(" .!?")

    async def _prefetch_from_event(self, payload: dict[str, Any]) -> None:
        meeting_id = payload.get("meeting_id")
        speaker_id = payload.get("speaker_id")
        text = payload.get("text")
        source_lang = payload.get("language")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (meeting_id, speaker_id, text, source_lang)
        ):
            return

        assert isinstance(meeting_id, str)
        assert isinstance(speaker_id, str)
        assert isinstance(text, str)
        assert isinstance(source_lang, str)
        sentences = split_into_sentences(text)
        if not sentences:
            return

        target_langs = await self._get_target_languages(meeting_id, speaker_id, source_lang)
        glossary_terms = await self._get_mt_glossary(meeting_id)
        recent_context = list(getattr(self, "_recent_source_contexts", {}).get(meeting_id, ()))
        static_context = await self._get_meeting_context(meeting_id)
        meeting_context = recent_context[-3:] + static_context
        translator = self._require_translator()
        cache = getattr(self, "_speculative_translations", None)
        if cache is None:
            cache = {}
            self._speculative_translations = cache

        tasks: list[asyncio.Task[_Translated]] = []
        now = time.monotonic()
        semaphore = getattr(self, "_speculative_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            self._speculative_semaphore = semaphore
        for target_lang in target_langs:
            # Redundant since _get_target_languages started filtering these out, and kept
            # anyway: speculation is the one path that must never pay for an echo.
            if is_same_language(source_lang, target_lang):
                continue
            for sentence in sentences:
                key = (
                    meeting_id,
                    speaker_id,
                    target_lang,
                    self._speculative_text_key(sentence),
                )
                existing = cache.get(key)
                if existing is not None and now - existing[1] <= self._SPECULATIVE_TTL_SECONDS:
                    continue
                # Speculation is disposable and must never queue behind itself or occupy
                # the hot Realtime pool needed by validated final traffic.
                if semaphore.locked():
                    continue

                async def run_speculative(
                    sentence_text: str = sentence,
                    language_target: str = target_lang,
                ) -> _Translated:
                    async with semaphore:
                        try:
                            async with asyncio.timeout(self._SPECULATIVE_TIMEOUT_SECONDS):
                                return await translator.translate_with_valence(
                                    sentence_text,
                                    source_lang=source_lang,
                                    target_lang=language_target,
                                    glossary_terms=glossary_terms,
                                    meeting_context=_select_relevance_context(
                                        sentence_text,
                                        meeting_context,
                                    ),
                                )
                        except Exception:
                            self.logger.info(
                                "speculative_translation_abandoned",
                                meeting_id=meeting_id,
                            )
                            # Empty text is the caller's signal to translate for real; the
                            # valence beside it is None because nothing was read.
                            return "", None

                task = asyncio.create_task(run_speculative())
                cache[key] = (task, now)
                tasks.append(task)

        if tasks:
            started = time.monotonic()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info(
                "speculative_translation_ready",
                meeting_id=meeting_id,
                sentences=len(tasks),
                latency_ms=int((time.monotonic() - started) * 1000),
            )

    def _take_speculative_translation(
        self,
        meeting_id: str,
        speaker_id: str,
        target_lang: str,
        text: str,
    ) -> asyncio.Task[_Translated] | None:
        cache = cast(
            dict[
                tuple[str, str, str, str],
                tuple[asyncio.Task[_Translated], float],
            ],
            getattr(self, "_speculative_translations", {}),
        )
        key = (
            meeting_id,
            speaker_id,
            target_lang,
            self._speculative_text_key(text),
        )
        cached = cache.pop(key, None)
        if cached is None:
            return None
        task, created_at = cached
        if time.monotonic() - created_at > self._SPECULATIVE_TTL_SECONDS:
            if not task.done():
                task.cancel()
            return None
        return task

    async def _consume_loop(self) -> None:
        """Dispatch process() concurrently instead of one-message-at-a-time.

        stt_worker now pipelines: a multi-sentence utterance arrives as SEPARATE
        stt:results messages, each published the moment that sentence is detected
        (see stt_worker's on_early_segment), specifically so translation doesn't have
        to wait for the whole STT chunk to finish. BaseWorker._consume_loop awaits
        process() for one message before reading the next, which would silently
        re-serialize those messages' translate() calls right back into the same
        sequential bottleneck the earlier per-sentence asyncio.gather() fix removed —
        just moved up from "within one process() call" to "across messages". Dispatch
        concurrently (bounded by _CONCURRENCY_LIMIT) so sentence 2's translation can
        start while sentence 1's is still in flight.

        RedisStreamClient.consume_concurrent ties XACK to successful handler
        completion. Failed work remains pending for BaseWorker's reclaim/DLQ path.
        """
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )

        async def _run(message_id: bytes, data: dict[bytes, bytes]) -> None:
            await self._process_and_log_errors(message_id, data)

        while not self._shutdown_event.is_set():
            try:
                await self._recover_stale_messages()
                await self.redis.consume_concurrent(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    handler=_run,
                    consumer=self._consumer_name,
                    block_ms=2000,
                    count=self._CONCURRENCY_LIMIT,
                    concurrency=self._CONCURRENCY_LIMIT,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger.exception("consume_loop_error")
                await asyncio.sleep(1.0)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Translate one STT result segment by chunking into sentences.

        Every participant is simultaneously a translation source (when they speak) and a
        target (their own listen language) — a meeting is not one fixed source->target
        pair. Fan out to every DISTINCT listen-language among the OTHER participants (not
        per-listener: two listeners both wanting Vietnamese share one translated+dubbed
        track, matching how tts_worker's LiveKitTTSPublisher already keys its bots by
        (meeting_id, target_lang)).
        """
        stt_result = STTResultMessage.from_redis(data)

        # Not speech. `stt:results` carries one synthetic segment per meeting — the
        # __MEETING_END__ sentinel MeetingRoomService publishes to wake the assistant worker —
        # and this stage translated it like anything else: one paid LLM call per target
        # language, whose output ("Meeting end", "Kết thúc cuộc họp") then went to tts_worker
        # for a paid render and onto the interpreter track, and to billing_worker, which
        # dead-lettered it twice per meeting because speaker_id="system" is not a UUID.
        #
        # Checked FIRST, before the pause and translation-active gates: whether the platform's
        # own control message gets translated is not a per-room setting.
        if is_control_marker(stt_result.text) or is_system_speaker(stt_result.speaker_id):
            self.logger.debug(
                "control_marker_skipped",
                meeting_id=stt_result.meeting_id,
                speaker_id=stt_result.speaker_id,
            )
            return

        if stt_result.meeting_id in self._paused_rooms:
            return

        # Translation is opt-in; transcription is not. This gate used to live in
        # livekit_ingress_worker, where it stopped audio reaching STT at all — so a meeting
        # nobody had started translation on produced no transcript either, and the two
        # features could not be used apart.
        #
        # It belongs here instead. STT now runs for any live meeting, which is what fills
        # the transcript panel, and this stage — the one that actually spends a translation
        # and a dubbed voice — stays closed until the room reports translation active.
        # `_translation_active_for`, not `_is_translation_active`: the async one recovers the
        # room's route snapshot from Redis when no broadcast has been seen for it. A worker that
        # restarted mid-meeting — which is what a deploy does — otherwise reads False here for a
        # room that is actively translating and drops every result until the room ends.
        if not await self._translation_active_for(stt_result.meeting_id):
            # INFO, not debug. This is the one branch that discards a paid-for STT result, and at
            # production's INFO level the debug line was invisible: the pipeline went silent with
            # no record of the decision anywhere. WT-373 was diagnosed from Redis stream offsets
            # because these logs did not exist.
            self.logger.info(
                "translation_skipped_not_started",
                meeting_id=stt_result.meeting_id,
            )
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - stt_result.timestamp_ms
        await self.redis.publish_telemetry(stt_result.meeting_id, self.worker_name, e2e_latency_ms)

        target_langs = await self._get_target_languages(
            stt_result.meeting_id, stt_result.speaker_id, stt_result.language
        )
        glossary_terms = await self._get_mt_glossary(stt_result.meeting_id)
        recent_context = list(
            getattr(self, "_recent_source_contexts", {}).get(stt_result.meeting_id, ())
        )
        static_context = await self._get_meeting_context(stt_result.meeting_id)
        # Keep the durable topic/description at the end so translator.py's bounded tail
        # always retains it even after several accepted utterances have accumulated.
        meeting_context = recent_context[-3:] + static_context

        # Split long STT results into smaller sentences (streaming mechanism)
        sentences = split_into_sentences(stt_result.text)

        if not sentences:
            if stt_result.is_final_chunk:
                for target_lang in target_langs:
                    result = TranslationResultMessage(
                        segment_id=f"{stt_result.segment_id}-{target_lang}",
                        meeting_id=stt_result.meeting_id,
                        speaker_id=stt_result.speaker_id,
                        original_text="",
                        translated_text="",
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        is_final_chunk=True,
                        timestamp_ms=stt_result.timestamp_ms,
                        translator_model=self._require_translator().model,
                        source_segment_id=stt_result.segment_id,
                        chunk_index=stt_result.chunk_index,
                        # Courier, not judge — billing_worker is what reads this. See
                        # STTResultMessage.is_early.
                        is_early=stt_result.is_early,
                        prosody=stt_result.prosody,
                    )
                    await self.publish(
                        "translate:results", stt_result.meeting_id, result.to_redis()
                    )
            return

        publish_results = await asyncio.gather(
            *(
                self._translate_and_publish(
                    stt_result,
                    sentences,
                    target_lang,
                    glossary_terms,
                    meeting_context,
                )
                for target_lang in target_langs
            )
        )
        if any(publish_results) and stt_result.confidence >= self._CONTEXT_MIN_CONFIDENCE:
            self._remember_source_context(stt_result.meeting_id, stt_result.text)

    async def _translate_and_publish(
        self,
        stt_result: STTResultMessage,
        sentences: list[str],
        target_lang: str,
        glossary_terms: list[dict[str, str]] | None = None,
        meeting_context: list[str] | None = None,
    ) -> bool:
        # Fire off translation for all sentences UP FRONT so they run concurrently instead
        # of one-sentence-at-a-time (each translate() call is a real OpenAI network
        # round-trip — awaiting them sequentially in the publish loop below used to add a
        # full round-trip of latency per extra sentence). Sentence 0 still gets its own
        # single-sentence call (not folded into the batch) so it can be awaited and
        # published on its own as soon as it's ready — TTS starts on it immediately,
        # same as before — while sentences 1..N-1 translate together in ONE batched call
        # that's already running in the background by the time we get to them.
        # is_same_language, not ==. A source of "vi" against a target of "vi-VN" is the
        # same language spelled two ways, and comparing the tags said otherwise — so the
        # worker translated Vietnamese into Vietnamese, paying for a model call to produce
        # a sentence it already had.
        passthrough = is_same_language(stt_result.language, target_lang)
        first_task: asyncio.Task[_Translated] | None = None
        rest_task: asyncio.Task[list[str]] | None = None
        rest_results: list[str] | None = None
        speculative_hit = False
        translation_started = time.monotonic()
        translator = self._require_translator()
        if not passthrough:
            first_context = _select_relevance_context(sentences[0], meeting_context)
            first_task = self._take_speculative_translation(
                stt_result.meeting_id,
                stt_result.speaker_id,
                target_lang,
                sentences[0],
            )
            speculative_hit = first_task is not None
            if first_task is None:
                first_task = asyncio.create_task(
                    translator.translate_with_valence(
                        sentences[0],
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                        meeting_context=first_context,
                    )
                )
            if len(sentences) > 1:
                rest_context = (
                    meeting_context
                    if any(
                        _select_relevance_context(sentence, meeting_context)
                        for sentence in sentences[1:]
                    )
                    else []
                )
                rest_task = asyncio.create_task(
                    translator.translate_batch(
                        sentences[1:],
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                        meeting_context=rest_context,
                    )
                )

        published_any = False
        for idx, sentence in enumerate(sentences):
            # Sequence the segment ID (per target language, so different listeners'
            # translations of the same STT segment don't collide) so the frontend gets
            # consecutive speech segments.
            chunk_segment_id = f"{stt_result.segment_id}-{target_lang}-c{idx}"

            if passthrough:
                # Same language as the speaker, so there is nothing to translate — but the text
                # is raw recogniser output: no sentence casing, little punctuation, and every
                # "ờ"/"à" that was actually said. A same-language pass makes it read like
                # writing. See transcript_guardian for why the result is verified rather than
                # trusted, and why a failed verification keeps the original instead of
                # publishing a fluent guess.
                # BOUNDED, because this path used to cost nothing at all.
                #
                # Passthrough previously forwarded the sentence with no network call, so adding
                # one puts a model round-trip in front of a line that used to publish instantly.
                # Tidier text is not worth a transcript that lags the speaker, so the tidy-up
                # gets a small budget and the raw sentence goes out the moment it is exceeded.
                try:
                    async with asyncio.timeout(self._POLISH_TIMEOUT_SECONDS):
                        polished = await translator.polish(sentence, stt_result.language)
                except Exception:
                    polished = sentence

                translated_text = choose_transcript(sentence, polished, stt_result.language)
                valence = None
            elif idx == 0:
                assert first_task is not None
                try:
                    translated_text, valence = await first_task
                except Exception:
                    translated_text, valence = "", None
                if not translated_text:
                    translated_text, valence = await translator.translate_with_valence(
                        sentence,
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                        meeting_context=first_context,
                    )
            else:
                if rest_results is None:
                    assert rest_task is not None
                    rest_results = await rest_task
                translated_text = rest_results[idx - 1]
                # translate_batch does not ask for a marker: its reply is a numbered list, and a
                # per-line marker would be one more thing for the line parser to get wrong. The
                # sentences after the first therefore carry no valence, which the pipeline reads
                # as NOT DETERMINED — the same as before this existed.
                valence = None

            if translated_text.strip().upper() == OUT_OF_MEETING_SCOPE:
                # SUPPRESSION GETS A SECOND OPINION, WITHOUT THE CONTEXT THAT CAUSED IT.
                #
                # This branch used to `continue`, which published nothing for THIS target and
                # nothing else — so the sentence reached every other listener and vanished for
                # one. It is judged per target language, in a separate model call per target, so
                # the same utterance could be in scope for one listener and out of scope for
                # their neighbour; the transcript still showed the original, which is what made
                # it read as "roughly one line in ten is not translated".
                #
                # Production, 12h: 10 suppressions against 228 translated chunks — 4% — and every
                # one of them was ordinary meeting speech. "Em mút mic của anh nè, anh có bị mút
                # không?" is a person asking about their microphone.
                #
                # The sentinel is an artefact of the relevance context: it is only offered when
                # meeting_context is attached (see _select_relevance_context), so the honest test
                # is whether the sentence is still out of scope WITHOUT it. Genuine background
                # noise — a television, someone else's conversation — fails that test too and is
                # still dropped. A real utterance comes back translated and gets delivered.
                retry_text = ""
                try:
                    retry_text, valence = await translator.translate_with_valence(
                        sentence,
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                        meeting_context=[],
                    )
                except Exception:
                    retry_text, valence = "", None

                if not retry_text.strip() or retry_text.strip().upper() == OUT_OF_MEETING_SCOPE:
                    self.logger.info(
                        "background_utterance_suppressed",
                        meeting_id=stt_result.meeting_id,
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        original=sentence[:60],
                        confirmed_without_context=True,
                    )
                    continue

                # Warning, not info: a sentence that only looked out of scope because of the
                # context we attached is a false positive of our own making, and the rate of
                # them is the thing worth watching.
                self.logger.warning(
                    "background_utterance_suppression_overturned",
                    meeting_id=stt_result.meeting_id,
                    source_lang=stt_result.language,
                    target_lang=target_lang,
                    original=sentence[:60],
                )
                translated_text = retry_text

            is_final = (idx == len(sentences) - 1) and stt_result.is_final_chunk

            # How long this sentence took to become available, from the start of the stage.
            #
            # Measured once and used twice — on the message and on the log line below — so the
            # number a dashboard sees and the number a log search finds cannot drift apart.
            #
            # Measured from `translation_started` rather than per-call on purpose: the sentences
            # are translated concurrently (sentence 0 alone, 1..N-1 in one batch), so a per-call
            # timer would report the batch's own duration and hide the wait in front of it. What a
            # listener experiences is the whole interval before their sentence arrived, which is
            # what this is.
            sentence_latency_ms = int((time.monotonic() - translation_started) * 1000)

            result = TranslationResultMessage(
                segment_id=chunk_segment_id,
                meeting_id=stt_result.meeting_id,
                speaker_id=stt_result.speaker_id,
                original_text=sentence,
                translated_text=translated_text,
                source_lang=stt_result.language,
                target_lang=target_lang,
                # WT-278: explicitly the SOURCE segment's STT confidence, carried for diagnostics
                # only. The translator returns no quality score, so this must never be presented as
                # one — see TranslationResultMessage.source_stt_confidence. The -1.0 "no logprobs"
                # sentinel collapses to None here so it is never persisted as data (WT-277).
                source_stt_confidence=optional_confidence(str(stt_result.confidence)),
                start_ms=stt_result.start_ms,
                end_ms=stt_result.end_ms,
                is_final_chunk=is_final,
                timestamp_ms=stt_result.timestamp_ms,
                translator_model=translator.model,
                source_segment_id=stt_result.segment_id,
                chunk_index=stt_result.chunk_index,
                # Courier, not judge — billing_worker is what reads this. See
                # STTResultMessage.is_early.
                is_early=stt_result.is_early,
                # Delivery is carried, not derived: how the speaker sounded is settled upstream
                # at the audio, and translating the words does not change it. VALENCE is the one
                # part that cannot come from the audio — anger and delight look alike on pitch
                # and energy — so it is folded in here, from the model that actually read the
                # sentence. See translation_worker/valence.py.
                prosody=_with_valence(stt_result.prosody, valence),
                # Carried so TranscriptService can finally fill translation_contents.latency_ms.
                # The column and its C# property have existed all along and were NULL on every
                # row ever written, which is why "translation is sometimes slow" has never been
                # answerable after the fact.
                latency_ms=sentence_latency_ms,
            )

            # Publish IMMEDIATELY so TTS can synthesize while next chunk is translated
            await self.publish("translate:results", stt_result.meeting_id, result.to_redis())
            published_any = True

            self.logger.info(
                "chunk_translated",
                meeting_id=stt_result.meeting_id,
                speaker_id=stt_result.speaker_id,
                # Both ids: this message's own, which is what tts_worker will log, and the STT
                # segment it came from, which is what stt_worker logged. One sentence can fan
                # out to several target languages, so the source id is what collapses them back
                # into the one thing the speaker actually said.
                segment_id=result.segment_id,
                source_segment_id=stt_result.segment_id,
                chunk_index=idx,
                source_lang=stt_result.language,
                target_lang=target_lang,
                original=sentence[:60],
                translated=translated_text[:60],
                # Flash mode: an early sentence is spoken but never billed. When a dub goes
                # missing this says which half of the pipeline it belonged to.
                is_early=stt_result.is_early,
                speculative_hit=speculative_hit if idx == 0 else False,
                stage_latency_ms=sentence_latency_ms,
                pipeline_latency_ms=max(0, int(time.time() * 1000) - stt_result.timestamp_ms),
            )
        return published_any

    async def _get_target_languages(
        self, meeting_id: str, speaker_id: str, source_lang: str = ""
    ) -> set[str]:
        """Every DISTINCT listen-language among the OTHER participants in this meeting,
        minus the language the speaker is already speaking.

        Reads from a Redis hash set by the backend when a user joins and selects their
        preferred output language: `translationRoom:{translationRoomId}:languages`,
        keyed by userId -> that user's own ListenLanguage (TranslationRoomHub.
        JoinTranslationRoom).

        NOTE: `meeting_id` here is actually the translation_room_id (see
        AudioChunkMessage.from_redis / RedisStreamService.PublishAudioChunkAsync).

        Previously this returned only the FIRST other participant's entry found — with
        >1 listener wanting different languages, everyone but the first got nothing
        (single target_lang per TranslationResultMessage). Fixed to fan out to every
        distinct language present instead of picking one.
        """
        all_languages = await self.redis.hgetall(f"translationRoom:{meeting_id}:languages")
        targets: set[str] = set()
        for raw_user_id, raw_lang in all_languages.items():
            user_id = raw_user_id.decode() if isinstance(raw_user_id, bytes) else raw_user_id
            if user_id == speaker_id:
                continue
            lang = raw_lang.decode() if isinstance(raw_lang, bytes) else raw_lang
            if lang:
                targets.add(lang)

        # No other participant registered yet — avoid assuming Vietnamese for all users.
        targets = targets or {"en"}

        # S6. The speaker's OWN language is not a translation target, and this is the only
        # place that can say so — the set excluded the speaker's user id but never their
        # language, so a listener who chose the room's source language got a full
        # TranslationResultMessage. Nothing downstream stopped it: `passthrough` only
        # skipped the LLM call (the message was still built and published), and
        # TTSWorker.process had no language comparison at all, so it synthesized the
        # speaker's own words and published them on an ai-interpreter LiveKit track. The
        # listener is already subscribed to that speaker's raw mic, so they heard the real
        # voice and a synthetic echo of the same sentence at once.
        #
        # Deliberately NOT relying on the backend to have filtered this out. The AI never
        # reads TranslationRoomAudioRouteService's route table; it reads the
        # translationRoom:{id}:languages hash, and TranslationRoomHub.SetListenLanguage
        # writes that hash with no policy validation at all. LanguagePolicy also explicitly
        # permits listening in the room's source language. Both are legitimate ways for a
        # same-language pair to exist, so this must hold on its own.
        #
        # This also disarms the `targets or {"en"}` fallback above: a lone English speaker
        # used to be given "en" as a target and hear an English echo of themselves.
        if source_lang:
            echoes = {lang for lang in targets if is_same_language(lang, source_lang)}
            if echoes:
                self.logger.info(
                    "same_language_targets_dropped",
                    meeting_id=meeting_id,
                    speaker_id=speaker_id,
                    source_lang=source_lang,
                    dropped=sorted(echoes),
                )
                targets -= echoes

        return targets

    async def _get_mt_glossary(self, meeting_id: str) -> list[dict[str, str]]:
        """This meeting's workspace glossary, as [{"source": ..., "target": ...}, ...] —
        published to `translationRoom:{meeting_id}:mt_glossary` by
        GlossaryStartedEventConsumer (TranscriptService) when the room starts. Cached for
        the room's lifetime (see self._mt_glossaries), same reasoning as stt_worker's
        _get_stt_prompt: the glossary is set before the meeting, not mid-meeting.

        Empty list (not an error) when the workspace has no active glossary — translate()/
        translate_batch() already treat that as "no glossary", falling back to their plain
        proper-nouns exception.
        """
        cached = self._mt_glossaries.get(meeting_id)
        if cached is not None:
            return cached

        raw = await self.redis.get(f"translationRoom:{meeting_id}:mt_glossary")
        glossary: list[dict[str, str]] = []
        if raw:
            try:
                parsed = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                if isinstance(parsed, list):
                    glossary = [cast(dict[str, str], t) for t in parsed if isinstance(t, dict)]
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.logger.warning("mt_glossary_parse_failed", meeting_id=meeting_id)

        self._mt_glossaries[meeting_id] = glossary
        if glossary:
            self.logger.info("mt_glossary_loaded", meeting_id=meeting_id, terms=len(glossary))
        return glossary

    async def _get_meeting_context(self, meeting_id: str) -> list[str]:
        """Load the bounded room title/description published on ``meeting.started``.

        This durable context disambiguates short phrases and code-switched technical terms;
        recent accepted source utterances are added separately in :meth:`process`.
        """
        contexts = cast(
            dict[str, list[str]] | None,
            getattr(self, "_meeting_contexts", None),
        )
        if contexts is None:
            contexts = {}
            self._meeting_contexts = contexts
        cached = contexts.get(meeting_id)
        if cached is not None:
            return cached

        raw = await self.redis.get(f"translationRoom:{meeting_id}:meeting_context")
        context: list[str] = []
        if raw:
            try:
                decoded = raw.decode() if isinstance(raw, bytes) else str(raw)
            except UnicodeDecodeError:
                self.logger.warning("meeting_context_decode_failed", meeting_id=meeting_id)
            else:
                normalized = " ".join(decoded.split())[:600].strip()
                if normalized:
                    context = [normalized]

        contexts[meeting_id] = context
        if context:
            self.logger.info("meeting_context_loaded", meeting_id=meeting_id)
        return context

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        self._mt_glossaries.pop(room_id, None)
        getattr(self, "_meeting_contexts", {}).pop(room_id, None)
        getattr(self, "_recent_source_contexts", {}).pop(room_id, None)
        speculative = getattr(self, "_speculative_translations", {})
        for key in [key for key in speculative if key[0] == room_id]:
            task, _created_at = speculative.pop(key)
            if not task.done():
                task.cancel()

    def _remember_source_context(self, meeting_id: str, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        contexts = getattr(self, "_recent_source_contexts", None)
        if contexts is None:
            contexts = {}
            self._recent_source_contexts = contexts
        window = contexts.get(meeting_id)
        if not isinstance(window, deque):
            window = deque(window or (), maxlen=self._CONTEXT_SEGMENTS)
            contexts[meeting_id] = window
        window.append(cleaned)

    async def _cleanup(self) -> None:
        listener = getattr(self, "_speculative_listener_task", None)
        if listener is not None:
            listener.cancel()
            try:
                await listener
            except asyncio.CancelledError:
                pass
        for task, _created_at in getattr(self, "_speculative_translations", {}).values():
            if not task.done():
                task.cancel()
        if self.translator is not None:
            await self.translator.close()

    def _require_translator(self) -> OpenAITranslator:
        if self.translator is None:
            raise RuntimeError("Translation model is not loaded")
        return self.translator
