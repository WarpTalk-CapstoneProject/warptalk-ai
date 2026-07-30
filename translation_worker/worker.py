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
from shared.openai_usage import TokenUsage
from shared.schemas import AIUsageMessage, STTResultMessage, TranslationResultMessage
from shared.text_utils import split_into_sentences
from translation_worker.translator import (
    OUT_OF_MEETING_SCOPE,
    OpenAITranslator,
    TranslationBatchWithUsage,
    TranslationWithUsage,
)

TRANSLATION_CHARGE_TYPE = "TRANSLATION"
TRANSLATION_BATCH_SIZE = 8

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
    normalized = " ".join(
        re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    )
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
            tuple[asyncio.Task[str], float],
        ] = {}
        self._speculative_semaphore = asyncio.Semaphore(1)
        self._speculative_listener_task: asyncio.Task[None] | None = None

    async def load_model(self) -> None:
        """Initialize OpenAI translation client."""
        self.translator = OpenAITranslator(
            api_key=resolve_openai_api_key(self.translation_settings.api_key),
            model=self.translation_settings.model,
            realtime_model=self.translation_settings.realtime_model,
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

        target_langs = await self._get_target_languages(meeting_id, speaker_id)
        glossary_terms = await self._get_mt_glossary(meeting_id)
        recent_context = list(
            getattr(self, "_recent_source_contexts", {}).get(meeting_id, ())
        )
        static_context = await self._get_meeting_context(meeting_id)
        meeting_context = recent_context[-3:] + static_context
        translator = self._require_translator()
        cache = getattr(self, "_speculative_translations", None)
        if cache is None:
            cache = {}
            self._speculative_translations = cache

        tasks: list[asyncio.Task[str]] = []
        now = time.monotonic()
        semaphore = getattr(self, "_speculative_semaphore", None)
        if semaphore is None:
            semaphore = asyncio.Semaphore(1)
            self._speculative_semaphore = semaphore
        for target_lang in target_langs:
            if source_lang.split("-", 1)[0] == target_lang.split("-", 1)[0]:
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
                ) -> str:
                    async with semaphore:
                        try:
                            async with asyncio.timeout(self._SPECULATIVE_TIMEOUT_SECONDS):
                                return await translator.translate(
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
                            return ""

                task = asyncio.create_task(
                    run_speculative()
                )
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
    ) -> asyncio.Task[str] | None:
        cache = cast(
            dict[
                tuple[str, str, str, str],
                tuple[asyncio.Task[str], float],
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

        if stt_result.meeting_id in self._paused_rooms:
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - stt_result.timestamp_ms
        await self.redis.publish_telemetry(stt_result.meeting_id, self.worker_name, e2e_latency_ms)

        target_langs = await self._get_target_languages(
            stt_result.meeting_id, stt_result.speaker_id
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
                    )
                    await self.publish(
                        "translate:results",
                        stt_result.meeting_id,
                        result.to_redis(),

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
        passthrough = stt_result.language == target_lang
        first_task: asyncio.Task[str] | asyncio.Task[TranslationWithUsage] | None = None
        rest_task: asyncio.Task[TranslationBatchWithUsage] | None = None
        rest_results: TranslationBatchWithUsage | None = None
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
                    self._translate_sentence(
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
                    self._translate_batch(
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
                translated_text = sentence
            elif idx == 0:
                assert first_task is not None
                first_result = await first_task
                if isinstance(first_result, TranslationWithUsage):
                    translated_text = first_result.text
                    await self._publish_ai_usage(
                        stt_result,
                        target_lang,
                        first_result.usage,
                        idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{chunk_segment_id}:usage",
                    )
                else:
                    translated_text = first_result
                if not translated_text:
                    fallback_result = await self._translate_sentence(
                        sentence,
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                        meeting_context=first_context,
                    )
                    translated_text = fallback_result.text
                    await self._publish_ai_usage(
                        stt_result,
                        target_lang,
                        fallback_result.usage,
                        idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{chunk_segment_id}:fallback",
                    )
            else:
                if rest_results is None:
                    assert rest_task is not None
                    rest_results = await rest_task
                    await self._publish_ai_usage(
                        stt_result,
                        target_lang,
                        rest_results.usage,
                        idempotency_key=(
                            f"{TRANSLATION_CHARGE_TYPE}:{stt_result.segment_id}:{target_lang}:"
                            f"batch:1:{len(sentences) - 1}"
                        ),
                    )
                translated_text = rest_results.texts[idx - 1]
                if not translated_text:
                    translated_text = ""

            if translated_text.strip().upper() == OUT_OF_MEETING_SCOPE:
                self.logger.info(
                    "background_utterance_suppressed",
                    meeting_id=stt_result.meeting_id,
                    source_lang=stt_result.language,
                    target_lang=target_lang,
                    original=sentence[:60],
                )
                continue

            is_final = (idx == len(sentences) - 1) and stt_result.is_final_chunk

            result = TranslationResultMessage(
                segment_id=chunk_segment_id,
                meeting_id=stt_result.meeting_id,
                speaker_id=stt_result.speaker_id,
                original_text=sentence,
                translated_text=translated_text,
                source_lang=stt_result.language,
                target_lang=target_lang,
                confidence=stt_result.confidence,
                start_ms=stt_result.start_ms,
                end_ms=stt_result.end_ms,
                is_final_chunk=is_final,
                timestamp_ms=stt_result.timestamp_ms,
                translator_model=translator.model,
                source_segment_id=stt_result.segment_id,
                chunk_index=stt_result.chunk_index,
            )

            # Publish IMMEDIATELY so TTS can synthesize while next chunk is translated
            await self.publish("translate:results", stt_result.meeting_id, result.to_redis())
            published_any = True

            self.logger.info(
                "chunk_translated",
                meeting_id=stt_result.meeting_id,
                chunk_index=idx,
                source_lang=stt_result.language,
                target_lang=target_lang,
                original=sentence[:60],
                translated=translated_text[:60],
                speculative_hit=speculative_hit if idx == 0 else False,
                stage_latency_ms=int((time.monotonic() - translation_started) * 1000),
                pipeline_latency_ms=max(0, int(time.time() * 1000) - stt_result.timestamp_ms),
            )
        return published_any

    async def _translate_sentence(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None,
        meeting_context: list[str] | None = None,
    ) -> TranslationWithUsage:
        if "translate_with_usage" in type(self.translator).__dict__:
            return await self.translator.translate_with_usage(
                text,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_terms=glossary_terms,
                meeting_context=meeting_context,
            )

        translated = await self.translator.translate(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_terms,
            meeting_context=meeting_context,
        )
        return TranslationWithUsage(translated)

    async def _translate_batch(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None,
        meeting_context: list[str] | None = None,
    ) -> TranslationBatchWithUsage:
        if "translate_batch_with_usage" in type(self.translator).__dict__:
            return await self.translator.translate_batch_with_usage(
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_terms=glossary_terms,
                meeting_context=meeting_context,
            )

        translated = await self.translator.translate_batch(
            texts,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_terms,
            meeting_context=meeting_context,
        )
        return TranslationBatchWithUsage(translated)

    async def _publish_ai_usage(
        self,
        stt_result: STTResultMessage,
        target_lang: str,
        usage: TokenUsage,
        *,
        idempotency_key: str,
    ) -> None:
        if not usage.has_tokens:
            return

        message = AIUsageMessage(
            room_id=stt_result.meeting_id,
            user_id=stt_result.speaker_id,
            charge_type=TRANSLATION_CHARGE_TYPE,
            model=self.translator.model,
            prompt_tokens=usage.prompt_tokens,
            cached_tokens=usage.cached_tokens,
            completion_tokens=usage.completion_tokens,
            source_lang=stt_result.language,
            target_lang=target_lang,
            idempotency_key=idempotency_key,
        )
        await self.publish("ai:usage", stt_result.meeting_id, message.to_redis())

    async def _get_target_languages(self, meeting_id: str, speaker_id: str) -> set[str]:
        """Every DISTINCT listen-language among the OTHER participants in this meeting.

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
        room_target_languages = await self._get_room_target_languages(meeting_id)
        if room_target_languages:
            targets.update(room_target_languages)

        # No participant or room target language registered yet. Keep English as the
        # conservative compatibility fallback instead of assuming one locale globally.
        return targets or {"en"}

    async def _get_room_target_languages(self, meeting_id: str) -> set[str]:
        """Room-level target languages configured at room creation time."""
        try:
            raw = await self.redis.get(f"meeting:{meeting_id}:target_languages")
        except Exception:
            self.logger.warning(
                "failed_to_read_room_target_languages",
                meeting_id=meeting_id,
                exc_info=True,
            )
            return set()

        if not raw:
            return set()

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            self.logger.warning(
                "invalid_room_target_languages",
                meeting_id=meeting_id,
            )
            return set()

        if not isinstance(parsed, list):
            return set()

        return {str(lang).strip() for lang in parsed if str(lang).strip()}

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
