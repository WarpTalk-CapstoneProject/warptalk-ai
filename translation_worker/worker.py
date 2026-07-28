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
import time

from shared.base_worker import BaseWorker
from shared.config import TranslationSettings, resolve_openai_api_key
from shared.openai_usage import TokenUsage
from shared.schemas import AIUsageMessage, STTResultMessage, TranslationResultMessage
from shared.text_utils import split_into_sentences
from translation_worker.translator import (
    OpenAITranslator,
    TranslationBatchWithUsage,
    TranslationWithUsage,
)

TRANSLATION_CHARGE_TYPE = "TRANSLATION"
TRANSLATION_BATCH_SIZE = 8


class TranslationWorker(BaseWorker):
    """Translation worker using OpenAI gpt-4.1-mini."""

    worker_name = "translation"
    input_stream = "stt:results"
    consumer_group = "translate-workers"

    # Bounds concurrent process() dispatch in _consume_loop — see its docstring.
    _CONCURRENCY_LIMIT = 8

    def __init__(
        self,
        translation_settings: TranslationSettings | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.translation_settings = translation_settings or TranslationSettings()
        self.translator: OpenAITranslator | None = None
        # meeting_id -> this workspace's glossary as [{"source": ..., "target": ...}, ...],
        # published to `translationRoom:{meeting_id}:mt_glossary` by
        # GlossaryStartedEventConsumer (TranscriptService) when the room starts. Cached for
        # the room's lifetime — same "don't hit Redis on every chunk" reasoning as
        # stt_worker's _stt_prompts (see docs/code-switching-research.md).
        self._mt_glossaries: dict[str, list[dict]] = {}

    async def load_model(self) -> None:
        """Initialize OpenAI translation client."""
        self.translator = OpenAITranslator(
            api_key=resolve_openai_api_key(self.translation_settings.api_key),
            model=self.translation_settings.model,
            max_tokens=self.translation_settings.max_tokens,
            temperature=self.translation_settings.temperature,
        )
        await self.translator.load()

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

        Trade-off: RedisStreamClient.consume() acks a message right after yielding it,
        not after process() returns, so a crash mid-flight loses whatever's currently
        dispatched instead of it being redelivered. Accepted here — an occasional lost
        translation chunk is a smaller cost than re-serializing every utterance.
        """
        self.logger.info(
            "consume_loop_started",
            stream=self.input_stream,
            group=self.consumer_group,
            consumer=self._consumer_name,
        )
        semaphore = asyncio.Semaphore(self._CONCURRENCY_LIMIT)

        async def _run(message_id: bytes, data: dict[bytes, bytes]) -> None:
            async with semaphore:
                try:
                    await self.process(message_id, data)
                except Exception:
                    self.logger.exception(
                        "process_error", message_id=message_id, stream=self.input_stream
                    )

        while not self._shutdown_event.is_set():
            try:
                async for message_id, data in self.redis.consume(
                    stream=self.input_stream,
                    group=self.consumer_group,
                    consumer=self._consumer_name,
                    block_ms=2000,
                ):
                    if self._shutdown_event.is_set():
                        break
                    asyncio.create_task(_run(message_id, data))
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
                        translator_model=self.translator.model,
                        source_segment_id=stt_result.segment_id,
                        chunk_index=0,
                    )
                    await self.publish(
                        "translate:results",
                        stt_result.meeting_id,
                        result.to_redis(),
                    )
            return

        await asyncio.gather(*(
            self._translate_and_publish(stt_result, sentences, target_lang, glossary_terms)
            for target_lang in target_langs
        ))

    async def _translate_and_publish(
        self,
        stt_result: STTResultMessage,
        sentences: list[str],
        target_lang: str,
        glossary_terms: list[dict] | None = None,
    ) -> None:
        # Single-sentence chunks keep the low-latency path; multi-sentence chunks are
        # grouped into bounded batches so prompt overhead is shared across sentences.
        passthrough = stt_result.language == target_lang
        first_task: asyncio.Task[TranslationWithUsage] | None = None
        batch_tasks: dict[int, asyncio.Task[TranslationBatchWithUsage]] = {}
        batch_results: dict[int, TranslationBatchWithUsage] = {}
        if not passthrough:
            if len(sentences) == 1:
                first_task = asyncio.create_task(
                    self._translate_sentence(
                        sentences[0],
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                        glossary_terms=glossary_terms,
                    )
                )
            else:
                for start in range(0, len(sentences), TRANSLATION_BATCH_SIZE):
                    batch = sentences[start : start + TRANSLATION_BATCH_SIZE]
                    batch_tasks[start] = asyncio.create_task(
                        self._translate_batch(
                            batch,
                            source_lang=stt_result.language,
                            target_lang=target_lang,
                            glossary_terms=glossary_terms,
                        )
                    )

        for idx, sentence in enumerate(sentences):
            # Sequence the segment ID (per target language, so different listeners'
            # translations of the same STT segment don't collide) so the frontend gets
            # consecutive speech segments.
            chunk_segment_id = f"{stt_result.segment_id}-{target_lang}-c{idx}"

            if passthrough:
                translated_text = sentence
            elif len(sentences) == 1:
                first_result = await first_task
                translated_text = first_result.text
                await self._publish_ai_usage(
                    stt_result,
                    target_lang,
                    first_result.usage,
                    idempotency_key=f"{TRANSLATION_CHARGE_TYPE}:{chunk_segment_id}:usage",
                )
            else:
                batch_start = (idx // TRANSLATION_BATCH_SIZE) * TRANSLATION_BATCH_SIZE
                if batch_start not in batch_results:
                    batch_results[batch_start] = await batch_tasks[batch_start]
                    current_batch = sentences[
                        batch_start : batch_start + TRANSLATION_BATCH_SIZE
                    ]
                    await self._publish_ai_usage(
                        stt_result,
                        target_lang,
                        batch_results[batch_start].usage,
                        idempotency_key=(
                            f"{TRANSLATION_CHARGE_TYPE}:{stt_result.segment_id}:{target_lang}:"
                            f"batch:{batch_start}:{len(current_batch)}"
                        ),
                    )
                translated_text = batch_results[batch_start].texts[idx - batch_start]

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
                translator_model=self.translator.model,
                source_segment_id=stt_result.segment_id,
                chunk_index=idx,
            )

            # Publish IMMEDIATELY so TTS can synthesize while next chunk is translated
            await self.publish("translate:results", stt_result.meeting_id, result.to_redis())

            self.logger.info(
                "chunk_translated",
                meeting_id=stt_result.meeting_id,
                chunk_index=idx,
                source_lang=stt_result.language,
                target_lang=target_lang,
                original=sentence[:60],
                translated=translated_text[:60],
            )

    async def _translate_sentence(
        self,
        text: str,
        *,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None,
    ) -> TranslationWithUsage:
        if "translate_with_usage" in type(self.translator).__dict__:
            return await self.translator.translate_with_usage(
                text,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_terms=glossary_terms,
            )

        translated = await self.translator.translate(
            text,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_terms,
        )
        return TranslationWithUsage(translated)

    async def _translate_batch(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
        glossary_terms: list[dict] | None,
    ) -> TranslationBatchWithUsage:
        if "translate_batch_with_usage" in type(self.translator).__dict__:
            return await self.translator.translate_batch_with_usage(
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                glossary_terms=glossary_terms,
            )

        translated = await self.translator.translate_batch(
            texts,
            source_lang=source_lang,
            target_lang=target_lang,
            glossary_terms=glossary_terms,
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
        return targets or {"en"}

    async def _get_mt_glossary(self, meeting_id: str) -> list[dict]:
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
        glossary: list[dict] = []
        if raw:
            try:
                parsed = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                if isinstance(parsed, list):
                    glossary = [t for t in parsed if isinstance(t, dict)]
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.logger.warning("mt_glossary_parse_failed", meeting_id=meeting_id)

        self._mt_glossaries[meeting_id] = glossary
        if glossary:
            self.logger.info("mt_glossary_loaded", meeting_id=meeting_id, terms=len(glossary))
        return glossary

    def _cleanup_room(self, room_id: str) -> None:
        super()._cleanup_room(room_id)
        self._mt_glossaries.pop(room_id, None)
