"""Translation Worker — Consumes STT results, produces translated text.

Pipeline:
    Redis Stream (stt:results:{meetingId})
    → OpenAI gpt-4.1-mini
    → Redis Stream (translate:results:{meetingId})

Passthrough: if source_lang == target_lang, forward without translation.
"""

from __future__ import annotations

import asyncio
import time

from shared.base_worker import BaseWorker
from shared.config import TranslationSettings, resolve_openai_api_key
from shared.schemas import STTResultMessage, TranslationResultMessage
from shared.text_utils import split_into_sentences
from translation_worker.translator import OpenAITranslator


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
        """Translate one STT result segment by chunking into sentences."""
        stt_result = STTResultMessage.from_redis(data)

        if stt_result.meeting_id in self._paused_rooms:
            return

        current_timestamp_ms = int(time.time() * 1000)
        e2e_latency_ms = current_timestamp_ms - stt_result.timestamp_ms
        await self.redis.publish_telemetry(stt_result.meeting_id, self.worker_name, e2e_latency_ms)

        # Get target language for this meeting/speaker
        target_lang = await self._get_target_language(
            stt_result.meeting_id, stt_result.speaker_id
        )

        # Split long STT results into smaller sentences (streaming mechanism)
        sentences = split_into_sentences(stt_result.text)

        if not sentences:
            if stt_result.is_final_chunk:
                result = TranslationResultMessage(
                    segment_id=stt_result.segment_id,
                    meeting_id=stt_result.meeting_id,
                    speaker_id=stt_result.speaker_id,
                    original_text="",
                    translated_text="",
                    source_lang=stt_result.language,
                    target_lang=target_lang,
                    is_final_chunk=True,
                    timestamp_ms=stt_result.timestamp_ms,
                    translator_model=self.translator.model,
                )
                await self.publish("translate:results", stt_result.meeting_id, result.to_redis())
            return

        # Fire off translation for all sentences UP FRONT so they run concurrently instead
        # of one-sentence-at-a-time (each translate() call is a real OpenAI network
        # round-trip — awaiting them sequentially in the publish loop below used to add a
        # full round-trip of latency per extra sentence). Sentence 0 still gets its own
        # single-sentence call (not folded into the batch) so it can be awaited and
        # published on its own as soon as it's ready — TTS starts on it immediately,
        # same as before — while sentences 1..N-1 translate together in ONE batched call
        # that's already running in the background by the time we get to them.
        passthrough = stt_result.language == target_lang
        first_task: asyncio.Task[str] | None = None
        rest_task: asyncio.Task[list[str]] | None = None
        rest_results: list[str] | None = None
        if not passthrough:
            first_task = asyncio.create_task(
                self.translator.translate(
                    sentences[0], source_lang=stt_result.language, target_lang=target_lang
                )
            )
            if len(sentences) > 1:
                rest_task = asyncio.create_task(
                    self.translator.translate_batch(
                        sentences[1:],
                        source_lang=stt_result.language,
                        target_lang=target_lang,
                    )
                )

        for idx, sentence in enumerate(sentences):
            # Sequence the segment ID so frontend gets consecutive speech segments
            chunk_segment_id = f"{stt_result.segment_id}-c{idx}"

            if passthrough:
                translated_text = sentence
            elif idx == 0:
                translated_text = await first_task
            else:
                if rest_results is None:
                    rest_results = await rest_task
                translated_text = rest_results[idx - 1]

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

    async def _get_target_language(self, meeting_id: str, speaker_id: str) -> str:
        """Get the target translation language for a speaker's audience.

        Reads from a Redis hash set by the backend when a user joins and selects their
        preferred output language: `translationRoom:{translationRoomId}:languages`,
        keyed by userId -> that user's own ListenLanguage (TranslationRoomHub.
        JoinTranslationRoom).

        NOTE: `meeting_id` here is actually the translation_room_id (see
        AudioChunkMessage.from_redis / RedisStreamService.PublishAudioChunkAsync).

        Previously this looked up `hget(hash, speaker_id)` — i.e. it read the SPEAKER's
        own listen-language entry, not any listener's. That is wrong: the host's own
        ListenLanguage is hardcoded to the room's source language at creation
        (TranslationRoomService.CreateTranslationRoomAsync sets
        hostParticipant.ListenLanguage = sourceLang), so when the host spoke, this
        always resolved target_lang == source_lang, short-circuited
        OpenAITranslator.translate()'s same-language passthrough, and silently
        produced zero real translations for the single most common case (host speaks,
        guest listens). Fixed to look at every OTHER participant's entry instead.

        Still a simplification: with >1 listener configured for different languages,
        this returns only the first one found (single target_lang per
        TranslationResultMessage) — true multi-listener fan-out (one message per
        distinct target language) is a separate, larger change, not attempted here.
        """
        all_languages = await self.redis.hgetall(f"translationRoom:{meeting_id}:languages")
        for raw_user_id, raw_lang in all_languages.items():
            user_id = raw_user_id.decode() if isinstance(raw_user_id, bytes) else raw_user_id
            if user_id == speaker_id:
                continue
            return raw_lang.decode() if isinstance(raw_lang, bytes) else raw_lang

        # No other participant registered yet — avoid assuming Vietnamese for all users.
        return "en"
