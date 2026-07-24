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
                    await self.publish("translate:results", stt_result.meeting_id, result.to_redis())
            return

        await asyncio.gather(*(
            self._translate_and_publish(stt_result, sentences, target_lang)
            for target_lang in target_langs
        ))

    async def _translate_and_publish(
        self, stt_result: STTResultMessage, sentences: list[str], target_lang: str
    ) -> None:
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
            # Sequence the segment ID (per target language, so different listeners'
            # translations of the same STT segment don't collide) so the frontend gets
            # consecutive speech segments.
            chunk_segment_id = f"{stt_result.segment_id}-{target_lang}-c{idx}"

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
