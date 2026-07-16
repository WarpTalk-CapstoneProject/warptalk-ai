"""Translation Worker — Consumes STT results, produces translated text.

Pipeline:
    Redis Stream (stt:results:{meetingId})
    → OpenAI gpt-4.1-mini
    → Redis Stream (translate:results:{meetingId})

Passthrough: if source_lang == target_lang, forward without translation.
"""

from __future__ import annotations

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
                )
                await self.publish("translate:results", stt_result.meeting_id, result.to_redis())
            return

        for idx, sentence in enumerate(sentences):
            # Sequence the segment ID so frontend gets consecutive speech segments
            chunk_segment_id = f"{stt_result.segment_id}-c{idx}"

            # Passthrough if same language
            if stt_result.language == target_lang:
                translated_text = sentence
            else:
                # gpt-4.1-mini: ~200-400ms per call, quality >> NLLB for all language pairs
                translated_text = await self.translator.translate(
                    sentence,
                    source_lang=stt_result.language,
                    target_lang=target_lang,
                )

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
        """Get the target translation language for a speaker.

        Reads from a Redis hash set by the backend when a user joins
        and selects their preferred output language.

        NOTE: `meeting_id` here is actually the translation_room_id (see
        AudioChunkMessage.from_redis / RedisStreamService.PublishAudioChunkAsync).
        The hash key MUST match TranslationRoomHub.JoinTranslationRoom, which writes
        to `translationRoom:{translationRoomId}:languages` — this used to read
        `meeting:{meeting_id}:languages` instead, a key nothing ever wrote to, so
        every listener's chosen language was silently ignored and this always fell
        through to the hardcoded "en" fallback below.
        """
        cached = await self.redis.hget(
            f"translationRoom:{meeting_id}:languages", speaker_id
        )
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached

        # Global fallback — avoid assuming Vietnamese for all users
        return "en"
