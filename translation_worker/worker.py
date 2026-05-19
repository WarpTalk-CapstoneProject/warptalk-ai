"""Translation Worker — Consumes STT results, produces translated text.

Pipeline:
    Redis Stream (stt:results:{meetingId})
    → NLLB / Google Translate
    → Redis Stream (translate:results:{meetingId})

Passthrough: if source_lang == target_lang, forward without translation.
"""

from __future__ import annotations

from shared.base_worker import BaseWorker
from shared.config import TranslationSettings
from shared.schemas import STTResultMessage, TranslationResultMessage
from shared.text_utils import split_into_sentences

from translation_worker.translator import (
    GoogleTranslator,
    NLLBTranslator,
    TranslatorWithFallback,
)


class TranslationWorker(BaseWorker):
    """Translation worker using NLLB-200 with Google Translate fallback."""

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
        self.translator: TranslatorWithFallback | None = None

    async def load_model(self) -> None:
        """Load translation model with optional fallback."""
        primary = NLLBTranslator(
            model_name=self.translation_settings.model,
            device=self.translation_settings.device,
            max_length=self.translation_settings.max_length,
        )

        fallback = None
        if self.translation_settings.fallback_provider == "google":
            fallback = GoogleTranslator()

        self.translator = TranslatorWithFallback(primary, fallback)
        await self.translator.load()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        """Translate one STT result segment by chunking into sentences."""
        stt_result = STTResultMessage.from_redis(data)

        # Get target language for this meeting/speaker
        target_lang = await self._get_target_language(
            stt_result.meeting_id, stt_result.speaker_id
        )

        # Split long STT results into smaller sentences (streaming mechanism)
        sentences = split_into_sentences(stt_result.text)
        
        if not sentences:
            return

        for idx, sentence in enumerate(sentences):
            # Passthrough if same language
            if stt_result.language == target_lang:
                translated_text = sentence
            else:
                # Translates quickly because NLLB handles partial sentences ~5-15 words
                translated_text = await self.translator.translate(
                    sentence,
                    source_lang=stt_result.language,
                    target_lang=target_lang,
                )

            result = TranslationResultMessage(
                segment_id=stt_result.segment_id,
                meeting_id=stt_result.meeting_id,
                speaker_id=stt_result.speaker_id,
                original_text=sentence,
                translated_text=translated_text,
                source_lang=stt_result.language,
                target_lang=target_lang,
                confidence=stt_result.confidence,
                start_ms=stt_result.start_ms,
                end_ms=stt_result.end_ms,
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
        """
        cached = await self.redis.hget(
            f"meeting:{meeting_id}:languages", speaker_id
        )
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached

        # Default fallback
        return "vi"
