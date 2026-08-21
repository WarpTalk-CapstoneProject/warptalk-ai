"""Translate the parts of a saved transcript the meeting never covered.

TranslationWorker is a live stage: it reads `stt:results`, refuses a room whose translation is
not started, and takes its target languages from the presence hash the gateway deletes as people
leave. Every one of those is correct while a meeting is running and useless afterwards — which is
why a finished transcript can only be read back in the language that happened to be selected at
the time, and a meeting that switched from English to Japanese half way through has neither
language covering the whole of it.

This worker is the after-the-fact half. TranscriptService works out which lines need translating
and publishes them here; this translates them and publishes ordinary translation results, which
TranscriptService persists through the same code path a live translation takes.

Two things send work here, and they differ by one field per line:

* a language the meeting never covered — the line has no translation in it at all;
* a line whose transcript was corrected — it HAS one, of a sentence nobody said, and the request
  names the translation_contents row being replaced so the consumer can record the chain.

Results go to `translate:backfill_results`, NOT `translate:results`: tts_worker consumes the
latter, and a backfill landing there would synthesise — and bill — speech for every line of a
meeting that ended hours ago.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from shared.base_worker import BaseWorker
from shared.config import TranslationSettings, WorkerSettings, resolve_openai_api_key
from shared.logger import setup_logging
from shared.schemas import TranslationResultMessage
from translation_worker.translator import OpenAITranslator

#: Where TranscriptService.TranscriptTranslationBackfillService publishes its work list.
BACKFILL_REQUEST_STREAM = "translate:backfill_requests"

#: Where the finished translations go. Read by TranscriptService's persistence consumer only.
BACKFILL_RESULT_STREAM = "translate:backfill_results"

#: How long a "failed" marker stays readable. Matches the run marker's TTL on the .NET side —
#: it is a hint for the reader's UI, never a lock anything depends on.
STATUS_TTL_SECONDS = 20 * 60


class TranslationBackfillWorker(BaseWorker):
    """Fills in a saved transcript's missing translations, one request batch at a time."""

    worker_name = "translation-backfill"
    input_stream = BACKFILL_REQUEST_STREAM
    consumer_group = "translation-backfill-workers"

    def __init__(
        self,
        translation_settings: TranslationSettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.translation_settings = translation_settings or TranslationSettings()
        self.translator: OpenAITranslator | None = None

    async def load_model(self) -> None:
        self.translator = OpenAITranslator(
            api_key=resolve_openai_api_key(self.translation_settings.api_key),
            model=self.translation_settings.model,
            max_tokens=self.translation_settings.max_tokens,
            temperature=self.translation_settings.temperature,
        )
        await self.translator.load()

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        fields = {
            key.decode() if isinstance(key, bytes) else key: value.decode()
            if isinstance(value, bytes)
            else value
            for key, value in data.items()
        }

        meeting_id = fields.get("meeting_id", "")
        target_lang = _bare(fields.get("target_lang", ""))
        status_key = fields.get("status_key", "")
        transcript_id = fields.get("transcript_id", "")

        try:
            segments = json.loads(fields.get("segments_json", "[]"))
        except json.JSONDecodeError:
            self.logger.error(
                "backfill_request_unreadable",
                transcript_id=transcript_id,
                message_id=message_id,
            )
            # Malformed input will never parse on a retry, so mark the run and let the message
            # be acknowledged rather than dead-lettering the same bytes five times.
            await self._mark_failed(status_key)
            return

        if not target_lang or not segments:
            return

        if self.translator is None:  # pragma: no cover - load_model always runs first
            raise RuntimeError("Backfill worker used before its translator was loaded")

        glossary = await self._glossary_for(meeting_id)

        # One request can carry lines from several speakers in several languages — a meeting
        # where a Vietnamese and a Japanese speaker alternate produces exactly that. translate_batch
        # takes a single source language, so the batch is split by the language it was spoken in
        # rather than being handed to the model as if it were all one.
        by_source: dict[str, list[dict[str, str]]] = {}
        for segment in segments:
            source_lang = _bare(str(segment.get("source_lang", "")))
            text = str(segment.get("text", "")).strip()
            segment_id = str(segment.get("segment_id", ""))
            if not text or not segment_id:
                continue
            if source_lang == target_lang:
                # Already in the requested language; TranscriptService does not count these as
                # missing, so reaching here means the two sides disagree about a language tag.
                continue
            by_source.setdefault(source_lang, []).append(
                {
                    "segment_id": segment_id,
                    "text": text,
                    # Present when this line already HAS a translation in the target language and
                    # is being redone — a human corrected what was said, so the stored translation
                    # is of a sentence nobody spoke. Absent for an ordinary gap fill.
                    "previous_translation_content_id": str(
                        segment.get("previous_translation_content_id") or ""
                    ),
                }
            )

        started = time.monotonic()
        translated_count = 0

        try:
            for source_lang, group in by_source.items():
                texts = [item["text"] for item in group]
                # No meeting_context: it turns on the out-of-scope suppression that lets the live
                # worker drop background speech. Every line here is already in the record and the
                # reader asked to see all of it, so nothing may be suppressed.
                translations = await self.translator.translate_batch(
                    texts,
                    source_lang,
                    target_lang,
                    glossary_terms=glossary or None,
                )

                for item, translated in zip(group, translations, strict=False):
                    if not translated or not translated.strip():
                        continue

                    result = TranslationResultMessage(
                        # The bare segment id, so the persistence consumer's
                        # ExtractUnderlyingSegmentId recovers it without stripping a suffix.
                        segment_id=item["segment_id"],
                        meeting_id=meeting_id,
                        speaker_id="",
                        original_text=item["text"],
                        translated_text=translated.strip(),
                        source_lang=source_lang,
                        target_lang=target_lang,
                        translator_model=self.translator.model,
                        source_segment_id=item["segment_id"],
                        is_final_chunk=True,
                        # Carried through rather than decided here: only the producer knows
                        # whether this line already had a translation, and the consumer needs
                        # both facts to write the correction chain that
                        # transcript.translation_contents models and has never recorded.
                        is_retranslated=bool(item["previous_translation_content_id"]),
                        previous_translation_content_id=(
                            item["previous_translation_content_id"] or None
                        ),
                        # No latency_ms on purpose. One API call produced N sentences, so no
                        # sentence has a duration of its own; the schema treats an absent field
                        # as "not measured" and the column stores NULL.
                    )
                    await self.publish(BACKFILL_RESULT_STREAM, meeting_id, result.to_redis())
                    translated_count += 1
        except Exception:
            await self._mark_failed(status_key)
            raise

        self.logger.info(
            "backfill_batch_translated",
            transcript_id=transcript_id,
            target_lang=target_lang,
            segments=translated_count,
            source_languages=sorted(by_source),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    async def _glossary_for(self, meeting_id: str) -> list[dict[str, str]]:
        """The room's glossary, if it outlived the meeting.

        GlossaryStartedEventConsumer writes it when a room starts and nothing renews it, so a
        transcript read back days later usually finds nothing. Best-effort by design: a backfill
        without the glossary is worth far more than no backfill.
        """
        if not meeting_id:
            return []
        try:
            raw = await self.redis.get(f"translationRoom:{meeting_id}:mt_glossary")
        except Exception:  # pragma: no cover - Redis errors must not fail the translation
            return []
        if not raw:
            return []
        try:
            terms = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            return []
        return terms if isinstance(terms, list) else []

    async def _mark_failed(self, status_key: str) -> None:
        if not status_key:
            return
        try:
            await self.redis.set_with_ttl(status_key, "failed", STATUS_TTL_SECONDS)
        except Exception:  # pragma: no cover - the marker is a hint, not state anything reads back
            self.logger.warning("backfill_status_not_written", status_key=status_key)


def _bare(language: str) -> str:
    """ "vi-VN" and "vi" are the same language; the transcript stores the second form."""
    return language.split("-")[0].split("_")[0].strip().lower()


async def main() -> None:
    """Entry point — run with `python -m translation_worker.backfill_worker`.

    Shares the ai-translation image with TranslationWorker rather than getting one of its own:
    the two need the same dependencies and the same translator, and the only thing that differs
    is which stream they read.
    """
    worker_settings = WorkerSettings()
    setup_logging(worker_settings.log_level)

    worker = TranslationBackfillWorker(
        translation_settings=TranslationSettings(),
        settings=worker_settings,
    )
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
