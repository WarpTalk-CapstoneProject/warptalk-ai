"""Transcribing a whole file, with all the time in the world.

WHY THIS IS NOT stt_worker.OpenAISTT
    That class is a realtime websocket session pool: it holds warm sockets, streams frames as they
    are spoken, and answers inside a two-second budget because a dub is waiting. Every design
    decision in it is about latency, and the config's own probe table records what that costs —
    `gpt-realtime-2.1-mini` at effort=minimal repaired 0 of 3 hard cases where the full model
    repaired 3.

    A second pass has no dub waiting. It sends the finished file to the transcriptions endpoint,
    which sees the whole meeting at once, and it can afford the model that gets it right. Same
    audio, different constraints, different call.

WHY CONTEXT IS PASSED AS A PROMPT
    The endpoint accepts a `prompt` that biases decoding towards names and terms it would otherwise
    guess at phonetically. A meeting already knows its own: the workspace glossary that
    GlossaryStartedEventConsumer feeds the live path, and the participants' own names. Handing them
    over is the cheapest accuracy this pass can buy — and the reason `stt_keywords` exists on the
    realtime side too.

    Bounded on purpose. `stt_worker/model.py` documents production copying an entire
    comma-separated keyword list into a transcript on marginal audio, and the filter it needed
    afterwards. A prompt is a suggestion to a model, not an instruction, and a long one is a longer
    thing to recite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from retranscribe_worker.merge import SpeakerSegment
from shared.logger import get_logger

logger = get_logger(__name__)

#: How much meeting vocabulary may go into the decoding prompt. Roughly the endpoint's own
#: documented ceiling, and far short of the length at which a model starts reciting it.
MAX_PROMPT_CHARS = 800


def build_prompt(terms: list[str], speakers: list[str] | None = None) -> str:
    """The meeting's own vocabulary, as a decoding hint.

    Names first: a mis-heard proper noun is the error people notice and the one a glossary is least
    likely to already contain. Truncated whole-term rather than mid-word, because half a name is a
    worse hint than no name.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for term in [*(speakers or []), *terms]:
        cleaned = term.strip()
        if not cleaned or cleaned.casefold() in seen:
            continue
        seen.add(cleaned.casefold())
        ordered.append(cleaned)

    prompt = ""
    for term in ordered:
        candidate = f"{prompt}, {term}" if prompt else term
        if len(candidate) > MAX_PROMPT_CHARS:
            break
        prompt = candidate
    return prompt


class BatchTranscriber:
    """One call per audio file, returning segments on the file's own clock."""

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def transcribe(
        self,
        path: Path,
        language: str | None = None,
        prompt: str = "",
    ) -> list[SpeakerSegment]:
        """Segments for one speaker's file, timed from the start of that file.

        `language` is passed when known rather than left to detection: these files are one speaker
        each and the room already recorded what they speak, so detection can only introduce an
        error the caller had the answer to. Passing None is a deliberate "we do not know".
        """
        options: dict[str, Any] = {"model": self._model}
        if language:
            options["language"] = language
        if prompt:
            options["prompt"] = prompt

        # NOT every transcription model returns timestamps. The accuracy-first family
        # (`gpt-transcribe`) rejects verbose_json outright — "Use \'json\' or \'text\' instead" —
        # while whisper-1 supports it. Asking for the richer format and accepting the refusal is
        # how one call site serves both, rather than a hardcoded table of model capabilities that
        # goes stale the next time a model ships.
        try:
            with path.open("rb") as handle:
                response = await self._client.audio.transcriptions.create(
                    file=handle,
                    response_format="verbose_json",
                    # Segment granularity, not word: the consumer places whole utterances on a
                    # meeting timeline, and word timings would be thousands of rows of precision
                    # nothing reads.
                    timestamp_granularities=["segment"],
                    **options,
                )
        except BadRequestError as error:
            if "verbose_json" not in str(error):
                raise
            # One untimed segment. Honest rather than convenient: a caller that needs to place
            # this on a meeting clock must notice it has no timings, instead of being handed
            # zeros that look like the utterance began at the start of the file.
            logger.info(
                "batch_transcription_model_has_no_timestamps",
                model=self._model,
            )
            with path.open("rb") as handle:
                response = await self._client.audio.transcriptions.create(
                    file=handle, response_format="json", **options
                )

        return _read_segments(response)

    async def close(self) -> None:
        await self._client.close()


def _read_segments(response: Any) -> list[SpeakerSegment]:
    """Segments out of a verbose_json response, whatever shape the SDK hands back.

    Tolerant because this is the one place a model-provider change lands, and the failure mode of
    being strict here is a meeting that silently re-transcribes to nothing.
    """
    raw = getattr(response, "segments", None)
    if raw is None and isinstance(response, dict):
        raw = response.get("segments")

    if not raw:
        # No segments but text is still an answer — one short utterance often comes back that way.
        text = getattr(response, "text", None) or (
            response.get("text") if isinstance(response, dict) else None
        )
        if not text or not str(text).strip():
            return []
        logger.info("batch_transcription_returned_text_without_segments")
        return [SpeakerSegment(start_ms=0, end_ms=0, text=str(text).strip())]

    segments: list[SpeakerSegment] = []
    for entry in raw:
        text = _field(entry, "text")
        if not text or not str(text).strip():
            continue

        start = _number(_field(entry, "start"))
        end = _number(_field(entry, "end"))
        segments.append(
            SpeakerSegment(
                start_ms=int(start * 1000),
                end_ms=int(end * 1000),
                text=str(text).strip(),
                # avg_logprob, the same unit TranscribedSegment.confidence carries — so the
                # second-pass rewrite guard compares like with like. Absent stays None: WT-277's
                # rule is that unknown confidence is never coalesced to a number.
                confidence=_optional_number(_field(entry, "avg_logprob")),
            )
        )
    return segments


def _field(entry: Any, name: str) -> Any:
    return entry.get(name) if isinstance(entry, dict) else getattr(entry, name, None)


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
