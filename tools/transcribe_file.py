"""Transcribe an audio file with the second pass, and optionally score it.

    python -m tools.transcribe_file --audio meeting.m4a --language vi
    python -m tools.transcribe_file --audio meeting.m4a --language vi --reference ref.txt

WHAT IT IS FOR
    Two things that were impossible before it. It is the front door of the accuracy benchmark —
    `shared/wer.py` has been a scale with nothing on it — and it is the only way to exercise the
    batch path at all without a full meeting, an egress and a Redis stream.

    Point it at a file, hand it what was actually said, and it prints the number the review asked
    for.

A WORD ABOUT THE REFERENCE
    Write it from the AUDIO, not from this tool's output. A reference typed while reading a machine
    transcript inherits that machine's mistakes, and the measurement then flatters itself in a way
    that looks rigorous. If you must correct rather than write from scratch, listen to the whole
    clip while you do it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from retranscribe_worker.batch_transcriber import BatchTranscriber, build_prompt
from shared.config import STTSettings, resolve_openai_api_key
from shared.wer import character_error_rate, word_error_rate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Audio file to transcribe.")
    parser.add_argument("--language", help="ISO code, e.g. vi or en. Omit to let the model detect.")
    parser.add_argument("--reference", help="Human transcript of the same audio, to score against.")
    parser.add_argument("--model", help="Override the transcription model.")
    parser.add_argument(
        "--terms",
        nargs="*",
        default=[],
        help="Meeting vocabulary — names, product terms — biased into decoding.",
    )
    parser.add_argument("--segments", action="store_true", help="Print each segment with its time.")
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"No such file: {audio}", file=sys.stderr)
        return 1

    settings = STTSettings()
    api_key = resolve_openai_api_key(settings.api_key)
    if not api_key:
        print("No OpenAI API key. Set OPENAI_API_KEY or STT_API_KEY.", file=sys.stderr)
        return 1

    model = args.model or settings.model
    transcriber = BatchTranscriber(api_key=api_key, model=model)

    try:
        segments = await transcriber.transcribe(
            audio,
            language=args.language,
            prompt=build_prompt(args.terms),
        )
    finally:
        await transcriber.close()

    text = " ".join(segment.text for segment in segments).strip()

    print(f"model    {model}")
    print(f"segments {len(segments)}")
    print()
    if args.segments:
        for segment in segments:
            stamp = f"{segment.start_ms // 60000:02d}:{segment.start_ms // 1000 % 60:02d}"
            confidence = "" if segment.confidence is None else f"  [{segment.confidence:.3f}]"
            print(f"  {stamp}{confidence}  {segment.text}")
        print()
    print(text)

    if not args.reference:
        return 0

    reference = Path(args.reference).read_text(encoding="utf-8")
    wer = word_error_rate(reference, text)
    cer = character_error_rate(reference, text)

    print()
    counts = f"S {wer.substitutions}  D {wer.deletions}  I {wer.insertions}"
    print(f"WER {wer.rate:.2%}   {counts}   over {wer.reference_length} reference words")
    print(f"CER {cer.rate:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
