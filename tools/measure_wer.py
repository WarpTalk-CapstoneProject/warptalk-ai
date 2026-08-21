"""Measure a transcript against a human reference, and compare two passes.

USAGE
    Establish the baseline the review is complaining about — this works today, before any second
    pass exists, and is the number to quote first:

        python -m tools.measure_wer --reference ref.txt --hypothesis pass1.txt

    Once a second pass produces a transcript for the same audio:

        python -m tools.measure_wer --reference ref.txt --hypothesis pass1.txt --second pass2.txt

    Add --alignment to print every substitution, deletion and insertion. A report that quotes a
    rate without ever having looked at the errors is a report nobody checked.

WHAT A REFERENCE IS
    A human transcription of the SAME audio, written without looking at the machine output.
    Written while reading it, it inherits the machine's mistakes and the measurement flatters
    itself — which is the one way to get this badly wrong, and it produces a number that looks
    rigorous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from shared.wer import Op, character_error_rate, compare, word_error_rate


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Human transcript of the audio.")
    parser.add_argument("--hypothesis", required=True, help="First-pass (realtime) transcript.")
    parser.add_argument("--second", help="Second-pass transcript of the same audio.")
    parser.add_argument("--alignment", action="store_true", help="Print every error.")
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    return parser.parse_args()


def _print_rate(label: str, reference: str, hypothesis: str) -> None:
    wer = word_error_rate(reference, hypothesis)
    cer = character_error_rate(reference, hypothesis)

    print(f"{label}")
    counts = f"S {wer.substitutions:4d}  D {wer.deletions:4d}  I {wer.insertions:4d}"
    print(f"  WER {wer.rate:6.2%}   {counts}")
    print(f"  CER {cer.rate:6.2%}   over {wer.reference_length} reference words")


def _print_alignment(reference: str, hypothesis: str) -> None:
    result = word_error_rate(reference, hypothesis)
    print("\n  errors:")
    for token in result.alignment:
        if token.op is Op.MATCH:
            continue
        if token.op is Op.SUBSTITUTION:
            print(f"    ~ {token.reference!r} -> {token.hypothesis!r}")
        elif token.op is Op.DELETION:
            print(f"    - {token.reference!r}")
        else:
            print(f"    + {token.hypothesis!r}")


def main() -> int:
    args = _parse_args()

    reference = _read(args.reference)
    first = _read(args.hypothesis)
    second = _read(args.second) if args.second else None

    if args.json:
        payload: dict[str, object] = {"first_pass": word_error_rate(reference, first).as_dict()}
        if second is not None:
            comparison = compare(reference, first, second)
            payload["second_pass"] = comparison.second.as_dict()
            payload["absolute_improvement"] = round(comparison.absolute_improvement, 4)
            payload["relative_improvement"] = round(comparison.relative_improvement, 4)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    _print_rate("first pass (realtime)", reference, first)
    if args.alignment:
        _print_alignment(reference, first)

    if second is None:
        return 0

    print()
    _print_rate("second pass (batch)", reference, second)
    if args.alignment:
        _print_alignment(reference, second)

    comparison = compare(reference, first, second)
    print()
    print(
        f"  {comparison.absolute_improvement:+.2%} absolute, "
        f"{comparison.relative_improvement:+.1%} of the first pass's errors removed"
    )
    # Said out loud rather than left for the reader to notice from a sign. A second pass that made
    # things worse is the outcome most likely to be skimmed past.
    if comparison.absolute_improvement < 0:
        print("  the second pass was WORSE than the first")

    return 0


if __name__ == "__main__":
    sys.exit(main())
