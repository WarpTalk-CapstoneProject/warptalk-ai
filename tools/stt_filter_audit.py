"""Audit what stt_worker's filter chain actually removes, and sweep for a better config.

WHY THIS EXISTS
---------------
Production correction data (4 rows, all correction_type='stt') shows corrected text
running ~90 characters LONGER than what STT produced. That is not a mis-heard word; it
is missing content. stt_worker/model.py:_filter_segments holds twelve independent drop
conditions and every one of them discards the WHOLE segment, so it is the first place
to look for content that vanishes without an error.

WHAT THIS MEASURES — AND WHAT IT DOES NOT
-----------------------------------------
This drives the REAL `_filter_segments` function over a labelled corpus, so it measures
the filter chain exactly as production runs it. It needs no audio, no API key, and no
live meeting, and it is deterministic — which is what makes a threshold sweep possible.

It does NOT measure transcription accuracy. Whether the model hears "Kubernetes" or
"Kuber" is an audio-level question that needs recordings and a live STT call; this tool
assumes the text it is given is what the model returned. Read a good score here as "the
filters are not eating real speech", never as "STT is accurate".

USAGE
    uv run python -m tools.stt_filter_audit            # audit current config
    uv run python -m tools.stt_filter_audit --sweep    # search for a better one
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from stt_worker import model as stt_model
from tools.stt_eval_corpus import DROP, KEEP, Utterance, full_corpus

# The prompt a real session carries: meeting title plus glossary terms. Several filters
# compare against it, so an audit without one would miss the prompt-echo guards entirely.
CONTEXT_PROMPT = "\n".join(
    [
        "WarpTalk transcript engineering sync",
        "Discussion of the realtime translation pipeline and its latency budget",
    ]
)
KEYWORDS = [
    "WarpTalk",
    "backend",
    "gRPC",
    "Kubernetes",
    "Redis",
    "staging",
    "latency",
    "translation worker",
]


@dataclass
class Outcome:
    utterance: Utterance
    kept: bool

    @property
    def correct(self) -> bool:
        return self.kept == (self.utterance.expected == KEEP)

    @property
    def failure_kind(self) -> str | None:
        if self.correct:
            return None
        return "DELETION" if self.utterance.expected == KEEP else "INSERTION"


def run_one(utterance: Utterance, **overrides: Any) -> tuple[bool, str]:
    """Push one utterance through the real filter chain.

    Returns (kept, reason). `reason` is the filter's own log event when it dropped —
    the chain logs a distinct name per condition, which is what makes attribution
    possible without instrumenting the function itself.
    """
    segment = {
        "text": utterance.text,
        "avg_logprob": utterance.avg_logprob,
        "no_speech_prob": 0.0,
        "start": 0.0,
        "end": utterance.duration_s,
    }

    # Every drop is reported through the module's own structlog logger, one distinct
    # event name per condition. Swap that logger for a recorder rather than re-deriving
    # the branch logic here, so attribution stays correct when a filter is added or moved.
    # (Capturing stderr does not work: structlog binds its stream at configure time.)
    events: list[str] = []

    class _Recorder:
        def _record(self, event: str = "", **_: Any) -> None:
            events.append(event)

        debug = info = warning = error = exception = _record

    original_logger = stt_model.logger
    stt_model.logger = _Recorder()  # type: ignore[assignment]
    try:
        kept = stt_model._filter_segments(
            [segment],
            utterance.language,
            0,
            allowed_languages={"vi", "en"},
            real_duration_s=utterance.duration_s,
            context_prompt=overrides.get("context_prompt", CONTEXT_PROMPT),
            keywords=overrides.get("keywords", KEYWORDS),
            min_avg_logprob=overrides.get("min_avg_logprob", -0.7),
        )
    finally:
        stt_model.logger = original_logger

    reason = next((e for e in events if e.startswith("filtered_")), "")
    return bool(kept), reason or ("kept" if kept else "dropped_unknown")


def audit(min_avg_logprob: float = -0.7) -> tuple[list[Outcome], dict[str, int]]:
    outcomes: list[Outcome] = []
    reasons: dict[str, int] = {}
    for utterance in full_corpus():
        kept, reason = run_one(utterance, min_avg_logprob=min_avg_logprob)
        outcomes.append(Outcome(utterance, kept))
        if not kept:
            reasons[reason] = reasons.get(reason, 0) + 1
    return outcomes, reasons


def score(outcomes: list[Outcome]) -> dict[str, float]:
    real = [o for o in outcomes if o.utterance.expected == KEEP]
    garbage = [o for o in outcomes if o.utterance.expected == DROP]
    retained = sum(1 for o in real if o.kept)
    blocked = sum(1 for o in garbage if not o.kept)
    return {
        # The metric the production correction data says actually matters.
        "content_retention": retained / len(real) if real else 0.0,
        "garbage_blocked": blocked / len(garbage) if garbage else 0.0,
        "deletions": float(len(real) - retained),
        "insertions": float(len(garbage) - blocked),
    }


def report(min_avg_logprob: float = -0.7) -> None:
    outcomes, reasons = audit(min_avg_logprob)
    s = score(outcomes)

    print(f"=== stt filter audit  (min_avg_logprob={min_avg_logprob}) ===\n")
    print(f"content retention : {s['content_retention']:.1%}  ({int(s['deletions'])} deletions)")
    print(f"garbage blocked   : {s['garbage_blocked']:.1%}  ({int(s['insertions'])} insertions)")

    deletions = [o for o in outcomes if o.failure_kind == "DELETION"]
    if deletions:
        print(f"\n--- real speech LOST ({len(deletions)}) ---")
        for o in deletions:
            _, reason = run_one(o.utterance, min_avg_logprob=min_avg_logprob)
            tags = ",".join(o.utterance.tags) or "-"
            print(f"  [{reason}] ({tags}) {o.utterance.text[:64]}")
            if o.utterance.note:
                print(f"      why it matters: {o.utterance.note}")

    insertions = [o for o in outcomes if o.failure_kind == "INSERTION"]
    if insertions:
        print(f"\n--- garbage KEPT ({len(insertions)}) ---")
        for o in insertions:
            print(f"  {o.utterance.text[:64]}")

    if reasons:
        print("\n--- drops by filter ---")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3d}  {reason}")


def sweep() -> None:
    """Grid-search the three tunables and show where the good region actually is.

    A single winning point proves nothing — a threshold that only works at one exact
    value is a value someone will break by nudging it. What matters is whether the
    chosen setting sits inside a PLATEAU of equally good settings, which is what the
    per-parameter tables below are for.
    """
    print("=== threshold sweep ===\n")
    print("Both metrics are on the labelled corpus in tools/stt_eval_corpus.py.\n")

    print("--- min_avg_logprob (the STT discard floor) ---")
    print(f"{'value':>8} {'retention':>10} {'blocked':>9}")
    for step in range(0, 13):
        value = round(-1.0 + step * 0.05, 2)
        s = score(audit(value)[0])
        print(f"{value:>8.2f} {s['content_retention']:>9.1%} {s['garbage_blocked']:>8.1%}")

    print("\n--- _BLOCKLIST_MARGINAL_LOGPROB (spoken vs hallucinated) ---")
    print(f"{'value':>8} {'retention':>10} {'blocked':>9}")
    original_blocklist = stt_model._BLOCKLIST_MARGINAL_LOGPROB
    try:
        for step in range(0, 13):
            value = round(-0.75 + step * 0.05, 2)
            stt_model._BLOCKLIST_MARGINAL_LOGPROB = value
            s = score(audit()[0])
            print(f"{value:>8.2f} {s['content_retention']:>9.1%} {s['garbage_blocked']:>8.1%}")
    finally:
        stt_model._BLOCKLIST_MARGINAL_LOGPROB = original_blocklist

    print("\n--- _MIN_DISTINCT_WORD_RATIO (speech vs repetition loop) ---")
    print(f"{'value':>8} {'retention':>10} {'blocked':>9}")
    original_ratio = stt_model._MIN_DISTINCT_WORD_RATIO
    try:
        for step in range(0, 13):
            value = round(0.30 + step * 0.05, 2)
            stt_model._MIN_DISTINCT_WORD_RATIO = value
            s = score(audit()[0])
            print(f"{value:>8.2f} {s['content_retention']:>9.1%} {s['garbage_blocked']:>8.1%}")
    finally:
        stt_model._MIN_DISTINCT_WORD_RATIO = original_ratio

    print(
        "\ncurrent config: "
        f"min_avg_logprob=-0.7, "
        f"_BLOCKLIST_MARGINAL_LOGPROB={original_blocklist}, "
        f"_MIN_DISTINCT_WORD_RATIO={original_ratio}"
    )


def by_language() -> None:
    """Score each meeting language separately.

    languages.ts offers six meeting-scope languages. The filter chain's hand-written
    assets — spelling repair, hallucination blocklists, the language guesser — cover two
    of them. This report is the direct measurement of what that costs the other four.
    """
    outcomes, _ = audit()
    languages: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        languages.setdefault(outcome.utterance.language, []).append(outcome)

    print("=== per-language filter quality ===\n")
    print(f"{'lang':>6} {'cases':>6} {'retention':>10} {'blocked':>9}  {'failures'}")
    for lang in sorted(languages):
        group = languages[lang]
        s = score(group)
        failures = [o for o in group if not o.correct]
        detail = ", ".join(f"{o.failure_kind}" for o in failures) or "-"
        print(
            f"{lang:>6} {len(group):>6} {s['content_retention']:>9.1%}"
            f" {s['garbage_blocked']:>8.1%}  {detail}"
        )

    print("\n--- every disagreement ---")
    for lang in sorted(languages):
        for outcome in languages[lang]:
            if outcome.correct:
                continue
            _, reason = run_one(outcome.utterance)
            print(f"  [{lang}] {outcome.failure_kind:9} [{reason}] {outcome.utterance.text[:48]}")
            if outcome.utterance.note:
                print(f"          {outcome.utterance.note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="sweep the three thresholds")
    parser.add_argument("--by-language", action="store_true", help="score each language")
    parser.add_argument("--min-avg-logprob", type=float, default=-0.7)
    args = parser.parse_args()
    if args.sweep:
        sweep()
    elif args.by_language:
        by_language()
    else:
        report(args.min_avg_logprob)


if __name__ == "__main__":
    main()
