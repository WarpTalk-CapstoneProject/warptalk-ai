"""Score the disfluency prepass against labelled fixtures — over-deletion first. WT-716.

USAGE
    Everything under tools/fixtures/disfluency, plain table:

        python -m tools.disfluency_eval

    One language, or a fixture file of your own:

        python -m tools.disfluency_eval --language vi
        python -m tools.disfluency_eval --fixtures my_cases.jsonl

    Score a tier-2 (LLM) run beside tier 1. The dump is JSON: either {"E1": "clean text", ...}
    or [{"id": "E1", "clean_text": "..."}, ...], produced by whatever ran the model — this
    tool never calls one.

        python -m tools.disfluency_eval --tier2 run-2026-09-24.json

    Machine-readable, and the shape CI should read:

        python -m tools.disfluency_eval --json --max-over-deletion 0.5

    Exit code 0 means every gate passed. Non-zero means at least one of: the over-deletion rate
    exceeded --max-over-deletion (a PERCENT, default 0.5), an invariant was violated, or a
    sentence that was already clean lost a token.

WHY OVER-DELETION IS THE PRIMARY METRIC AND F1 IS NOT
    The ruling this whole ticket is built on is faithfulness first: deleting too little is
    better than deleting too much. F1 cannot express that — it trades a missed "um" against a
    deleted "not" at par, so a change that cleans more aggressively and eats one real word in
    fifty can raise F1 while making the transcript untrustworthy. Over-deletion counts only the
    second kind of error: the share of tokens that SHOULD HAVE SURVIVED and did not. It is
    printed first, it is what the exit code is about, and it is the number to quote.

    Recall is reported and is deliberately NOT gated. A prepass that leaves a filler in has
    produced a transcript that is merely less tidy; one that deletes a word nobody can get back
    has produced a transcript that is wrong. Do not add a recall floor here: it would be an
    instruction to delete more, aimed at exactly the case where the rules were unsure.

FIXTURE FORMAT (JSONL, one object per line)
    {"id": "E1", "language": "en", "raw": "...", "expected_clean": "...", "kind": "filler"}

    `kind` is one of filler | stutter | self_repair | clean_control | trap.
        clean_control  a sentence with no disfluency in it at all. Any deletion here is a
                       regression and fails the run — this is the cheapest possible detector
                       for a rule that has started firing on ordinary speech.
        trap           a line that LOOKS like a disfluency and is not: "very very", "had had",
                       "ba ba", "まだまだ", an "uh-huh" that is the whole answer. These are the
                       cases a more aggressive prepass gets wrong first.

    Two optional booleans carry the turn context `prepass` takes, for fixtures whose expected
    output depends on it: "prev_turn_is_question" and "standalone_turn". Absent means "let the
    prepass infer it", which is what a live worker mostly does.

HOW A DELETION IS COUNTED
    Tokens, not characters, and by POSITION in the raw line. Both the expected and the actual
    clean text are aligned back to the raw tokens by the same greedy leftmost subsequence walk
    over `normalize_key`, so the alignment cannot favour one of them — and comparing keys means
    the capitalisation and terminal punctuation the prepass is allowed to change never register
    as a deletion. A clean text that is not a subsequence of the raw line at all is not scored
    as deletions: it is reported as `not_subsequence`, which is a bug in the prepass (invariant
    I1), not a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from shared.disfluency import check_invariants, normalize_key, prepass, tokenize
from shared.disfluency.tokenize import ja_morphology_available

#: Where the seeded fixtures live, relative to the repository root.
DEFAULT_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "disfluency"

#: Every `kind` a fixture may declare. Closed on purpose: a typo in a label would otherwise
#: create a silent new bucket and quietly leave those cases out of the per-kind breakdown.
KINDS = ("filler", "stutter", "self_repair", "clean_control", "trap")

#: A line ends as a question in any of the three languages.
QUESTION_MARKS = ("?", "？")


@dataclass(frozen=True)
class Fixture:
    """One labelled case: a raw line and the clean line the prepass is supposed to produce."""

    id: str
    language: str
    raw: str
    expected_clean: str
    kind: str
    prev_turn_is_question: bool = False
    standalone_turn: bool | None = None


@dataclass(frozen=True)
class Row:
    """What one fixture scored under one tier."""

    fixture: Fixture
    actual_clean: str
    tokens: int
    #: Tokens the fixture says should go.
    expected_deleted: int
    #: Tokens that actually went.
    actual_deleted: int
    true_positives: int
    #: Deleted and should not have been. THE number this harness exists to report.
    over_deletions: int
    #: Should have gone and stayed. Reported, never gated — see the module docstring.
    under_deletions: int
    exact: bool
    invariants: tuple[str, ...]
    question_ok: bool
    not_subsequence: bool

    @property
    def survivors(self) -> int:
        """Tokens that should have survived — the denominator of the over-deletion rate."""
        return self.tokens - self.expected_deleted


@dataclass
class Metrics:
    """Totals for one slice (a language, a kind, or everything)."""

    fixtures: int = 0
    tokens: int = 0
    survivors: int = 0
    expected_deleted: int = 0
    true_positives: int = 0
    over_deletions: int = 0
    under_deletions: int = 0
    exact: int = 0
    invariant_violations: int = 0
    question_ok: int = 0
    not_subsequence: int = 0

    def add(self, row: Row) -> None:
        self.fixtures += 1
        self.tokens += row.tokens
        self.survivors += row.survivors
        self.expected_deleted += row.expected_deleted
        self.true_positives += row.true_positives
        self.over_deletions += row.over_deletions
        self.under_deletions += row.under_deletions
        self.exact += int(row.exact)
        self.invariant_violations += len(row.invariants)
        self.question_ok += int(row.question_ok)
        self.not_subsequence += int(row.not_subsequence)

    @property
    def over_deletion_rate(self) -> float:
        """Share of tokens that should have survived and did not. A fraction, not a percent."""
        return self.over_deletions / self.survivors if self.survivors else 0.0

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.over_deletions
        return self.true_positives / predicted if predicted else 1.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.expected_deleted if self.expected_deleted else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def exact_rate(self) -> float:
        return self.exact / self.fixtures if self.fixtures else 1.0

    @property
    def question_accuracy(self) -> float:
        return self.question_ok / self.fixtures if self.fixtures else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fixtures": self.fixtures,
            "tokens": self.tokens,
            "survivors": self.survivors,
            "overDeletionRate": self.over_deletion_rate,
            "overDeletions": self.over_deletions,
            "underDeletions": self.under_deletions,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exactMatchRate": self.exact_rate,
            "exactMatches": self.exact,
            "invariantViolations": self.invariant_violations,
            "questionPunctuationAccuracy": self.question_accuracy,
            "notSubsequence": self.not_subsequence,
        }


@dataclass
class Report:
    """One tier scored over one fixture set."""

    tier: str
    rows: list[Row] = field(default_factory=list)
    overall: Metrics = field(default_factory=Metrics)
    by_language: dict[str, Metrics] = field(default_factory=dict)
    by_kind: dict[str, Metrics] = field(default_factory=dict)
    #: Tokens deleted from sentences that were already clean. MUST be 0.
    clean_control_regressions: int = 0
    #: Fixtures the tier dump had no answer for. Tier 1 always answers; a tier-2 dump may not.
    missing: list[str] = field(default_factory=list)

    def failures(self, max_over_deletion: float) -> list[str]:
        """Every reason this run should fail CI, in the order they matter."""
        reasons: list[str] = []
        if self.overall.over_deletion_rate > max_over_deletion:
            reasons.append(
                f"over-deletion {self.overall.over_deletion_rate:.3%} exceeds the "
                f"{max_over_deletion:.3%} ceiling "
                f"({self.overall.over_deletions} of {self.overall.survivors} tokens)"
            )
        if self.clean_control_regressions:
            reasons.append(
                f"{self.clean_control_regressions} token(s) deleted from clean_control "
                "sentences, which must never lose one"
            )
        if self.overall.invariant_violations:
            reasons.append(f"{self.overall.invariant_violations} invariant violation(s)")
        if self.overall.not_subsequence:
            reasons.append(
                f"{self.overall.not_subsequence} output(s) are not a deletion of their input"
            )
        return reasons

    def as_dict(self, max_over_deletion: float) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "overall": self.overall.as_dict(),
            "byLanguage": {lang: m.as_dict() for lang, m in sorted(self.by_language.items())},
            "byKind": {kind: m.as_dict() for kind, m in sorted(self.by_kind.items())},
            "cleanControlRegressions": self.clean_control_regressions,
            "missing": self.missing,
            "mismatches": [
                {
                    "id": row.fixture.id,
                    "language": row.fixture.language,
                    "kind": row.fixture.kind,
                    "raw": row.fixture.raw,
                    "expected": row.fixture.expected_clean,
                    "actual": row.actual_clean,
                    "overDeletions": row.over_deletions,
                    "underDeletions": row.under_deletions,
                    "invariants": list(row.invariants),
                }
                for row in self.rows
                if not row.exact
            ],
            "failures": self.failures(max_over_deletion),
        }


# --- loading ----------------------------------------------------------------------------------


def load_fixtures(paths: Sequence[Path]) -> list[Fixture]:
    """Every fixture in the given .jsonl files, or in the given directories.

    Raises on a malformed row rather than skipping it. A silently dropped fixture is a case
    that stops being checked without anybody noticing — the one failure mode a harness whose
    job is to catch regressions cannot have.
    """
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.jsonl")))
        else:
            files.append(path)

    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for file in files:
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{file.name}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: not JSON ({exc})") from exc
            fixture = _fixture_from(row, where)
            if fixture.id in seen:
                raise ValueError(f"{where}: duplicate fixture id {fixture.id!r}")
            seen.add(fixture.id)
            fixtures.append(fixture)
    return fixtures


def _fixture_from(row: Any, where: str) -> Fixture:
    if not isinstance(row, dict):
        raise ValueError(f"{where}: expected a JSON object")
    for key in ("id", "language", "raw", "expected_clean", "kind"):
        if not isinstance(row.get(key), str):
            raise ValueError(f"{where}: missing or non-string {key!r}")
    if row["kind"] not in KINDS:
        raise ValueError(f"{where}: kind {row['kind']!r} is not one of {', '.join(KINDS)}")
    standalone = row.get("standalone_turn")
    return Fixture(
        id=row["id"],
        language=row["language"],
        raw=row["raw"],
        expected_clean=row["expected_clean"],
        kind=row["kind"],
        prev_turn_is_question=bool(row.get("prev_turn_is_question", False)),
        standalone_turn=None if standalone is None else bool(standalone),
    )


def load_tier2(path: Path) -> dict[str, str]:
    """A tier-2 run's outputs, keyed by fixture id. See USAGE for the two shapes accepted."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return {str(e["id"]): str(e.get("clean_text", "")) for e in payload}
    raise ValueError(f"{path}: expected a JSON object or a list of objects")


# --- scoring ----------------------------------------------------------------------------------


def deleted_positions(raw: str, clean: str, language: str) -> tuple[set[int], bool]:
    """Which raw token positions `clean` dropped, and whether it is a deletion of `raw` at all.

    A greedy leftmost walk: every raw token that matches the next unconsumed clean token is
    kept, everything else is deleted. Greedy is exactly right for a subsequence — taking the
    earliest match can never make a later one impossible — and using it for BOTH the expected
    and the actual clean text means an ambiguous repeat ("the the contract" losing one "the")
    resolves the same way on both sides and cancels out.

    Comparison is on `normalize_key`, so the two edits the prepass IS allowed to make — a
    sentence-initial capital and terminal punctuation — never look like deletions.
    """
    raw_keys = [normalize_key(t, language) for t in tokenize(raw, language)]
    clean_keys = [normalize_key(t, language) for t in tokenize(clean, language)]

    deleted: set[int] = set()
    cursor = 0
    for index, key in enumerate(raw_keys):
        if cursor < len(clean_keys) and clean_keys[cursor] == key:
            cursor += 1
        else:
            deleted.add(index)
    # Leftover clean tokens mean the output holds something the input did not: not a deletion,
    # so the positions above describe nothing real. Invariant I1 is what catches this for real.
    return deleted, cursor < len(clean_keys)


def _ends_as_question(text: str) -> bool:
    return text.rstrip().endswith(QUESTION_MARKS)


def score_row(fixture: Fixture, actual_clean: str) -> Row:
    language = fixture.language
    tokens = len(tokenize(fixture.raw, language))
    expected, expected_broken = deleted_positions(fixture.raw, fixture.expected_clean, language)
    actual, actual_broken = deleted_positions(fixture.raw, actual_clean, language)

    if expected_broken:
        # The fixture itself is wrong: its expected_clean is not a deletion of its raw line.
        # Louder than a bad score, because no score computed from it would mean anything.
        raise ValueError(
            f"{fixture.id}: expected_clean is not a deletion of raw "
            f"({fixture.expected_clean!r} vs {fixture.raw!r})"
        )

    return Row(
        fixture=fixture,
        actual_clean=actual_clean,
        tokens=tokens,
        expected_deleted=len(expected),
        actual_deleted=len(actual),
        true_positives=len(actual & expected),
        over_deletions=len(actual - expected),
        under_deletions=len(expected - actual),
        exact=actual_clean == fixture.expected_clean,
        invariants=tuple(check_invariants(fixture.raw, actual_clean, language)),
        question_ok=_ends_as_question(actual_clean) == _ends_as_question(fixture.expected_clean),
        not_subsequence=actual_broken,
    )


def run_prepass(fixture: Fixture) -> str:
    result = prepass(
        fixture.raw,
        fixture.language,
        prev_turn_is_question=fixture.prev_turn_is_question,
        standalone_turn=fixture.standalone_turn,
    )
    return result.clean_text


def score(fixtures: Iterable[Fixture], outputs: dict[str, str] | None, tier: str) -> Report:
    """Score one tier. `outputs` None runs the prepass; a dict is a recorded tier-2 run."""
    report = Report(tier=tier)
    for fixture in fixtures:
        if outputs is None:
            actual = run_prepass(fixture)
        elif fixture.id in outputs:
            actual = outputs[fixture.id]
        else:
            # Not scored as a perfect pass and not scored as a failure either: a tier-2 dump
            # that skipped a line says nothing about that line. Counted and named instead.
            report.missing.append(fixture.id)
            continue

        row = score_row(fixture, actual)
        report.rows.append(row)
        report.overall.add(row)
        report.by_language.setdefault(fixture.language, Metrics()).add(row)
        report.by_kind.setdefault(fixture.kind, Metrics()).add(row)
        if fixture.kind == "clean_control":
            report.clean_control_regressions += row.actual_deleted
    return report


# --- printing ---------------------------------------------------------------------------------


def _print_report(report: Report, max_over_deletion: float, show_mismatches: bool) -> None:
    overall = report.overall
    print(f"\n{report.tier} — {overall.fixtures} fixtures, {overall.tokens} tokens")
    print("=" * 78)

    # FIRST AND ALONE ON ITS OWN LINES. Everything below this is context for it.
    verdict = "PASS" if overall.over_deletion_rate <= max_over_deletion else "FAIL"
    print(
        f"OVER-DELETION  {overall.over_deletion_rate:7.3%}   "
        f"{overall.over_deletions} of {overall.survivors} tokens that should have survived"
    )
    print(f"               ceiling {max_over_deletion:.3%}  ->  {verdict}")
    print(
        f"  clean_control regressions {report.clean_control_regressions}   "
        f"invariant violations {overall.invariant_violations}"
    )

    header = f"\n  {'':<6}{'n':>4}  {'over-del':>9}  {'prec':>6} {'recall':>6} {'f1':>6}"
    header += f"  {'exact':>7}  {'quest':>6}  {'inv':>4}"
    print(header)
    for language, metrics in sorted(report.by_language.items()):
        _print_metrics_line(language, metrics)
    if len(report.by_language) > 1:
        _print_metrics_line("ALL", overall)

    print(f"\n  by kind{'':<12}{'n':>4}  {'over-del':>9}  {'recall':>6}  {'exact':>7}")
    for kind in KINDS:
        by_kind = report.by_kind.get(kind)
        if by_kind is None:
            continue
        print(
            f"  {kind:<19}{by_kind.fixtures:>4}  {by_kind.over_deletion_rate:>8.2%}  "
            f"{by_kind.recall:>6.3f}  {by_kind.exact:>3}/{by_kind.fixtures:<3}"
        )

    if report.missing:
        print(f"\n  not answered by this tier: {', '.join(report.missing)}")

    if show_mismatches:
        _print_mismatches(report)


def _print_metrics_line(label: str, metrics: Metrics) -> None:
    print(
        f"  {label:<6}{metrics.fixtures:>4}  {metrics.over_deletion_rate:>8.2%}  "
        f"{metrics.precision:>6.3f} {metrics.recall:>6.3f} {metrics.f1:>6.3f}  "
        f"{metrics.exact:>3}/{metrics.fixtures:<3}  {metrics.question_accuracy:>6.1%}  "
        f"{metrics.invariant_violations:>4}"
    )


def _print_mismatches(report: Report) -> None:
    wrong = [row for row in report.rows if not row.exact]
    if not wrong:
        print("\n  every fixture matched exactly.")
        return
    print(f"\n  {len(wrong)} fixture(s) did not match exactly:")
    for row in wrong:
        direction = []
        if row.over_deletions:
            direction.append(f"-{row.over_deletions} over")
        if row.under_deletions:
            direction.append(f"-{row.under_deletions} under")
        if row.invariants:
            direction.append("!" + ",".join(row.invariants))
        note = f"  [{' '.join(direction)}]" if direction else "  [punctuation/case only]"
        print(f"\n    {row.fixture.id} ({row.fixture.language}, {row.fixture.kind}){note}")
        print(f"      raw      {row.fixture.raw}")
        print(f"      expected {row.fixture.expected_clean}")
        print(f"      actual   {row.actual_clean}")


# --- entry point ------------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score shared.disfluency.prepass against labelled fixtures (WT-716).",
    )
    parser.add_argument(
        "--fixtures",
        action="append",
        type=Path,
        help=f"A .jsonl file or a directory of them. Repeatable. Default: {DEFAULT_FIXTURE_DIR}",
    )
    parser.add_argument("--language", help="Score only this language (en, vi, ja).")
    parser.add_argument("--tier2", type=Path, help="A recorded tier-2 run to score as well.")
    parser.add_argument(
        "--max-over-deletion",
        type=float,
        default=0.5,
        help="Ceiling on the over-deletion rate, as a PERCENT. Default 0.5.",
    )
    parser.add_argument("--json", action="store_true", help="Machine-readable output.")
    parser.add_argument(
        "--quiet", action="store_true", help="Table only; do not print the mismatching fixtures."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # The fixtures are Vietnamese and Japanese; a Windows console defaults to cp1252 and would
    # raise UnicodeEncodeError before printing a single number.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    fixtures = load_fixtures(args.fixtures or [DEFAULT_FIXTURE_DIR])
    if args.language:
        fixtures = [f for f in fixtures if f.language == args.language]
    if not fixtures:
        print("No fixtures matched.", file=sys.stderr)
        return 2

    ceiling = args.max_over_deletion / 100.0
    reports = [score(fixtures, None, tier="tier 1 (prepass)")]
    if args.tier2:
        reports.append(score(fixtures, load_tier2(args.tier2), tier="tier 2 (llm)"))

    # Said once, up front, because it changes what the ja numbers mean: without fugashi the
    # prepass falls back to a coarse tokenizer and deliberately does less.
    morphology = ja_morphology_available()
    if args.json:
        print(
            json.dumps(
                {
                    "maxOverDeletion": ceiling,
                    "jaMorphologyAvailable": morphology,
                    "reports": [r.as_dict(ceiling) for r in reports],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        if not morphology and any(f.language == "ja" for f in fixtures):
            print(
                "note: fugashi/unidic-lite is not installed, so Japanese is scored on the "
                "fallback tokenizer."
            )
        for report in reports:
            _print_report(report, ceiling, show_mismatches=not args.quiet)

    failures = [reason for report in reports for reason in report.failures(ceiling)]
    if failures and not args.json:
        print("\nFAILED:")
        for reason in failures:
            print(f"  - {reason}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
