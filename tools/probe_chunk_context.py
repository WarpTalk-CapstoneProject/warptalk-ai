"""Can caching the previous sentences as context rescue chunked transcription?

WHY THIS PROBE EXISTS
    `tools/probe_commit_semantics.py` established two things about OpenAI Realtime:

      E1  `commit()` CONSUMES the buffer, so rolling commits yield disjoint slices rather
          than a growing hypothesis.
      E4  At 1000ms slices the model hallucinates ACROSS LANGUAGES — Chinese and English
          appeared in a Vietnamese-hinted session, on audio that a single commit
          transcribed correctly.

    E1 was read as fatal. On reflection it is not, and that is what this probe tests.
    LocalAgreement exists to answer "how do I emit early without emitting something that
    will later be revised?" — but when commits consume the buffer there is nothing to
    revise: every slice is final and disjoint, which is append-only monotonic output by
    construction, exactly what a TTS stage needs. The architecture is not the blocker.

    The blocker is E4: quality. A one-second slice carries no linguistic context, so the
    model guesses. The obvious counter is to GIVE it the context — the transcription
    session already accepts a `prompt` (`OpenAISTT._session_payload` -> transcription
    config, `model.py:1829`), and production already uses that field to carry a room's
    glossary. Nothing stops it also carrying what the speaker has said so far.

    If priming works, chunked streaming on OpenAI is viable and no provider change is
    needed. If it does not, the case for a true streaming ASR stops being an argument
    and becomes a measurement.

WHAT IS MEASURED
    A matrix of slice length x priming, every cell against the same audio and the same
    single-commit baseline:

      * FOREIGN SCRIPTS — reuses production's own `_scripts_in()` (model.py:296), the
        detector `_filter_segments` uses to catch cross-script hallucination. A Vietnamese
        clip should yield an empty set. Anything else is the failure E4 found, counted
        rather than eyeballed.
      * WORD COUNT vs baseline — words invented or lost by slicing.
      * LATENCY — mean commit -> completed per slice, because a fix that costs a second
        per slice is not a fix for a cabin.

    Priming is cumulative and bounded: after each slice the running transcript is fed back
    as the next slice's prompt, capped so a long turn cannot grow the prompt without limit.

THE ANSWER, MEASURED (2026-08-20, gpt-transcribe, 8.72s of connected Vietnamese, --language vi)

      slice  primed    n  words  foreign scripts  mean ms  implied lag
     1000ms      no    9     34              han      949       1949ms
     1000ms     yes    9     38                —      961       1961ms
     2000ms      no    5     38                —     1103       3103ms
     2000ms     yes    5     34                —     1064       3064ms
     3000ms      no    3     38                —      899       3899ms
     3000ms     yes    3     38                —     1096       4096ms

    PRIMING WORKS, AND IT IS NOT ENOUGH.
      * At 1000ms it removed the Chinese hallucination outright (`給我展開東西` -> `Kế hoạch
        triển khai tuần sau`) and recovered a phrase the SINGLE-COMMIT BASELINE got wrong —
        the baseline said "cái hoạt", primed slices said "kế hoạch". Context is a real lever.
      * At 3000ms primed also beat the baseline on that same phrase.
      * At 2000ms primed was WORSE than raw, returning pseudo-phonetic noise
        ("Mìn se sam-lái ki ê hoat tsin-khai tōng-san"). With n=1 per cell that may be
        variance rather than signal; it is recorded because it was not predicted.

    THE BLOCKER IS THE CLOCK, NOT THE TRANSCRIPT.
    `commit -> completed` cost ~900-1100ms at EVERY slice length — it does not shrink with
    the slice. So a chunked design's floor is `slice + ~1s`, before translation or synthesis
    are charged at all. Against a 1500ms cabin budget: the only cells near it (1000ms, ~1.95s)
    are the ones whose transcripts break, and every clean cell sits at 3.1-4.1s. The
    requirements move against each other; no slice length satisfies both.

    CONCLUSION: a commit-gated STT cannot reach the cabin target at any slice length, and
    priming cannot close the gap because it changes transcript quality rather than the cost
    of obtaining one. A true streaming ASR is the design.

    CAVEATS: n=1 per cell, one clip, and the audio is Cartesia `sonic-3.5` output rather than
    a human recording (the repo carries no speech fixture). The ~1s commit floor is the robust
    part — it reproduced across both this probe and probe_commit_semantics. The per-cell
    transcript differences are not yet separable from run-to-run variance.

WHAT THE LITERATURE SAYS THIS PROBE GOT WRONG — read before trusting the transcript column

    THIS PROBE SLICES THE AUDIO. THAT IS NOT WHAT STREAMING WHISPER DOES.
    Whisper-Streaming (arXiv 2307.14743, IJCNLP-AACL 2023) measures WER 8.5% at 0.5s chunks
    against 8.0% at 2.0s — half a point, not a collapse — because LocalAgreement-2 runs over a
    GROWING AUDIO BUFFER and chunks only the EMISSION. The model always hears full acoustic
    context; only the output is incremental.

    So the transcript degradation measured above is a property of slicing the audio, which is
    the one thing a consuming `commit()` forces you to do. The precise reason to leave this API
    is therefore not "it hallucinates" but "it forces the wrong thing to be chunked" — and it
    also explains why TEXT context could not rescue it: what a 1-second slice lacks is ACOUSTIC
    context, which no prompt supplies.

    PRIMING'S PUBLISHED UPSIDE IS 0.2 WER ABSOLUTE.
    Whisper's own long-form ablation (arXiv 2212.04356, Table 7) moves 10.2 -> 10.0 by adding
    previous-text conditioning. The "primed beat the baseline on `kế hoạch`" observation above
    is one sample against an effect that small; treat it as noise until repeated.

    CONDITIONING IS NOT FREE. WhisperX (Interspeech 2023) reports it "is more prone to
    hallucination and repetition", and both WhisperX and faster-whisper's batched pipeline
    default it to False.

METHOD
    Imports the session/append/commit helpers from probe_commit_semantics rather than
    restating them, so the two probes cannot drift apart on the one thing they share.

    No Redis, no LiveKit. Needs OPENAI_API_KEY (or STT_API_KEY) and one wav of real
    connected speech in the language being tested.

USAGE
    uv run python -m tools.probe_chunk_context --wav sample-vi.wav --language vi --dry-run
    uv run python -m tools.probe_chunk_context --wav sample-vi.wav --language vi
    uv run python -m tools.probe_chunk_context --wav sample-vi.wav --language vi \
        --slices 1000,2000,3000
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from typing import Any, cast

from shared.config import STTSettings
from stt_worker.model import REALTIME_SAMPLE_RATE, OpenAISTT, _resample_pcm16, _scripts_in
from tools.probe_commit_semantics import (
    CommitResult,
    _append,
    _commit_and_collect,
    _load_wav,
    _open_session,
    _resolve_key,
)

# How much prior transcript to hand the next slice. Whisper-family prompts are a
# continuation hint, not a document: past a few sentences they stop steering and start
# competing with the audio for the model's attention. Bounded here for the same reason
# translation_worker bounds its meeting context.
#
# 600 IS PROBABLY TOO HIGH, AND MAY HAVE CORRUPTED THIS PROBE'S OWN 2000ms CELL.
# The IWSLT 2025 winning system (arXiv 2506.17077) gained from context up to 250 tokens and
# then reported "hallucinations, mostly repetitions of long sentences" past roughly 300-500 —
# which is exactly the pseudo-phonetic noise the 2000ms primed cell returned. 600 Vietnamese
# characters lands in that band. Drop this to ~200 before re-running, and treat the recorded
# 2000ms result as suspect rather than as a finding.
CONTEXT_CHARS = 600

# The booth lag the cabin architecture is built around. A chunked design's lag is at BEST its
# slice length plus what one commit costs to come back, so this is the bar every cell in the
# matrix is measured against — and it is the bar that decides the design, not the transcript
# quality on its own.
CABIN_LAG_TARGET_MS = 1500


@dataclass
class RunResult:
    """One cell of the matrix."""

    slice_ms: int
    primed: bool
    slices: list[CommitResult]

    @property
    def text(self) -> str:
        return " ".join(s.completed_text for s in self.slices if s.completed_text).strip()

    @property
    def foreign_scripts(self) -> set[str]:
        """Non-Latin scripts that leaked in — production's own hallucination detector."""
        return _scripts_in(self.text)

    @property
    def mean_completed_ms(self) -> float | None:
        values = [s.completed_ms for s in self.slices if s.completed_ms is not None]
        return sum(values) / len(values) if values else None

    @property
    def implied_lag_ms(self) -> float | None:
        """Best-case lag a cabin built on this cell could reach.

        A slice cannot be committed before it has been spoken, and its text does not exist
        until the commit comes back — so `slice_ms + mean_completed_ms` is a floor, before
        translation and synthesis are charged at all. This is the number that decides the
        design; transcript quality only decides whether a cell is admissible in the first place.
        """
        mean = self.mean_completed_ms
        return None if mean is None else self.slice_ms + mean


async def run_sliced(
    stt: OpenAISTT,
    pcm_24k: bytes,
    language: str | None,
    slice_ms: int,
    primed: bool,
) -> RunResult:
    """Commit the clip in `slice_ms` pieces, optionally priming each with the transcript so far."""
    slice_bytes = (REALTIME_SAMPLE_RATE * 2 * slice_ms) // 1000
    slice_bytes -= slice_bytes % 2
    manager, conn = await _open_session(stt, language)
    slices: list[CommitResult] = []
    context = ""
    try:
        for start in range(0, len(pcm_24k), slice_bytes):
            piece = pcm_24k[start : start + slice_bytes]
            # Sub-100ms tails are rejected by the API and would fail the run for a reason
            # that has nothing to do with the question.
            if len(piece) < REALTIME_SAMPLE_RATE // 10:
                break
            if primed and context:
                await conn.session.update(
                    session=cast(Any, stt._session_payload(language, context, None, None))
                )
            await _append(conn, piece)
            result = await _commit_and_collect(conn)
            slices.append(result)
            if primed and result.completed_text:
                context = f"{context} {result.completed_text}".strip()[-CONTEXT_CHARS:]
        return RunResult(slice_ms=slice_ms, primed=primed, slices=slices)
    finally:
        await manager.__aexit__(None, None, None)


async def run_baseline(stt: OpenAISTT, pcm_24k: bytes, language: str | None) -> CommitResult:
    manager, conn = await _open_session(stt, language)
    try:
        await _append(conn, pcm_24k)
        return await _commit_and_collect(conn)
    finally:
        await manager.__aexit__(None, None, None)


def _report(baseline: CommitResult, runs: list[RunResult]) -> None:
    base_words = len(baseline.completed_text.split())
    base_scripts = _scripts_in(baseline.completed_text)

    print("\n" + "=" * 84)
    print("BASELINE — one commit, whole clip")
    print("=" * 84)
    print(f"  {baseline.completed_text!r}")
    print(
        f"  words={base_words}  foreign_scripts={sorted(base_scripts) or '(none)'}  "
        f"completed={baseline.completed_ms:.0f}ms"
        if baseline.completed_ms
        else ""
    )

    print("\n" + "=" * 84)
    print("MATRIX — slice length x priming")
    print("=" * 84)
    print(
        f"  {'slice':>7}  {'primed':>6}  {'n':>3}  {'words':>5}  "
        f"{'foreign scripts':>16}  {'mean ms':>8}  {'implied lag':>11}"
    )
    for run in runs:
        scripts = sorted(run.foreign_scripts)
        mean = run.mean_completed_ms
        lag = run.implied_lag_ms
        flag = "" if lag is None or lag <= CABIN_LAG_TARGET_MS else "  ✗"
        print(
            f"  {run.slice_ms:>5}ms  {('yes' if run.primed else 'no'):>6}  "
            f"{len(run.slices):>3}  {len(run.text.split()):>5}  "
            f"{(','.join(scripts) if scripts else '—'):>16}  "
            f"{(f'{mean:.0f}' if mean else '—'):>8}  "
            f"{(f'{lag:.0f}ms' if lag else '—'):>11}{flag}"
        )
    print(
        f"\n  implied lag = slice length + mean commit->completed. Cabin target is "
        f"{CABIN_LAG_TARGET_MS}ms; ✗ marks a cell that cannot reach it."
    )

    print("\n" + "=" * 84)
    print("TRANSCRIPTS")
    print("=" * 84)
    for run in runs:
        label = f"{run.slice_ms}ms {'primed' if run.primed else 'raw   '}"
        print(f"  [{label}] {run.text!r}\n")

    print("=" * 84)
    print("WHAT THIS DECIDES")
    print("=" * 84)
    primed_runs = [r for r in runs if r.primed]
    raw_runs = [r for r in runs if not r.primed]
    primed_clean = [r for r in primed_runs if not r.foreign_scripts]
    raw_clean = [r for r in raw_runs if not r.foreign_scripts]

    print(
        f"  cells with NO foreign-script leak:  primed {len(primed_clean)}/{len(primed_runs)}"
        f"   raw {len(raw_clean)}/{len(raw_runs)}"
    )
    # Admissible = clean transcript AND inside the cabin lag budget. Either test alone gives
    # a half-answer: a clean cell that lands at 4 seconds is not a cabin, and a fast cell that
    # invents Chinese is not a transcript. Judging on scripts alone is what made the first
    # version of this report say "slice length governs" while missing the actual blocker.
    admissible = [
        r
        for r in runs
        if not r.foreign_scripts
        and r.implied_lag_ms is not None
        and r.implied_lag_ms <= CABIN_LAG_TARGET_MS
    ]
    timed = [r for r in runs if r.implied_lag_ms is not None]
    fastest = min(timed, key=lambda r: r.implied_lag_ms or 0.0, default=None)

    if admissible:
        best = min(admissible, key=lambda r: r.implied_lag_ms or 0.0)
        print(f"""
  {len(admissible)} cell(s) are BOTH clean and inside the {CABIN_LAG_TARGET_MS}ms budget.
  Best: {best.slice_ms}ms slices, primed={best.primed}, implied lag {best.implied_lag_ms:.0f}ms.
  → Chunked commits on this API CAN reach the cabin target. Each slice is final and
    disjoint, which is already the append-only output a TTS stage needs. Build on it.""")
    else:
        floor = f"{fastest.implied_lag_ms:.0f}ms" if fastest else "unknown"
        print(f"""
  NO cell is both clean and inside the {CABIN_LAG_TARGET_MS}ms budget. The fastest cell of
  any quality implies {floor}, and every clean cell is slower still.

  The decisive quantity is the CLOCK, not the transcript. `commit -> completed` costs
  roughly the same whatever the slice length, so shortening a slice buys latency that the
  fixed cost immediately takes back — while the shortened slice is exactly what destroys
  accuracy. The two requirements move against each other, and no slice length satisfies both.

  → A commit-gated STT cannot reach the cabin target at ANY slice length. Priming changes
    how good the transcript is, not what it costs to obtain, so priming cannot close this.
  → A true streaming ASR — interim results, no commit — is the design, and that is now a
    measurement rather than an argument.""")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--slices",
        default="1000,2000,3000",
        help="comma-separated slice lengths in ms (default 1000,2000,3000)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    slice_lengths = [int(v) for v in args.slices.split(",") if v.strip()]

    pcm, rate = _load_wav(args.wav)
    duration_s = len(pcm) / 2 / rate
    pcm_24k = _resample_pcm16(pcm, rate, REALTIME_SAMPLE_RATE)

    settings = STTSettings()
    model = args.model or settings.model
    calls = 1 + sum(2 * max(1, int(duration_s * 1000) // ms) for ms in slice_lengths)

    print(f"wav       : {args.wav}  ({duration_s:.2f}s @ {rate}Hz)")
    print(f"model     : {model}")
    print(f"language  : {args.language or '(auto)'}")
    print(f"matrix    : slices {slice_lengths} x primed/raw  + 1 baseline")
    print(f"calls     : ~{calls} transcription calls")

    if args.dry_run:
        print("\ndry run — no API calls made")
        return

    api_key = _resolve_key(settings.api_key)
    if not api_key:
        raise SystemExit("no STT_API_KEY or OPENAI_API_KEY in the environment or .env")

    stt = OpenAISTT(api_key=api_key, model=model)

    baseline = await run_baseline(stt, pcm_24k, args.language)
    runs: list[RunResult] = []
    for slice_ms in slice_lengths:
        for primed in (False, True):
            runs.append(await run_sliced(stt, pcm_24k, args.language, slice_ms, primed))

    _report(baseline, runs)


if __name__ == "__main__":
    asyncio.run(main())
