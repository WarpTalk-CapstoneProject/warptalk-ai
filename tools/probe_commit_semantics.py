"""Does `input_audio_buffer.commit()` CONSUME the buffer, or does the buffer GROW?

WHY THIS PROBE EXISTS
    The cabin architecture needs the STT stage to emit revisable hypotheses while the speaker
    is still talking. Two designs can deliver that, and exactly one fact decides between them:

      A. ROLLING COMMIT on OpenAI Realtime — commit every ~500ms instead of at segment end.
         Viable ONLY if a commit leaves the previously appended audio in place, so each commit
         re-transcribes a GROWING buffer and consecutive transcripts can be compared
         (LocalAgreement: emit the longest common prefix of two consecutive hypotheses).

      B. SWITCH to a true streaming ASR (Nemotron 3.5 ASR Streaming / Deepgram Nova-3 /
         ElevenLabs Scribe v2 Realtime) that emits interim results with no commit at all.

    If commit CONSUMES the buffer, then rolling commit yields consecutive DISJOINT SLICES, not
    hypotheses. There is nothing for two reads to agree on, LocalAgreement does not apply, and
    design A is dead — the decision collapses to B before a line of it is written.

    `stt_worker/model.py:2076` has a separate `input_audio_buffer.clear()` for "whatever has
    been appended but never committed", which only makes sense if commit already drained the
    buffer. That is suggestive, not proof: it is one reading of one helper. This measures it.

WHY IT MATTERS THAT THIS IS CHEAP TO ANSWER
    Both designs cost weeks. This costs one afternoon and a handful of API calls, and it
    eliminates one of them outright. Run it before writing either.

THE ANSWER, MEASURED (2026-08-20, gpt-transcribe, 8.72s of connected Vietnamese, --language vi)
    E1  commit CONSUMES the buffer. Commit 2 returned only the second half; it did not
        contain a word of commit 1. Design A is dead — there is no growing window to
        compare, so LocalAgreement cannot be built on this API at any commit rate.

    E2  0/3 commits had `completed` contradict its own deltas. So the deltas do not revise
        either: even delta-level agreement has nothing to work with. The
        `stt_delta_final_mismatch` guard at model.py:1694 is defensive, not evidence of a
        common case — this clip never triggered it.

    E3  The gate costs, commit -> completed: 1630ms for one 8.72s commit, 716ms mean for
        1s rolling commits (467ms mean to first delta). Note the floor: even a 1-second
        slice costs ~700ms to come back, so rolling commits do not amortise — each one pays
        most of the fixed cost again.

    E4  Slicing does not merely mangle boundaries, it makes the model hallucinate a
        DIFFERENT LANGUAGE. Same audio, same session, same `vi` hint:

          single  : "Mình sẽ xem lại cái hoạt triển khai tuần sau trước khi hết giờ. Phần
                     backend thì ổn rồi, vấn đề nằm ở phía client nên bạn gửi tài liệu cho
                     cả nhóm sau cuộc họp này nhé."
          rolling : "Mình sẽ xem lại. 因發展開動。 Ow, jerky heads are. From backend. Thì ôm
                     rồi, vâng. They name a fear client. Nên bạn gửi tay. Tài liệu cho cả
                     nhóm sau. Cuộc họp này nhé."

        A 1-second slice carries no linguistic context, so the model guesses — and guesses
        across languages. This is the same failure `_filter_segments` exists to catch, and
        here it would arrive at a rate no filter should be asked to absorb.

    CAVEAT ON THE FIXTURE: the audio was Cartesia `sonic-3.5` output, not a human recording,
    because the repo carries no speech fixture. E1 is a protocol fact and is unaffected by
    that. E4's severity might not be: synthetic speech may be easier OR harder to hallucinate
    over than a real microphone. Re-run on a real recording before quoting E4 as a rate.

WHAT ELSE FALLS OUT OF THE SAME RUN
    E2  Do the deltas of ONE commit revise, or only append?
        `model.py:1694` already carries a guard for the model revising inside the
        already-flushed delta prefix ("stt_delta_final_mismatch"), so revision is known to
        happen — this measures how often, which is what decides whether delta-level
        LocalAgreement is worth anything on its own.

    E3  What does the commit gate actually cost? commit -> first delta, commit -> completed.
        Flash mode (`append_streamed_audio`) uploads audio early but does NOT open this gate:
        the model starts decoding at commit. This is the number that says what a rolling
        commit would cost per commit, and what removing the gate entirely would save.

    E4  Rolling commit vs single commit over the SAME audio. Even if the buffer grows, slicing
        may mangle word boundaries. This compares the concatenated rolling transcript against
        the single-commit transcript on identical input.

METHOD
    Reuses the PRODUCTION session config (`OpenAISTT._session_payload`) and the production
    resampler, so what is measured is the session this pipeline actually opens — not a
    hand-rolled approximation that could differ in a way that changes the answer. Only the
    append/commit TIMING is driven by this file, because that is the whole subject.

    No Redis, no LiveKit. Needs OPENAI_API_KEY and one wav of real speech.

USAGE
    uv run python -m tools.probe_commit_semantics --wav sample-vi.wav --language vi
    uv run python -m tools.probe_commit_semantics --wav sample-vi.wav --dry-run

    `--dry-run` prints the exact call count and billed seconds for your clip before spending
    anything; a 6s clip at the default 1000ms slices is 9 calls / ~39s of billed audio. Use
    real connected speech, not a single word: the whole question is about word boundaries.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import os
import time
import wave
from dataclasses import dataclass, field
from typing import Any, cast

from openai import AsyncOpenAI

from shared.config import STTSettings
from stt_worker.model import REALTIME_SAMPLE_RATE, OpenAISTT, _resample_pcm16

# Production's own bound (TRANSCRIBE_EVENT_TIMEOUT_S). A commit that has not completed by then
# is a failure, not a slow success, and hanging here would hide the result behind a wedged run.
EVENT_TIMEOUT_S = 15.0

# One append may carry at most this much raw PCM, mirroring model.py:1623. The Realtime API
# rejects oversized frames, and matching production means a size limit can never be the reason
# this probe and the pipeline disagree.
APPEND_BYTES = REALTIME_SAMPLE_RATE * 2 * 2


@dataclass
class CommitResult:
    """Everything one commit produced, with the clock started at the commit call."""

    deltas: list[str] = field(default_factory=list)
    delta_ms: list[float] = field(default_factory=list)
    completed_text: str = ""
    first_delta_ms: float | None = None
    completed_ms: float | None = None

    @property
    def delta_concat(self) -> str:
        return "".join(self.deltas).strip()

    @property
    def revised(self) -> bool:
        """Did `completed` differ from what the deltas already said?

        Not an equality check on the whole string: trailing punctuation and whitespace differ
        harmlessly. A revision is the delta stream having claimed something the final transcript
        then contradicts — which is exactly what a prefix test detects and an equality test
        would drown in noise.
        """
        concat = self.delta_concat
        if not concat:
            return False
        return not self.completed_text.startswith(concat)


def _resolve_key(stage_key: str) -> str:
    """The same precedence `shared.config.resolve_openai_api_key` applies, plus a `.env` read.

    The production workers get their environment injected by compose, so `resolve_openai_api_key`
    can rely on `os.environ` alone. A probe run from a shell cannot, and `STT_API_KEY` is
    routinely empty in `.env` (it is `OPENAI_API_KEY` that carries the value) — reading only the
    stage key would report "no key" while a perfectly good one sat two lines below it in the
    same file. Same pattern as tools/probe_tts_first_audio.py.
    """
    for candidate in (stage_key, os.getenv("STT_API_KEY", ""), os.getenv("OPENAI_API_KEY", "")):
        if candidate:
            return candidate
    try:
        with open(".env") as handle:
            values = dict(
                line.strip().split("=", 1)
                for line in handle
                if "=" in line and not line.lstrip().startswith("#")
            )
    except OSError:
        return ""
    return values.get("STT_API_KEY", "") or values.get("OPENAI_API_KEY", "")


def _load_wav(path: str) -> tuple[bytes, int]:
    """Read a wav into raw PCM16 mono plus its sample rate."""
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM, got {handle.getsampwidth() * 8}-bit")
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels == 1:
        return frames, rate
    # Downmix by taking the left channel rather than averaging: averaging two channels of a
    # phase-shifted room recording attenuates exactly the speech this probe is about.
    mono = bytearray()
    for i in range(0, len(frames), 4):
        mono += frames[i : i + 2]
    return bytes(mono), rate


async def _open_session(stt: OpenAISTT, language: str | None) -> tuple[Any, Any]:
    """Open a transcription session using the production payload."""
    client = AsyncOpenAI(api_key=stt.api_key)
    manager = client.realtime.connect(extra_query={"intent": "transcription"})
    conn = await manager.__aenter__()
    await conn.session.update(session=cast(Any, stt._session_payload(language, None, None, None)))
    return manager, conn


async def _append(conn: Any, pcm_24k: bytes) -> None:
    for i in range(0, len(pcm_24k), APPEND_BYTES):
        frame = pcm_24k[i : i + APPEND_BYTES]
        await conn.input_audio_buffer.append(audio=base64.b64encode(frame).decode())


async def _commit_and_collect(conn: Any) -> CommitResult:
    """Commit, then read events until `completed`, timing everything from the commit."""
    result = CommitResult()
    started = time.monotonic()
    await conn.input_audio_buffer.commit()

    async def _read() -> None:
        async for event in conn:
            etype = getattr(event, "type", "")
            if etype == "conversation.item.input_audio_transcription.delta":
                delta = getattr(event, "delta", "") or ""
                if not delta:
                    continue
                elapsed = (time.monotonic() - started) * 1000
                if result.first_delta_ms is None:
                    result.first_delta_ms = elapsed
                result.deltas.append(delta)
                result.delta_ms.append(elapsed)
            elif etype == "conversation.item.input_audio_transcription.completed":
                result.completed_text = (getattr(event, "transcript", "") or "").strip()
                result.completed_ms = (time.monotonic() - started) * 1000
                return
            elif etype == "error":
                raise RuntimeError(f"realtime error: {getattr(event, 'error', event)!r}")

    await asyncio.wait_for(_read(), timeout=EVENT_TIMEOUT_S)
    return result


# ----------------------------------------------------------------------------------
# E1 + E2 + E3 — two commits on one session, second appended AFTER the first committed
# ----------------------------------------------------------------------------------


async def probe_commit_semantics(
    stt: OpenAISTT, pcm_24k: bytes, language: str | None, split_ms: int
) -> tuple[CommitResult, CommitResult]:
    """Append the first half, commit, append the second half, commit.

    If commit CONSUMES, the second transcript describes only the second half.
    If the buffer GROWS, the second transcript describes the whole clip and therefore starts
    with the first transcript's text.
    """
    split_bytes = (REALTIME_SAMPLE_RATE * 2 * split_ms) // 1000
    split_bytes -= split_bytes % 2  # never split mid-sample
    first, second = pcm_24k[:split_bytes], pcm_24k[split_bytes:]
    if not second:
        raise SystemExit("--split-ms lands past the end of the clip; use a longer wav")

    manager, conn = await _open_session(stt, language)
    try:
        await _append(conn, first)
        head = await _commit_and_collect(conn)
        await _append(conn, second)
        tail = await _commit_and_collect(conn)
        return head, tail
    finally:
        await manager.__aexit__(None, None, None)


# ----------------------------------------------------------------------------------
# E4 — rolling commit vs one commit, same audio
# ----------------------------------------------------------------------------------


async def probe_single_commit(stt: OpenAISTT, pcm_24k: bytes, language: str | None) -> CommitResult:
    manager, conn = await _open_session(stt, language)
    try:
        await _append(conn, pcm_24k)
        return await _commit_and_collect(conn)
    finally:
        await manager.__aexit__(None, None, None)


async def probe_rolling_commit(
    stt: OpenAISTT, pcm_24k: bytes, language: str | None, slice_ms: int
) -> list[CommitResult]:
    slice_bytes = (REALTIME_SAMPLE_RATE * 2 * slice_ms) // 1000
    slice_bytes -= slice_bytes % 2
    manager, conn = await _open_session(stt, language)
    results: list[CommitResult] = []
    try:
        for i in range(0, len(pcm_24k), slice_bytes):
            piece = pcm_24k[i : i + slice_bytes]
            if len(piece) < REALTIME_SAMPLE_RATE // 10:  # <100ms tail — the API rejects these
                break
            await _append(conn, piece)
            results.append(await _commit_and_collect(conn))
        return results
    finally:
        await manager.__aexit__(None, None, None)


# ----------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:7.0f}ms"


def _report(
    head: CommitResult,
    tail: CommitResult,
    single: CommitResult,
    rolling: list[CommitResult],
    slice_ms: int,
) -> None:
    # Compare against a PREFIX of the first transcript, not all of it: if the buffer grows, the
    # second read may legitimately reword the tail of what the first read said (that is E2's
    # subject) while still clearly containing its opening. A whole-string test would call that
    # "consumes" and get the architecture decision backwards.
    marker = head.completed_text[: max(8, len(head.completed_text) // 2)]
    consumes = not tail.completed_text.startswith(marker)

    print("\n" + "=" * 78)
    print("E1 — DOES commit() CONSUME THE BUFFER?")
    print("=" * 78)
    print(f"  commit 1 (first half) : {head.completed_text!r}")
    print(f"  commit 2 (second half): {tail.completed_text!r}")
    print()
    if consumes:
        print("  → commit 2 does NOT contain commit 1's text.")
        print("  → VERDICT: commit CONSUMES the buffer.")
    else:
        print("  → commit 2 STARTS WITH commit 1's text — the buffer carried over.")
        print("  → VERDICT: the buffer GROWS across commits.")

    print("\n" + "=" * 78)
    print("E2 — DO THE DELTAS OF ONE COMMIT REVISE, OR ONLY APPEND?")
    print("=" * 78)
    for label, res in (("first", head), ("second", tail), ("single", single)):
        print(f"  {label:6} deltas={len(res.deltas):3d}  revised={'YES' if res.revised else 'no'}")
        if res.revised:
            print(f"         delta concat : {res.delta_concat!r}")
            print(f"         completed    : {res.completed_text!r}")
    revisions = sum(1 for r in (head, tail, single) if r.revised)
    print(f"\n  → {revisions}/3 commits had the final transcript contradict its own deltas.")
    print("    (model.py:1694 already guards this case — this is its rate, on this clip.)")

    print("\n" + "=" * 78)
    print("E3 — WHAT DOES THE COMMIT GATE COST?")
    print("=" * 78)
    print(f"  {'commit':>8}  {'->first delta':>14}  {'->completed':>13}")
    for label, res in (("first", head), ("second", tail), ("single", single)):
        print(f"  {label:>8}  {_ms(res.first_delta_ms):>14}  {_ms(res.completed_ms):>13}")
    if rolling:
        firsts = [r.first_delta_ms for r in rolling if r.first_delta_ms is not None]
        comps = [r.completed_ms for r in rolling if r.completed_ms is not None]
        if firsts:
            mean_first = _ms(sum(firsts) / len(firsts))
            mean_comp = _ms(sum(comps) / len(comps) if comps else None)
            print(f"  {'rolling':>8}  {mean_first:>14}  {mean_comp:>13}  (mean of {len(rolling)})")

    print("\n" + "=" * 78)
    print(f"E4 — ROLLING COMMIT ({slice_ms}ms SLICES) vs ONE COMMIT, SAME AUDIO")
    print("=" * 78)
    joined = " ".join(r.completed_text for r in rolling if r.completed_text).strip()
    print(f"  single  : {single.completed_text!r}")
    print(f"  rolling : {joined!r}")
    print(
        f"  slices  : {len(rolling)}   words single={len(single.completed_text.split())} "
        f"rolling={len(joined.split())}"
    )

    print("\n" + "=" * 78)
    print("WHAT THIS DECIDES")
    print("=" * 78)
    if consumes:
        print("""  Rolling commit on OpenAI Realtime yields DISJOINT SLICES, not hypotheses.
  Two consecutive reads describe different audio, so there is nothing for them to
  agree on and LocalAgreement cannot be built on this API.

  → Design A (rolling commit, keep OpenAI) is DEAD.
  → Go straight to design B: a true streaming ASR that emits interim results with
    no commit — Nemotron 3.5 ASR Streaming, Deepgram Nova-3, or ElevenLabs Scribe
    v2 Realtime. All three are also cheaper per audio-hour than gpt-transcribe.
  → Compare E4's two transcripts before accepting this: mangled word boundaries at
    the slice edges are the visible symptom, and they are what a listener would hear.""")
    else:
        print("""  The buffer survives a commit, so each commit re-transcribes a GROWING window and
  consecutive transcripts ARE comparable.

  → Design A (rolling commit + LocalAgreement on OpenAI) is VIABLE.
  → Its price is E3's "->completed" figure paid once per commit, times the commit
    rate. At 500ms commits that is ~12x the current call count on a 6s utterance;
    weigh it against STT being ~12% of provider spend.
  → Design B is still worth pricing: it removes the gate entirely rather than
    paying to reopen it repeatedly.""")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True, help="16-bit PCM wav of real connected speech")
    parser.add_argument("--language", default=None, help="language hint, e.g. vi")
    parser.add_argument("--model", default=None, help="override STT_MODEL")
    parser.add_argument(
        "--split-ms",
        type=int,
        default=3000,
        help="where E1 splits the clip into two commits (default 3000)",
    )
    parser.add_argument(
        "--slice-ms", type=int, default=1000, help="E4 rolling commit interval (default 1000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the wav and print the plan without calling the API",
    )
    args = parser.parse_args()

    pcm, rate = _load_wav(args.wav)
    duration_s = len(pcm) / 2 / rate
    pcm_24k = _resample_pcm16(pcm, rate, REALTIME_SAMPLE_RATE)

    settings = STTSettings()
    model = args.model or settings.model
    slices = max(1, int(duration_s * 1000) // args.slice_ms)
    billed = duration_s * 2 + duration_s + (duration_s * (slices + 1) / 2)

    print(f"wav       : {args.wav}  ({duration_s:.2f}s @ {rate}Hz -> {REALTIME_SAMPLE_RATE}Hz)")
    print(f"model     : {model}")
    print(f"language  : {args.language or '(auto)'}")
    print(f"plan      : E1 2 commits · E4 1 single + {slices} rolling")
    print(f"billed    : ~{billed:.0f}s of audio across {3 + slices} transcription calls")

    if args.dry_run:
        print("\ndry run — no API calls made")
        return

    api_key = _resolve_key(settings.api_key)
    if not api_key:
        raise SystemExit("no STT_API_KEY or OPENAI_API_KEY in the environment or .env")

    stt = OpenAISTT(api_key=api_key, model=model)

    head, tail = await probe_commit_semantics(stt, pcm_24k, args.language, args.split_ms)
    single = await probe_single_commit(stt, pcm_24k, args.language)
    rolling = await probe_rolling_commit(stt, pcm_24k, args.language, args.slice_ms)

    _report(head, tail, single, rolling, args.slice_ms)


if __name__ == "__main__":
    asyncio.run(main())
