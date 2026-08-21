"""How far behind a live speaker does each STT provider actually run?

WHY THIS PROBE EXISTS
    The two probes before this one dumped audio into the socket as fast as it would go and
    measured `commit -> completed`. That answers a protocol question and nothing else: a
    listener does not experience "how long after I sent 8 seconds of audio did text arrive",
    they experience "how far behind the speaker is the caption right now".

    So this one PACES THE AUDIO AT 1x, exactly as a microphone would, and timestamps every
    transcript event against the audio clock. That is the number the cabin architecture is
    budgeted in, and it is not derivable from the earlier runs.

WHAT IT DELIBERATELY DOES NOT DO
    It never sends `commit` or `finalize` until the audio is exhausted. That IS the test:
    a provider that streams gives text while the speaker talks; a provider that does not
    gives nothing until the flush. `gpt-transcribe` is expected to produce a single burst at
    the end — it is included precisely as the negative control, since OpenAI's own docs say
    to use it "only when you specifically need transcription to begin after a committed
    audio turn".

PROVIDERS
    openai:<model>   e.g. openai:gpt-live-transcribe (the streaming model, released
                     2026-07-29), or openai:gpt-transcribe (production today, the control).
    cartesia:<model> manual-finalize websocket. NOTE: the SDK's type literals say
                     auto_finalize accepts ONLY `ink-2` (English), while `ink-whisper` — the
                     Vietnamese one — is manual_finalize only. Whether it emits interim text
                     WITHOUT a finalize is the whole question for Cartesia.
                     The SDK also types `language` as Literal['en'], so a non-English tag is
                     passed through `extra_query` to find out whether the wire accepts what
                     the type hint refuses.

THE METRICS, AND WHICH ONES ARE EXACT
    EXACT — no modelling:
      * first_text_ms   ms from the first audio frame to the first character of transcript.
      * tail_ms         ms from the LAST audio frame to the final transcript. What a listener
                        waits through after the speaker stops.
      * events          how many transcript updates arrived, i.e. the cadence.
      * streamed        whether ANY text arrived before the audio ended. The yes/no that
                        decides whether a provider is usable for a cabin at all.

    APPROXIMATE — labelled as such wherever it is printed:
      * word_lag_ms     for each event, `elapsed − (words_so_far / total_words) × duration`.
                        Assumes an even speaking rate, which is wrong in detail and adequate
                        in aggregate. Use it to compare providers on the same clip, never as
                        an absolute figure to quote.

THE ANSWER, MEASURED (2026-08-20, 8.72s of connected Vietnamese, --language vi)

      provider                      streamed?   1st text     tail   events   scripts
      openai:gpt-live-transcribe          YES     1309ms    872ms       46       han
      openai:gpt-transcribe                no     9730ms   1450ms       45         —
      cartesia:ink-whisper                YES     3396ms    207ms        3         —

    PRODUCTION IS ON THE WRONG MODEL, AND THIS IS THE PROOF.
    `gpt-transcribe` — what `STT_MODEL` is pinned to today — produced NOTHING until the
    audio ran out and was committed: first text at 9730ms against an 8720ms clip. It behaved
    exactly as OpenAI documents it. `gpt-live-transcribe` on the identical audio gave first
    text at 1309ms across 46 updates. That is a 7.4x improvement from a config line.

    CARTESIA `ink-whisper` STREAMS, AND `language=vi` WORKS DESPITE THE TYPE STUB.
    The SDK types manual_finalize's `language` as Literal['en']; passed through extra_query
    the wire accepted `vi` and returned Vietnamese. It also has the FASTEST TAIL of the three
    (207ms from last audio frame to final text, vs 872ms) and leaked no foreign script.
    Against that: only 3 events for 8.7s — far too coarse for a cabin — and the worst
    accuracy of the three on this clip ("Phần bạc anh đi ôm rồi" for "Phần backend thì ổn
    rồi"). Its `finalize` was NOT required to receive transcripts, which is the structural
    property OpenAI's commit lacks.

    NEITHER STREAMING PROVIDER IS CLEAN. `gpt-live-transcribe` still emitted Chinese
    (`sam來計劃展開動`) with `languages: ["vi"]` set. Pinning the language does not stop
    cross-script hallucination — consistent with the literature, where the fix is gating the
    audio, not the language flag.

    THE FULL DUB BUDGET, EVERY TERM MEASURED
      STT lag (gpt-live-transcribe)   ~1797ms   word-rate median, this probe, APPROX
      Translation                       685ms   p50, tools/probe_translate_ttfb, same day
      TTS first audio (warm)            180ms   p50, tools/probe_tts_first_audio, 2026-08-18
                                      -------
      Dub lag                          ~2.7s

    So a 1000ms dub target is NOT reachable with this cascade: it would need STT to answer
    within ~120ms, and the fastest streaming provider measured here takes 1309ms to say
    anything at all. Transcript, by contrast, IS near-realtime at 1309ms/46 updates.
    Note for framing: a human simultaneous interpreter's Ear-Voice Span is 2-4s, so ~2.7s
    is inside professional range even though it misses the stated target.

    CAVEATS: one clip, one run per provider, and the audio is Cartesia `sonic-3.5` output
    rather than a human recording. `first_text` and `tail` are exact; `word-rate lag` assumes
    an even speaking rate and this clip has no pauses, which flatters every provider equally.

USAGE
    uv run python -m tools.probe_streaming_lag --wav vi.wav --language vi \
        --provider openai:gpt-live-transcribe
    uv run python -m tools.probe_streaming_lag --wav vi.wav --language vi \
        --provider cartesia:ink-whisper
    uv run python -m tools.probe_streaming_lag --wav vi.wav --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from dataclasses import dataclass, field
from typing import Any, cast

from openai import AsyncOpenAI

from shared.config import STTSettings, TTSSettings
from stt_worker.model import REALTIME_SAMPLE_RATE, OpenAISTT, _resample_pcm16, _scripts_in
from tools.probe_commit_semantics import _load_wav, _resolve_key

# One frame of audio per tick. 20ms is what WebRTC uses and what LiveKit delivers, so pacing
# at anything else would measure a cadence production never sees.
FRAME_MS = 20

# A provider that has said nothing this long after the audio ran out has failed the run
# rather than been slow — production's own TRANSCRIBE_EVENT_TIMEOUT_S.
DRAIN_TIMEOUT_S = 15.0


@dataclass
class Event:
    """One transcript update, stamped against the audio clock."""

    at_ms: float
    text: str
    is_final: bool


@dataclass
class Run:
    provider: str
    duration_ms: float
    events: list[Event] = field(default_factory=list)
    audio_end_ms: float = 0.0
    error: str = ""

    @property
    def final_text(self) -> str:
        finals = [e.text for e in self.events if e.is_final]
        return (
            " ".join(finals) if finals else (self.events[-1].text if self.events else "")
        ).strip()

    @property
    def first_text_ms(self) -> float | None:
        return next((e.at_ms for e in self.events if e.text.strip()), None)

    @property
    def tail_ms(self) -> float | None:
        """From the last audio frame to the last transcript event."""
        return (self.events[-1].at_ms - self.audio_end_ms) if self.events else None

    @property
    def streamed(self) -> bool:
        """Did ANY text arrive while the speaker was still talking?"""
        return any(e.at_ms < self.audio_end_ms and e.text.strip() for e in self.events)

    def word_lags(self) -> list[float]:
        """Per-event lag, under an even-speaking-rate assumption. Approximate by construction."""
        total = len(self.final_text.split())
        if not total:
            return []
        out = []
        for e in self.events:
            spoken = len(e.text.split())
            covered = min(1.0, spoken / total) * self.duration_ms
            out.append(e.at_ms - covered)
        return out


async def _pace(pcm_24k: bytes, send: Any, started: float) -> float:
    """Feed `send` one 20ms frame at a time, in real time. Returns audio-end offset in ms."""
    frame_bytes = (REALTIME_SAMPLE_RATE * 2 * FRAME_MS) // 1000
    for i in range(0, len(pcm_24k), frame_bytes):
        target = started + (i / (REALTIME_SAMPLE_RATE * 2))
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await send(pcm_24k[i : i + frame_bytes])
    return (time.monotonic() - started) * 1000


# ----------------------------------------------------------------------------------
# OpenAI
# ----------------------------------------------------------------------------------


async def run_openai(model: str, pcm_24k: bytes, language: str | None, duration_ms: float) -> Run:
    run = Run(provider=f"openai:{model}", duration_ms=duration_ms)
    settings = STTSettings()
    stt = OpenAISTT(api_key=_resolve_key(settings.api_key), model=model)
    client = AsyncOpenAI(api_key=stt.api_key)
    manager = client.realtime.connect(extra_query={"intent": "transcription"})
    conn = await manager.__aenter__()
    try:
        await conn.session.update(
            session=cast(Any, stt._session_payload(language, None, None, None))
        )
        started = time.monotonic()
        acc = {"text": ""}

        async def reader() -> None:
            async for event in conn:
                etype = getattr(event, "type", "")
                at = (time.monotonic() - started) * 1000
                if etype.endswith("input_audio_transcription.delta"):
                    acc["text"] += getattr(event, "delta", "") or ""
                    run.events.append(Event(at, acc["text"], False))
                elif etype.endswith("input_audio_transcription.completed"):
                    text = (getattr(event, "transcript", "") or "").strip()
                    run.events.append(Event(at, text, True))
                    acc["text"] = ""
                elif etype == "error":
                    run.error = repr(getattr(event, "error", event))
                    return

        task = asyncio.create_task(reader())
        run.audio_end_ms = await _pace(
            pcm_24k,
            lambda b: conn.input_audio_buffer.append(audio=base64.b64encode(b).decode()),
            started,
        )
        # Only NOW flush — everything before this point had to arrive on the provider's own.
        await conn.input_audio_buffer.commit()
        try:
            await asyncio.wait_for(task, timeout=DRAIN_TIMEOUT_S)
        except TimeoutError:
            task.cancel()
        return run
    except Exception as exc:  # noqa: BLE001 — a probe reports failures, it does not raise them
        run.error = run.error or repr(exc)
        return run
    finally:
        await manager.__aexit__(None, None, None)


# ----------------------------------------------------------------------------------
# Cartesia
# ----------------------------------------------------------------------------------


async def run_cartesia(model: str, pcm_24k: bytes, language: str | None, duration_ms: float) -> Run:
    from cartesia import AsyncCartesia

    run = Run(provider=f"cartesia:{model}", duration_ms=duration_ms)
    client = AsyncCartesia(api_key=TTSSettings().api_key)
    # `language` is typed Literal['en'] in the SDK; ink-whisper's docs list `vi`. Pass it as a
    # query param so the WIRE decides, not the type stub.
    extra = {"language": language} if language else {}
    manager = client.stt.manual_finalize.websocket(
        encoding="pcm_s16le",
        model=cast(Any, model),
        sample_rate=REALTIME_SAMPLE_RATE,
        extra_query=cast(Any, extra),
    )
    conn = await manager.enter()
    try:
        started = time.monotonic()

        async def reader() -> None:
            # Responses are a discriminated union of pydantic models (transcript / flush_done /
            # done / error), so read them as attributes rather than dict keys.
            while True:
                msg = await conn.recv()
                at = (time.monotonic() - started) * 1000
                mtype = getattr(msg, "type", "")
                if mtype == "transcript":
                    text = getattr(msg, "text", "") or ""
                    if text.strip():
                        run.events.append(Event(at, text, bool(getattr(msg, "is_final", False))))
                elif mtype == "error":
                    run.error = str(msg)
                    return
                elif mtype == "done":
                    return

        task = asyncio.create_task(reader())
        run.audio_end_ms = await _pace(pcm_24k, conn.send, started)
        # Same contract as the OpenAI path: finalize only after the audio is gone. The request
        # type is a bare literal ("finalize" | "close"), not an object.
        await conn.send("finalize")
        try:
            await asyncio.wait_for(task, timeout=DRAIN_TIMEOUT_S)
        except TimeoutError:
            task.cancel()
        return run
    except Exception as exc:  # noqa: BLE001
        run.error = run.error or repr(exc)
        return run
    finally:
        with_close = getattr(conn, "close", None)
        if with_close:
            await with_close()


# ----------------------------------------------------------------------------------


def _report(runs: list[Run], target_ms: int) -> None:
    print("\n" + "=" * 88)
    print("STREAMING LAG — audio paced at 1x, no commit/finalize until the audio ran out")
    print("=" * 88)
    head = f"  {'provider':<34}  {'streamed?':>9}  {'1st text':>9}  {'tail':>8}"
    print(f"{head}  {'events':>6}  {'scripts':>8}")
    for r in runs:
        if r.error:
            print(f"  {r.provider:<34}  {'ERROR':>9}  {r.error[:60]}")
            continue
        first = f"{r.first_text_ms:.0f}ms" if r.first_text_ms is not None else "—"
        tail = f"{r.tail_ms:.0f}ms" if r.tail_ms is not None else "—"
        scripts = ",".join(sorted(_scripts_in(r.final_text))) or "—"
        print(
            f"  {r.provider:<34}  {('YES' if r.streamed else 'no'):>9}  {first:>9}  "
            f"{tail:>8}  {len(r.events):>6}  {scripts:>8}"
        )

    print("\n" + "=" * 88)
    print("TRANSCRIPTS")
    print("=" * 88)
    for r in runs:
        print(f"  [{r.provider}]")
        print(f"    {r.final_text!r}" if not r.error else f"    ERROR: {r.error}")
        lags = r.word_lags()
        if lags:
            mid = sorted(lags)[len(lags) // 2]
            print(f"    word-rate lag (APPROX): median {mid:.0f}ms over {len(lags)} events")
        print()

    print("=" * 88)
    print("WHAT THIS DECIDES")
    print("=" * 88)
    streaming = [r for r in runs if not r.error and r.streamed]
    if not streaming:
        print("""  No provider produced text before the speaker stopped. Every one tested is
  commit-gated in practice, whatever its docs say, and none can drive a cabin.""")
    else:
        best = min(streaming, key=lambda r: r.first_text_ms or 1e9)
        print(f"  {len(streaming)} provider(s) emitted text DURING speech.")
        print(f"  Earliest first text: {best.provider} at {best.first_text_ms:.0f}ms.")
        print(f"""
  For the dub budget, first-text is only the STT share. Against a {target_ms}ms target the
  remaining stages measured elsewhere are ~700ms translation + ~180ms TTS first audio, so
  an STT that answers later than ~{max(0, target_ms - 880)}ms cannot fit whatever else is done.""")
    print()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wav", required=True)
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--provider",
        action="append",
        default=None,
        help="repeatable: openai:<model> | cartesia:<model>",
    )
    parser.add_argument("--target-ms", type=int, default=1000, help="dub lag target (default 1000)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    providers = args.provider or [
        "openai:gpt-live-transcribe",
        "openai:gpt-transcribe",
        "cartesia:ink-whisper",
    ]

    pcm, rate = _load_wav(args.wav)
    duration_ms = len(pcm) / 2 / rate * 1000
    pcm_24k = _resample_pcm16(pcm, rate, REALTIME_SAMPLE_RATE)

    print(f"wav       : {args.wav}  ({duration_ms / 1000:.2f}s @ {rate}Hz)")
    print(f"language  : {args.language or '(auto)'}")
    print(f"providers : {', '.join(providers)}")
    print(
        f"pacing    : 1x realtime, {FRAME_MS}ms frames — each run takes ~{duration_ms / 1000:.0f}s"
    )

    if args.dry_run:
        print("\ndry run — no API calls made")
        return

    runs: list[Run] = []
    for spec in providers:
        vendor, _, model = spec.partition(":")
        print(f"\n… running {spec}")
        if vendor == "openai":
            runs.append(await run_openai(model, pcm_24k, args.language, duration_ms))
        elif vendor == "cartesia":
            runs.append(await run_cartesia(model, pcm_24k, args.language, duration_ms))
        else:
            raise SystemExit(f"unknown provider vendor: {vendor!r}")

    _report(runs, args.target_ms)


if __name__ == "__main__":
    asyncio.run(main())
