"""Probe which context fields and confidence signals each STT model actually supports.

WHY
---
stt_worker/model.py:_session_payload branches on a hardcoded model set:

    is_next_generation_transcribe = self.model in {"gpt-transcribe", "gpt-live-transcribe"}

Only those two receive `keywords` (structured glossary) and `languages` (plural, the
multi-language hint that makes code-switching work). Production runs
gpt-4o-mini-transcribe, which takes the other branch and gets `language` SINGULAR — it
pins the model to one language, which is the mechanism that turns an English proper noun
like "Codex" into a Vietnamese phonetic rendering ("cô đích").

The same branch also decides confidence: the `include` logprobs selector is sent ONLY to
models outside that set, because the comment states the gpt-transcribe family does not
expose confidence scores. That matters because two filters in this repo now gate on
avg_logprob (_BLOCKLIST_MARGINAL_LOGPROB and min_avg_logprob). If a model returns no
logprobs, every segment reads as marginal and the blocklist starts deleting ordinary
speech again — "Okay", "Ừ", "Xin chào".

So the model choice is a three-way trade nobody can resolve from the docs, which do not
document logprob support at all. This script answers it against the live API.

WHAT IT DOES
------------
For each candidate model: opens a real transcription session through the project's own
OpenAISTT class (not a parallel reimplementation, so the answer applies to production),
transcribes one short clip containing an English proper noun inside Vietnamese speech,
and reports:

    * did the session accept `keywords` / `languages` / `prompt`?
    * did the completed event carry token logprobs, or the -1.0 sentinel?
    * how was the proper noun actually transcribed?

COST AND CREDENTIALS
--------------------
Reads OPENAI_API_KEY from warptalk-ai/.env (via the same load_dotenv() the workers use).
It is never printed. Generates one short TTS clip per run (reused across models) and
makes one transcription call per model — a handful of cents in total.

USAGE
    uv run python -m tools.stt_capability_probe
    uv run python -m tools.stt_capability_probe --audio path/to/real_recording.wav

Prefer --audio with a genuine recording of someone saying the phrase: synthesised speech
is cleaner than a real meeting and will make every model look better than it is.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import wave
from dataclasses import dataclass

from openai import AsyncOpenAI

from shared.config import resolve_openai_api_key
from stt_worker.model import OpenAISTT

# Candidates, in the order the decision actually cares about.
CANDIDATES = [
    "gpt-4o-mini-transcribe",  # what production runs today
    "gpt-transcribe",          # what shared/config.py defaults to
    "gpt-live-transcribe",     # tunable latency, $0.017/min
    "gpt-realtime-whisper",    # streaming counterpart
]

# A Vietnamese sentence carrying an English product name — the exact failure the
# glossary is meant to prevent. Keep the proper noun in `KEYWORDS` so the probe also
# shows whether supplying it as a structured keyword changes the outcome.
PROBE_SENTENCE = "Mình dùng Codex để review code trước khi deploy lên staging."
PROPER_NOUNS = ["Codex", "WarpTalk", "Kubernetes"]
KEYWORDS = [*PROPER_NOUNS, "staging", "deploy"]
PROMPT = "WarpTalk engineering meeting. Product names: Codex, WarpTalk, Kubernetes."


@dataclass
class ProbeResult:
    model: str
    session_ok: bool
    transcript: str = ""
    avg_logprob: float | None = None
    error: str = ""

    @property
    def has_logprobs(self) -> bool:
        return self.avg_logprob is not None and self.avg_logprob != -1.0

    @property
    def heard_proper_nouns(self) -> list[str]:
        lowered = self.transcript.casefold()
        return [noun for noun in PROPER_NOUNS if noun.casefold() in lowered]


async def synthesise_probe_audio(client: AsyncOpenAI, path: str) -> None:
    """Generate the probe clip once, as 24 kHz mono PCM wrapped in a WAV container."""
    response = await client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=PROBE_SENTENCE,
        response_format="wav",
    )
    with open(path, "wb") as handle:
        handle.write(response.content)


def read_wav_pcm(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise SystemExit(
                f"{path}: need 16-bit mono PCM, got "
                f"{handle.getsampwidth() * 8}-bit / {handle.getnchannels()}ch"
            )
        return handle.readframes(handle.getnframes()), handle.getframerate()


async def probe_model(model: str, pcm: bytes, sample_rate: int, api_key: str) -> ProbeResult:
    stt = OpenAISTT(api_key=api_key, model=model)
    try:
        await stt.load()
    except Exception as exc:  # noqa: BLE001 - the failure itself is the finding
        return ProbeResult(model, session_ok=False, error=f"load failed: {exc!r}")

    try:
        segments = await stt.transcribe(
            pcm,
            sample_rate=sample_rate,
            language="vi",
            meeting_id="probe",
            speaker_id="probe",
            allowed_languages={"vi", "en"},
            prompt=PROMPT,
            keywords=KEYWORDS,
        )
    except TypeError as exc:
        # transcribe()'s signature differs by version; surface it rather than guessing.
        return ProbeResult(model, session_ok=False, error=f"signature mismatch: {exc}")
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(model, session_ok=False, error=repr(exc)[:200])
    finally:
        try:
            await stt.close()
        except Exception:
            pass

    if not segments:
        return ProbeResult(
            model,
            session_ok=True,
            error="session accepted the config but returned no surviving segment "
            "(the filter chain may have dropped it — rerun with the audit tool)",
        )

    best = segments[0]
    return ProbeResult(
        model,
        session_ok=True,
        transcript=best.text,
        avg_logprob=best.confidence,
    )


async def run(audio_path: str | None) -> None:
    api_key = resolve_openai_api_key()
    if not api_key:
        raise SystemExit(
            "No OPENAI_API_KEY. Put it in warptalk-ai/.env or export it; it is never printed."
        )

    client = AsyncOpenAI(api_key=api_key)
    generated = False
    if audio_path is None:
        audio_path = "/tmp/warptalk_stt_probe.wav"
        print(f"synthesising probe audio -> {audio_path}")
        print("  NOTE: synthetic speech is cleaner than a real meeting. Treat these")
        print("        numbers as an upper bound, and rerun with --audio when you can.\n")
        await synthesise_probe_audio(client, audio_path)
        generated = True

    pcm, sample_rate = read_wav_pcm(audio_path)
    duration = len(pcm) / 2 / sample_rate
    print(f'probe phrase : "{PROBE_SENTENCE}"')
    print(f"audio        : {audio_path} ({duration:.1f}s @ {sample_rate} Hz)")
    print(f"keywords     : {KEYWORDS}\n")

    results = []
    for model in CANDIDATES:
        print(f"probing {model} ...")
        results.append(await probe_model(model, pcm, sample_rate, api_key))

    print("\n=== capability matrix ===\n")
    print(f"{'model':<26} {'session':<9} {'logprobs':<10} {'proper nouns heard'}")
    for r in results:
        if not r.session_ok:
            print(f"{r.model:<26} {'FAILED':<9} {'-':<10} {r.error[:60]}")
            continue
        logprobs = f"yes ({r.avg_logprob:.3f})" if r.has_logprobs else "NO (sentinel)"
        heard = ", ".join(r.heard_proper_nouns) or "none"
        print(f"{r.model:<26} {'ok':<9} {logprobs:<10} {heard}")

    print("\n=== transcripts ===\n")
    for r in results:
        print(f"  {r.model}")
        print(f"    {r.transcript or '(none)  ' + r.error}")

    print("\n=== how to read this ===\n")
    print("  logprobs NO  -> _BLOCKLIST_MARGINAL_LOGPROB and min_avg_logprob both go blind")
    print("                  on that model; every segment reads as marginal and the")
    print("                  blocklist deletes ordinary speech again. Use the")
    print("                  duration/no-speech fallback in _filter_segments instead.")
    print("  proper nouns -> a model that heard 'Codex' with keywords supplied, and a")
    print("                  model that did not, is the whole argument for the")
    print("                  keywords-capable branch of _session_payload.")

    if generated:
        os.unlink(audio_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", help="16-bit mono WAV of the probe phrase")
    args = parser.parse_args()
    asyncio.run(run(args.audio))


if __name__ == "__main__":
    main()
