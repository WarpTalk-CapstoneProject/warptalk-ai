"""Benchmark realtime transcription models on a retained deterministic PCM chunk."""

from __future__ import annotations

import argparse
import asyncio
import time

from redis.asyncio import Redis

from shared.config import STTSettings, WorkerSettings, resolve_openai_api_key
from shared.schemas import AudioChunkMessage
from stt_worker.model import OpenAISTT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument("--offset-ms", type=int, default=0)
    parser.add_argument("--duration-ms", type=int, default=3000)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-transcribe", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
    )
    return parser.parse_args()


async def _load_fixture(
    room: str,
    stream_id: str,
    offset_ms: int,
    duration_ms: int,
) -> tuple[bytes, int, str | None]:
    settings = WorkerSettings()
    redis = Redis.from_url(
        settings.redis.url,
        password=settings.redis.password or None,
        decode_responses=False,
    )
    try:
        entries = await redis.xrange("audio:chunks", min=stream_id, max=stream_id, count=1)
        prompt = await redis.get(f"translationRoom:{room}:stt_prompt")
    finally:
        await redis.aclose()
    if not entries or entries[0][1] is None:
        raise RuntimeError(f"Redis stream entry not found: {stream_id}")
    chunk = AudioChunkMessage.from_redis(entries[0][1])
    start = offset_ms * chunk.sample_rate // 1000 * 2
    end = start + duration_ms * chunk.sample_rate // 1000 * 2
    prompt_text = prompt.decode() if isinstance(prompt, bytes) else prompt
    return chunk.audio_data[start:end], chunk.sample_rate, prompt_text


async def benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    audio, sample_rate, prompt = await _load_fixture(
        args.room,
        args.stream_id,
        args.offset_ms,
        args.duration_ms,
    )
    api_key = resolve_openai_api_key(STTSettings().api_key)
    results: list[dict[str, object]] = []
    for model_name in args.models:
        model = OpenAISTT(api_key=api_key, model=model_name)
        await model.load()
        await model.warm_up(pool_size=1)
        speaker = f"qa-benchmark-{model_name}"
        await model.prepare_session(
            args.room,
            speaker,
            language="vi",
            prompt=prompt,
        )
        try:
            for run in range(2):
                started = time.monotonic()
                segments = await model.transcribe(
                    audio,
                    sample_rate=sample_rate,
                    language="vi",
                    meeting_id=args.room,
                    speaker_id=speaker,
                    prompt=prompt,
                    allowed_languages={"vi", "en"},
                )
                results.append(
                    {
                        "model": model_name,
                        "run": run + 1,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "text": " ".join(segment.text for segment in segments),
                        "confidence": [segment.confidence for segment in segments],
                    }
                )
        finally:
            await model.close()
    return results


if __name__ == "__main__":
    print(asyncio.run(benchmark(_parse_args())))
