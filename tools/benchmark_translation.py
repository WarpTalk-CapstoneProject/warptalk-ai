"""Benchmark translation prompt/context variants on the hot Realtime pool."""

from __future__ import annotations

import argparse
import asyncio
import time

from redis.asyncio import Redis

from shared.config import TranslationSettings, WorkerSettings, resolve_openai_api_key
from translation_worker.translator import OpenAITranslator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--source", default="vi")
    parser.add_argument("--target", default="en")
    return parser.parse_args()


async def benchmark(args: argparse.Namespace) -> list[dict[str, object]]:
    settings = WorkerSettings()
    redis = Redis.from_url(
        settings.redis.url,
        password=settings.redis.password or None,
        decode_responses=True,
    )
    try:
        raw_context = await redis.get(f"translationRoom:{args.room}:meeting_context")
    finally:
        await redis.aclose()
    static_context = raw_context.decode() if isinstance(raw_context, bytes) else raw_context

    translation = TranslationSettings()
    translator = OpenAITranslator(
        api_key=resolve_openai_api_key(translation.api_key),
        model=translation.model,
        realtime_model=translation.realtime_model,
        realtime_reasoning_effort=translation.realtime_reasoning_effort,
        realtime_pool_size=1,
        realtime_timeout_seconds=translation.realtime_timeout_seconds,
        realtime_max_output_tokens=translation.realtime_max_output_tokens,
        max_tokens=translation.max_tokens,
        temperature=translation.temperature,
    )
    await translator.load()
    await translator.warm_up()
    results: list[dict[str, object]] = []
    try:
        for run in range(2):
            for mode, context in (
                ("full", [static_context] if static_context else []),
                ("none", []),
            ):
                started = time.monotonic()
                output = await translator.translate(
                    args.text,
                    args.source,
                    args.target,
                    meeting_context=context,
                )
                results.append(
                    {
                        "run": run + 1,
                        "mode": mode,
                        "context_chars": len(static_context or "") if mode == "full" else 0,
                        "latency_ms": int((time.monotonic() - started) * 1000),
                        "output": output,
                    }
                )
    finally:
        await translator.close()
    return results


if __name__ == "__main__":
    print(asyncio.run(benchmark(_parse_args())))
