from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from metrics_exporter.metrics import collect_metrics
from shared.redis_client import RedisStreamClient

redis_client = RedisStreamClient()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await redis_client.connect()
    try:
        yield
    finally:
        await redis_client.disconnect()


app = FastAPI(title="WarpTalk Redis Metrics Exporter", lifespan=lifespan)


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
async def ready() -> dict[str, str]:
    try:
        await redis_client.redis.ping()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    return {"status": "ready"}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> PlainTextResponse:
    body = await collect_metrics(redis_client.redis)
    return PlainTextResponse(
        body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
