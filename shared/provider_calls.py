"""Per-provider call outcomes and latency, per UTC hour, for the admin Providers page.

WHY THIS EXISTS
    `warptalk:outcome:{stage}` says whether a pipeline STAGE worked. It cannot say which vendor
    failed or how: a Cartesia 402 (credits spent), an OpenAI 429 (rate limit) and a bug in our
    own code are all "vendor_error" or "error" there, and it keeps one running total per stage,
    so it cannot answer "how was OpenAI on the 3rd of August" either. The Providers page needs
    exactly that — a 90-day row of daily bars per vendor, and a live success rate — from what
    our own calls saw, not from a vendor's marketing status page.

WHERE IT IS RECORDED
    - Every OpenAI HTTP call (chat, responses, embeddings, transcription, moderation) goes
      through `observed_openai_http_client`, and every Cartesia HTTP call (voice cloning, the
      voice catalog, the bytes fallback) through `observed_cartesia_http_client`: each is an
      httpx transport that times the call to its response headers and classifies the status.
      The SDK's own retries are separate calls here, which is the point: a 429 that the SDK
      retried into a 200 still happened.
    - Websocket calls (Cartesia TTS, OpenAI realtime STT) have no httpx request to observe; the
      worker that owns the call records it with `record_provider_call` where it already knows
      the outcome.

THE REDIS CONTRACT (read by billing-service ProviderCallStatsSyncWorker — keep in step)
    key    warptalk:provider_calls:{YYYY-MM-DD}            one hash per UTC day, TTL 3 days
    field  {provider}|{HH}|{operation}|{model}|{outcome}  count of calls with that outcome
           {provider}|{HH}|{operation}|{model}|lat:{le}    latency histogram bucket (not cumulative)
           {provider}|{HH}|{operation}|{model}|lat_sum     Σ latency in ms
    HH is the UTC hour (00-23), model is "-" when unknown. The billing sync copies the hash into
    Postgres every few minutes; the TTL only has to outlive a missed sync or two.

Best effort throughout: nothing here may fail or slow down the call it measures.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from shared.logger import get_logger

if TYPE_CHECKING:
    from shared.redis_client import RedisStreamClient

logger = get_logger(__name__)

PROVIDER_CALLS_KEY_PREFIX = "warptalk:provider_calls:"
PROVIDER_CALLS_KEY_TTL_SECONDS = 3 * 24 * 60 * 60

#: Every outcome a provider call can have. `ok` is any 1xx-3xx; the rest are failures, split by
#: whose problem they are: `quota` (402), `rate_limited` (429) and `auth` (401/403) are the
#: account; `server_error` (5xx), `timeout` and `network_error` are the vendor or the path to it;
#: `client_error` (any other 4xx) is almost always our request; `error` is an exception that
#: carried no status at all.
PROVIDER_CALL_OUTCOMES = (
    "ok",
    "quota",
    "rate_limited",
    "auth",
    "client_error",
    "server_error",
    "timeout",
    "network_error",
    "error",
)

#: Finer at the bottom than the stage ladder: a healthy embeddings or chat call returns in well
#: under a second, and a ladder starting at 250ms would put every one of them in one bucket.
PROVIDER_LATENCY_BUCKETS_MS = (100, 250, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000)

_SEPARATOR = "|"
_UNSAFE = re.compile(r"[|\s]+")
#: A body bigger than this is not parsed for its model name (audio uploads, long prompts).
_MAX_MODEL_SNIFF_BYTES = 256 * 1024
_STATUS_IN_MESSAGE = re.compile(r"(?:status(?:[ _]code)?|http)[\s:=]*([1-5]\d\d)\b", re.IGNORECASE)


def _label(value: str | None) -> str:
    cleaned = _UNSAFE.sub("-", (value or "").strip())[:80]
    return cleaned or "-"


def classify_status(status: int) -> str:
    """The outcome of a call that got an HTTP answer."""
    if status < 400:
        return "ok"
    if status == 402:
        return "quota"
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth"
    if status >= 500:
        return "server_error"
    return "client_error"


def classify_exception(exc: BaseException) -> str:
    """The outcome of a call that raised. Status first (SDK errors carry one), then the kind."""
    for attribute in ("status_code", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return classify_status(value)
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int) and 100 <= status <= 599:
        return classify_status(status)

    name = type(exc).__name__.lower()
    if (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException))
        or "timeout" in name
    ):
        return "timeout"

    # Websocket SDKs put the status in the message ("status code 402", "HTTP 429").
    message = str(exc)
    match = _STATUS_IN_MESSAGE.search(message)
    if match:
        return classify_status(int(match.group(1)))
    lowered = message.lower()
    if "payment required" in lowered or "insufficient credits" in lowered or "quota" in lowered:
        return "quota"
    if "rate limit" in lowered or "too many requests" in lowered:
        return "rate_limited"

    if isinstance(exc, (httpx.TransportError, ConnectionError, OSError)) or "connection" in name:
        return "network_error"
    return "error"


def latency_bucket(latency_ms: int) -> str:
    return next((str(edge) for edge in PROVIDER_LATENCY_BUCKETS_MS if latency_ms <= edge), "+Inf")


def provider_calls_key(at: datetime) -> str:
    return f"{PROVIDER_CALLS_KEY_PREFIX}{at.astimezone(UTC):%Y-%m-%d}"


def provider_call_fields(
    provider: str,
    operation: str,
    model: str | None,
    outcome: str,
    latency_ms: int | None,
    at: datetime,
) -> list[tuple[str, int]]:
    """The (field, increment) pairs one call adds to its day's hash. Pure, for the tests."""
    if outcome not in PROVIDER_CALL_OUTCOMES:
        outcome = "error"
    prefix = _SEPARATOR.join(
        (
            _label(provider).lower(),
            f"{at.astimezone(UTC).hour:02d}",
            _label(operation),
            _label(model),
        )
    )
    fields = [(f"{prefix}{_SEPARATOR}{outcome}", 1)]
    if latency_ms is not None and latency_ms >= 0:
        fields.append((f"{prefix}{_SEPARATOR}lat:{latency_bucket(latency_ms)}", 1))
        fields.append((f"{prefix}{_SEPARATOR}lat_sum", int(latency_ms)))
    return fields


# ── recording ────────────────────────────────────────────────────────────────────────────────

_client: RedisStreamClient | None = None
_pending: set[asyncio.Task[None]] = set()


def bind_provider_calls(client: RedisStreamClient | None) -> None:
    """Where calls are recorded. Set once per process by BaseWorker after Redis connects."""
    global _client
    _client = client


async def record_provider_call(
    provider: str,
    operation: str,
    outcome: str,
    latency_ms: int | None = None,
    model: str | None = None,
    at: datetime | None = None,
) -> None:
    """Count one call. A no-op before `bind_provider_calls`; never raises."""
    client = _client
    if client is None:
        return
    try:
        moment = at or datetime.now(UTC)
        await client.record_provider_call_fields(
            provider_calls_key(moment),
            provider_call_fields(provider, operation, model, outcome, latency_ms, moment),
            PROVIDER_CALLS_KEY_TTL_SECONDS,
        )
    except Exception:
        logger.debug(
            "provider_call_record_failed", provider=provider, operation=operation, exc_info=True
        )


def record_provider_call_soon(
    provider: str,
    operation: str,
    outcome: str,
    latency_ms: int | None = None,
    model: str | None = None,
) -> None:
    """`record_provider_call` without waiting on Redis: the HTTP path must not pay for it."""
    if _client is None:
        return
    try:
        task = asyncio.get_running_loop().create_task(
            record_provider_call(provider, operation, outcome, latency_ms, model, datetime.now(UTC))
        )
    except RuntimeError:
        return
    _pending.add(task)
    task.add_done_callback(_pending.discard)


# ── the httpx transport ──────────────────────────────────────────────────────────────────────


def model_of_request(request: httpx.Request) -> str | None:
    """The `model` of a JSON request body, when there is a small one to read."""
    if "json" not in request.headers.get("content-type", ""):
        return None
    try:
        content = request.content
    except httpx.RequestNotRead:
        return None
    if not content or len(content) > _MAX_MODEL_SNIFF_BYTES:
        return None
    try:
        body: Any = json.loads(content)
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    # OpenAI names it `model`, Cartesia `model_id`.
    model = body.get("model") or body.get("model_id")
    return model if isinstance(model, str) else None


class ObservedTransport(httpx.AsyncBaseTransport):
    """Wraps a transport; records every request's outcome and time to response headers."""

    def __init__(
        self, provider: str, operation: str, inner: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._provider = provider
        self._operation = operation
        self._inner = inner or httpx.AsyncHTTPTransport(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        model = model_of_request(request)
        try:
            response = await self._inner.handle_async_request(request)
        except asyncio.CancelledError:
            # Our own cancellation (a meeting ended, a timeout above us) says nothing of the vendor.
            raise
        except Exception as exc:
            record_provider_call_soon(
                self._provider,
                self._operation,
                classify_exception(exc),
                _elapsed_ms(started),
                model,
            )
            raise
        record_provider_call_soon(
            self._provider,
            self._operation,
            classify_status(response.status_code),
            _elapsed_ms(started),
            model,
        )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def observed_openai_http_client(operation: str) -> httpx.AsyncClient:
    """The OpenAI SDK's own default client, with every request observed as `openai/{operation}`."""
    from openai import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(transport=ObservedTransport("openai", operation))


def observed_cartesia_http_client(operation: str) -> httpx.AsyncClient:
    """Cartesia's default client, observed as `cartesia/{operation}` (HTTP calls only)."""
    from cartesia import DefaultAsyncHttpxClient

    return DefaultAsyncHttpxClient(transport=ObservedTransport("cartesia", operation))
