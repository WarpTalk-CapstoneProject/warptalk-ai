"""Which third-party integrations this worker process is configured for, as seen from inside it.

THE CROSS-REPO CONTRACT
    Redis hash   platform:integrations:v1:ai-{worker_name}   (EXPIRE 600s, refreshed every 300s)
    Field        integration key: "openai", "cartesia", "livekit"
    Value        JSON {"configured": bool, "detail": str | null, "reportedAt": ISO-8601 UTC}
    Read by      the platform settings console (Integrations), which cannot see a worker's env.

NEVER A SECRET. `detail` carries model names and nothing else: no key, no secret, no URL, not
even a prefix of one. "configured" is the only thing a credential contributes, and it is a bool.

Best effort, like the heartbeat that calls it: a report that cannot be written is logged and
dropped, and the worker carries on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)

KEY_PREFIX = "platform:integrations:v1:"
TTL_SECONDS = 600
REPORT_INTERVAL_SECONDS = 300.0

OPENAI = "openai"
CARTESIA = "cartesia"
LIVEKIT = "livekit"

# LiveKitSettings' shipped defaults. A worker still carrying them has not been configured.
_LIVEKIT_PLACEHOLDER_PREFIX = "YOUR_"


def integrations_key(worker_name: str) -> str:
    return f"{KEY_PREFIX}ai-{worker_name}"


@dataclass(frozen=True)
class IntegrationReport:
    configured: bool
    detail: str | None = None

    def to_json(self, reported_at: datetime) -> str:
        return json.dumps(
            {
                "configured": self.configured,
                "detail": self.detail,
                "reportedAt": reported_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            }
        )


def credential_report(credential: str | None, detail: str | None = None) -> IntegrationReport:
    """Configured when the credential is non-empty. The credential itself goes nowhere."""
    return IntegrationReport(configured=bool((credential or "").strip()), detail=detail)


def livekit_report(livekit: Any, detail: str | None = None) -> IntegrationReport:
    """Configured when URL, key and secret are all set and none is a shipped placeholder."""
    try:
        url = str(getattr(livekit, "url", "") or "").strip()
        key = str(getattr(livekit, "api_key", "") or "").strip()
        secret = str(getattr(livekit, "api_secret", "") or "").strip()
    except Exception:
        return IntegrationReport(configured=False, detail=detail)
    configured = bool(url and key and secret) and not any(
        value.startswith(_LIVEKIT_PLACEHOLDER_PREFIX) for value in (key, secret)
    )
    return IntegrationReport(configured=configured, detail=detail)


async def publish_integration_status(
    redis: Any,
    worker_name: str,
    reports: Mapping[str, IntegrationReport],
    *,
    now: datetime | None = None,
) -> bool:
    """Write this worker's reports. True when written; never raises."""
    if not reports:
        return False
    try:
        key = integrations_key(worker_name)
        reported_at = now or datetime.now(UTC)
        for name, report in reports.items():
            await redis.hset(key, name, report.to_json(reported_at))
        await redis.expire(key, TTL_SECONDS)
        return True
    except Exception:
        logger.warning("integration_status_publish_failed", worker=worker_name, exc_info=True)
        return False
