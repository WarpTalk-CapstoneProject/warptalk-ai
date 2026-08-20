"""Putting a finished local file somewhere it will outlive the process.

WHY THIS IS SEPARATE FROM WHAT PRODUCES THE FILES
    Writing an audio track is microseconds against local disk; uploading it is seconds
    against a network that fails. Folding the two together would put a retry loop inside the
    audio path, which is the one place in this worker that must never block.

WHY boto3 IS IMPORTED LAZILY
    warptalk-ai runs ten workers and only one of them uploads anything. Importing an AWS SDK
    at module scope would pay for it in every image, including the ones built without the
    dependency at all. A missing boto3 is therefore a normal, reportable condition here
    rather than an import error that takes a worker down at startup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ObjectStorageSettings:
    bucket: str
    prefix: str = ""
    endpoint: str = ""
    region: str = "auto"
    access_key: str = ""
    secret_key: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.bucket)


class ObjectStorage:
    """Uploads local files under a key prefix, and reports honestly when it cannot.

    Every method returns a value rather than raising. The caller is a cleanup path on a
    meeting that has already ended: there is nothing useful it could do with an exception,
    and letting one escape would abandon the remaining tracks in the same meeting.
    """

    def __init__(self, settings: ObjectStorageSettings) -> None:
        self.settings = settings
        self._client = None
        self._unavailable = False

    def _resolve_client(self) -> Any:
        if self._client is not None or self._unavailable:
            return self._client

        try:
            import boto3  # type: ignore[import-untyped]  # noqa: PLC0415 — lazy on purpose
        except ImportError:
            logger.error(
                "object_storage_unavailable",
                reason="boto3 is not installed in this image",
                bucket=self.settings.bucket,
            )
            self._unavailable = True
            return None

        try:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.settings.endpoint or None,
                region_name=self.settings.region or None,
                aws_access_key_id=self.settings.access_key or None,
                aws_secret_access_key=self.settings.secret_key or None,
            )
        except Exception:
            logger.error("object_storage_client_failed", bucket=self.settings.bucket, exc_info=True)
            self._unavailable = True
            return None

        return self._client

    def upload(self, path: Path, key: str) -> str | None:
        """Upload one file; the object's URI on success, None on any failure.

        Blocking. Call it from a thread — `asyncio.to_thread` — never straight off the event
        loop, because a slow bucket would otherwise stall every room this worker carries.
        """
        if not self.settings.configured:
            return None

        client = self._resolve_client()
        if client is None:
            return None

        full_key = f"{self.settings.prefix.strip('/')}/{key}".lstrip("/")
        try:
            client.upload_file(str(path), self.settings.bucket, full_key)
        except Exception:
            logger.warning(
                "object_storage_upload_failed",
                bucket=self.settings.bucket,
                key=full_key,
                exc_info=True,
            )
            return None

        return f"s3://{self.settings.bucket}/{full_key}"
