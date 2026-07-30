import json
from typing import Any

from openai import AsyncOpenAI

from security_worker.scanners import OpenAISecurityScanner
from shared.base_worker import BaseWorker
from shared.config import SecuritySettings, resolve_openai_api_key

RESULT_TTL_SECONDS = 300


class SecurityWorker(BaseWorker):
    """Consumes document scan requests, coordinates multi-language PII/DLP
    scanning and masking through OpenAI, and stores results in Redis."""

    worker_name = "security"
    input_stream = "security:scan_requests"
    consumer_group = "security-workers"

    def __init__(
        self,
        security_settings: SecuritySettings | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.security_settings = security_settings or SecuritySettings()
        self.openai_client: AsyncOpenAI | None = None
        self.openai_scanner: OpenAISecurityScanner | None = None

    async def load_model(self) -> None:
        api_key = resolve_openai_api_key(self.security_settings.api_key)
        if not api_key:
            self.logger.warning("OPENAI_API_KEY is not configured for security_worker")
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.openai_scanner = OpenAISecurityScanner(self.openai_client, self.security_settings)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        decoded_data = {
            k.decode() if isinstance(k, bytes) else k: v.decode()
            if isinstance(v, bytes)
            else str(v)
            for k, v in data.items()
        }

        scan_id = decoded_data.get("scan_id")
        if not scan_id:
            self.logger.error("Missing scan_id in request. Skipping.")
            return

        self.logger.info("processing_scan_request", scan_id=scan_id)

        try:
            content = decoded_data.get("content", "")
            pii_enabled = decoded_data.get("pii_enabled", "false").lower() in {"true", "1", "yes"}
            dlp_enabled = decoded_data.get("dlp_enabled", "false").lower() in {"true", "1", "yes"}
            keywords_raw = decoded_data.get("keywords", "[]")

            try:
                keywords_blacklist = json.loads(keywords_raw)
            except Exception:
                keywords_blacklist = []

            # Fast-path 1: Both disabled -> Return original clean content
            if not pii_enabled and not dlp_enabled:
                await self._save_result(
                    scan_id,
                    pii_detected=False,
                    dlp_detected=False,
                    violation_found=False,
                    masked_content=content,
                )
                return

            # Local DLP Keyword Check (<1ms)
            dlp_detected_local = False
            if dlp_enabled and keywords_blacklist:
                content_lower = content.lower()
                dlp_detected_local = any(
                    kw.lower() in content_lower
                    for kw in keywords_blacklist
                    if isinstance(kw, str) and kw.strip()
                )

            # Fast-path 2: PII disabled & DLP enabled -> Return local evaluation immediately
            if not pii_enabled and dlp_enabled:
                await self._save_result(
                    scan_id,
                    pii_detected=False,
                    dlp_detected=dlp_detected_local,
                    violation_found=dlp_detected_local,
                    masked_content=content,
                )
                self.logger.info(
                    "completed_local_dlp_scan", scan_id=scan_id, violation_found=dlp_detected_local
                )
                return

            # Check OpenAI client
            if not self.openai_client or not self.openai_client.api_key or not self.openai_scanner:
                await self._save_result(
                    scan_id,
                    pii_detected=False,
                    dlp_detected=dlp_detected_local,
                    violation_found=True,
                    masked_content="",
                    scan_failed=True,
                )
                return

            # Dynamic OpenAI Multi-Language PII & DLP Scanning & Masking
            (
                pii_detected,
                llm_dlp_detected,
                violation_found,
                final_masked_content,
            ) = await self.openai_scanner.scan_and_mask(
                content,
                pii_enabled,
                dlp_enabled,
                keywords_blacklist,
            )

            dlp_detected = llm_dlp_detected or dlp_detected_local
            violation_found = violation_found or dlp_detected

            await self._save_result(
                scan_id, pii_detected, dlp_detected, violation_found, final_masked_content
            )
            self.logger.info(
                "completed_scan_request", scan_id=scan_id, violation_found=violation_found
            )

        except Exception:
            self.logger.exception("failed_scan_request", scan_id=scan_id)
            await self._save_result(
                scan_id,
                pii_detected=False,
                dlp_detected=False,
                violation_found=True,
                masked_content="",
                scan_failed=True,
            )

    async def _save_result(
        self,
        scan_id: str,
        pii_detected: bool,
        dlp_detected: bool,
        violation_found: bool,
        masked_content: str = "",
        scan_failed: bool = False,
    ) -> None:
        key = f"security:scan_result:{scan_id}"
        result_payload = {
            "pii_detected": pii_detected,
            "dlp_detected": dlp_detected,
            "violation_found": violation_found,
            "masked_content": masked_content,
            "scan_failed": scan_failed,
        }
        await self.redis.set_with_ttl(key, json.dumps(result_payload), RESULT_TTL_SECONDS)
