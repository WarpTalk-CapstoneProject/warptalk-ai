import json
import os
from openai import AsyncOpenAI
from shared.base_worker import BaseWorker
from shared.config import SecuritySettings, resolve_openai_api_key

class SecurityWorker(BaseWorker):
    """Consumes document text scan requests, calls OpenAI for PII/DLP scanning, and stores the scan result in Redis."""

    worker_name = "security"
    input_stream = "security:scan_requests"
    consumer_group = "security-workers"

    def __init__(self, security_settings: SecuritySettings | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.security_settings = security_settings or SecuritySettings()
        self.openai_client: AsyncOpenAI | None = None

    async def load_model(self) -> None:
        api_key = resolve_openai_api_key(self.security_settings.api_key)
        if not api_key:
            self.logger.warning("OPENAI_API_KEY is not configured for security_worker")
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def process(self, message_id: bytes, data: dict[bytes, bytes]) -> None:
        # Decode the request data
        decoded_data = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else str(v)
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

            # Fast-path 1: If both are disabled, return clean scan result immediately
            if not pii_enabled and not dlp_enabled:
                await self._save_result(scan_id, pii_detected=False, dlp_detected=False, violation_found=False)
                return

            # Fast-path 2: Local DLP Keyword Check (Instant local evaluation < 1ms)
            dlp_detected_local = False
            if dlp_enabled and keywords_blacklist:
                content_lower = content.lower()
                dlp_detected_local = any(kw.lower() in content_lower for kw in keywords_blacklist if isinstance(kw, str) and kw.strip())

            # Fast-path 3: If PII is disabled and only DLP is enabled, return local result without calling OpenAI
            if not pii_enabled and dlp_enabled:
                await self._save_result(scan_id, pii_detected=False, dlp_detected=dlp_detected_local, violation_found=dlp_detected_local)
                self.logger.info("completed_local_dlp_scan", scan_id=scan_id, violation_found=dlp_detected_local)
                return

            if not self.openai_client or not self.openai_client.api_key:
                raise ValueError("OpenAI client or API key is not configured")

            text_to_analyze = content
            if len(text_to_analyze) > 20000:
                text_to_analyze = text_to_analyze[:20000] + "... [truncated]"

            keywords_json = json.dumps(keywords_blacklist)

            system_prompt = (
                "You are a document security scanner. Analyze the provided text for PII (Personally Identifiable Information like email, phone number, SSN, ID card numbers, full names, addresses) and DLP (Data Loss Prevention) keyword violations.\n\n"
                "Instructions:\n"
                "1. If PII Detection is enabled (pii_enabled is true), check if the text contains PII. Set piiDetected to true if found, otherwise false.\n"
                "2. If DLP Detection is enabled (dlp_enabled is true), check if the text contains any of the blacklisted keywords (case-insensitive). Set dlpDetected to true if found, otherwise false.\n"
                "3. Set violationFound to true if either piiDetected or dlpDetected is true.\n\n"
                "Respond ONLY in JSON format matching this schema:\n"
                "{\n"
                "  \"piiDetected\": boolean,\n"
                "  \"dlpDetected\": boolean,\n"
                "  \"violationFound\": boolean\n"
                "}"
            )

            user_prompt = (
                f"Settings:\n"
                f"- PII Enabled: {pii_enabled}\n"
                f"- DLP Enabled: {dlp_enabled}\n"
                f"- DLP Blacklisted Keywords: {keywords_json}\n\n"
                f"Text to analyze:\n{text_to_analyze}"
            )

            completion = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=60,
            )

            content_str = completion.choices[0].message.content
            if not content_str:
                raise ValueError("Empty response from OpenAI")

            result = json.loads(content_str)
            pii_detected = result.get("piiDetected", False)
            dlp_detected = result.get("dlpDetected", False) or dlp_detected_local
            violation_found = result.get("violationFound", False) or pii_detected or dlp_detected

            await self._save_result(scan_id, pii_detected, dlp_detected, violation_found)
            self.logger.info("completed_scan_request", scan_id=scan_id, violation_found=violation_found)

        except Exception as e:
            self.logger.exception("failed_scan_request", scan_id=scan_id)
            # Do not write result on failure; C# side will timeout and apply fail-safe fallback

    async def _save_result(self, scan_id: str, pii_detected: bool, dlp_detected: bool, violation_found: bool) -> None:
        key = f"security:scan_result:{scan_id}"
        result_payload = {
            "pii_detected": pii_detected,
            "dlp_detected": dlp_detected,
            "violation_found": violation_found
        }
        # Save result in Redis as a string with 5 minutes expiration
        await self.redis.set_with_ttl(key, json.dumps(result_payload), 300)

