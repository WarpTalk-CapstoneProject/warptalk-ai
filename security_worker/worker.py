import json
from typing import Any

from openai import AsyncOpenAI

from security_worker.scanners import OpenAISecurityScanner
from shared.base_worker import BaseWorker
from shared.config import SecuritySettings, resolve_openai_api_key

RESULT_TTL_SECONDS = 300


def keywords_present_in(content: str, keywords: list[str]) -> tuple[str, ...]:
    """The blacklist entries this document actually contains.

    Exact, case-insensitive, over the WHOLE document — not the truncated slice the model is shown.
    This is the DLP verdict; nothing else is.
    """
    lowered = content.lower()
    return tuple(
        keyword
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip() and keyword.lower() in lowered
    )


def split_claims_by_evidence(
    content: str, claimed: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the model's claimed DLP hits into (in the text, not in the text).

    A claim the document does not contain is a hallucination, and the second half of this tuple
    exists so it gets said out loud. Before this, an invented match was indistinguishable from a
    real one the moment it left the scanner, and it silently killed the document.
    """
    lowered = content.lower()
    supported = tuple(term for term in claimed if term.lower() in lowered)
    unsupported = tuple(term for term in claimed if term.lower() not in lowered)
    return supported, unsupported


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

            # Local DLP Keyword Check (<1ms). The whole DLP verdict, on every path below.
            matched_keywords: tuple[str, ...] = ()
            if dlp_enabled and keywords_blacklist:
                matched_keywords = keywords_present_in(content, keywords_blacklist)
            dlp_detected_local = bool(matched_keywords)

            # Fast-path 2: PII disabled & DLP enabled -> Return local evaluation immediately.
            # This path was always right; it is the one below, which reached for the model, that
            # was not.
            if not pii_enabled and dlp_enabled:
                await self._save_result(
                    scan_id,
                    pii_detected=False,
                    dlp_detected=dlp_detected_local,
                    violation_found=dlp_detected_local,
                    masked_content=content,
                    dlp_flagged_terms=matched_keywords,
                )
                self.logger.info(
                    "completed_local_dlp_scan",
                    scan_id=scan_id,
                    violation_found=dlp_detected_local,
                    dlp_terms=list(matched_keywords),
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

            # Dynamic OpenAI Multi-Language PII Scanning & Masking. The model also cites any DLP
            # spans it thinks it saw, but it does not decide whether there was a violation.
            report = await self.openai_scanner.scan_and_mask(
                content,
                pii_enabled,
                dlp_enabled,
                keywords_blacklist,
            )

            # THE DLP VERDICT IS THE LOCAL CHECK AND NOTHING ELSE.
            #
            # This used to be `llm_dlp_detected or dlp_detected_local`, and that `or` is the bug.
            # The local check answers the question exactly, over the whole document, for free; the
            # model answers it approximately, over at most `max_analyze_length` characters of it.
            # OR-ing the approximate answer on top of the exact one can only ever ADD false
            # positives — it has no way to correct one — and a DLP hit is not appealable, because
            # `ProcessDocumentUploadAsync` builds `canIndex` with `&& !DlpDetected` and offers no
            # masking route the way PII has one. So the document is dropped for good.
            #
            # Measured in production 2026-09-06: across all thirteen DLP-enabled scans on record,
            # the blacklist occurred zero times in the submitted text. Three documents were blocked
            # as DLP violations anyway. Three out of three were invented.
            dlp_detected = dlp_detected_local

            supported_claims: tuple[str, ...] = ()
            if dlp_enabled:
                supported_claims, unsupported_claims = split_claims_by_evidence(
                    content, report.dlp_terms_claimed
                )
                if unsupported_claims:
                    # The line that would have exposed this on day one instead of after three
                    # documents. A claim about text that is not in the document is a fact about
                    # the model, not about the document.
                    self.logger.warning(
                        "dlp_claim_absent_from_text",
                        scan_id=scan_id,
                        terms=list(unsupported_claims),
                    )

            # A supported claim that is not itself a blacklist entry is a variant the exact search
            # cannot see — "F*ck" for "Fuck". It is worth surfacing and NOT worth blocking on:
            # blocking on a fuzzy match is precisely what cost three documents.
            dlp_flagged_terms = tuple(dict.fromkeys(matched_keywords + supported_claims))
            violation_found = report.pii_detected or dlp_detected

            await self._save_result(
                scan_id,
                report.pii_detected,
                dlp_detected,
                violation_found,
                report.masked_content,
                dlp_flagged_terms=dlp_flagged_terms,
            )
            self.logger.info(
                "completed_scan_request",
                scan_id=scan_id,
                violation_found=violation_found,
                dlp_detected=dlp_detected,
                dlp_terms=list(dlp_flagged_terms),
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
        dlp_flagged_terms: tuple[str, ...] = (),
    ) -> None:
        key = f"security:scan_result:{scan_id}"
        result_payload = {
            "pii_detected": pii_detected,
            "dlp_detected": dlp_detected,
            "violation_found": violation_found,
            "masked_content": masked_content,
            "scan_failed": scan_failed,
            # The evidence behind the verdict. Additive: the backend's ScanResponse ignores
            # members it does not declare, so this reaches nothing yet — it exists so that a
            # blocked document can eventually say WHICH word blocked it, rather than only that
            # something did.
            "dlp_flagged_terms": list(dlp_flagged_terms),
        }
        await self.redis.set_with_ttl(key, json.dumps(result_payload), RESULT_TTL_SECONDS)
