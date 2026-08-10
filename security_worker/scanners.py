import json

from openai import AsyncOpenAI

from shared.config import SecuritySettings
from shared.openai_options import completion_options

# --- Constants ---
# Fallbacks only. The live values come from SecuritySettings (SECURITY_MODEL,
# SECURITY_MAX_TOKENS, SECURITY_TEMPERATURE, SECURITY_MAX_ANALYZE_LENGTH) — this
# scanner used to read the module constants directly and ignore the settings object
# entirely, which made SECURITY_MODEL a dead environment variable: production could
# set it to anything and the scanner still called gpt-4o-mini.
DEFAULT_MODEL = "gpt-4o-mini"
MAX_ANALYZE_LENGTH = 20000
MAX_TOKENS = 2000
TEMPERATURE = 0.0


class OpenAISecurityScanner:
    """OpenAI-backed document scanner for dynamic multi-language PII/DLP
    inspection and masking."""

    def __init__(self, client: AsyncOpenAI, settings: SecuritySettings) -> None:
        self.client = client
        self.settings = settings

    async def scan_and_mask(
        self,
        text: str,
        pii_enabled: bool,
        dlp_enabled: bool,
        keywords_blacklist: list[str],
    ) -> tuple[bool, bool, bool, str]:
        """Scans and dynamically masks PII/DLP in text using OpenAI LLM.

        Returns (pii_detected, dlp_detected, violation_found, masked_content).
        """
        if not text:
            return False, False, False, text

        max_analyze_length = self.settings.max_analyze_length or MAX_ANALYZE_LENGTH
        text_to_analyze = text
        if len(text_to_analyze) > max_analyze_length:
            text_to_analyze = text_to_analyze[:max_analyze_length] + "... [truncated]"

        keywords_json = json.dumps(keywords_blacklist)

        system_prompt = (
            "You are a multi-language document security scanner supporting all "
            "languages (English, Japanese, Vietnamese, etc.).\n"
            "Analyze the provided text for PII (emails, phone numbers, SSN, My "
            "Number, CCCD/ID numbers, credit cards, full names, addresses) and "
            "DLP keyword violations.\n\n"
            "Instructions:\n"
            "1. If PII Detection is enabled (pii_enabled is true), detect any PII "
            "in the text. Mask detected PII using [PII_REDACTED], "
            "[EMAIL_REDACTED], [PHONE_REDACTED], [ID_REDACTED], "
            "[CARD_REDACTED]. Set piiDetected to true if PII is found, otherwise "
            "false.\n"
            "2. If DLP Detection is enabled (dlp_enabled is true), check if the "
            "text contains any of the blacklisted keywords (case-insensitive). "
            "Set dlpDetected to true if found, otherwise false.\n"
            "3. Provide the complete final text with all PII masked in "
            "maskedContent. If no PII is found or PII Detection is disabled, "
            "keep maskedContent equal to the input text.\n"
            "4. Set violationFound to true if either piiDetected or dlpDetected is true.\n\n"
            "Respond ONLY in JSON format matching this schema:\n"
            "{\n"
            '  "piiDetected": boolean,\n'
            '  "dlpDetected": boolean,\n'
            '  "violationFound": boolean,\n'
            '  "maskedContent": string\n'
            "}"
        )

        user_prompt = (
            f"Settings:\n"
            f"- PII Enabled: {pii_enabled}\n"
            f"- DLP Enabled: {dlp_enabled}\n"
            f"- DLP Blacklisted Keywords: {keywords_json}\n\n"
            f"Text to analyze:\n{text_to_analyze}"
        )

        model = self.settings.model or DEFAULT_MODEL
        completion = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            **completion_options(
                model,
                self.settings.max_tokens or MAX_TOKENS,
                self.settings.temperature if self.settings.temperature is not None else TEMPERATURE,
            ),
        )

        content_str = completion.choices[0].message.content
        if not content_str:
            raise ValueError("Empty response from OpenAI")

        result = json.loads(content_str)
        pii_detected = bool(result.get("piiDetected", False))
        dlp_detected = bool(result.get("dlpDetected", False))
        violation_found = bool(result.get("violationFound", False)) or pii_detected or dlp_detected
        masked_content = str(result.get("maskedContent", text))

        return pii_detected, dlp_detected, violation_found, masked_content
