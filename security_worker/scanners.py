import json
from dataclasses import dataclass

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

# WT-460. The output budget has to be able to hold the INPUT, because this prompt asks the model
# to return `maskedContent` — the whole analysed text, echoed back with PII replaced.
#
# The two caps above were set independently and never compared: input was allowed 20,000
# CHARACTERS while the reply was capped at 2,000 TOKENS. For anything past roughly six thousand
# characters the model physically cannot finish the JSON object, the reply is cut mid-string, and
# `json.loads` raises. That is the whole of "approved documents never embed": the scan throws, the
# document never becomes AiEligible, and nothing reaches Qdrant. Every hypothesis in the ticket —
# missing OpenAI key, Qdrant refused, VectorDb:Url unset — was wrong; the scan was reaching
# OpenAI perfectly well and being truncated on the way back.
#
# Deliberately conservative: 2 characters per token, when English averages closer to 4. Vietnamese
# and Japanese are far denser per character, and this scanner exists to read exactly those. Being
# generous costs output budget the model only spends if it needs it; being tight costs the
# document.
CHARS_PER_OUTPUT_TOKEN = 2

# Room for the JSON envelope, the flags, the cited matches, and the redaction markers that make
# masked text longer than the original.
JSON_ENVELOPE_TOKENS = 512


@dataclass(frozen=True)
class SecurityScanReport:
    """What the model OBSERVED — deliberately not what the system decides.

    The asymmetry between the two fields is the point of this type.

    `pii_detected` is a judgement only a model can make: "is this string somebody's name" has no
    exact answer, and a PII hit is RECOVERABLE, because the masked text is still indexable.

    A DLP hit is neither. "Does this document contain one of these words" is answered exactly, for
    free, by a substring search over the whole file — and a hit is TERMINAL, because the ingestion
    path has no masking route for DLP the way it has one for PII. So the model does not get to
    return `dlpDetected`. It returns the spans it BELIEVES are blacklist hits, and `SecurityWorker`
    decides, after checking each one against the document.

    Production, 2026-09-06: of thirteen DLP-enabled scans, the blacklist appeared zero times in the
    submitted text — and three documents were still blocked as DLP violations. Every DLP block this
    system has ever produced was a hallucination, OR-ed on top of a local check that had it right.
    """

    pii_detected: bool
    masked_content: str
    dlp_terms_claimed: tuple[str, ...]


def _claimed_terms(raw: object) -> tuple[str, ...]:
    """The model's `dlpMatches`, read defensively.

    Anything that is not a list of non-empty strings is not a claim. A malformed field must not
    become a verdict by accident — which is exactly how a stray boolean used to block a document.
    """
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item.strip())


class OpenAISecurityScanner:
    """OpenAI-backed document scanner for dynamic multi-language PII/DLP
    inspection and masking."""

    def __init__(self, client: AsyncOpenAI, settings: SecuritySettings) -> None:
        self.client = client
        self.settings = settings

    def _output_token_budget(self, text_to_analyze: str) -> int:
        """How much room the reply needs, given it carries the analysed text back.

        Never smaller than the configured cap: SECURITY_MAX_TOKENS stays a floor a deployment can
        raise, it just stops being a ceiling that silently truncates the answer. A short document
        therefore behaves exactly as before, and a long one gets the room it actually needs
        instead of failing.
        """
        configured = self.settings.max_tokens or MAX_TOKENS
        required = len(text_to_analyze) // CHARS_PER_OUTPUT_TOKEN + JSON_ENVELOPE_TOKENS
        return max(configured, required)

    async def scan_and_mask(
        self,
        text: str,
        pii_enabled: bool,
        dlp_enabled: bool,
        keywords_blacklist: list[str],
    ) -> SecurityScanReport:
        """Scans and dynamically masks PII in text using OpenAI, and cites any DLP spans it sees.

        Returns what the model reported. The DLP verdict is not in here on purpose — see
        `SecurityScanReport`.
        """
        if not text:
            return SecurityScanReport(pii_detected=False, masked_content=text, dlp_terms_claimed=())

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
            "2. If DLP Detection is enabled (dlp_enabled is true), look for the blacklisted "
            "keywords (case-insensitive). In dlpMatches, list the matching substrings copied "
            "VERBATIM out of the text, character for character. Return an empty list if there "
            "are none. Do not paraphrase, normalise, translate, explain or invent an entry: "
            "every string you return is searched for in the document, and one that is not "
            "there is discarded.\n"
            "3. Provide the complete final text with all PII masked in "
            "maskedContent. If no PII is found or PII Detection is disabled, "
            "keep maskedContent equal to the input text.\n\n"
            "Respond ONLY in JSON format matching this schema:\n"
            "{\n"
            '  "piiDetected": boolean,\n'
            '  "dlpMatches": string[],\n'
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
                self._output_token_budget(text_to_analyze),
                self.settings.temperature if self.settings.temperature is not None else TEMPERATURE,
            ),
        )

        choice = completion.choices[0]
        content_str = choice.message.content
        if not content_str:
            raise ValueError("Empty response from OpenAI")

        # WT-460: say WHY, while the reason is still knowable.
        #
        # `response_format={"type": "json_object"}` guarantees the model AIMS at valid JSON; it
        # does not guarantee the reply fits inside max_tokens. A cut-off object is still invalid
        # JSON, so this used to surface as a bare JSONDecodeError from `json.loads` three frames
        # down — which reads like a malformed model reply and sent the whole investigation at the
        # API key and at Qdrant. `finish_reason` is the API telling us plainly that it ran out of
        # room, and it costs nothing to look.
        if choice.finish_reason == "length":
            raise ValueError(
                "OpenAI reply was truncated by the output token limit "
                f"({len(text_to_analyze)} chars analysed). The scan returns the masked text in "
                "full, so the output budget must exceed the input; raise SECURITY_MAX_TOKENS or "
                "lower SECURITY_MAX_ANALYZE_LENGTH."
            )

        result = json.loads(content_str)
        return SecurityScanReport(
            pii_detected=bool(result.get("piiDetected", False)),
            masked_content=str(result.get("maskedContent", text)),
            # Only ever what the model CLAIMS. Note the model is shown at most
            # `max_analyze_length` characters, so its claims cover a slice of a long document
            # while the check that decides reads all of it.
            dlp_terms_claimed=_claimed_terms(result.get("dlpMatches")),
        )
