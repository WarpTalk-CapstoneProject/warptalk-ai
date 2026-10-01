"""Local fast-path regex scanners for common structured PII
(CCCD/CMND, VN phone, email, credit card)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Email pattern matching standard email formats
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Vietnam Phone Numbers: +84 or 0 followed by 3, 5, 7, 8, 9 and 8 digits
# (with optional spaces, dots, or dashes)
VN_PHONE_REGEX = re.compile(r"(?<!\d)(?:\+84|0)[-. ]?(?:3|5|7|8|9)(?:[-. ]?\d){8}(?!\d)")

# Vietnam CCCD (12 digits starting with 0, optionally separated by spaces, dots, dashes)
VN_CCCD_REGEX = re.compile(
    r"(?<!\d)0(?:"
    r"\d{11}|"
    r"\d{2}[-. ]\d{3}[-. ]\d{6}|"
    r"\d{2}[-. ]\d{3}[-. ]\d{3}[-. ]\d{3}"
    r")(?!\d)"
)

# Vietnam CMND (9 digits)
VN_CMND_REGEX = re.compile(r"(?<!\d)\d{9}(?!\d)")

# Potential Credit Card numbers: 13-19 digits (with optional spaces or dashes)
CREDIT_CARD_REGEX = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _is_luhn_valid(number_str: str) -> bool:
    """Validate numeric string with Luhn algorithm."""
    digits = [int(c) for c in number_str if c.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


@dataclass(frozen=True)
class RegexPiiResult:
    detected: bool
    masked_text: str
    matches_count: int


def scan_and_mask_regex_pii(text: str) -> RegexPiiResult:
    """Scan and mask structured PII locally using high-precision regexes.

    Executed in < 1ms to prevent leakage and reduce LLM burden.
    """
    if not text:
        return RegexPiiResult(detected=False, masked_text=text, matches_count=0)

    # Normalize unicode to NFC
    normalized_text = unicodedata.normalize("NFC", text)
    count = 0

    # 1. Mask Email
    def replace_email(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[EMAIL_REDACTED]"

    masked = EMAIL_REGEX.sub(replace_email, normalized_text)

    # 2. Mask VN Phone
    def replace_phone(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[PHONE_REDACTED]"

    masked = VN_PHONE_REGEX.sub(replace_phone, masked)

    # 3. Mask Credit Cards with Luhn check
    def replace_card(match: re.Match[str]) -> str:
        nonlocal count
        matched_str = match.group(0)
        digits_only = re.sub(r"\D", "", matched_str)
        if _is_luhn_valid(digits_only):
            count += 1
            return "[CARD_REDACTED]"
        return matched_str

    masked = CREDIT_CARD_REGEX.sub(replace_card, masked)

    # 4. Mask VN CCCD (12 digits starting with 0)
    def replace_cccd(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "[ID_REDACTED]"

    masked = VN_CCCD_REGEX.sub(replace_cccd, masked)

    return RegexPiiResult(
        detected=count > 0,
        masked_text=masked,
        matches_count=count,
    )
