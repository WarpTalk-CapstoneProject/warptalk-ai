"""Generate billing pricing tables from provider prices and estimation inputs.

The script is intentionally deterministic: it keeps provider prices, markup, FX, and
workload assumptions in one place, then emits Markdown tables for the master plan.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal
from pathlib import Path

D = Decimal

MASTER_PLAN_PATH = Path(__file__).resolve().parents[2] / "credit-billing-master-plan.md"

FX_RATE_USD_VND = D("26300")
CREDIT_VALUE_VND = D("4")

CHARS_PER_SEC = D("12.5")
TALK_DENSITY = D("0.70")
VAD_PADDING = D("1.15")
SEGMENT_CHARS = D("60")
CHARS_PER_TOKEN = D("3.2")
GLOSSARY_OVERHEAD_TOKENS = D("620")
BASE_TRANSLATION_OVERHEAD_TOKENS = D("100")

WALL_CLOCK_SECONDS = D("3600")
SUMMARY_CREDITS_PER_HOUR = D("30")


@dataclass(frozen=True)
class MasterPlanInputs:
    fx_rate_usd_vnd: Decimal
    chars_per_sec: Decimal
    talk_density: Decimal
    vad_padding: Decimal
    segment_chars: Decimal
    chars_per_token: Decimal
    glossary_overhead_tokens: Decimal


@dataclass(frozen=True)
class Rate:
    charge_type: str
    unit: str
    provider: str
    model: str
    provider_unit_cost_usd: Decimal
    markup_multiplier: Decimal
    log_only: bool = False

    @property
    def vnd_per_unit(self) -> Decimal:
        return self.provider_unit_cost_usd * FX_RATE_USD_VND

    @property
    def unit_price(self) -> Decimal:
        if self.log_only:
            return D("0")
        return self.vnd_per_unit * self.markup_multiplier / CREDIT_VALUE_VND


RATES = [
    Rate("STT", "second", "openai", "gpt-4o-transcribe", D("0.000100"), D("2.50")),
    Rate("TRANSLATION", "token_in", "openai", "gpt-4.1-mini", D("0.00000040"), D("2.50")),
    Rate(
        "TRANSLATION",
        "token_in_cached",
        "openai",
        "gpt-4.1-mini",
        D("0.00000010"),
        D("2.50"),
    ),
    Rate("TRANSLATION", "token_out", "openai", "gpt-4.1-mini", D("0.00000160"), D("2.50")),
    Rate(
        "AUDIO_DUBBING_STANDARD",
        "character",
        "cartesia",
        "sonic-3.5",
        D("0.00003920"),
        D("3.00"),
    ),
    Rate(
        "AUDIO_DUBBING_VOICE_CLONE",
        "character",
        "cartesia",
        "sonic-3.5-clone",
        D("0.00005880"),
        D("3.50"),
    ),
    Rate(
        "VOICE_CLONE_ENROLLMENT",
        "profile",
        "cartesia",
        "cartesia-localizing-voice",
        D("0.00882000"),
        D("3.50"),
    ),
    Rate("AI_ASSISTANT", "token_in", "openai", "gpt-4.1", D("0.00000200"), D("2.50")),
    Rate(
        "AI_ASSISTANT",
        "token_in_cached",
        "openai",
        "gpt-4.1",
        D("0.00000050"),
        D("2.50"),
    ),
    Rate("AI_ASSISTANT", "token_out", "openai", "gpt-4.1", D("0.00000800"), D("2.50")),
    Rate("AI_SUMMARY", "token_in", "openai", "gpt-4o-mini", D("0.00000015"), D("2.50")),
    Rate(
        "AI_SUMMARY",
        "token_in_cached",
        "openai",
        "gpt-4o-mini",
        D("0.000000075"),
        D("2.50"),
    ),
    Rate("AI_SUMMARY", "token_out", "openai", "gpt-4o-mini", D("0.00000060"), D("2.50")),
    Rate("EMBEDDING", "token", "openai", "text-embedding-3-small", D("0.00000002"), D("0"), True),
]


def parse_vn_decimal(value: str) -> Decimal:
    return D(value.replace(".", "").replace(",", "."))


def read_master_plan_inputs(path: Path = MASTER_PLAN_PATH) -> MasterPlanInputs:
    text = path.read_text(encoding="utf-8")

    def find(pattern: str) -> str:
        match = re.search(pattern, text)
        if not match:
            raise RuntimeError(f"Cannot read pricing input from {path}: {pattern}")
        return match.group(1)

    return MasterPlanInputs(
        fx_rate_usd_vnd=parse_vn_decimal(find(r"1 USD = ([\d.]+) ₫")),
        chars_per_sec=parse_vn_decimal(find(r"`CHARS_PER_SEC`\s*\|\s*([\d,]+)")),
        talk_density=parse_vn_decimal(find(r"`TALK_DENSITY`\s*\|\s*([\d,]+)%")) / D("100"),
        vad_padding=parse_vn_decimal(find(r"`VAD_PADDING`\s*\|\s*\*\*([\d,]+)×\*\*")),
        segment_chars=parse_vn_decimal(find(r"`SEGMENT_CHARS`\s*\|\s*([\d,]+)")),
        chars_per_token=parse_vn_decimal(find(r"`CHARS_PER_TOKEN`\s*\|\s*([\d,]+)")),
        glossary_overhead_tokens=parse_vn_decimal(
            find(r"`GLOSSARY_OVERHEAD`\s*\|\s*~([\d,]+)")
        ),
    )


def verify_master_plan_inputs(inputs: MasterPlanInputs) -> None:
    expected = MasterPlanInputs(
        fx_rate_usd_vnd=FX_RATE_USD_VND,
        chars_per_sec=CHARS_PER_SEC,
        talk_density=TALK_DENSITY,
        vad_padding=VAD_PADDING,
        segment_chars=SEGMENT_CHARS,
        chars_per_token=CHARS_PER_TOKEN,
        glossary_overhead_tokens=GLOSSARY_OVERHEAD_TOKENS,
    )
    if inputs != expected:
        raise RuntimeError(
            "pricing_calc.py constants no longer match credit-billing-master-plan.md "
            f"§3.1/§3.2 inputs. expected={expected!r} actual={inputs!r}"
        )


def q(value: Decimal, places: str) -> Decimal:
    return value.quantize(D(places), rounding=ROUND_HALF_UP)


def ceil(value: Decimal) -> Decimal:
    return value.to_integral_value(rounding=ROUND_CEILING)


def rate(charge_type: str, unit: str) -> Rate:
    return next(r for r in RATES if r.charge_type == charge_type and r.unit == unit)


def credits(charge_type: str, unit: str, quantity: Decimal) -> Decimal:
    return quantity * rate(charge_type, unit).unit_price


def provider_vnd(charge_type: str, unit: str, quantity: Decimal) -> Decimal:
    return quantity * rate(charge_type, unit).vnd_per_unit


@dataclass(frozen=True)
class Workload:
    name: str
    languages: Decimal = D("1")
    dubbing: str | None = None
    glossary: bool = True
    assistant_question: bool = False


def translation_quantities(languages: Decimal, glossary: bool) -> tuple[Decimal, Decimal, Decimal]:
    spoken_seconds = WALL_CLOCK_SECONDS * TALK_DENSITY
    source_chars = spoken_seconds * CHARS_PER_SEC
    segments = ceil(source_chars / SEGMENT_CHARS)
    content_tokens = source_chars / CHARS_PER_TOKEN
    overhead = GLOSSARY_OVERHEAD_TOKENS if glossary else BASE_TRANSLATION_OVERHEAD_TOKENS
    input_tokens = (content_tokens + (segments * overhead)) * languages
    output_tokens = content_tokens * languages
    return input_tokens, D("0"), output_tokens


def workload_row(
    workload: Workload,
) -> tuple[str, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if workload.assistant_question:
        in_tokens = D("4000")
        out_tokens = D("500")
        assistant_credits = ceil(
            credits("AI_ASSISTANT", "token_in", in_tokens)
            + credits("AI_ASSISTANT", "token_out", out_tokens)
        )
        provider_cost = provider_vnd("AI_ASSISTANT", "token_in", in_tokens) + provider_vnd(
            "AI_ASSISTANT", "token_out", out_tokens
        )
        return workload.name, D("0"), D("0"), D("0"), D("0"), assistant_credits, provider_cost

    spoken_seconds = WALL_CLOCK_SECONDS * TALK_DENSITY
    stt_seconds = spoken_seconds * VAD_PADDING
    source_chars = spoken_seconds * CHARS_PER_SEC

    stt_credits = credits("STT", "second", stt_seconds)
    stt_provider = provider_vnd("STT", "second", stt_seconds)

    token_in, token_in_cached, token_out = translation_quantities(
        workload.languages, workload.glossary
    )
    translation_credits = (
        credits("TRANSLATION", "token_in", token_in)
        + credits("TRANSLATION", "token_in_cached", token_in_cached)
        + credits("TRANSLATION", "token_out", token_out)
    )
    translation_provider = (
        provider_vnd("TRANSLATION", "token_in", token_in)
        + provider_vnd("TRANSLATION", "token_in_cached", token_in_cached)
        + provider_vnd("TRANSLATION", "token_out", token_out)
    )

    tts_credits = D("0")
    tts_provider = D("0")
    if workload.dubbing == "standard":
        tts_chars = source_chars * workload.languages
        tts_credits = credits("AUDIO_DUBBING_STANDARD", "character", tts_chars)
        tts_provider = provider_vnd("AUDIO_DUBBING_STANDARD", "character", tts_chars)
    elif workload.dubbing == "voice_clone":
        tts_chars = source_chars * workload.languages
        tts_credits = credits("AUDIO_DUBBING_VOICE_CLONE", "character", tts_chars)
        tts_provider = provider_vnd("AUDIO_DUBBING_VOICE_CLONE", "character", tts_chars)

    provider_cost = stt_provider + translation_provider + tts_provider + (
        SUMMARY_CREDITS_PER_HOUR * CREDIT_VALUE_VND / D("2.50")
    )
    total_credits = stt_credits + translation_credits + tts_credits + SUMMARY_CREDITS_PER_HOUR
    return (
        workload.name,
        stt_credits,
        translation_credits,
        tts_credits,
        SUMMARY_CREDITS_PER_HOUR,
        total_credits,
        provider_cost,
    )


def fmt_decimal(value: Decimal, places: str = "0.000000") -> str:
    return f"{q(value, places):f}".replace(".", ",")


def fmt_int(value: Decimal) -> str:
    return f"{int(q(value, '1')):,}".replace(",", ".")


def render_rate_card() -> str:
    rows = [
        "| `charge_type` | `unit` | provider | model | "
        "`provider_unit_cost` (USD) | ₫/đơn vị | `markup_multiplier` | "
        "**`unit_price`** |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in RATES:
        markup = "-" if r.log_only else fmt_decimal(r.markup_multiplier, "0.00")
        unit_price = "**0,000000** (log-only)" if r.log_only else f"**{fmt_decimal(r.unit_price)}**"
        rows.append(
            f"| `{r.charge_type}` | `{r.unit}` | {r.provider} | {r.model} | "
            f"{fmt_decimal(r.provider_unit_cost_usd, '0.000000000')} | "
            f"{fmt_decimal(r.vnd_per_unit)} | {markup} | {unit_price} |"
        )
    return "\n".join(rows)


def render_workloads() -> str:
    workloads = [
        Workload("Phụ đề, 1 ngôn ngữ"),
        Workload("Phụ đề, 1 ngôn ngữ, **không glossary**", glossary=False),
        Workload("Phụ đề, 3 ngôn ngữ", languages=D("3")),
        Workload("**Dub standard, 1 ngôn ngữ**", dubbing="standard"),
        Workload("Dub standard, 3 ngôn ngữ", languages=D("3"), dubbing="standard"),
        Workload("Dub voice-clone, 1 ngôn ngữ", dubbing="voice_clone"),
        Workload("**1 câu hỏi AI Assistant** (4.000 in / 500 out)", assistant_question=True),
    ]
    rows = [
        "| Cấu hình | STT | Translation | TTS | Summary | **Tổng credit** | "
        "**₫ bán lẻ** | Chi phí provider | **Markup** |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for item in workloads:
        name, stt, translation, tts, summary, total, provider_cost = workload_row(item)
        retail = total * CREDIT_VALUE_VND
        markup = retail / provider_cost if provider_cost else D("0")
        rows.append(
            f"| {name} | {fmt_int(stt) if stt else '-'} | "
            f"{fmt_int(translation) if translation else '-'} | "
            f"{fmt_int(tts) if tts else '-'} | {fmt_int(summary) if summary else '-'} | "
            f"**{fmt_int(total)}** | {fmt_int(retail)} ₫ | {fmt_int(provider_cost)} ₫ | "
            f"**{fmt_decimal(markup, '0.00')}×** |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=MASTER_PLAN_PATH,
        help="Path to credit-billing-master-plan.md used to verify §3.1/§3.2 inputs.",
    )
    parser.add_argument(
        "--skip-master-plan-check",
        action="store_true",
        help="Generate from local constants without verifying against the master plan.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.skip_master_plan_check:
        verify_master_plan_inputs(read_master_plan_inputs(args.plan_file))

    print("### 3.3 Rate card")
    print()
    print(render_rate_card())
    print()
    print("### 3.5 Workload tham chiếu - 1 giờ họp")
    print()
    print(render_workloads())


if __name__ == "__main__":
    main()
