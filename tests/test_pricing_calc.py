"""Tests for Phase 5 pricing table generation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_pricing_calc():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "pricing_calc.py"
    spec = importlib.util.spec_from_file_location("pricing_calc", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pricing_calc_reads_master_plan_inputs() -> None:
    pricing_calc = _load_pricing_calc()

    inputs = pricing_calc.read_master_plan_inputs()

    assert inputs.fx_rate_usd_vnd == pricing_calc.D("26300")
    assert inputs.chars_per_sec == pricing_calc.D("12.5")
    assert inputs.talk_density == pricing_calc.D("0.70")
    assert inputs.vad_padding == pricing_calc.D("1.15")
    pricing_calc.verify_master_plan_inputs(inputs)


def test_pricing_calc_generates_core_rate_card_values() -> None:
    pricing_calc = _load_pricing_calc()

    assert pricing_calc.rate("STT", "second").unit_price == pricing_calc.D("1.643750")
    assert pricing_calc.rate("TRANSLATION", "token_in").unit_price == pricing_calc.D(
        "0.006575"
    )
    assert pricing_calc.rate("AUDIO_DUBBING_STANDARD", "character").unit_price == (
        pricing_calc.D("0.773220")
    )

    output = pricing_calc.render_rate_card()
    assert "`AI_ASSISTANT` | `token_out`" in output
    assert "**0,131500**" in output
