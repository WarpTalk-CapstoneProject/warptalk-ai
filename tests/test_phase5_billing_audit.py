"""Tests for Phase 5 real-data billing audit helpers."""

from __future__ import annotations

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path


def _load_phase5_audit():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "phase5_billing_audit.py"
    spec = importlib.util.spec_from_file_location("phase5_billing_audit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_markup_matches_phase5_thresholds() -> None:
    audit = _load_phase5_audit()

    assert audit.classify_markup(None) == "missing_provider_dashboard_cost"
    assert audit.classify_markup(Decimal("0.99")) == "loss"
    assert audit.classify_markup(Decimal("1.00")) == "outside_phase5_target"
    assert audit.classify_markup(Decimal("1.49")) == "outside_phase5_target"
    assert audit.classify_markup(Decimal("1.50")) == "within_phase5_target"
    assert audit.classify_markup(Decimal("4.00")) == "within_phase5_target"
    assert audit.classify_markup(Decimal("4.01")) == "outside_phase5_target"


def test_audit_sql_uses_required_phase5_measurements() -> None:
    audit = _load_phase5_audit()

    assert "SUM(quantity)::numeric AS stt_billed_seconds" in audit.AUDIT_SQL_TEMPLATE
    assert "{charge_type_column} = 'STT'" in audit.AUDIT_SQL_TEMPLATE
    assert "translation_room.translation_room_sessions" in audit.AUDIT_SQL_TEMPLATE
    assert "COALESCE(SUM(credits_consumed), 0)::numeric * 4 AS retail_vnd" in (
        audit.AUDIT_SQL_TEMPLATE
    )
    assert "ROUND(c.retail_vnd / NULLIF(pc.provider_vnd, 0), 4) AS markup" in (
        audit.AUDIT_SQL_TEMPLATE
    )


def test_estimated_provider_cost_uses_rate_card_baseline() -> None:
    audit = _load_phase5_audit()

    assert audit.BASELINE_FX_RATE_USD_VND == Decimal("26300")
    assert "usage_rate_card" in audit.ESTIMATED_PROVIDER_COST_SQL
    assert "provider_unit_cost" in audit.ESTIMATED_PROVIDER_COST_SQL
    assert "jsonb_array_elements(" in (audit.ESTIMATED_PROVIDER_COST_SQL)
    assert "ur.details->'unit_breakdown'" in audit.ESTIMATED_PROVIDER_COST_SQL
    assert "ELSE '[]'::jsonb" in audit.ESTIMATED_PROVIDER_COST_SQL
    assert "(item->>'quantity')::numeric * urc.provider_unit_cost * $5::numeric" in (
        audit.ESTIMATED_PROVIDER_COST_SQL
    )
    assert "ur.quantity * urc.provider_unit_cost * $5::numeric" in (
        audit.ESTIMATED_PROVIDER_COST_SQL
    )
