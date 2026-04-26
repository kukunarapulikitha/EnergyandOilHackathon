"""Unit tests for the well economics math module.

Run:  pytest tests/test_well_model.py -v

These tests are deliberately self-contained — no Streamlit, no Gold data,
no network. They verify the mathematical contracts of each function.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.economics.well_model import (
    arps_production,
    compute_eur,
    irr,
    monthly_cashflow,
    npv,
    payback_period_months,
    total_revenue,
)


# ------------------------------------------------------------------ arps


def test_arps_at_t0_returns_ip():
    """First month rate must equal qi exactly (no elapsed time yet)."""
    q = arps_production(qi=1000.0, Di=0.60, b=1.0, months=120)
    assert abs(q[0] - 1000.0) < 1e-6


def test_arps_exponential_decay_b_zero():
    """When b=0 the curve is pure exponential: q(t) = qi * exp(-Di*t)."""
    q = arps_production(qi=1000.0, Di=0.50, b=0.0, months=25)
    # After 1 year (month 12): expected = 1000 * exp(-0.5) ≈ 606.53
    expected = 1000.0 * np.exp(-0.50)
    assert abs(q[12] - expected) < 1.0, f"got {q[12]:.2f}, expected {expected:.2f}"


def test_arps_rate_is_monotonically_decreasing():
    """Production never increases month-over-month (Arps is strictly declining)."""
    q = arps_production(qi=900.0, Di=0.65, b=1.1, months=240)
    diffs = np.diff(q)
    assert np.all(diffs <= 1e-9), "found a month where production increased"


def test_arps_single_month_returns_length_one():
    """Edge case: months=1 should return a single-element array."""
    q = arps_production(qi=500.0, Di=0.5, b=1.0, months=1)
    assert len(q) == 1
    assert abs(q[0] - 500.0) < 1e-6


# ------------------------------------------------------------------ EUR


def test_eur_monotonically_increasing_with_months():
    """EUR grows (or stays flat) as well life is extended — never shrinks."""
    base = arps_production(qi=900.0, Di=0.60, b=1.1, months=120)
    ext = arps_production(qi=900.0, Di=0.60, b=1.1, months=240)
    assert compute_eur(ext) >= compute_eur(base)


# ------------------------------------------------------------------ cashflow / NPV


def test_npv_positive_for_permian_defaults():
    """Permian-like well (IP=950, Di=0.62, b=1.1, capex=$8.5M, price=$78) should pencil."""
    q = arps_production(qi=950.0, Di=0.62, b=1.1, months=360)
    cf = monthly_cashflow(q, price=78.0, loe=10.0, severance_pct=0.075, capex=8_500_000)
    assert npv(cf, 0.10) > 0, "Permian defaults at $78 WTI should be NPV-positive"


def test_npv_negative_for_clearly_uneconomic_well():
    """Tiny well, low price, huge capex should yield negative NPV."""
    q = arps_production(qi=50.0, Di=0.80, b=1.0, months=60)
    cf = monthly_cashflow(q, price=20.0, loe=18.0, severance_pct=0.10, capex=12_000_000)
    assert npv(cf, 0.10) < 0


def test_cashflow_t0_includes_capex():
    """Month-0 cashflow must be (revenue_0 - costs_0 - capex)."""
    q = arps_production(qi=1000.0, Di=0.60, b=1.0, months=24)
    capex = 5_000_000
    cf = monthly_cashflow(q, price=70.0, loe=12.0, severance_pct=0.07, capex=capex)
    # Month-0 revenue
    vol_0 = q[0] * 30.4375
    rev_0 = vol_0 * 70.0
    cost_0 = rev_0 * 0.07 + vol_0 * 12.0
    expected_cf0 = rev_0 - cost_0 - capex
    assert abs(cf[0] - expected_cf0) < 0.01


# ------------------------------------------------------------------ IRR


def test_irr_returns_none_when_well_never_recovers():
    """A well that never produces enough to cover capex should return None or <0."""
    q = arps_production(qi=10.0, Di=0.90, b=1.0, months=36)
    cf = monthly_cashflow(q, price=15.0, loe=12.0, severance_pct=0.10, capex=8_000_000)
    result = irr(cf)
    assert result is None or result < 0, f"expected None or negative IRR, got {result}"


def test_irr_positive_for_profitable_well():
    """A clearly profitable well should return a positive annualized IRR.

    Skipped if neither numpy_financial nor scipy is installed (CI without extras).
    """
    try:
        import numpy_financial  # noqa: F401
    except ImportError:
        try:
            import scipy  # noqa: F401
        except ImportError:
            pytest.skip("Neither numpy_financial nor scipy installed — IRR solver unavailable")
    q = arps_production(qi=950.0, Di=0.62, b=1.1, months=360)
    cf = monthly_cashflow(q, price=78.0, loe=10.0, severance_pct=0.075, capex=8_500_000)
    result = irr(cf)
    assert result is not None and result > 0, f"expected positive IRR, got {result}"


# ------------------------------------------------------------------ payback


def test_payback_returns_int_for_profitable_well():
    """Payback must be an int in a plausible range for Permian defaults."""
    q = arps_production(qi=950.0, Di=0.62, b=1.1, months=360)
    cf = monthly_cashflow(q, price=78.0, loe=10.0, severance_pct=0.075, capex=8_500_000)
    pb = payback_period_months(cf)
    assert pb is not None, "expected a payback period, got None"
    assert isinstance(pb, int)
    assert 1 <= pb <= 60, f"payback {pb} months outside expected 1–60 range"


def test_payback_returns_none_when_unrecoverable():
    """If cumulative cashflow never turns positive, payback must be None."""
    q = arps_production(qi=10.0, Di=0.90, b=1.0, months=24)
    cf = monthly_cashflow(q, price=15.0, loe=12.0, severance_pct=0.10, capex=8_000_000)
    assert payback_period_months(cf) is None


# ------------------------------------------------------------------ total revenue


def test_total_revenue_equals_sum_of_monthly_revenues():
    """total_revenue() must match manual summation of volume × price."""
    q = arps_production(qi=800.0, Di=0.55, b=1.0, months=120)
    price = 75.0
    expected = float(np.sum(q * 30.4375 * price))
    assert abs(total_revenue(q, price) - expected) < 1e-3
