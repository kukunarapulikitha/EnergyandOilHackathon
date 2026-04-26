"""Pure-Python well economics: Arps decline curve + cashflow + NPV/IRR/payback.

No Streamlit imports. Fully testable in isolation via pytest.

Conventions:
- Crude:   qi in bbl/d, price in $/bbl, LOE in $/bbl
- Gas:     qi in Mcf/d, price in $/Mcf, LOE in $/Mcf
- Severance: decimal fraction (0.075 = 7.5%)
- Discount rate: annual decimal (0.10 = 10%)
"""

from __future__ import annotations

import numpy as np

DAYS_PER_MONTH = 30.4375  # 365.25 / 12 — keeps annual cumulative in line


def arps_production(
    qi: float,
    Di: float,
    b: float,
    months: int,
    min_decline: float = 0.06,
) -> np.ndarray:
    """Hyperbolic Arps decline with a terminal exponential switch.

    qi: initial daily rate (bbl/d for crude, Mcf/d for gas).
    Di: initial nominal annual decline rate (e.g. 0.65 = 65%/yr).
    b:  hyperbolic exponent. b=0 → exponential, b=1 → harmonic, 1.0–1.3
        typical for unconventional shale.
    months: well-life horizon (rate is sampled at month index 0 = first month).
    min_decline: terminal exponential floor. When the instantaneous decline
        rate falls below this, switch to exponential at min_decline. This
        keeps the long tail realistic (bare hyperbolic over-recovers).
    """
    months = max(int(months), 1)
    t_years = np.arange(months) / 12.0

    if b < 1e-6:
        return qi * np.exp(-Di * t_years)

    # Hyperbolic phase
    q_hyp = qi / (1 + b * Di * t_years) ** (1.0 / b)

    # Instantaneous nominal decline of the hyperbolic curve at time t:
    #   D(t) = Di / (1 + b * Di * t)
    # Find the year at which D(t) drops to min_decline:
    #   t_switch = (Di / min_decline - 1) / (b * Di)
    if Di <= min_decline:
        return q_hyp  # already below the floor — pure hyperbolic
    t_switch = (Di / min_decline - 1.0) / (b * Di)

    if t_switch >= t_years[-1]:
        return q_hyp  # switch never happens within the well life

    q_at_switch = qi / (1 + b * Di * t_switch) ** (1.0 / b)
    q_exp_tail = q_at_switch * np.exp(-min_decline * (t_years - t_switch))
    return np.where(t_years < t_switch, q_hyp, q_exp_tail)


def compute_eur(
    monthly_rate: np.ndarray,
    days_per_month: float = DAYS_PER_MONTH,
) -> float:
    """Estimated Ultimate Recovery — cumulative production over the horizon.
    Returns volume in bbl (crude) or Mcf (gas).
    """
    return float(np.sum(monthly_rate * days_per_month))


def monthly_cashflow(
    monthly_rate: np.ndarray,
    price: float,
    loe: float,
    severance_pct: float,
    capex: float,
    days_per_month: float = DAYS_PER_MONTH,
) -> np.ndarray:
    """Pre-tax monthly cashflow with day-zero capex booked at index 0.

    Returns: array of cashflows in $.
    """
    monthly_volume = monthly_rate * days_per_month
    revenue = monthly_volume * price
    severance = revenue * severance_pct
    opex = monthly_volume * loe
    cf = revenue - severance - opex
    cf = cf.copy()
    cf[0] = cf[0] - capex
    return cf


def npv(
    cashflows: np.ndarray,
    annual_rate: float = 0.10,
    periods_per_year: int = 12,
) -> float:
    """Discounted NPV of monthly cashflows at the given annual rate."""
    monthly_rate = (1 + annual_rate) ** (1.0 / periods_per_year) - 1
    discount = (1 + monthly_rate) ** np.arange(len(cashflows))
    return float(np.sum(cashflows / discount))


def irr(
    cashflows: np.ndarray,
    periods_per_year: int = 12,
) -> float | None:
    """Annualized IRR. Returns None if no real solution within [-99%, +500%].

    Tries numpy_financial first (fast), falls back to scipy.optimize.brentq.
    """
    try:
        import numpy_financial as nf  # type: ignore[import-not-found]
        monthly = nf.irr(cashflows)
    except Exception:
        try:
            from scipy.optimize import brentq

            def _npv_at(rate: float) -> float:
                return float(
                    np.sum(cashflows / (1 + rate) ** np.arange(len(cashflows)))
                )

            try:
                monthly = brentq(_npv_at, -0.99, 5.0)
            except (ValueError, RuntimeError):
                return None
        except ImportError:
            return None

    if monthly is None or (isinstance(monthly, float) and np.isnan(monthly)):
        return None
    return float((1 + monthly) ** periods_per_year - 1)


def payback_period_months(cashflows: np.ndarray) -> int | None:
    """First month-index at which cumulative cashflow reaches >= 0.

    Returns None if the well never pays back within the modelled horizon.
    """
    cum = np.cumsum(cashflows)
    pos = np.where(cum >= 0)[0]
    return int(pos[0]) if len(pos) else None


def total_revenue(
    monthly_rate: np.ndarray,
    price: float,
    days_per_month: float = DAYS_PER_MONTH,
) -> float:
    """Gross revenue (no costs deducted) over the well life."""
    return float(np.sum(monthly_rate * days_per_month * price))
