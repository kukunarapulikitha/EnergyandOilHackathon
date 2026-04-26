"""KPI calculations for regional investment analysis.

KPI set (defined in detail in docs/kpi_definitions.md):
    1. projected_production_estimate  (required by Tier 1)
    2. yoy_growth_rate                 (Tier 2)
    3. volatility_score                (Tier 2)
    4. revenue_potential               (Tier 2)
    5. investment_score                (Tier 3 — composite)

All KPIs accept annual DataFrames produced by normalize.aggregate_annual()
or a ForecastResult for forward-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecast.linear import ForecastResult

DEFAULT_WTI_PRICE_USD_PER_BBL = 78.0  # rolling ~2024-2025 average
DAYS_PER_YEAR = 365


def projected_production_estimate(
    forecast: ForecastResult, target_year: int
) -> float:
    """Projected thousand barrels/day for a target year.

    If the target_year is within the actuals range, returns the actual.
    Otherwise returns the linear forecast.
    """
    if not forecast.actuals.empty:
        match = forecast.actuals[forecast.actuals["year"] == target_year]
        if not match.empty:
            return float(match.iloc[0]["production"])
    if not forecast.forecast.empty:
        match = forecast.forecast[forecast.forecast["year"] == target_year]
        if not match.empty:
            return float(match.iloc[0]["production"])
    # Fallback: extrapolate from the fit
    return float(forecast.slope * target_year + forecast.intercept)


def yoy_growth_rate(annual_df: pd.DataFrame, region_id: str, year: int) -> float:
    """Year-over-year production growth (%) for a region at a given year."""
    region = annual_df[annual_df["region_id"] == region_id].sort_values("year")
    prev = region[region["year"] == year - 1]
    curr = region[region["year"] == year]
    if prev.empty or curr.empty:
        return float("nan")
    p, c = prev.iloc[0]["production"], curr.iloc[0]["production"]
    if p == 0:
        return float("nan")
    return float((c - p) / p * 100)


def volatility_score(annual_df: pd.DataFrame, region_id: str) -> float:
    """Coefficient of variation (std / mean) across all historical years.

    Lower = more consistent production. Returned as a percentage.
    """
    region = annual_df[annual_df["region_id"] == region_id]
    if region.empty or region["production"].mean() == 0:
        return float("nan")
    return float(region["production"].std() / region["production"].mean() * 100)


def decline_rate(annual_df: pd.DataFrame, region_id: str, window: int = 5) -> float:
    """Annualized production decline rate over the last `window` years (%).

    Positive value = declining. Useful for mature basins. Computed as the
    slope of the last N years divided by the mean — then negated so that
    a POSITIVE number indicates decline.
    """
    region = (
        annual_df[annual_df["region_id"] == region_id]
        .sort_values("year")
        .tail(window)
    )
    if len(region) < 2:
        return float("nan")
    years = region["year"].to_numpy(dtype=float)
    prod = region["production"].to_numpy(dtype=float)
    mean = prod.mean()
    if mean == 0:
        return float("nan")
    # simple OLS slope = cov(x,y) / var(x)
    slope = ((years - years.mean()) * (prod - mean)).sum() / (
        (years - years.mean()) ** 2
    ).sum()
    return float(-slope / mean * 100)


def relative_performance_index(
    annual_df: pd.DataFrame, region_id: str, year: int
) -> float:
    """Percentile rank of (growth × 1/volatility) vs peers at a given year.

    0 = worst performer among peers, 100 = best. Combines recent growth
    with production stability — higher is better on both dimensions.
    Only compares within the same fuel_type.
    """
    peers = annual_df[annual_df["year"] == year]
    if peers.empty:
        return float("nan")

    scores: dict[str, float] = {}
    for rid in peers["region_id"].unique():
        g = yoy_growth_rate(annual_df, rid, year)
        v = volatility_score(annual_df, rid)
        if any(pd.isna(x) for x in (g, v)) or v == 0:
            continue
        scores[rid] = g / v  # growth per unit of volatility

    if region_id not in scores or len(scores) < 2:
        return float("nan")

    ordered = sorted(scores.values())
    rank = ordered.index(scores[region_id])
    return float(rank / (len(ordered) - 1) * 100)


def revenue_potential(
    projected_mbd: float,
    wti_price: float = DEFAULT_WTI_PRICE_USD_PER_BBL,
) -> float:
    """Estimated annual revenue (USD) from projected production.

    projected_mbd: thousand barrels per day
    revenue = projected_mbd * 1000 * days_per_year * wti_price
    """
    if np.isnan(projected_mbd):
        return float("nan")
    return float(projected_mbd * 1000 * DAYS_PER_YEAR * wti_price)


def investment_score(
    forecast: ForecastResult,
    annual_df: pd.DataFrame,
    region_id: str,
    target_year: int,
) -> dict:
    """Composite 0-100 investment attractiveness score.

    Components (equal-weighted, normalized):
        - Projected production (scale-relative to max region)
        - Growth rate (positive is good)
        - Reliability (R² of the linear fit)
        - Inverse volatility (stable production preferred)

    Returns a dict with the score and component breakdown for transparency.
    """
    proj = projected_production_estimate(forecast, target_year)
    growth = yoy_growth_rate(annual_df, region_id, min(target_year, int(annual_df["year"].max())))
    vol = volatility_score(annual_df, region_id)
    max_prod = annual_df["production"].max()

    prod_score = np.clip(proj / max_prod, 0, 1) * 100 if max_prod else 0
    growth_score = np.clip((growth + 20) / 40, 0, 1) * 100 if not np.isnan(growth) else 50
    reliability_score = np.clip(forecast.r_squared, 0, 1) * 100
    stability_score = np.clip(1 - (vol / 50), 0, 1) * 100 if not np.isnan(vol) else 50

    composite = float(
        np.mean([prod_score, growth_score, reliability_score, stability_score])
    )

    return {
        "score": round(composite, 1),
        "components": {
            "projected_production": round(float(prod_score), 1),
            "growth": round(float(growth_score), 1),
            "reliability": round(float(reliability_score), 1),
            "stability": round(float(stability_score), 1),
        },
        "raw": {
            "projected_mbd": round(float(proj), 2),
            "growth_pct": round(float(growth), 2) if not np.isnan(growth) else None,
            "r_squared": round(float(forecast.r_squared), 3),
            "volatility_pct": round(float(vol), 2) if not np.isnan(vol) else None,
        },
    }
