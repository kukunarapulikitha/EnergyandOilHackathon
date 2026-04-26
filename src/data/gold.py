"""Gold layer — business-ready analytical tables.

Gold is split into two tables so that data-backed facts and model-based
inferences remain structurally separated. This directly supports the
"distinguish data-backed claims from model-generated inference"
requirement in the problem statement.

Outputs (consumed by Streamlit and AI prompts):

    data/gold/regional_actuals.csv    — one row per (region, year, fuel)
        Observed production + KPIs computed from observed data only.
        No forecasts. Fully reproducible from Silver.

    data/gold/region_forecasts.csv    — one row per (region, fuel)
        Linear-regression parameters (slope, intercept, R²) and the
        training window. Projections are computed at read time as
        `slope * year + intercept`. Re-fitting for alternate cutoff years
        happens live in the UI via `src/forecast/linear.py`.

Streamlit reads both and stitches the chart on every slider change.
"""

from __future__ import annotations

import pandas as pd

from src.forecast.linear import fit_and_forecast
from src.kpi.calculations import (
    decline_rate,
    investment_score,
    relative_performance_index,
    revenue_potential,
    volatility_score,
    yoy_growth_rate,
)

FORECAST_METHOD = "linear_ols"
DEFAULT_HORIZON_END = 2035
CRUDE = "crude_oil"


# ============================================================ ACTUALS


def build_regional_actuals(
    annual_silver: pd.DataFrame,
    wti_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Gold table of observed annual production + derived KPIs.

    Every row is a real measurement (no projections). KPIs in this table
    depend only on historical data (growth, volatility, revenue at the
    latest spot price).
    """
    if annual_silver.empty:
        return pd.DataFrame(
            columns=[
                "region_id", "region_name", "year", "fuel_type",
                "production", "unit",
                "growth_pct", "volatility_pct", "decline_rate_pct",
                "relative_performance_index", "revenue_potential_usd",
                "source", "fetched_at",
            ]
        )

    latest_wti = _latest_wti(wti_prices)
    rows: list[dict] = []

    for (fuel, region_id), g in annual_silver.groupby(
        ["fuel_type", "region_id"], sort=False
    ):
        g = g.sort_values("year")
        fuel_scope = annual_silver[annual_silver["fuel_type"] == fuel]
        region_name = g["region_name"].iloc[0]
        unit = g["unit"].iloc[0]
        source = g["source"].iloc[0]
        vol = volatility_score(g, region_id)
        decline = decline_rate(g, region_id)

        for _, r in g.iterrows():
            year = int(r["year"])
            production = float(r["production"])
            growth = yoy_growth_rate(g, region_id, year)
            rpi = relative_performance_index(fuel_scope, region_id, year)
            rev = _revenue_for_fuel(fuel, production, latest_wti)

            rows.append({
                "region_id": region_id,
                "region_name": region_name,
                "year": year,
                "fuel_type": fuel,
                "production": production,
                "unit": unit,
                "growth_pct": None if pd.isna(growth) else round(growth, 2),
                "volatility_pct": None if pd.isna(vol) else round(vol, 2),
                "decline_rate_pct": None if pd.isna(decline) else round(decline, 2),
                "relative_performance_index": None if pd.isna(rpi) else round(rpi, 1),
                "revenue_potential_usd": rev,
                "source": source,
                "fetched_at": r.get("fetched_at"),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(["fuel_type", "region_id", "year"])
        .reset_index(drop=True)
    )


# ============================================================ FORECASTS


def build_region_forecasts(
    annual_silver: pd.DataFrame,
    horizon_end: int = DEFAULT_HORIZON_END,
) -> pd.DataFrame:
    """Gold table of linear-regression parameters per (region, fuel).

    Stores model parameters only — no projected values. Projections are
    computed at read time: `production = slope * year + intercept`.

    The training window uses ALL available history (latest actual year).
    For alternate cutoffs (e.g., when the UI slider is dragged backward),
    Streamlit calls `fit_and_forecast()` live.
    """
    cols = [
        "region_id", "region_name", "fuel_type",
        "slope", "intercept", "r_squared",
        "trained_through_year", "horizon_end",
        "method", "investment_score", "source",
    ]
    if annual_silver.empty:
        return pd.DataFrame(columns=cols)

    latest_year = int(annual_silver["year"].max())
    rows: list[dict] = []

    for (fuel, region_id), g in annual_silver.groupby(
        ["fuel_type", "region_id"], sort=False
    ):
        g = g.sort_values("year")
        if len(g) < 2:
            continue

        try:
            fc = fit_and_forecast(
                annual_silver[annual_silver["fuel_type"] == fuel],
                region_id=region_id,
                selected_year=latest_year,
                horizon_end=horizon_end,
            )
        except ValueError:
            continue

        score = investment_score(
            fc,
            annual_silver[annual_silver["fuel_type"] == fuel],
            region_id,
            latest_year,
        )

        rows.append({
            "region_id": region_id,
            "region_name": g["region_name"].iloc[0],
            "fuel_type": fuel,
            "slope": round(fc.slope, 4),
            "intercept": round(fc.intercept, 4),
            "r_squared": round(fc.r_squared, 4),
            "trained_through_year": latest_year,
            "horizon_end": horizon_end,
            "method": FORECAST_METHOD,
            "investment_score": score["score"],
            "source": f"model:{FORECAST_METHOD} trained on "
                      f"EIA annual {g['year'].min()}–{latest_year}",
        })

    return (
        pd.DataFrame(rows, columns=cols)
        .sort_values(["fuel_type", "region_id"])
        .reset_index(drop=True)
    )


def project(forecasts_row: pd.Series, year: int) -> float:
    """Apply a forecast row to a target year: slope * year + intercept."""
    return float(forecasts_row["slope"] * year + forecasts_row["intercept"])


# ============================================================ helpers


def _latest_wti(wti: pd.DataFrame | None) -> float | None:
    if wti is None or wti.empty:
        return None
    return float(wti.sort_values("date").iloc[-1]["price_usd_per_bbl"])


def _revenue_for_fuel(
    fuel: str, production_value: float, wti_price: float | None
) -> float | None:
    if fuel != CRUDE:
        return None
    if wti_price is None:
        return round(revenue_potential(production_value), 2)
    return round(revenue_potential(production_value, wti_price=wti_price), 2)
