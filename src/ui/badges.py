"""Status badges for regions — the "chip" labels next to region names.

Classifies each region into a short qualitative label based on its current
production, growth, and decline profile. Used in the map tooltip and the
Workspace KPI header to add at-a-glance context.

Rules are intentionally simple — these aren't quantitative outputs, just a
way to make the numbers immediately interpretable.
"""

from __future__ import annotations

import pandas as pd

LEADING_PRODUCER_TOP_N = 1      # rank 1 → leader
HIGH_GROWTH_THRESHOLD = 5.0     # YoY % to qualify as "High Growth"
MATURE_DECLINE_THRESHOLD = 2.0  # decline rate %/yr to qualify as "Mature Decline"


def classify_region(
    region_id: str,
    fuel: str,
    year: int,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> tuple[str, str]:
    """Return (label, emoji) for a region at the given year.

    Precedence (first match wins):
        🏆 Leading Producer   — highest production in its fuel cohort at `year`
        📈 High Growth        — YoY > +5%
        📉 Mature Decline     — decline_rate > +2%/yr
        ⚪ Stable             — everything else
    """
    fuel_scope = actuals[actuals["fuel_type"] == fuel]
    year_rows = fuel_scope[fuel_scope["year"] == year]

    # Leading producer? Top production this year within its fuel cohort.
    if not year_rows.empty:
        top = year_rows.sort_values("production", ascending=False).head(
            LEADING_PRODUCER_TOP_N
        )
        if region_id in set(top["region_id"]):
            return ("Leading Producer", "🏆")

    # If year is beyond latest actuals, fall back to latest-actual growth
    region_rows = fuel_scope[fuel_scope["region_id"] == region_id]
    growth = _latest(region_rows, "growth_pct", year)
    decline = _latest(region_rows, "decline_rate_pct", year)

    if growth is not None and growth > HIGH_GROWTH_THRESHOLD:
        return ("High Growth", "📈")

    if decline is not None and decline > MATURE_DECLINE_THRESHOLD:
        return ("Mature Decline", "📉")

    return ("Stable", "⚪")


def badge_markdown(region_id: str, fuel: str, year: int,
                   actuals: pd.DataFrame, forecasts: pd.DataFrame) -> str:
    """Return a short markdown chip like '🏆 Leading Producer'."""
    label, emoji = classify_region(region_id, fuel, year, actuals, forecasts)
    return f"{emoji} **{label}**"


def _latest(df: pd.DataFrame, col: str, target_year: int) -> float | None:
    if df.empty or col not in df.columns:
        return None
    exact = df[df["year"] == target_year]
    if not exact.empty and pd.notna(exact[col].iloc[0]):
        return float(exact[col].iloc[0])
    hist = df.sort_values("year").dropna(subset=[col])
    if hist.empty:
        return None
    return float(hist.iloc[-1][col])
