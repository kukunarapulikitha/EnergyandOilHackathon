"""Investment Thesis builder — narrative verdict for a region.

Replaces the bare 0-100 Investment Score with a rule-based thesis that
explains WHY a region is attractive, citing real KPI numbers.

Verdict thresholds:
    score >= 70  →  "Pursue" (green)
    score >= 50  →  "Watch"  (amber)
    score <  50  →  "Pass"   (red)
"""

from __future__ import annotations

import pandas as pd

DEFAULT_WTI = 78.0
DEFAULT_HENRY_HUB = 3.40


def _fmt_billions(usd: float) -> str:
    if usd >= 1e9:
        return f"${usd/1e9:.1f}B"
    if usd >= 1e6:
        return f"${usd/1e6:.0f}M"
    return f"${usd:,.0f}"


def _verdict_from_score(score: float) -> tuple[str, str, str]:
    """Return (label, color_hex, action_text)."""
    if score >= 70:
        return ("Pursue", "#1b8a3a", "worth pursuing")
    if score >= 50:
        return ("Watch", "#d97706", "worth watching")
    return ("Pass", "#b91c1c", "low priority")


def build_investment_thesis(
    region_id: str,
    fuel: str,
    year: int,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    wti_price: float = DEFAULT_WTI,
    henry_hub: float = DEFAULT_HENRY_HUB,
) -> dict:
    """Return a structured investment thesis for one region in one year.

    Pulls real KPI values from actuals + forecasts; emits 3 narrative
    bullets and a verdict label backed by the investment_score.
    """
    fa = actuals[(actuals["fuel_type"] == fuel) & (actuals["region_id"] == region_id)]
    ff = forecasts[(forecasts["fuel_type"] == fuel) & (forecasts["region_id"] == region_id)]
    if fa.empty or ff.empty:
        return {
            "region_id": region_id,
            "region_name": region_id,
            "verdict": "—",
            "verdict_color": "#666",
            "title": f"{region_id} — insufficient data.",
            "bullets": [],
            "rationale": "No actuals or forecasts available for this region.",
            "score": None,
        }

    region_name = fa["region_name"].iloc[0]
    forecast_row = ff.iloc[0]
    score = float(forecast_row.get("investment_score") or 0)
    verdict, verdict_color, action_text = _verdict_from_score(score)

    # ---- compute live numbers ----------------------------------------------
    slope = float(forecast_row["slope"])
    intercept = float(forecast_row["intercept"])
    projected = max(slope * year + intercept, 0.0)

    # Revenue
    if fuel == "crude_oil":
        # Mb/d × 1000 bbl × 365 × $/bbl
        revenue_usd = projected * 1000 * 365 * wti_price
        price_label = f"${wti_price:.2f}/bbl WTI"
        unit_label = "Mb/d"
    else:
        # MMcf × 1,000,000 × $/Mcf — note: gas annual is MMcf not MMcf/month
        revenue_usd = projected * 1_000_000 * henry_hub
        price_label = f"${henry_hub:.2f}/Mcf Henry Hub"
        unit_label = "MMcf"

    # YoY growth at the selected year (or latest year for forecast targets)
    available_years = sorted(fa["year"].unique())
    last_actual_year = available_years[-1] if available_years else year
    growth_year = min(year, last_actual_year)
    growth_row = fa[fa["year"] == growth_year]
    yoy_growth = float(growth_row["growth_pct"].iloc[0]) if not growth_row.empty and pd.notna(growth_row["growth_pct"].iloc[0]) else None

    # Peer mean YoY for the same fuel (excluding this region)
    peers = actuals[
        (actuals["fuel_type"] == fuel)
        & (actuals["year"] == growth_year)
        & (actuals["region_id"] != region_id)
    ]
    peer_yoy = float(peers["growth_pct"].mean()) if not peers.empty else None

    # 5-year decline rate — production_now vs production_5yr_ago
    decline_5y = None
    if len(available_years) >= 6:
        latest_y = available_years[-1]
        five_back = latest_y - 5
        latest_p = fa[fa["year"] == latest_y]["production"].iloc[0] if latest_y in fa["year"].values else None
        old_p = fa[fa["year"] == five_back]["production"].iloc[0] if five_back in fa["year"].values else None
        if latest_p and old_p and old_p > 0:
            decline_5y = ((latest_p - old_p) / old_p) / 5 * 100

    # Volatility (CoV %) over trailing 10 years
    trail = fa[fa["year"] >= (available_years[-1] - 9 if available_years else year - 9)]
    if len(trail) >= 5 and trail["production"].mean() > 0:
        volatility_pct = float(trail["production"].std() / trail["production"].mean() * 100)
    else:
        volatility_pct = None

    if volatility_pct is None:
        vol_label = "limited history"
    elif volatility_pct < 8:
        vol_label = "very stable production"
    elif volatility_pct < 15:
        vol_label = "stable production"
    elif volatility_pct < 25:
        vol_label = "moderate volatility"
    else:
        vol_label = "high volatility"

    # ---- build bullets -----------------------------------------------------
    bullets: list[str] = []

    # B1: revenue scale
    bullets.append(
        f"Projected {year} revenue of {_fmt_billions(revenue_usd)}/yr at {price_label} "
        f"× {projected:,.0f} {unit_label} — "
        + ("meaningful scale for a BD pursuit." if revenue_usd >= 5e9 else "modest scale; verify niche fit.")
    )

    # B2: YoY growth + peer comparison
    if yoy_growth is not None:
        peer_clause = ""
        if peer_yoy is not None:
            delta = yoy_growth - peer_yoy
            direction = "outperforming" if delta > 0 else "underperforming"
            peer_clause = f", {direction} peer average by {abs(delta):+.1f}%"
        movement = "growth" if yoy_growth >= 0 else "contraction"
        bullets.append(
            f"Year-over-year {movement} of {yoy_growth:+.1f}% ({growth_year - 1}→{growth_year})"
            + peer_clause + "."
        )

    # B3: decline rate + volatility
    if decline_5y is not None:
        d_label = "declining" if decline_5y < 0 else "growing"
        bullets.append(
            f"{d_label.capitalize()} at {decline_5y:+.1f}% annualized over trailing 5 years"
            f" with {vol_label} (CoV {volatility_pct:.1f}% )."
            if volatility_pct is not None
            else f"{d_label.capitalize()} at {decline_5y:+.1f}% annualized over trailing 5 years."
        )
    elif volatility_pct is not None:
        bullets.append(f"Volatility profile: {vol_label} (CoV {volatility_pct:.1f}%).")

    # Forecast quality
    r2 = float(forecast_row.get("r_squared") or 0)
    if r2 < 0.6:
        bullets.append(
            f"⚠️ Forecast R² is {r2:.2f} — trend is noisy; treat the projection as directional only."
        )
    elif r2 >= 0.85:
        bullets.append(
            f"Forecast R² of {r2:.2f} — strong linear trend, projection is grounded."
        )

    return {
        "region_id": region_id,
        "region_name": region_name,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "title": f"{region_name} {fuel.replace('_', ' ')} is {action_text}.",
        "bullets": bullets,
        "rationale": (
            "Rule-based verdict — scored from revenue scale, YoY growth, decline rate, "
            "volatility, and forecast quality (Investment Score = "
            f"{score:.0f}/100). Every number is computed live from EIA Gold-layer data."
        ),
        "score": score,
        "projected": projected,
        "revenue_usd": revenue_usd,
    }


def build_revenue_sensitivity_matrix(
    forecast_row: pd.Series,
    target_year: int,
    wti_price: float = DEFAULT_WTI,
    price_shocks: tuple[float, ...] = (-0.40, -0.20, -0.10, 0.0, 0.10, 0.20),
    production_shocks: tuple[float, ...] = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20),
) -> tuple[pd.DataFrame, float]:
    """2D shock matrix: rows = price shock %, cols = production shock %.

    Each cell = base_revenue × (1 + price_shock) × (1 + prod_shock).
    Returns (matrix_df, baseline_usd).
    """
    slope = float(forecast_row["slope"])
    intercept = float(forecast_row["intercept"])
    base_production = max(slope * target_year + intercept, 0.0)
    base_revenue = base_production * 1000 * 365 * wti_price  # crude only

    rows = []
    for ps in price_shocks:
        row = []
        for prs in production_shocks:
            rev = base_revenue * (1 + ps) * (1 + prs)
            row.append(rev)
        rows.append(row)

    df = pd.DataFrame(
        rows,
        index=[f"{ps*100:+.0f}%" for ps in price_shocks],
        columns=[f"{prs*100:+.0f}%" for prs in production_shocks],
    )
    df.index.name = "↓ Price / Production →"
    return df, base_revenue


def build_decline_price_matrix(
    forecast_row: pd.Series,
    target_year: int,
    fuel: str,
    decline_scenarios: tuple[float, ...] = (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25),
    crude_prices: tuple[float, ...] = (50.0, 60.0, 70.0, 80.0, 90.0, 100.0),
    gas_prices: tuple[float, ...] = (2.0, 2.50, 3.0, 3.50, 4.0, 4.50),
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    """Decline rate × price scenario matrix for production and revenue.

    Rows (↓): annual decline rate scenarios applied from trained_through_year
              to target_year. Negative rate = production growth.
    Columns (→): price assumptions (WTI $/bbl for crude; Henry Hub $/MMBtu for gas).

    Each cell:
        years_ahead = max(target_year - trained_through_year, 0)
        adjusted_prod = base_production × (1 − decline_rate)^years_ahead
        revenue (crude) = adjusted_prod × 1000 × 365 × price
        revenue (gas)   = adjusted_prod × 1_000_000 × price

    Note: production is constant within each row (price doesn't affect volume).
    Returns (prod_df, rev_df, base_production, trained_through_year).
    """
    slope = float(forecast_row["slope"])
    intercept = float(forecast_row["intercept"])
    trained_through = int(forecast_row["trained_through_year"])
    base_production = max(slope * target_year + intercept, 0.0)
    years_ahead = max(target_year - trained_through, 0)

    prices = crude_prices if fuel == "crude_oil" else gas_prices

    prod_rows, rev_rows = [], []
    for dr in decline_scenarios:
        adjusted = max(base_production * ((1 - dr) ** years_ahead), 0.0)
        prod_row, rev_row = [], []
        for price in prices:
            revenue = (
                adjusted * 1000 * 365 * price
                if fuel == "crude_oil"
                else adjusted * 1_000_000 * price
            )
            prod_row.append(adjusted)
            rev_row.append(revenue)
        prod_rows.append(prod_row)
        rev_rows.append(rev_row)

    row_labels = [f"{dr*100:+.0f}%/yr" for dr in decline_scenarios]
    col_labels = (
        [f"${p:.0f}/bbl" for p in prices]
        if fuel == "crude_oil"
        else [f"${p:.2f}/MMBtu" for p in prices]
    )

    prod_df = pd.DataFrame(prod_rows, index=row_labels, columns=col_labels)
    rev_df = pd.DataFrame(rev_rows, index=row_labels, columns=col_labels)
    prod_df.index.name = "↓ Decline Rate / Price →"
    rev_df.index.name = "↓ Decline Rate / Price →"

    return prod_df, rev_df, base_production, float(trained_through)
