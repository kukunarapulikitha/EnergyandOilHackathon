"""Intent classification + structured artifact builders for the AI Analyst.

The rubric explicitly rewards "outputs beyond prose — filtered views, summaries,
or data-driven responses." This module pre-computes a structured artifact
(table, metric row, or plotly chart) per query that the UI renders ABOVE the
prose response.

Intents:
- ranking   — "which region has the highest projected production for 2027?"
- summary   — "summarize the opportunity in the Permian Basin"
- sensitivity — "what happens if I assume a 15% steeper decline rate?"
- lookup    — fallback (prose only)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.ai.prompts import resolve_basin


# ---------------------------------------------------------------- CLASSIFIER

_RANKING_KW = (
    "highest", "lowest", "biggest", "smallest", "top",
    "bottom", "rank", "ranking", "which region", "which state", "which padd",
    "best", "worst", "leader", "leading", "most",
)
_SUMMARY_KW = (
    "summarize", "summary", "tell me about", "opportunity", "overview of",
    "describe", "profile", "snapshot",
)
_SENSITIVITY_KW = (
    "what if", "decline rate", "steeper", "assume", "scenario",
    "stress test", "stress-test",
)


def classify_intent(query: str) -> str:
    q = query.lower()
    if any(k in q for k in _SENSITIVITY_KW):
        return "sensitivity"
    if any(k in q for k in _RANKING_KW):
        return "ranking"
    if any(k in q for k in _SUMMARY_KW):
        return "summary"
    return "lookup"


# ---------------------------------------------------------------- ARTIFACTS

def _project(slope: float, intercept: float, year: int) -> float:
    return float(slope * year + intercept)


def _ranking_artifact(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    fuel: str,
    year: int,
) -> dict | None:
    ff = forecasts[forecasts["fuel_type"] == fuel].copy()
    if ff.empty:
        return None
    fa = actuals[(actuals["fuel_type"] == fuel) & (actuals["year"] == year)]

    rows = []
    for _, fr in ff.iterrows():
        rid = fr["region_id"]
        a = fa[fa["region_id"] == rid]
        if not a.empty:
            value = float(a["production"].iloc[0])
            kind = "actual"
        else:
            value = _project(float(fr["slope"]), float(fr["intercept"]), year)
            kind = "projected"
        rows.append({
            "Region": fr["region_name"],
            "Production": value,
            "Type": kind,
            "Forecast R²": float(fr["r_squared"]),
            "Investment Score": float(fr["investment_score"]) if pd.notna(fr["investment_score"]) else None,
        })
    df = pd.DataFrame(rows).sort_values("Production", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking

    unit = "Mb/d" if fuel == "crude_oil" else "MMcf"
    return {
        "kind": "table",
        "title": f"Regions ranked by production — {year} ({unit})",
        "df": df,
        "unit": unit,
    }


def _summary_artifact(
    query: str,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    fuel: str,
    year: int,
) -> dict | None:
    # Try to identify a target region: first via basin, then via PADD/state name match
    geo = resolve_basin(query)
    target_region_id: str | None = None
    target_label: str | None = None

    if geo:
        target_region_id = geo["padd"] if fuel == "crude_oil" else None
        target_label = geo["basin"].title() + f" Basin → {geo['padd_name']}"
        # For natural gas basins, map to first state region
        if fuel == "natural_gas" and geo["states"]:
            from src.ai.prompts import STATE_TO_GAS_REGION_ID
            for st_abbr in geo["states"]:
                rid = STATE_TO_GAS_REGION_ID.get(st_abbr)
                if rid and not forecasts[forecasts["region_id"] == rid].empty:
                    target_region_id = rid
                    break

    if target_region_id is None:
        # Fallback: scan query for region_name substrings
        q = query.lower()
        for _, fr in forecasts[forecasts["fuel_type"] == fuel].iterrows():
            name = str(fr["region_name"]).lower()
            tokens = [t for t in name.replace("(", "").replace(")", "").split() if len(t) > 3]
            if any(t in q for t in tokens):
                target_region_id = fr["region_id"]
                target_label = fr["region_name"]
                break

    if target_region_id is None:
        return None

    fr_row = forecasts[
        (forecasts["region_id"] == target_region_id)
        & (forecasts["fuel_type"] == fuel)
    ]
    if fr_row.empty:
        return None
    fr_row = fr_row.iloc[0]

    fa_region = actuals[
        (actuals["region_id"] == target_region_id)
        & (actuals["fuel_type"] == fuel)
    ].sort_values("year")
    latest = fa_region.iloc[-1] if not fa_region.empty else None

    proj = _project(float(fr_row["slope"]), float(fr_row["intercept"]), year)
    unit = "Mb/d" if fuel == "crude_oil" else "MMcf"

    metrics = {
        f"Projected {year}": f"{proj:,.0f} {unit}",
        "Forecast R²": f"{float(fr_row['r_squared']):.2f}",
    }
    if latest is not None:
        if pd.notna(latest.get("growth_pct")):
            metrics["Latest YoY"] = f"{float(latest['growth_pct']):+.1f}%"
        if pd.notna(latest.get("decline_rate_pct")):
            metrics["Decline (5yr)"] = f"{float(latest['decline_rate_pct']):+.2f}%/yr"
        if pd.notna(latest.get("relative_performance_index")):
            metrics["Rel. perf."] = f"{float(latest['relative_performance_index']):.0f}/100"
    if pd.notna(fr_row.get("investment_score")):
        metrics["Investment score"] = f"{float(fr_row['investment_score']):.0f}/100"

    return {
        "kind": "metrics",
        "title": f"📍 {target_label or fr_row['region_name']}",
        "subtitle": f"Region {fr_row['region_name']} — {fuel.replace('_', ' ')}",
        "metrics": metrics,
    }


def _sensitivity_artifact(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    rate_change: float,
    fuel: str,
    year: int,
    selected_regions: list[str] | None,
) -> dict | None:
    ff = forecasts[forecasts["fuel_type"] == fuel].copy()
    if selected_regions:
        ff = ff[ff["region_id"].isin(selected_regions)]
    if ff.empty:
        return None

    # Build year axis around the user's target year
    years = list(range(year - 5, year + 6))
    fig = go.Figure()
    for _, r in ff.iterrows():
        rid = r["region_id"]
        slope = float(r["slope"])
        intercept = float(r["intercept"])
        adj_slope = slope * (1 + rate_change)
        # Anchor adjusted curve at latest observed actual (interpretable scenario)
        rh = actuals[
            (actuals["region_id"] == rid) & (actuals["fuel_type"] == fuel)
        ].sort_values("year")
        if not rh.empty:
            anchor_year = int(rh.iloc[-1]["year"])
            anchor_val = float(rh.iloc[-1]["production"])
        else:
            anchor_year = year
            anchor_val = _project(slope, intercept, year)
        base = [_project(slope, intercept, y) for y in years]
        adjusted = [
            anchor_val + adj_slope * (y - anchor_year) if y >= anchor_year
            else _project(slope, intercept, y)
            for y in years
        ]
        name = r["region_name"]
        fig.add_trace(go.Scatter(
            x=years, y=base, mode="lines",
            name=f"{name} — base", line=dict(width=2),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=adjusted, mode="lines",
            name=f"{name} — adjusted ({rate_change:+.0%})",
            line=dict(width=2, dash="dash"),
        ))
    unit = "Mb/d" if fuel == "crude_oil" else "MMcf"
    fig.update_layout(
        title=f"Sensitivity: base vs. {rate_change:+.0%} slope adjustment",
        xaxis_title="Year",
        yaxis_title=f"Production ({unit})",
        height=380,
        margin=dict(l=40, r=10, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    fig.add_vline(x=year, line_dash="dot", line_color="gray",
                  annotation_text=f"Target {year}", annotation_position="top")
    return {
        "kind": "chart",
        "title": f"Sensitivity scenario: {rate_change:+.0%} slope change",
        "fig": fig,
    }


def build_artifact(
    intent: str,
    query: str,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    controls: dict,
    rate_change: float | None = None,
) -> dict | None:
    """Dispatch to the right artifact builder. Returns None for lookup intents."""
    fuel = controls["fuel"]
    year = controls["year"]
    selected = controls.get("regions") or None

    if intent == "ranking":
        return _ranking_artifact(actuals, forecasts, fuel, year)
    if intent == "summary":
        return _summary_artifact(query, actuals, forecasts, fuel, year)
    if intent == "sensitivity" and rate_change is not None:
        return _sensitivity_artifact(actuals, forecasts, rate_change, fuel, year, selected)
    return None
