"""Ranked bar chart for the sidebar — regions sorted by the active metric.

Mirrors the "Regional Rankings" panel from the reference dashboard design.
Compact Plotly horizontal bar colored by the same palette as the map's
main overlay, so the visual language stays consistent across the page.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_METRIC_CFG = {
    "production": {
        "label": "Production",
        "scale": "Blues",
        "fmt": ",.0f",
        "is_diverging": False,
    },
    "growth_pct": {
        "label": "YoY Growth (%)",
        "scale": "RdYlGn",
        "fmt": "+.1f",
        "is_diverging": True,
    },
    "relative_performance_index": {
        "label": "Relative Perf.",
        "scale": "RdYlGn",
        "fmt": ".0f",
        "is_diverging": True,
    },
    "investment_score": {
        "label": "Investment Score",
        "scale": "RdYlGn",
        "fmt": ".0f",
        "is_diverging": False,
    },
}


def ranked_bar(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    metric: str,
    year: int,
    fuel: str,
    top_n: int = 10,
    height: int = 240,
) -> go.Figure | None:
    """Horizontal bar chart of regions ranked by the selected metric."""
    if metric not in _METRIC_CFG:
        return None
    cfg = _METRIC_CFG[metric]

    rows: list[dict] = []
    fuel_actuals = actuals[actuals["fuel_type"] == fuel]
    fuel_forecasts = forecasts[forecasts["fuel_type"] == fuel]

    if metric == "investment_score":
        # Pulled from forecasts table, not year-dependent
        if fuel_forecasts.empty:
            return None
        for _, r in fuel_forecasts.iterrows():
            rows.append({
                "region": r["region_name"],
                "value": float(r["investment_score"]),
            })
    elif metric == "production":
        year_rows = fuel_actuals[fuel_actuals["year"] == year]
        if not year_rows.empty:
            for _, r in year_rows.iterrows():
                rows.append({
                    "region": r["region_name"],
                    "value": float(r["production"]),
                })
        else:
            # Projected year — use forecast params
            for _, r in fuel_forecasts.iterrows():
                proj = float(r["slope"] * year + r["intercept"])
                rows.append({"region": r["region_name"], "value": proj})
    else:
        year_rows = fuel_actuals[fuel_actuals["year"] == year]
        source = year_rows if not year_rows.empty else fuel_actuals.sort_values("year")
        for region_id, g in source.groupby("region_id"):
            val = g[metric].dropna()
            if val.empty:
                continue
            rows.append({
                "region": g["region_name"].iloc[0],
                "value": float(val.iloc[-1]),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows).dropna(subset=["value"])
    if df.empty:
        return None

    df = df.sort_values("value", ascending=True).tail(top_n)

    bar_kwargs = dict(
        x=df["value"],
        y=df["region"],
        orientation="h",
        marker=dict(
            color=df["value"],
            colorscale=cfg["scale"],
            showscale=False,
        ),
        text=[format(v, cfg["fmt"]) for v in df["value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:" + cfg["fmt"] + "}<extra></extra>",
    )
    if cfg["is_diverging"]:
        bar_kwargs["marker"]["cmid"] = 0 if metric == "growth_pct" else 50

    fig = go.Figure(go.Bar(**bar_kwargs))
    fig.update_layout(
        title=dict(
            text=f"Top by {cfg['label']}",
            font=dict(size=12, color="#CCCCCC"),
        ),
        height=height,
        margin=dict(l=8, r=40, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=10, color="#CCCCCC")),
        showlegend=False,
    )
    return fig
