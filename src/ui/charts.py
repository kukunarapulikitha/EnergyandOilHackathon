"""Plotly chart helpers for the Energy Intelligence System.

Works against the split Gold tables:
    actuals  — data/gold/regional_actuals.csv    (facts)
    forecasts — data/gold/region_forecasts.csv   (linear model params)

Conventions:
    Actuals  → solid line, filled markers
    Forecast → dashed line, hollow markers, light shaded band
    Consistent color per region across all charts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_REGION_ORDER = ["R10", "R20", "R30", "R40", "R50", "SLA", "SOK", "SPA", "STX", "SWV"]
_PALETTE = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#00BCD4",
    "#FF9800", "#795548", "#607D8B", "#E91E63", "#009688",
]
_COLOR_MAP: dict[str, str] = dict(zip(_REGION_ORDER, _PALETTE))


def _region_color(region_id: str) -> str:
    return _COLOR_MAP.get(region_id, "#888888")


def _hex_to_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


def _unit_label(fuel_type: str) -> str:
    return "Mb/d" if fuel_type == "crude_oil" else "MMcf"


def _project(row: pd.Series, year: int) -> float:
    return float(row["slope"] * year + row["intercept"])


# ---------------------------------------------------------------- public


def actuals_forecast_chart(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    region_id: str,
    fuel_type: str,
    selected_year: int,
    horizon_end: int = 2035,
    title: str | None = None,
) -> go.Figure:
    """Single-region chart: actuals (solid, ≤ selected_year) + forecast (dashed, >)."""
    a = actuals[
        (actuals["region_id"] == region_id) & (actuals["fuel_type"] == fuel_type)
    ].sort_values("year")
    if a.empty:
        return go.Figure()

    region_name = a["region_name"].iloc[0]
    color = _region_color(region_id)
    unit = _unit_label(fuel_type)

    hist = a[a["year"] <= selected_year]
    fig = go.Figure()

    if not hist.empty:
        fig.add_trace(go.Scatter(
            x=hist["year"], y=hist["production"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate="%{x}: <b>%{y:,.0f}</b> " + unit + "<extra>Actual</extra>",
        ))

    f = forecasts[
        (forecasts["region_id"] == region_id) & (forecasts["fuel_type"] == fuel_type)
    ]
    if not f.empty and selected_year < horizon_end:
        row = f.iloc[0]
        future = list(range(selected_year + 1, horizon_end + 1))
        proj = [_project(row, y) for y in future]

        # Connector between last actual and first forecast
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=[hist["year"].iloc[-1], future[0]],
                y=[hist["production"].iloc[-1], proj[0]],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))

        # Uncertainty band widens with horizon — 5% at +1yr, scaling up
        spread = [0.05 * (i + 1) / len(future) for i in range(len(future))]
        upper = [p * (1 + s) for p, s in zip(proj, spread)]
        lower = [p * (1 - s) for p, s in zip(proj, spread)]
        fig.add_trace(go.Scatter(
            x=future + future[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=f"rgba({_hex_to_rgb(color)}, 0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=future, y=proj,
            mode="lines+markers",
            name=f"Forecast (R²={row['r_squared']:.2f})",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=6, color="white", line=dict(color=color, width=2)),
            hovertemplate="%{x}: <b>%{y:,.0f}</b> " + unit + "<extra>Forecast</extra>",
        ))

    fig.add_vline(
        x=selected_year,
        line_dash="dot",
        line_color="rgba(128,128,128,0.6)",
        annotation_text=f"Selected: {selected_year}",
        annotation_position="top right",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=title or f"{region_name} — {fuel_type.replace('_', ' ').title()}",
        xaxis_title="Year",
        yaxis_title=f"Production ({unit})",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=50),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
    )
    return fig


def top_regions_bar(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected_year: int,
    fuel_type: str,
    n: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Horizontal bar — top regions by production at selected_year.

    If selected_year is historical, uses actuals; otherwise projects via forecast params.
    """
    hist = actuals[
        (actuals["fuel_type"] == fuel_type) & (actuals["year"] == selected_year)
    ]
    actual_map = dict(zip(hist["region_id"], hist["production"]))

    f = forecasts[forecasts["fuel_type"] == fuel_type].copy()
    if f.empty:
        return go.Figure()

    f["value"] = f.apply(
        lambda r: actual_map.get(r["region_id"], _project(r, selected_year)), axis=1
    )
    f["is_actual"] = f["region_id"].isin(actual_map)
    f = f.dropna(subset=["value"]).nlargest(n, "value")

    unit = _unit_label(fuel_type)
    colors = [_region_color(rid) for rid in f["region_id"]]
    patterns = ["" if a else "/" for a in f["is_actual"]]

    fig = go.Figure(go.Bar(
        x=f["value"],
        y=f["region_name"],
        orientation="h",
        marker=dict(color=colors, pattern_shape=patterns),
        text=[f"{v:,.0f}" for v in f["value"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:,.0f} " + unit
                      + "<br>%{customdata}<extra></extra>",
        customdata=["actual" if a else "projected" for a in f["is_actual"]],
    ))
    fig.update_layout(
        title=title or f"Top {fuel_type.replace('_', ' ').title()} Regions — {selected_year}",
        xaxis_title=unit,
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=160, r=60, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        height=max(240, 40 * len(f) + 120),
    )
    return fig


def multi_region_comparison(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    region_ids: list[str],
    selected_year: int,
    fuel_type: str,
    horizon_end: int = 2035,
) -> go.Figure:
    """Overlay multiple regions — actuals solid, projections dashed."""
    unit = _unit_label(fuel_type)
    fig = go.Figure()

    for rid in region_ids:
        a = actuals[
            (actuals["region_id"] == rid) & (actuals["fuel_type"] == fuel_type)
        ].sort_values("year")
        if a.empty:
            continue
        color = _region_color(rid)
        name = a["region_name"].iloc[0]

        hist = a[a["year"] <= selected_year]
        if not hist.empty:
            fig.add_trace(go.Scatter(
                x=hist["year"], y=hist["production"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=2),
                marker=dict(size=5),
                legendgroup=rid,
                hovertemplate=f"<b>{name}</b> %{{x}}: %{{y:,.0f}} {unit}<extra></extra>",
            ))

        f = forecasts[
            (forecasts["region_id"] == rid) & (forecasts["fuel_type"] == fuel_type)
        ]
        if f.empty or selected_year >= horizon_end:
            continue
        row = f.iloc[0]
        future = list(range(selected_year + 1, horizon_end + 1))
        proj = [_project(row, y) for y in future]
        xs = ([hist["year"].iloc[-1]] if not hist.empty else []) + future
        ys = ([hist["production"].iloc[-1]] if not hist.empty else []) + proj
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            name=f"{name} (proj.)",
            line=dict(color=color, width=2, dash="dash"),
            legendgroup=rid, showlegend=False,
            hovertemplate=f"<b>{name}</b> %{{x}}: %{{y:,.0f}} {unit}<extra>proj</extra>",
        ))

    fig.add_vline(x=selected_year, line_dash="dot", line_color="rgba(128,128,128,0.5)")
    fig.update_layout(
        title="Regional Comparison",
        xaxis_title="Year",
        yaxis_title=f"Production ({unit})",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=70, b=50),
        xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)"),
        height=500,
    )
    return fig


def small_multiples(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    region_ids: list[str],
    selected_year: int,
    fuel_type: str,
    horizon_end: int = 2035,
) -> go.Figure:
    """Grid of per-region charts for the Compare tab."""
    n = len(region_ids)
    if n == 0:
        return go.Figure()
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    unit = _unit_label(fuel_type)
    names = []
    for rid in region_ids:
        sub = actuals[(actuals["region_id"] == rid) & (actuals["fuel_type"] == fuel_type)]
        names.append(sub["region_name"].iloc[0] if not sub.empty else rid)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=names, shared_yaxes=False)

    for i, rid in enumerate(region_ids):
        row, col = divmod(i, cols)
        color = _region_color(rid)
        a = actuals[
            (actuals["region_id"] == rid) & (actuals["fuel_type"] == fuel_type)
        ].sort_values("year")
        if a.empty:
            continue

        hist = a[a["year"] <= selected_year]
        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["year"], y=hist["production"],
                    mode="lines", line=dict(color=color, width=1.8),
                    showlegend=False,
                    hovertemplate=f"%{{x}}: %{{y:,.0f}} {unit}<extra>Actual</extra>",
                ),
                row=row + 1, col=col + 1,
            )

        f = forecasts[
            (forecasts["region_id"] == rid) & (forecasts["fuel_type"] == fuel_type)
        ]
        if not f.empty and selected_year < horizon_end:
            fr = f.iloc[0]
            future = list(range(selected_year + 1, horizon_end + 1))
            proj = [_project(fr, y) for y in future]
            xs = ([hist["year"].iloc[-1]] if not hist.empty else []) + future
            ys = ([hist["production"].iloc[-1]] if not hist.empty else []) + proj
            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="lines", line=dict(color=color, width=1.8, dash="dash"),
                    showlegend=False,
                    hovertemplate=f"%{{x}}: %{{y:,.0f}} {unit}<extra>Forecast</extra>",
                ),
                row=row + 1, col=col + 1,
            )

    fig.update_layout(
        height=280 * rows,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.12)")
    return fig


# ============================================================ sensitivity


def sensitivity_heatmap(
    forecast_row: pd.Series,
    target_year: int,
    price_range: tuple[float, float] | None = None,
    price_steps: int = 9,
    decline_adj_range: tuple[float, float] = (-0.20, 0.20),
    decline_steps: int = 9,
) -> go.Figure:
    """Heat map — revenue sensitivity to (price × decline-rate adjustment).

    Production_adjusted = slope × (1 + decline_adj) × year + intercept.
    Revenue (oil) = production × 1000 bbl × 365 × WTI ($/bbl).
    Revenue (gas) = production × 1,000,000 × Henry Hub ($/MMBtu).
    """
    fuel = forecast_row.get("fuel_type", "crude_oil")
    is_oil = fuel == "crude_oil"

    if price_range is None:
        price_range = (40.0, 120.0) if is_oil else (2.0, 8.0)

    price_values = np.linspace(price_range[0], price_range[1], price_steps)
    decline_values = np.linspace(decline_adj_range[0], decline_adj_range[1], decline_steps)

    slope = float(forecast_row["slope"])
    intercept = float(forecast_row["intercept"])

    z = np.zeros((decline_steps, price_steps))
    for i, dadj in enumerate(decline_values):
        adj_slope = slope * (1 + dadj)
        production = max(adj_slope * target_year + intercept, 0)
        for j, price in enumerate(price_values):
            if is_oil:
                revenue = production * 1000 * 365 * price
            else:
                revenue = production * 1_000_000 * price
            z[i, j] = revenue / 1e9  # $ billions

    region_name = forecast_row.get("region_name", forecast_row.get("region_id", "?"))
    price_unit = "USD/bbl (WTI)" if is_oil else "USD/MMBtu (Henry Hub)"
    price_fmt = (lambda p: f"${p:.0f}") if is_oil else (lambda p: f"${p:.2f}")
    hover_price = "WTI: %{x}/bbl" if is_oil else "Henry Hub: %{x}/MMBtu"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[price_fmt(p) for p in price_values],
            y=[f"{d*100:+.0f}%" for d in decline_values],
            colorscale="RdYlGn",
            zmid=float(np.median(z)) if z.size else 0,
            hovertemplate=(
                f"{hover_price}<br>"
                "Decline adj: %{y}<br>"
                "Revenue: $%{z:,.2f}B<extra></extra>"
            ),
            colorbar=dict(title=dict(text="Revenue ($B/yr)", side="right")),
            text=[[f"${v:,.1f}B" for v in row] for row in z],
            texttemplate="%{text}",
            textfont=dict(size=10),
        )
    )
    fig.update_layout(
        title=(
            f"Sensitivity: Revenue at {target_year} — "
            f"{region_name} ({fuel.replace('_', ' ').title()})"
        ),
        xaxis_title=f"Price assumption ({price_unit})",
        yaxis_title="Decline-rate adjustment (− = steeper decline · + = growth)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=120, r=80, t=70, b=60),
        height=500,
    )
    return fig
