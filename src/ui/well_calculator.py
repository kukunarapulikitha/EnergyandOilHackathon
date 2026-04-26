"""Streamlit UI for the Well Economics Calculator.

Inputs run through `st.session_state` keys (well_calc_*) so the map-prefill
flow can mutate them before widgets render. Map click → load_region_defaults()
pushes new values into session_state → next render picks them up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.economics.region_defaults import (
    GENERIC_DEFAULT,
    REGION_DEFAULTS,
    get_defaults,
    list_regions,
)
from src.economics.well_model import (
    arps_production,
    compute_eur,
    irr,
    monthly_cashflow,
    npv,
    payback_period_months,
    total_revenue,
)

# ----------------------------------------------------------------- session state

_KEYS = (
    "well_calc_region", "well_calc_ip", "well_calc_Di", "well_calc_b",
    "well_calc_well_life", "well_calc_capex", "well_calc_loe",
    "well_calc_price", "well_calc_severance", "well_calc_discount",
    "well_calc_fuel",
)


def _load_into_state(defaults: dict) -> None:
    """Push defaults into session_state under the well_calc_* keys."""
    st.session_state["well_calc_region"] = defaults.get("region_id")
    st.session_state["well_calc_fuel"] = defaults["fuel"]
    st.session_state["well_calc_ip"] = float(defaults["ip"])
    st.session_state["well_calc_Di"] = float(defaults["Di"])
    st.session_state["well_calc_b"] = float(defaults["b"])
    st.session_state["well_calc_well_life"] = int(defaults["well_life_months"])
    st.session_state["well_calc_capex"] = int(defaults["capex"])
    st.session_state["well_calc_loe"] = float(defaults["loe"])
    st.session_state["well_calc_price"] = float(defaults["price"])
    st.session_state["well_calc_severance"] = float(defaults["severance_pct"]) * 100
    st.session_state.setdefault("well_calc_discount", 10.0)


def _ensure_initialized() -> None:
    """First render: seed with the focused region's defaults if any, else generic."""
    if "well_calc_initialized" in st.session_state:
        return
    focus_id = st.session_state.get("map_focus_region")
    _load_into_state(get_defaults(focus_id))
    st.session_state["well_calc_last_loaded_region"] = focus_id
    st.session_state["well_calc_initialized"] = True


def _maybe_reload_for_focus_change() -> None:
    """If the user clicked a NEW region on the map since last render, reload."""
    focus_id = st.session_state.get("map_focus_region")
    last_loaded = st.session_state.get("well_calc_last_loaded_region")
    if focus_id and focus_id != last_loaded and focus_id in REGION_DEFAULTS:
        _load_into_state(get_defaults(focus_id))
        st.session_state["well_calc_last_loaded_region"] = focus_id
        defaults = REGION_DEFAULTS[focus_id]
        try:
            st.toast(f"📍 Loaded {defaults['label']} defaults", icon="🎯")
        except Exception:
            pass  # toast unavailable in some Streamlit versions


# ----------------------------------------------------------------- charts

def _decline_chart(monthly_rate: np.ndarray, fuel: str, well_life: int) -> go.Figure:
    months = np.arange(len(monthly_rate))
    unit = "bbl/d" if fuel == "crude_oil" else "Mcf/d"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months, y=monthly_rate, mode="lines",
        line=dict(color="#1f77b4", width=2.2),
        name="Production rate",
        hovertemplate="Month %{x}<br>Rate: %{y:,.0f} " + unit + "<extra></extra>",
    ))
    # Year-1 cumulative annotation
    if len(monthly_rate) >= 12:
        y1_cum = float(np.sum(monthly_rate[:12]) * 30.4375)
        y1_label = (
            f"Year 1 cum: {y1_cum/1000:,.0f} Mbbl"
            if fuel == "crude_oil"
            else f"Year 1 cum: {y1_cum/1000:,.0f} MMcf"
        )
        fig.add_annotation(
            x=12, y=monthly_rate[12] if len(monthly_rate) > 12 else monthly_rate[-1],
            text=y1_label, showarrow=True, arrowhead=2, ax=40, ay=-40,
            font=dict(size=11, color="#444"),
        )
    fig.update_layout(
        title=f"Production Decline Curve ({well_life} months)",
        xaxis_title="Month", yaxis_title=f"Rate ({unit})",
        height=360, margin=dict(l=40, r=10, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def _cashflow_chart(cf: np.ndarray, payback_mo: int | None) -> go.Figure:
    months = np.arange(len(cf))
    cum = np.cumsum(cf)
    fig = go.Figure()
    # Shade above/below zero
    pos = np.where(cum >= 0, cum, 0)
    neg = np.where(cum < 0, cum, 0)
    fig.add_trace(go.Scatter(
        x=months, y=neg, fill="tozeroy", mode="none",
        fillcolor="rgba(220, 60, 60, 0.18)", showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=pos, fill="tozeroy", mode="none",
        fillcolor="rgba(40, 160, 90, 0.18)", showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=months, y=cum, mode="lines",
        line=dict(color="#222", width=2.2),
        name="Cumulative cashflow",
        hovertemplate="Month %{x}<br>Cum CF: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="#666")
    if payback_mo is not None and payback_mo < len(cum):
        fig.add_vline(x=payback_mo, line_dash="dash", line_color="#2a8",
                      annotation_text=f"Payback: month {payback_mo}",
                      annotation_position="top")
    fig.update_layout(
        title="Cumulative Cashflow ($)",
        xaxis_title="Month", yaxis_title="Cumulative CF ($)",
        height=360, margin=dict(l=40, r=10, t=50, b=40),
        hovermode="x unified",
    )
    return fig


# ----------------------------------------------------------------- main render

def render_well_calculator(actuals: pd.DataFrame, forecasts: pd.DataFrame, ctrl: dict) -> None:
    st.subheader("🛢️ Well Economics Calculator")
    st.caption(
        "Single-well horizontal economic model — Arps decline curve + "
        "discounted cashflow analysis. All inputs editable; outputs recompute "
        "on every change. Click a region on the **Dashboard map** to auto-load "
        "regional defaults here."
    )

    _ensure_initialized()
    _maybe_reload_for_focus_change()

    # ---------------- region preset row
    preset_col, reset_col, info_col = st.columns([0.45, 0.20, 0.35])
    with preset_col:
        all_regions = [("__none__", "— Generic / no region —")] + list_regions()
        rid_options = [r[0] for r in all_regions]
        labels = {r[0]: r[1] for r in all_regions}
        current = st.session_state.get("well_calc_region") or "__none__"
        if current not in rid_options:
            current = "__none__"
        idx = rid_options.index(current)
        chosen = st.selectbox(
            "Region preset",
            options=rid_options,
            index=idx,
            format_func=lambda rid: labels[rid],
            key="well_calc_preset_select",
        )
        if chosen != current:
            target = None if chosen == "__none__" else chosen
            _load_into_state(get_defaults(target))
            st.session_state["well_calc_last_loaded_region"] = target
            st.rerun()
    with reset_col:
        st.markdown("&nbsp;")  # vertical spacer
        if st.button("🔄 Reset to defaults", use_container_width=True):
            target = st.session_state.get("well_calc_region")
            _load_into_state(get_defaults(target))
            st.rerun()
    with info_col:
        focus_id = st.session_state.get("map_focus_region")
        if focus_id and focus_id in REGION_DEFAULTS:
            st.info(f"📍 Map-focused region: **{focus_id}** ({REGION_DEFAULTS[focus_id]['label']})")
        else:
            st.caption(
                "💡 Click a region on the Dashboard map to auto-load its defaults here."
            )

    fuel = st.session_state.get("well_calc_fuel", "crude_oil")
    is_crude = fuel == "crude_oil"
    rate_unit = "bbl/d" if is_crude else "Mcf/d"
    price_unit = "$/bbl" if is_crude else "$/Mcf"

    # ---------------- inputs (two columns)
    st.markdown("#### Inputs")
    left, right = st.columns(2)
    with left:
        st.markdown("**Production parameters**")
        st.number_input(
            f"Initial production rate ({rate_unit})",
            min_value=1.0, max_value=50_000.0, step=10.0,
            key="well_calc_ip",
            help="IP — peak monthly average rate at the start of well life.",
        )
        st.slider(
            "Initial decline rate (Di, /yr)",
            min_value=0.05, max_value=1.20, step=0.01,
            key="well_calc_Di",
            help="Nominal annual decline at t=0. Shale wells: 0.55–0.85.",
        )
        st.slider(
            "Hyperbolic exponent (b)",
            min_value=0.0, max_value=1.5, step=0.05,
            key="well_calc_b",
            help="0=exponential, 1=harmonic. 1.0–1.3 typical for unconventional shale.",
        )
        st.slider(
            "Well life (months)",
            min_value=60, max_value=480, step=12,
            key="well_calc_well_life",
        )
    with right:
        st.markdown("**Economics**")
        st.number_input(
            "Drilling + completion capex ($)",
            min_value=500_000, max_value=30_000_000, step=100_000,
            key="well_calc_capex",
        )
        st.number_input(
            f"Lease operating expense ({price_unit})",
            min_value=0.0, max_value=50.0, step=0.10,
            key="well_calc_loe",
        )
        st.number_input(
            f"Commodity price ({price_unit})",
            min_value=0.10, max_value=200.0, step=0.50,
            key="well_calc_price",
        )
        st.slider(
            "Severance + ad valorem (%)",
            min_value=0.0, max_value=15.0, step=0.25,
            key="well_calc_severance",
        )
        st.slider(
            "Discount rate (%)",
            min_value=5.0, max_value=20.0, step=0.5,
            key="well_calc_discount",
        )

    # ---------------- compute
    rate = arps_production(
        qi=st.session_state["well_calc_ip"],
        Di=st.session_state["well_calc_Di"],
        b=st.session_state["well_calc_b"],
        months=st.session_state["well_calc_well_life"],
    )
    cf = monthly_cashflow(
        monthly_rate=rate,
        price=st.session_state["well_calc_price"],
        loe=st.session_state["well_calc_loe"],
        severance_pct=st.session_state["well_calc_severance"] / 100.0,
        capex=st.session_state["well_calc_capex"],
    )
    eur_val = compute_eur(rate)
    npv_val = npv(cf, st.session_state["well_calc_discount"] / 100.0)
    irr_val = irr(cf)
    payback = payback_period_months(cf)
    revenue = total_revenue(rate, st.session_state["well_calc_price"])

    # ---------------- output cards
    st.markdown("#### Outputs")
    c1, c2, c3, c4, c5 = st.columns(5)
    if is_crude:
        c1.metric("EUR", f"{eur_val/1000:,.0f} Mbbl")
    else:
        c1.metric("EUR", f"{eur_val/1000:,.0f} MMcf")
    c2.metric(
        f"NPV @ {st.session_state['well_calc_discount']:.0f}%",
        f"${npv_val/1e6:,.2f}M",
        delta="profitable" if npv_val > 0 else "uneconomic",
        delta_color="normal" if npv_val > 0 else "inverse",
    )
    c3.metric(
        "IRR",
        f"{irr_val*100:.1f}%" if (irr_val is not None and not np.isnan(irr_val)) else "—",
    )
    c4.metric(
        "Payback",
        f"{payback} mo" if payback is not None else "no payback",
    )
    c5.metric("Total revenue", f"${revenue/1e6:,.1f}M")

    # ---------------- charts
    st.markdown("#### Visualizations")
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.plotly_chart(
            _decline_chart(rate, fuel, st.session_state["well_calc_well_life"]),
            use_container_width=True,
        )
    with chart_right:
        st.plotly_chart(
            _cashflow_chart(cf, payback),
            use_container_width=True,
        )

    # ---------------- methodology disclosure
    with st.expander("📖 Methodology & assumptions"):
        st.markdown(
            """
            **Decline curve:** hyperbolic Arps with terminal exponential switch
            at 6%/yr (prevents bare-hyperbolic over-recovery).
            `q(t) = qi / (1 + b·Di·t)^(1/b)` for t < t_switch, then exponential
            at the floor rate.

            **EUR:** cumulative production over the modelled well life.
            EUR is sensitive to well life — extend to 30+ years for a fairer
            comparison vs. industry "true EUR" figures.

            **Cashflow:** revenue = volume × price; severance + ad valorem
            applied to revenue; LOE applied to volume; capex booked at t=0.
            Pre-federal-tax. No hedging, no royalty deduction (assume working
            interest figures or apply a manual NRI haircut to price).

            **NPV / IRR:** discounted at user-selected annual rate, monthly
            compounding. IRR uses `numpy_financial` with a `scipy.optimize`
            fallback. IRR returns `—` when no real solution exists.

            **Payback:** first month at which cumulative cashflow turns
            non-negative. Undiscounted.

            _All figures are demonstration-grade. Real-world AFEs / type curves
            require operator-level data not in this dataset._
            """
        )

    # ---------------- export
    monthly_df = pd.DataFrame({
        "month": np.arange(len(rate)),
        "rate": rate,
        "cashflow": cf,
        "cumulative_cashflow": np.cumsum(cf),
    })
    st.download_button(
        "📥 Export monthly schedule (CSV)",
        data=monthly_df.to_csv(index=False).encode("utf-8"),
        file_name="well_economics_schedule.csv",
        mime="text/csv",
    )
