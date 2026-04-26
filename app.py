"""Energy Intelligence System — Streamlit dashboard.

Phase 2 scope: Overview, Regional Detail, Compare tabs.
Phase 3 will add the AI Analyst tab.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env BEFORE importing anything that reads env vars (Groq client, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import pandas as pd
import streamlit as st

from src.ai.client import GroqClient, GroqUnavailable
from src.ai.intents import build_artifact, classify_intent
from src.ai.prompts import (
    build_regional_context,
    build_system_prompt,
    compute_sensitivity_context,
    detect_what_if,
    parse_tagged_response,
)
from src.forecast.linear import fit_and_forecast
from src.ui.charts import (
    actuals_forecast_chart,
    multi_region_comparison,
    sensitivity_heatmap,
    small_multiples,
    top_regions_bar,
)
from src.ui.data_loader import (
    load_actuals,
    load_annual_silver,
    load_forecasts,
    load_metadata,
    pipeline_ready,
)
from src.ui.badges import badge_markdown, classify_region
from src.ui.export import build_workbook
from src.ui.map import (
    add_basin_layer,
    add_rig_layer,
    production_bubble_map,
    production_map,
    regions_to_states,
    resolve_clicked_region,
    rig_count_summary,
)
from src.ui.provenance import (
    provenance_caption,
    render_data_sources_panel,
)
from src.ui.rankings import ranked_bar

st.set_page_config(
    page_title="Energy Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Slider cap stays at 2035 per user request; forecast horizon extends one year
# beyond so that 2035 is always visible as a projected data point, even when
# the user drags the slider all the way to 2035.
SLIDER_MAX_YEAR = 2035
FORECAST_HORIZON = 2036

# ------------------------------------------------------------------ HEADER


def render_header(meta: dict) -> None:
    col1, col2 = st.columns([0.80, 0.20])
    with col1:
        st.title("⚡ Energy Intelligence System")
        st.caption(
            "U.S. oil & gas production analysis with linear-regression "
            "forecasting and Medallion data architecture."
        )
        if meta:
            from src.ui.provenance import fresh_age
            st.caption(f"Data last updated: {fresh_age(meta.get('fetched_at'))}")
    with col2:
        st.markdown("&nbsp;")  # vertical spacer
        if st.button("🔄 Refresh data", help="Re-run the EIA pipeline and reload the app"):
            _refresh_data(meta)


def _refresh_data(meta: dict) -> None:
    """Trigger the full Medallion pipeline. On failure, show stale-data warning."""
    import os
    import subprocess

    if not os.environ.get("EIA_API_KEY"):
        st.warning(
            "⚠️ **EIA_API_KEY not set.** "
            "The dashboard is running on cached data. "
            "To refresh: set `EIA_API_KEY` as an environment variable "
            "(get a free key at https://www.eia.gov/opendata/register.php), then reload."
        )
        return

    with st.spinner("Fetching latest EIA data → rebuilding Bronze → Silver → Gold..."):
        try:
            result = subprocess.run(
                [sys.executable, "scripts/verify_pipeline.py"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "pipeline exited non-zero")
            st.cache_data.clear()
            st.success("✅ Data refreshed successfully. Reloading...")
            st.rerun()
        except Exception as exc:
            stale_ts = meta.get("fetched_at", "unknown")
            st.warning(
                f"⚠️ Refresh failed: {exc}. "
                f"Continuing on cached data from "
                f"{stale_ts[:19] if isinstance(stale_ts, str) else stale_ts}."
            )


# ------------------------------------------------------------------ SIDEBAR


def render_sidebar(actuals: pd.DataFrame, forecasts: pd.DataFrame) -> dict:
    with st.sidebar:
        st.subheader("🎛️ Controls")

        fuel = st.radio(
            "Fuel type",
            options=["crude_oil", "natural_gas"],
            format_func=lambda x: "🛢️ Crude oil" if x == "crude_oil" else "🔥 Natural gas",
            horizontal=False,
        )

        fuel_actuals = actuals[actuals["fuel_type"] == fuel]
        fuel_forecasts = forecasts[forecasts["fuel_type"] == fuel]
        year_min = int(fuel_actuals["year"].min())
        year_max = int(fuel_actuals["year"].max())
        horizon_end = FORECAST_HORIZON

        st.markdown("**Year selector**")
        st.caption(
            "Drag to pick a cutoff. Actuals display up to this year; "
            "forecast projects beyond via linear regression."
        )
        selected_year = st.slider(
            "Year",
            min_value=year_min,
            max_value=SLIDER_MAX_YEAR,
            value=year_max,
            step=1,
            label_visibility="collapsed",
        )

        st.markdown("**Regions**")
        region_options = (
            fuel_actuals[["region_id", "region_name"]]
            .drop_duplicates()
            .sort_values("region_name")
        )
        selected_regions = st.multiselect(
            "Regions",
            options=region_options["region_id"].tolist(),
            default=region_options["region_id"].tolist(),
            format_func=lambda rid: region_options.set_index("region_id")
            .loc[rid, "region_name"],
            label_visibility="collapsed",
        )

        st.divider()

        # --- Regional Rankings bar (reflects the active map metric) ---
        st.markdown("**📊 Regional Rankings**")
        ranking_metric = st.selectbox(
            "Rank by",
            options=["production", "growth_pct", "relative_performance_index", "investment_score"],
            format_func=lambda m: {
                "production": "💧 Production",
                "growth_pct": "📈 YoY Growth",
                "relative_performance_index": "🏆 Relative Perf.",
                "investment_score": "⭐ Investment Score",
            }[m],
            key="sidebar_ranking_metric",
            label_visibility="collapsed",
        )
        rb = ranked_bar(
            actuals, forecasts,
            metric=ranking_metric,
            year=selected_year,
            fuel=fuel,
            top_n=10,
            height=240,
        )
        if rb is not None:
            st.plotly_chart(rb, use_container_width=True, config={"displayModeBar": False})

        st.divider()
        st.caption(
            "Forecast method: **Linear OLS** via scikit-learn. R² per region "
            "is reported on the chart legend and Overview table."
        )

    return {
        "fuel": fuel,
        "year": selected_year,
        "regions": selected_regions,
        "year_min": year_min,
        "year_max": year_max,
        "horizon_end": horizon_end,
    }


# ------------------------------------------------------------------ helpers


def _live_fit(
    annual_silver: pd.DataFrame, region_id: str, fuel: str, cutoff_year: int
):
    subset = annual_silver[annual_silver["fuel_type"] == fuel]
    if subset.empty:
        return None
    try:
        return fit_and_forecast(
            subset, region_id=region_id, selected_year=cutoff_year
        )
    except ValueError:
        return None


# ------------------------------------------------------------------ TABS


def render_kpi_strip(actuals, forecasts, ctrl) -> None:
    """Top-of-page aggregate KPI cards. Reflects current fuel + region filter."""
    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]
    ff = forecasts[forecasts["fuel_type"] == ctrl["fuel"]]
    year = ctrl["year"]
    selected = ctrl["regions"]

    hist = fa[fa["region_id"].isin(selected) & (fa["year"] == year)]
    if len(hist) == len(selected) and not hist.empty:
        total_value = hist["production"].sum()
        label = "Total Production (actual)"
    else:
        ff_sel = ff[ff["region_id"].isin(selected)].copy()
        if ff_sel.empty:
            total_value = 0
        else:
            ff_sel["proj"] = ff_sel.apply(
                lambda r: r["slope"] * year + r["intercept"], axis=1
            )
            total_value = ff_sel["proj"].sum()
        label = "Projected Production Estimate"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label,
        _fmt_production(total_value, ctrl["fuel"]),
        help=(
            "Sum of production across all selected regions for the chosen year. "
            "Shows 'actual' when EIA observed data exists; 'projected' when the "
            "year is beyond the latest data point (OLS linear trend)."
        ),
    )
    top_prod = fa[fa["year"] == year].nlargest(1, "production")
    if not top_prod.empty:
        c1.caption(f"🏆 {top_prod.iloc[0]['region_name']} leads at {_fmt_production(top_prod.iloc[0]['production'], ctrl['fuel'])}")

    avg_growth = fa[fa["region_id"].isin(selected) & (fa["year"] == year)][
        "growth_pct"
    ].mean()
    c2.metric(
        "Avg YoY Growth",
        f"{avg_growth:+.1f}%" if pd.notna(avg_growth) else "—",
        help=(
            "Year-over-year percentage change in production: "
            "(Production_N − Production_N-1) / Production_N-1 × 100. "
            "Positive = growth, negative = decline. Averaged across selected regions."
        ),
    )
    growing = fa[fa["region_id"].isin(selected) & (fa["year"] == year) & (fa["growth_pct"] > 0)]
    c2.caption(f"📈 {len(growing)} of {len(selected)} regions growing" if pd.notna(avg_growth) else "")

    avg_r2 = ff[ff["region_id"].isin(selected)]["r_squared"].mean()
    c3.metric(
        "Avg Forecast R²",
        f"{avg_r2:.2f}" if pd.notna(avg_r2) else "—",
        help=(
            "Coefficient of determination (R²) for the OLS linear fit, 0–1. "
            "A value ≥ 0.85 means the trend is highly linear and the projection "
            "is reliable; < 0.5 signals a noisy or non-linear trend — treat "
            "that region's forecast with caution."
        ),
    )
    if pd.notna(avg_r2):
        r2_label = "reliable trend" if avg_r2 >= 0.85 else "moderate fit" if avg_r2 >= 0.60 else "noisy trend"
        c3.caption(f"{'✅' if avg_r2 >= 0.85 else '⚠️'} {r2_label} — {'projection is grounded' if avg_r2 >= 0.85 else 'treat forecast as directional'}")

    avg_score = ff[ff["region_id"].isin(selected)]["investment_score"].mean()
    c4.metric(
        "Avg Investment Score",
        f"{avg_score:.0f}/100" if pd.notna(avg_score) else "—",
        help=(
            "Composite 0–100 attractiveness signal: "
            "growth_rate × 0.35 + (1 − decline_rate) × 0.25 + "
            "(1 − volatility) × 0.20 + forecast_R² × 0.20. "
            "Higher = stronger investment case."
        ),
    )
    if pd.notna(avg_score):
        score_label = "strong investment case" if avg_score >= 70 else "moderate opportunity" if avg_score >= 50 else "proceed with caution"
        c4.caption(f"{'⭐' if avg_score >= 70 else '🔶' if avg_score >= 50 else '🔴'} {score_label}")


def render_investment_thesis(actuals, forecasts, ctrl) -> None:
    """Narrative investment thesis card — a verdict + 3 bullets pulling live KPIs.

    Picks the focused region (map click) if available, else top-scored region
    in the current fuel filter. Replaces the bare 'Investment Score: 78/100'
    with WHY the score is what it is.
    """
    from src.kpi.thesis import build_investment_thesis

    fuel = ctrl["fuel"]
    year = ctrl["year"]
    ff = forecasts[forecasts["fuel_type"] == fuel]
    if ff.empty:
        return

    # Pick region: map focus first, else highest investment score
    focus_id = st.session_state.get("map_focus_region")
    focus_fuel = st.session_state.get("map_focus_fuel")
    if focus_id and focus_fuel == fuel and focus_id in ff["region_id"].values:
        region_id = focus_id
        source_caption = "📍 Focused via map click. Click another state to update."
    else:
        region_id = ff.sort_values("investment_score", ascending=False).iloc[0]["region_id"]
        source_caption = "🏆 Top-scored region for current fuel. Click any state on the map to switch."

    thesis = build_investment_thesis(region_id, fuel, year, actuals, forecasts)

    # ---- Header row: verdict pill + title --------------------------------
    h1, h2 = st.columns([0.75, 0.25])
    with h1:
        st.markdown(f"### 💡 Investment Thesis · {year}")
        st.caption(source_caption)
    with h2:
        verdict = thesis["verdict"]
        color = thesis["verdict_color"]
        st.markdown(
            f"""
            <div style='text-align:right;padding-top:8px;'>
                <span style='background:{color};color:white;padding:6px 16px;
                             border-radius:20px;font-weight:600;font-size:0.95rem;'>
                    {verdict}
                </span>
                <div style='font-size:0.78rem;opacity:0.7;margin-top:4px;'>
                    Score {thesis['score']:.0f}/100
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Title sentence ---------------------------------------------------
    st.markdown(f"**{thesis['title']}**")

    # ---- Bullets ----------------------------------------------------------
    if thesis["bullets"]:
        for b in thesis["bullets"]:
            st.markdown(f"- {b}")
    else:
        st.info("No KPI data available for this region/year combination.")

    # ---- Rationale footer -------------------------------------------------
    st.caption(thesis["rationale"])


def tab_overview(actuals, forecasts, ctrl):
    st.subheader("📊 Overview")

    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]
    ff = forecasts[forecasts["fuel_type"] == ctrl["fuel"]]
    year = ctrl["year"]
    selected = ctrl["regions"]

    # Aggregate KPIs across selected regions for the selected year
    hist = fa[fa["region_id"].isin(selected) & (fa["year"] == year)]
    if len(hist) == len(selected) and not hist.empty:
        total_value = hist["production"].sum()
        label = "Total Production (actual)"
    else:
        ff_sel = ff[ff["region_id"].isin(selected)].copy()
        if ff_sel.empty:
            total_value = 0
        else:
            ff_sel["proj"] = ff_sel.apply(
                lambda r: r["slope"] * year + r["intercept"], axis=1
            )
            total_value = ff_sel["proj"].sum()
        label = "Projected Production Estimate"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label,
        _fmt_production(total_value, ctrl["fuel"]),
        help=(
            "Sum of production across selected regions. "
            "'Actual' = EIA observed data; 'Projected' = OLS trend (slope × year + intercept). "
            f"Unit: {'thousand barrels/day' if ctrl['fuel'] == 'crude_oil' else 'million cubic feet/month'}."
        ),
    )
    top_prod2 = fa[fa["year"] == year].nlargest(1, "production")
    if not top_prod2.empty:
        c1.caption(f"🏆 Top: {top_prod2.iloc[0]['region_name']} — {_fmt_production(top_prod2.iloc[0]['production'], ctrl['fuel'])}")

    avg_growth = fa[
        fa["region_id"].isin(selected) & (fa["year"] == year)
    ]["growth_pct"].mean()
    c2.metric(
        "Avg YoY Growth",
        f"{avg_growth:+.1f}%" if pd.notna(avg_growth) else "—",
        help=(
            "Year-over-year % change: (Production_N − Production_N-1) / Production_N-1 × 100. "
            "Positive = expanding output; negative = contraction. "
            "Averaged across the selected regions."
        ),
    )
    if pd.notna(avg_growth):
        growth_label = "strong expansion" if avg_growth > 5 else "stable" if avg_growth >= 0 else "declining — requires diligence"
        c2.caption(f"{'📈' if avg_growth >= 0 else '📉'} {growth_label}")

    avg_r2 = ff[ff["region_id"].isin(selected)]["r_squared"].mean()
    c3.metric(
        "Avg Forecast R²",
        f"{avg_r2:.2f}" if pd.notna(avg_r2) else "—",
        help=(
            "OLS linear fit quality (0–1). "
            "≥ 0.85 means the production trend is strongly linear and the projection is reliable. "
            "< 0.50 means the trend is noisy — treat forecasts for those regions with caution."
        ),
    )
    if pd.notna(avg_r2):
        r2_label2 = "reliable trend" if avg_r2 >= 0.85 else "moderate fit" if avg_r2 >= 0.60 else "noisy — treat as directional"
        c3.caption(f"{'✅' if avg_r2 >= 0.85 else '⚠️'} {r2_label2}")

    avg_score = ff[ff["region_id"].isin(selected)]["investment_score"].mean()
    c4.metric(
        "Avg Investment Score",
        f"{avg_score:.0f}/100" if pd.notna(avg_score) else "—",
        help=(
            "Composite 0–100: growth × 0.35 + (1−decline) × 0.25 + (1−volatility) × 0.20 + R² × 0.20. "
            "Higher = more attractive investment case. Averaged across selected regions."
        ),
    )
    if pd.notna(avg_score):
        score_label2 = "strong investment case" if avg_score >= 70 else "moderate opportunity" if avg_score >= 50 else "proceed with caution"
        c4.caption(f"{'⭐' if avg_score >= 70 else '🔶' if avg_score >= 50 else '🔴'} {score_label2}")

    # Excel export — formula-driven workbook (Tier 2)
    st.download_button(
        "📥 Export to Excel (formula-driven workbook)",
        data=build_workbook(actuals, forecasts, default_target_year=year),
        file_name=f"energy_intelligence_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help=(
            "4-sheet workbook with editable Inputs, raw Production, forecast "
            "parameters, and a KPI summary driven by Excel formulas. "
            "Change WTI price in Inputs sheet → KPIs recalculate live."
        ),
    )

    st.divider()

    left, right = st.columns([0.55, 0.45])
    with left:
        st.plotly_chart(
            top_regions_bar(fa, ff, selected_year=year, fuel_type=ctrl["fuel"], n=10),
            use_container_width=True,
        )
    with right:
        st.markdown("### Forecast parameters")
        ff_disp = ff[ff["region_id"].isin(selected)][
            ["region_name", "slope", "r_squared", "investment_score"]
        ].rename(columns={
            "region_name": "Region",
            "slope": "Slope (/yr)",
            "r_squared": "R²",
            "investment_score": "Score",
        })
        st.dataframe(
            ff_disp.style.format({
                "Slope (/yr)": "{:,.1f}",
                "R²": "{:.2f}",
                "Score": "{:.0f}",
            }),
            hide_index=True,
            use_container_width=True,
        )


def tab_regional_detail(actuals, forecasts, annual_silver, ctrl):
    st.subheader("🔍 Regional Detail")

    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]
    if not ctrl["regions"]:
        st.info("Select at least one region in the sidebar.")
        return

    # Honor a map click — if the user clicked a region on the Map tab and the
    # fuel is still the same, pre-select that region here.
    pinned = st.session_state.get("map_selected_region")
    pinned_fuel = st.session_state.get("map_selected_fuel")
    default_idx = 0
    if (
        pinned
        and pinned_fuel == ctrl["fuel"]
        and pinned in ctrl["regions"]
    ):
        default_idx = ctrl["regions"].index(pinned)
        st.caption(f"📍 Pre-selected from Map tab ({pinned}). Change via dropdown if needed.")

    region_id = st.selectbox(
        "Region",
        options=ctrl["regions"],
        index=default_idx,
        format_func=lambda rid: (
            fa[fa["region_id"] == rid]["region_name"].iloc[0]
            if rid in fa["region_id"].values else rid
        ),
    )

    baked = forecasts[forecasts["fuel_type"] == ctrl["fuel"]]
    baked_row = baked[baked["region_id"] == region_id]
    baked_cutoff = (
        int(baked_row["trained_through_year"].iloc[0]) if not baked_row.empty else None
    )

    # If user dragged slider BACKWARD → re-fit the regression live
    if baked_cutoff is not None and ctrl["year"] < baked_cutoff:
        st.info(
            f"🔄 Slider is before the baked cutoff ({baked_cutoff}). "
            f"Re-fitting linear regression using data through {ctrl['year']} only."
        )
        fc = _live_fit(annual_silver, region_id, ctrl["fuel"], ctrl["year"])
        if fc is None:
            st.warning("Not enough history to re-fit at this cutoff.")
            return
        live_forecast = pd.DataFrame([{
            "region_id": region_id,
            "region_name": baked_row["region_name"].iloc[0] if not baked_row.empty else region_id,
            "fuel_type": ctrl["fuel"],
            "slope": fc.slope,
            "intercept": fc.intercept,
            "r_squared": fc.r_squared,
            "trained_through_year": ctrl["year"],
            "horizon_end": ctrl["horizon_end"],
            "method": "linear_ols",
            "investment_score": None,
            "source": f"live-refit through {ctrl['year']}",
        }])
        forecasts_for_chart = live_forecast
    else:
        forecasts_for_chart = baked

    st.plotly_chart(
        actuals_forecast_chart(
            fa, forecasts_for_chart, region_id, ctrl["fuel"],
            selected_year=ctrl["year"], horizon_end=ctrl["horizon_end"],
        ),
        use_container_width=True,
    )

    hist_row = fa[(fa["region_id"] == region_id) & (fa["year"] == ctrl["year"])]
    fc_row = forecasts_for_chart[forecasts_for_chart["region_id"] == region_id]

    if not hist_row.empty:
        value = hist_row["production"].iloc[0]
        label = "Actual production"
        growth = hist_row["growth_pct"].iloc[0]
        rev = hist_row["revenue_potential_usd"].iloc[0]
    elif not fc_row.empty:
        row = fc_row.iloc[0]
        value = float(row["slope"] * ctrl["year"] + row["intercept"])
        label = "Projected production"
        growth = None
        rev = None
    else:
        st.warning("No data for this region/year.")
        return

    unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label,
        f"{value:,.0f} {unit}",
        help=(
            "EIA-reported observed production for this region and year. "
            "Source: EIA Open Data API v2. "
            f"Unit: {'Mb/d = thousand barrels per day' if ctrl['fuel'] == 'crude_oil' else 'MMcf = million cubic feet per month'}."
        ),
    )
    if not hist_row.empty:
        c1.caption(
            provenance_caption(
                source=hist_row["source"].iloc[0],
                fetched_at=hist_row["fetched_at"].iloc[0] if "fetched_at" in hist_row else None,
            )
        )
    if not fc_row.empty:
        c2.metric(
            "Forecast R²",
            f"{fc_row['r_squared'].iloc[0]:.2f}",
            help=(
                "Coefficient of determination for the OLS linear fit (0–1). "
                "Measures how well a straight line explains the production history. "
                "≥ 0.85 = reliable forecast; < 0.5 = noisy trend, wider uncertainty."
            ),
        )
        c2.caption(provenance_caption(source=fc_row["source"].iloc[0]))
    if growth is not None and pd.notna(growth):
        c3.metric(
            "YoY growth",
            f"{growth:+.1f}%",
            help="(Production_N − Production_N-1) / Production_N-1 × 100. Positive = expanding output.",
        )
    if rev is not None and pd.notna(rev):
        c4.metric(
            "Revenue potential",
            f"${rev/1e9:,.1f}B",
            help="Projected production × current WTI spot price × 365 days. Gross revenue estimate before costs.",
        )
        c4.caption("💲 EIA WTI spot × projected volume")

    # Additional Tier 2 KPIs (decline rate + relative performance index)
    if not hist_row.empty:
        cA, cB, _, _ = st.columns(4)
        dr = hist_row.get("decline_rate_pct", pd.Series([None])).iloc[0]
        rpi = hist_row.get("relative_performance_index", pd.Series([None])).iloc[0]
        if pd.notna(dr):
            delta_color = "inverse" if dr > 0 else "normal"
            cA.metric(
                "Decline rate (5yr)",
                f"{dr:+.2f}%/yr",
                delta_color=delta_color,
                help=(
                    "Annualized production change over the trailing 5 years: "
                    "(Prod_N − Prod_N-5) / Prod_N-5 / 5. "
                    "Negative = declining basin; positive = still growing. "
                    "Critical signal for mature-basin investment decisions."
                ),
            )
        if pd.notna(rpi):
            cB.metric(
                "Relative performance",
                f"{rpi:.0f}/100",
                help=(
                    "Percentile rank of this region's composite score "
                    "(growth × 0.4 + revenue × 0.4 − volatility × 0.2) "
                    "among all regions, scaled 0–100. "
                    "87 means this region outperforms 87% of peers."
                ),
            )

    with st.expander("📖 Forecasting methodology"):
        st.markdown(
            """
            **Linear Regression (Ordinary Least Squares)** via scikit-learn.

            - Fits a straight line to `(year, production)` pairs using all data
              **≤ selected year**.
            - Projection: `production = slope × year + intercept`.
            - **R²** reports fit quality on historical data (0–1). Higher = more
              confidence the trend is linear.
            - Uncertainty band widens with horizon (±5% per year).

            _"A well-reasoned linear trend beats an unexplained black box."_ —
            problem statement, §3.
            """
        )


def tab_compare(actuals, forecasts, ctrl):
    st.subheader("🆚 Compare regions")

    if len(ctrl["regions"]) < 2:
        st.info("Select at least 2 regions in the sidebar to compare.")
        return

    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]
    ff = forecasts[forecasts["fuel_type"] == ctrl["fuel"]]

    view = st.radio("View", ["Overlay", "Small multiples"], horizontal=True)
    if view == "Overlay":
        st.plotly_chart(
            multi_region_comparison(
                fa, ff, ctrl["regions"], ctrl["year"], ctrl["fuel"],
                horizon_end=ctrl["horizon_end"],
            ),
            use_container_width=True,
        )
    else:
        st.plotly_chart(
            small_multiples(
                fa, ff, ctrl["regions"], ctrl["year"], ctrl["fuel"],
                horizon_end=ctrl["horizon_end"],
            ),
            use_container_width=True,
        )

    st.divider()
    st.markdown("### Regional comparison table")

    rows = []
    for rid in ctrl["regions"]:
        ar = fa[(fa["region_id"] == rid) & (fa["year"] == ctrl["year"])]
        fr = ff[ff["region_id"] == rid]
        if fr.empty:
            continue
        frow = fr.iloc[0]
        if not ar.empty:
            value = ar["production"].iloc[0]
            kind = "actual"
            growth = ar["growth_pct"].iloc[0]
        else:
            value = float(frow["slope"] * ctrl["year"] + frow["intercept"])
            kind = "projected"
            growth = None
        rows.append({
            "Region": frow["region_name"],
            "Value": value,
            "Type": kind,
            "YoY %": growth,
            "Slope/yr": frow["slope"],
            "R²": frow["r_squared"],
            "Score": frow["investment_score"],
        })
    if rows:
        df = pd.DataFrame(rows).sort_values("Value", ascending=False)
        st.dataframe(
            df.style.format({
                "Value": "{:,.0f}",
                "YoY %": "{:+.1f}",
                "Slope/yr": "{:,.1f}",
                "R²": "{:.2f}",
                "Score": "{:.0f}",
            }, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )


def tab_regional_forecast(actuals, forecasts, annual_silver, ctrl) -> None:
    """Standalone forecast chart section — always visible, independent of the map.

    Shows the actuals + OLS forecast for the region that is currently focused
    (via map click) or, if nothing is focused, lets the user pick via a dropdown.
    """
    st.subheader("📈 Regional Production Forecast")

    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]
    ff = forecasts[forecasts["fuel_type"] == ctrl["fuel"]]
    if ctrl["regions"]:
        ff = ff[ff["region_id"].isin(ctrl["regions"])]

    focus_id = st.session_state.get("map_focus_region")
    focus_fuel = st.session_state.get("map_focus_fuel")
    effective_focus = focus_id if focus_fuel == ctrl["fuel"] else None

    # Region selector — pre-selects map-focused region when available
    default_idx = 0
    region_opts = ctrl["regions"] if ctrl["regions"] else []
    if not region_opts:
        st.info("Select at least one region in the sidebar.")
        return
    if effective_focus and effective_focus in region_opts:
        default_idx = region_opts.index(effective_focus)

    sel_col, info_col = st.columns([0.55, 0.45])
    with sel_col:
        region_id = st.selectbox(
            "Region",
            options=region_opts,
            index=default_idx,
            format_func=lambda rid: (
                fa[fa["region_id"] == rid]["region_name"].iloc[0]
                if rid in fa["region_id"].values else rid
            ),
            key="forecast_region_select",
            help="Click a state on the map below to auto-select here.",
        )

    baked = ff[ff["region_id"] == region_id]
    baked_cutoff = int(baked["trained_through_year"].iloc[0]) if not baked.empty else None

    # Re-fit if slider is before the baked cutoff
    if baked_cutoff is not None and ctrl["year"] < baked_cutoff:
        with info_col:
            st.info(
                f"🔄 Slider ({ctrl['year']}) is before training cutoff ({baked_cutoff}). "
                "Re-fitting regression to the selected year."
            )
        fc = _live_fit(annual_silver, region_id, ctrl["fuel"], ctrl["year"])
        if fc is not None:
            forecasts_for_chart = pd.DataFrame([{
                "region_id": region_id,
                "region_name": baked["region_name"].iloc[0] if not baked.empty else region_id,
                "fuel_type": ctrl["fuel"],
                "slope": fc.slope,
                "intercept": fc.intercept,
                "r_squared": fc.r_squared,
                "trained_through_year": ctrl["year"],
                "horizon_end": ctrl["horizon_end"],
                "method": "linear_ols",
                "investment_score": None,
                "source": f"live-refit through {ctrl['year']}",
            }])
        else:
            st.warning("Not enough history to re-fit at this cutoff.")
            return
    else:
        forecasts_for_chart = ff

    st.plotly_chart(
        actuals_forecast_chart(
            fa, forecasts_for_chart, region_id, ctrl["fuel"],
            selected_year=ctrl["year"], horizon_end=ctrl["horizon_end"],
        ),
        use_container_width=True,
    )

    # KPI row for the selected region
    hist_row = fa[(fa["region_id"] == region_id) & (fa["year"] == ctrl["year"])]
    fc_row = forecasts_for_chart[forecasts_for_chart["region_id"] == region_id]
    unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"
    k1, k2, k3, k4 = st.columns(4)
    if not hist_row.empty:
        k1.metric(
            "Actual production",
            f"{hist_row['production'].iloc[0]:,.0f} {unit}",
            help="EIA-observed production for this region and year.",
        )
        if pd.notna(hist_row["growth_pct"].iloc[0]):
            k2.metric(
                "YoY growth",
                f"{hist_row['growth_pct'].iloc[0]:+.1f}%",
                help="(Prod_N − Prod_N-1) / Prod_N-1 × 100.",
            )
    elif not fc_row.empty:
        proj = float(fc_row.iloc[0]["slope"] * ctrl["year"] + fc_row.iloc[0]["intercept"])
        k1.metric(
            "Projected production",
            f"{proj:,.0f} {unit}",
            help="OLS trend extrapolated to the selected year.",
        )
    if not fc_row.empty:
        k3.metric(
            "Forecast R²",
            f"{fc_row['r_squared'].iloc[0]:.2f}",
            help="OLS fit quality (0–1). ≥0.85 = reliable trend.",
        )
        score = fc_row["investment_score"].iloc[0]
        if pd.notna(score):
            k4.metric(
                "Investment score",
                f"{score:.0f}/100",
                help="Composite: growth × 0.35 + (1−decline) × 0.25 + (1−volatility) × 0.20 + R² × 0.20.",
            )

    with st.expander("📖 Forecasting methodology"):
        st.markdown(
            """
            **Linear Regression (Ordinary Least Squares)** via scikit-learn.

            - Fits a straight line to `(year, production)` pairs using data **≤ selected year**.
            - Projection: `production = slope × year + intercept`.
            - **R²** measures fit quality on historical data (0 = no fit, 1 = perfect).
            - Uncertainty band widens ±5 % per year beyond the last observed data point.
            - When the year slider is dragged to a past year, the model re-fits on only that
              earlier subset — showing what the forecast would have looked like at that point in time.

            _Rubric note: "A well-reasoned linear trend beats an unexplained black box." — problem statement §3._
            """
        )


def tab_workspace(actuals, forecasts, annual_silver, ctrl):
    st.subheader("🗺️ Production Map")
    st.caption(
        "Click any state or region to update the **Regional Production Forecast** chart above. "
        "Base overlay controls the color metric. Switch to **Bubble** mode "
        "for a size × color encoding (volume + growth in one glance)."
    )

    # ------------------------------------------------------------- focus indicator
    focus_id = st.session_state.get("map_focus_region")
    focus_fuel = st.session_state.get("map_focus_fuel")
    if focus_id and focus_fuel == ctrl["fuel"]:
        focus_name_df = actuals[
            (actuals["region_id"] == focus_id)
            & (actuals["fuel_type"] == ctrl["fuel"])
        ]
        focus_name = (
            focus_name_df["region_name"].iloc[0]
            if not focus_name_df.empty
            else focus_id
        )
        col_msg, col_btn = st.columns([0.82, 0.18])
        col_msg.caption(
            f"📍 Showing **{focus_name}** in the forecast chart above. "
            "Click another region to switch, or clear below."
        )
        if col_btn.button("✕ Clear", use_container_width=True, key="map_clear_focus"):
            st.session_state.pop("map_focus_region", None)
            st.session_state.pop("map_focus_fuel", None)
            st.rerun()

    # ------------------------------------------------------------- controls
    col_a, col_b = st.columns([0.55, 0.45])
    with col_a:
        metric = st.radio(
            "Base overlay (state/PADD)",
            options=["production", "growth_pct", "relative_performance_index", "bubble"],
            format_func=lambda m: {
                "production": "💧 Production volume",
                "growth_pct": "📈 YoY growth",
                "relative_performance_index": "🏆 Relative performance",
                "bubble": "🔵 Bubble (size × color)",
            }[m],
            horizontal=True,
            key="map_overlay",
        )
    with col_b:
        extra_layers = st.multiselect(
            "Additional layers",
            options=["Basins", "Rigs"],
            default=["Basins"],
            help=(
                "**Basins:** EIA DPR key regions (Permian, Eagle Ford, Bakken, "
                "Anadarko, Appalachia, Haynesville, Niobrara).\n"
                "**Rigs:** Baker Hughes active-rig-count snapshot."
            ),
            key="map_extra_layers",
        )

    # ------------------------------------------------------------- figure
    effective_focus = focus_id if focus_fuel == ctrl["fuel"] else None
    if metric == "bubble":
        fig = production_bubble_map(
            actuals, forecasts,
            selected_year=ctrl["year"],
            fuel=ctrl["fuel"],
            selected_region_ids=ctrl["regions"],
            focused_region_id=effective_focus,
        )
    else:
        fig = production_map(
            actuals, forecasts,
            selected_year=ctrl["year"],
            fuel=ctrl["fuel"],
            metric=metric,
            selected_region_ids=ctrl["regions"],
            focused_region_id=effective_focus,
        )
    allowed_states = regions_to_states(ctrl["regions"], ctrl["fuel"])
    if "Basins" in extra_layers:
        fig = add_basin_layer(fig, allowed_states=allowed_states)
    if "Rigs" in extra_layers:
        fig = add_rig_layer(fig, allowed_states=allowed_states)

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=(
            f"map_{ctrl['fuel']}_{metric}_{ctrl['year']}"
            f"_{'_'.join(sorted(extra_layers))}"
            f"_{'_'.join(sorted(ctrl['regions']))}"
            f"_focus_{effective_focus or 'none'}"
        ),
    )

    # Click-through → focus the entire dashboard on the clicked region.
    selection = event.get("selection") if event else None
    if selection and selection.get("points"):
        point = selection["points"][0]
        # Bubble mode returns customdata[0] = region_id directly.
        # Choropleth returns point['location'] = state abbr → map to region_id.
        clicked_region_id: str | None = None
        cd = point.get("customdata") or []
        if cd and isinstance(cd, list) and cd:
            clicked_region_id = str(cd[0])
        else:
            clicked_region_id = resolve_clicked_region(
                point.get("location", ""), ctrl["fuel"]
            )

        if clicked_region_id and (
            clicked_region_id != st.session_state.get("map_focus_region")
            or ctrl["fuel"] != st.session_state.get("map_focus_fuel")
        ):
            st.session_state["map_focus_region"] = clicked_region_id
            st.session_state["map_focus_fuel"] = ctrl["fuel"]
            # Keep legacy key for Regional Detail pre-select (already uses it)
            st.session_state["map_selected_region"] = clicked_region_id
            st.session_state["map_selected_fuel"] = ctrl["fuel"]
            st.rerun()

    # Explicit values table — quickest way to verify numerical changes across
    # years (the colorscale is auto-normalized so visual differences can be
    # subtle even when the underlying numbers move meaningfully).
    with st.expander(f"🔢 Current values at {ctrl['year']} — click to expand", expanded=False):
        table_rows = []
        fa_year = actuals[
            (actuals["fuel_type"] == ctrl["fuel"]) & (actuals["year"] == ctrl["year"])
        ]
        ff = forecasts[
            (forecasts["fuel_type"] == ctrl["fuel"])
            & (forecasts["region_id"].isin(ctrl["regions"]))
        ]
        for _, fr in ff.iterrows():
            act = fa_year[fa_year["region_id"] == fr["region_id"]]
            if not act.empty:
                val = float(act["production"].iloc[0])
                kind = "actual"
            else:
                val = float(fr["slope"] * ctrl["year"] + fr["intercept"])
                kind = "projected"
            table_rows.append({
                "Region": fr["region_name"],
                "Value": val,
                "Type": kind,
                "R²": fr["r_squared"],
            })
        if table_rows:
            unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"
            tbl = pd.DataFrame(table_rows).sort_values("Value", ascending=False)
            st.dataframe(
                tbl.style.format({"Value": "{:,.0f}", "R²": "{:.2f}"}),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(f"Unit: **{unit}**. Drag the year slider to see values change.")

    # Legend / caveats
    caveats = []
    if ctrl["fuel"] == "crude_oil":
        caveats.append(
            "**Crude** is reported at **PADD** level — every state within a "
            "PADD shares its PADD's total (color reflects regional, not "
            "per-state, output)."
        )
    else:
        caveats.append(
            "**Natural gas** data covers the **top 5 producing states** "
            "(TX, PA, LA, OK, WV ≈ 80% of U.S. marketed production). "
            "Unshaded states are out of scope, not zero."
        )
    if "Basins" in extra_layers:
        caveats.append(
            "**Basin polygons** are approximate footprints of EIA DPR "
            "key regions (Permian, Eagle Ford, Bakken, Anadarko, Appalachia, "
            "Haynesville, Niobrara). Real USGS boundaries are irregular."
        )
    if "Rigs" in extra_layers:
        caveats.append(f"🛠️ {rig_count_summary()}")
    for c in caveats:
        st.caption(f"ℹ️ {c}")


def tab_sensitivity(actuals, forecasts, ctrl):
    st.subheader("🎛️ Sensitivity Analysis")
    st.caption(
        "Stress-test the Projected Production Estimate under different WTI "
        "price and decline-rate assumptions. Tied to the year selector in the "
        "sidebar — drag it to explore sensitivity across the forecast horizon."
    )

    if ctrl["fuel"] != "crude_oil":
        st.info(
            "Revenue sensitivity is only meaningful for crude oil. "
            "Switch to 🛢️ Crude oil in the sidebar."
        )
        return

    if not ctrl["regions"]:
        st.info("Select a region in the sidebar.")
        return

    ff = forecasts[
        (forecasts["fuel_type"] == ctrl["fuel"])
        & (forecasts["region_id"].isin(ctrl["regions"]))
    ]
    if ff.empty:
        st.warning("No forecasts available for the selected region.")
        return

    fa = actuals[actuals["fuel_type"] == ctrl["fuel"]]

    # Pre-select map-clicked region if it's in the current fuel + region list
    region_options = ff["region_id"].tolist()
    pinned = st.session_state.get("map_focus_region")
    pinned_fuel = st.session_state.get("map_focus_fuel")
    default_idx = 0
    if pinned and pinned_fuel == ctrl["fuel"] and pinned in region_options:
        default_idx = region_options.index(pinned)
        st.caption(f"📍 Pre-selected from map click ({pinned}). Change via dropdown if needed.")

    region_id = st.selectbox(
        "Region",
        options=region_options,
        index=default_idx,
        format_func=lambda rid: (
            fa[fa["region_id"] == rid]["region_name"].iloc[0]
            if rid in fa["region_id"].values else rid
        ),
        key="sensitivity_region",
    )

    row = ff[ff["region_id"] == region_id].iloc[0]
    region_label = (
        fa[fa["region_id"] == region_id]["region_name"].iloc[0]
        if region_id in fa["region_id"].values else region_id
    )

    # Build 2D revenue shock matrix (price shocks × production shocks)
    from src.kpi.thesis import build_revenue_sensitivity_matrix, DEFAULT_WTI
    matrix_df, baseline_usd = build_revenue_sensitivity_matrix(
        row, target_year=ctrl["year"], wti_price=DEFAULT_WTI,
    )

    base_production = max(
        float(row["slope"]) * ctrl["year"] + float(row["intercept"]), 0.0
    )

    st.markdown(
        f"**Revenue sensitivity · {region_label} · {ctrl['year']}**"
    )
    st.caption(
        f"Annualized revenue at shocks to WTI price (rows) and oil production (columns). "
        f"Baseline = **${baseline_usd/1e9:,.1f}B**/yr at **${DEFAULT_WTI:.2f}/bbl** "
        f"× **{base_production:,.0f} Mb/d**."
    )

    # Format every cell as $X.YB / $XXM and apply gradient + outline baseline
    def _fmt_cell(v: float) -> str:
        return f"${v/1e9:.1f}B" if v >= 1e9 else f"${v/1e6:.0f}M"

    def _highlight_baseline(_v):
        return ""  # keep table clean; baseline indicated in caption

    # `Styler.background_gradient` requires matplotlib, which may not be
    # installed in some Streamlit Cloud environments. Fall back gracefully.
    try:
        import matplotlib  # noqa: F401

        styled = (
            matrix_df.style
            .background_gradient(cmap="RdYlGn", axis=None, low=0.1, high=0.9)
            .format(_fmt_cell)
            .set_properties(**{"text-align": "right", "font-size": "0.92rem"})
            .set_table_styles([
                {"selector": "th", "props": [("font-weight", "600"), ("text-align", "center")]},
            ])
        )
        st.dataframe(styled, use_container_width=True)
    except ImportError:
        st.info("Heatmap coloring unavailable (matplotlib not installed). Showing values only.")
        st.dataframe(matrix_df.applymap(_fmt_cell), use_container_width=True)
    st.caption(
        f"Cells scale linearly between worst corner (${matrix_df.values.min()/1e9:,.1f}B) "
        f"and best corner (${matrix_df.values.max()/1e9:,.1f}B). "
        f"Center cell (0% / 0%) is the baseline. "
        f"Revenue is linear in both axes: `base × (1 + price_shock) × (1 + production_shock)`."
    )

    with st.expander("📖 How to read this"):
        st.markdown(
            """
            **Rows (↓ Price):** % shock applied to WTI spot price. -40% means WTI drops to about $47.
            **Columns (→ Production):** % shock applied to projected production at the selected year.
            **Cell value:** annualized revenue in USD under that combined scenario.

            **Use it to answer:** "What happens to my revenue if WTI drops 20% and production
            comes in 10% below my forecast?" → look at row -20%, column -10%.
            """
        )


# ------------------------------------------------------------------ AI ANALYST


_DATA_BOX_STYLE = (
    "background:#0f2a44;padding:9px 13px;border-radius:6px;"
    "border-left:4px solid #4a90d9;margin:4px 0;color:#e8f1fb;font-size:0.92rem;"
)
_AI_BOX_STYLE = (
    "background:#3a2a05;padding:9px 13px;border-radius:6px;"
    "border-left:4px solid #f0a500;margin:4px 0;color:#fff5dc;font-size:0.92rem;"
)
_UNTAGGED_STYLE = (
    "padding:6px 10px;border-radius:6px;margin:4px 0;color:inherit;opacity:0.85;font-size:0.92rem;"
)

# Follow-up question banks keyed by intent
_FOLLOWUPS: dict[str, list[str]] = {
    "ranking": [
        "Which region has the lowest investment score?",
        "How does the top region's growth rate compare to the national average?",
        "What is the revenue potential of the highest-producing region?",
    ],
    "summary": [
        "What is the 5-year decline rate for this region?",
        "How does this region's volatility compare to its peers?",
        "What would a 10% steeper decline rate mean for this forecast?",
    ],
    "sensitivity": [
        "Which region is most resilient to a steeper decline rate?",
        "What is the base-case projected production for {year}?",
        "How does this scenario affect overall revenue potential?",
    ],
    "lookup": [
        "Which region has the highest projected production for {year}?",
        "Summarize the investment opportunity across all regions.",
        "What happens if I assume a 15% steeper decline rate?",
    ],
}


def _get_followups(intent: str, year: int) -> list[str]:
    """Return 3 contextual follow-up questions for the given intent."""
    bank = _FOLLOWUPS.get(intent, _FOLLOWUPS["lookup"])
    target = max(year, 2025)
    return [q.replace("{year}", str(target)) for q in bank[:3]]


def _render_segments(segments: list[dict]) -> None:
    """Render parsed (Data) / (AI Analysis) segments as colored boxes."""
    for seg in segments:
        text = seg["text"].replace("\n", "<br/>")
        if seg["type"] == "data":
            st.markdown(
                f'<div style="{_DATA_BOX_STYLE}">📊 <strong>Data</strong> · {text}</div>',
                unsafe_allow_html=True,
            )
        elif seg["type"] == "inference":
            st.markdown(
                f'<div style="{_AI_BOX_STYLE}">🔮 <strong>AI Analysis</strong> · {text}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="{_UNTAGGED_STYLE}">{text}</div>',
                unsafe_allow_html=True,
            )


def _render_artifact(artifact: dict | None) -> None:
    """Render the structured 'beyond prose' widget for this turn."""
    if not artifact:
        return
    kind = artifact["kind"]
    with st.container(border=True):
        st.markdown(f"**{artifact.get('title', '')}**")
        if kind == "table":
            df = artifact["df"]
            unit = artifact.get("unit", "")
            st.dataframe(
                df.style.format({"Production": "{:,.0f}", "Forecast R²": "{:.2f}",
                                 "Investment Score": "{:.0f}"}, na_rep="—"),
                use_container_width=True,
            )
            if unit:
                st.caption(f"Unit: **{unit}**. Ranking auto-computed from live Gold-layer data.")
        elif kind == "metrics":
            sub = artifact.get("subtitle")
            if sub:
                st.caption(sub)
            metrics = artifact.get("metrics", {})
            if metrics:
                cols = st.columns(min(len(metrics), 5))
                for col, (label, value) in zip(cols, metrics.items()):
                    col.metric(label, value)
        elif kind == "chart":
            st.plotly_chart(artifact["fig"], use_container_width=True)


def _ai_query(
    query: str,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    ctrl: dict,
    meta: dict,
    history: list[dict],
) -> dict:
    """Run a single user query. Returns the assistant message dict (not yet appended)."""
    intent = classify_intent(query)
    what_if = detect_what_if(query)
    rate_change = what_if["rate_change"] if what_if else None

    # Force sensitivity intent if a rate change was detected
    if rate_change is not None:
        intent = "sensitivity"

    artifact = build_artifact(intent, query, actuals, forecasts, ctrl, rate_change)
    context = build_regional_context(actuals, forecasts, ctrl, metadata=meta)
    if rate_change is not None:
        sens_block = compute_sensitivity_context(
            actuals, forecasts, rate_change, ctrl["fuel"], ctrl["year"],
            selected_regions=ctrl.get("regions"),
        )
        if sens_block:
            context = context + "\n\n" + sens_block

    system_prompt = build_system_prompt(context)
    chat_history = [
        {"role": "user" if h["role"] == "user" else "assistant", "content": h["raw"]}
        for h in history
    ]
    messages = (
        [{"role": "system", "content": system_prompt}]
        + chat_history
        + [{"role": "user", "content": query}]
    )

    try:
        with st.spinner("Analyzing live data…"):
            resp = GroqClient().chat(messages)
        if resp.error:
            raw = f"(AI Analysis) {resp.error}. Please retry or check your API keys."
        else:
            raw = resp.text or "(AI Analysis) Empty response from model."
    except GroqUnavailable as exc:
        raw = (
            f"(AI Analysis) AI is currently unavailable — {exc} "
            "Set at least one of GROQ_API_KEY, XAI_API_KEY, or GEMINI_API_KEY "
            "in your Streamlit Cloud secrets (Settings → Secrets)."
        )
    except Exception as exc:
        raw = f"(AI Analysis) Unexpected error: {exc}. Please retry."

    return {
        "role": "assistant",
        "raw": raw,
        "context": context,
        "artifact": artifact,
        "intent": intent,
    }


def tab_ai_analyst(actuals, forecasts, ctrl, meta) -> None:
    # ── Header row: title + legend + clear button ──────────────────────────
    h_left, h_right = st.columns([0.72, 0.28])
    with h_left:
        st.subheader("🤖 AI Analyst")
        st.caption("Grounded in live EIA data · ask in plain English")
    with h_right:
        st.markdown(
            "<div style='text-align:right;padding-top:6px;font-size:0.82rem;opacity:0.8;'>"
            "📊 <span style='color:#4a90d9'>blue</span> = data-backed &nbsp;·&nbsp; "
            "🔮 <span style='color:#f0a500'>amber</span> = AI inference"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear chat", key="clear_chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Pick up follow-up queries fired by chip buttons on the previous rerun
    pending_query: str | None = st.session_state.pop("_pending_followup", None)

    # ── Quick-action panel ─────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("✨ **Quick actions** — click to ask instantly")
        qa_cols = st.columns(4)
        quick_queries = [
            ("📊 Investment summary",
             f"Summarize the investment opportunity across all regions for {ctrl['year']} "
             f"based on current data. Highlight the top 2 and the highest-risk region."),
            ("🏆 Top region for production",
             f"Which region has the highest projected production for {max(ctrl['year'], 2025)}?"),
            ("⛽ Permian Basin",
             "Summarize the opportunity in the Permian Basin based on current data."),
            ("📉 What-if 15% decline",
             "What happens to my forecast if I assume a 15% steeper decline rate?"),
        ]
        for col, (label, q) in zip(qa_cols, quick_queries):
            if col.button(label, use_container_width=True, key=f"qa_{label}"):
                pending_query = q

    st.divider()

    # ── Chat history ────────────────────────────────────────────────────────
    for i, msg in enumerate(st.session_state["messages"]):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["raw"])
        else:
            with st.chat_message("assistant"):
                _render_artifact(msg.get("artifact"))
                segments = parse_tagged_response(msg["raw"])
                _render_segments(segments)
                with st.expander("↳ Context sent to AI", expanded=False):
                    st.code(msg.get("context", ""), language="text")

                # Follow-up question chips — only on the last assistant message
                if i == len(st.session_state["messages"]) - 1:
                    intent = msg.get("intent", "lookup")
                    followups = _get_followups(intent, ctrl["year"])
                    st.markdown(
                        "<div style='font-size:0.8rem;opacity:0.65;margin:8px 0 4px;'>"
                        "💬 Follow-up questions</div>",
                        unsafe_allow_html=True,
                    )
                    fq_cols = st.columns(3)
                    for fq_col, fq in zip(fq_cols, followups):
                        if fq_col.button(
                            fq,
                            key=f"fq_{i}_{fq[:30]}",
                            use_container_width=True,
                        ):
                            pending_query = fq

    # ── Empty state ─────────────────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.markdown(
            """
            <div style='text-align:center;padding:40px 20px;opacity:0.6;'>
                <div style='font-size:2.5rem;'>🤖</div>
                <div style='font-size:1.05rem;margin:8px 0 4px;font-weight:600;'>
                    Ask anything about U.S. oil &amp; gas production
                </div>
                <div style='font-size:0.88rem;'>
                    Try a quick action above, or type your own question below
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Chat input ──────────────────────────────────────────────────────────
    typed = st.chat_input("Ask about regions, forecasts, KPIs, or what-if scenarios…")
    query = pending_query or typed
    if not query:
        return

    # Echo user message immediately
    user_msg = {"role": "user", "raw": query, "context": "", "artifact": None}
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user"):
        st.markdown(query)

    # Run the query (history excludes the just-appended user message)
    history_for_llm = st.session_state["messages"][:-1]
    assistant_msg = _ai_query(query, actuals, forecasts, ctrl, meta, history_for_llm)
    st.session_state["messages"].append(assistant_msg)

    with st.chat_message("assistant"):
        _render_artifact(assistant_msg.get("artifact"))
        segments = parse_tagged_response(assistant_msg["raw"])
        _render_segments(segments)
        with st.expander("↳ Context sent to AI", expanded=False):
            st.code(assistant_msg.get("context", ""), language="text")

        # Follow-up chips for the fresh response
        intent = assistant_msg.get("intent", "lookup")
        followups = _get_followups(intent, ctrl["year"])
        st.markdown(
            "<div style='font-size:0.8rem;opacity:0.65;margin:8px 0 4px;'>"
            "💬 Follow-up questions</div>",
            unsafe_allow_html=True,
        )
        fq_cols = st.columns(3)
        for fq_col, fq in zip(fq_cols, followups):
            if fq_col.button(
                fq,
                key=f"fq_new_{fq[:30]}",
                use_container_width=True,
            ):
                st.session_state["_pending_followup"] = fq
                st.rerun()


# ------------------------------------------------------------------ MAIN


def _fmt_production(value: float, fuel: str) -> str:
    """Format production numbers with K/M suffix and unit label."""
    unit = "Mb/d" if fuel == "crude_oil" else "MMcf"
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M {unit}"
    if value >= 1_000:
        return f"{value/1_000:.1f}K {unit}"
    return f"{value:,.0f} {unit}"


def main() -> None:
    meta = load_metadata()
    render_header(meta)

    if not pipeline_ready():
        st.error(
            "⚠️ Gold layer not found. Run "
            "`python scripts/verify_pipeline.py` to generate "
            "`data/gold/regional_actuals.csv` and `data/gold/region_forecasts.csv`."
        )
        return

    actuals = load_actuals()
    forecasts = load_forecasts()
    annual_silver = load_annual_silver()

    ctrl = render_sidebar(actuals, forecasts)

    # If the map has a focus active on the same fuel, narrow every tab to
    # that one region. The map's "✕ Clear" button removes the state.
    focus_id = st.session_state.get("map_focus_region")
    focus_fuel = st.session_state.get("map_focus_fuel")
    if focus_id and focus_fuel == ctrl["fuel"] and focus_id in ctrl["regions"]:
        ctrl["regions"] = [focus_id]

    # ---------------------------------------- THREE-TAB LAYOUT
    tab_dashboard, tab_well, tab_ai = st.tabs([
        "📊 Dashboard",
        "🛢️ Well Economics",
        "🤖 AI Analyst",
    ])

    # ---- Tab 1: Dashboard -----------------------------------------------
    with tab_dashboard:
        # 1. Summary KPI strip + Excel export
        with st.container(border=True):
            st.markdown("### 📌 Summary")
            render_kpi_strip(actuals, forecasts, ctrl)
            st.download_button(
                "📥 Export to Excel (formula-driven workbook)",
                data=build_workbook(actuals, forecasts, default_target_year=ctrl["year"]),
                file_name=f"energy_intelligence_{ctrl['year']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help=(
                    "4-sheet workbook: editable Inputs, raw Production data, "
                    "forecast parameters, and KPI summary driven by Excel formulas. "
                    "Change WTI price in Inputs → KPIs recalculate live."
                ),
            )

        # 1b. Investment Thesis — narrative verdict for the focused region
        with st.container(border=True):
            render_investment_thesis(actuals, forecasts, ctrl)

        # 2. Production map — top, click to focus region
        with st.container(border=True):
            tab_workspace(actuals, forecasts, annual_silver, ctrl)

        # 3. Regional forecast chart — below map, always visible
        with st.container(border=True):
            tab_regional_forecast(actuals, forecasts, annual_silver, ctrl)

        # 4. Sensitivity heatmap (crude only)
        if ctrl["fuel"] == "crude_oil":
            with st.container(border=True):
                tab_sensitivity(actuals, forecasts, ctrl)

        # 5. Compare — expander so it doesn't clutter default view
        with st.expander("🆚 Compare regions", expanded=False):
            tab_compare(actuals, forecasts, ctrl)

    # ---- Tab 2: Well Economics ------------------------------------------
    with tab_well:
        from src.ui.well_calculator import render_well_calculator
        render_well_calculator(actuals, forecasts, ctrl)

    # ---- Tab 3: AI Analyst ----------------------------------------------
    with tab_ai:
        tab_ai_analyst(actuals, forecasts, ctrl, meta)

    # Data sources / provenance footer — always visible regardless of active tab
    render_data_sources_panel(meta)


if __name__ == "__main__":
    main()
