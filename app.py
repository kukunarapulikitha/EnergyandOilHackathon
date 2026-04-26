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
    merge_uploaded_actuals,
    parse_excel_upload,
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
    col1, col2, col3 = st.columns([0.60, 0.20, 0.20])
    with col1:
        st.title("⚡ Energy Intelligence System")
        st.caption(
            "U.S. oil & gas production analysis with linear-regression "
            "forecasting and Medallion data architecture."
        )
    with col2:
        if meta:
            dqs = meta.get("dqs", {})
            score = dqs.get("score", 0)
            color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            st.metric("Data Quality", f"{color} {score:.1f}/100")
            from src.ui.provenance import fresh_age
            st.caption(f"Updated {fresh_age(meta.get('fetched_at'))}")
    with col3:
        st.markdown("&nbsp;")  # vertical spacer
        if st.button("🔄 Refresh data", help="Re-run the EIA pipeline and reload the app"):
            _refresh_data(meta)


def _refresh_data(meta: dict) -> None:
    """Trigger the full Medallion pipeline. On failure, show stale-data warning."""
    import subprocess
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
                f"Continuing on cached data from {stale_ts[:19] if isinstance(stale_ts, str) else stale_ts}."
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

        # ---- Excel import ------------------------------------------------
        st.divider()
        st.markdown("**📂 Import custom data (Excel)**")
        st.caption(
            "Upload a `.xlsx` file to overlay your own production figures on "
            "the Gold-layer data. Required columns: `region_id`, `fuel_type`, "
            "`year`, `production`. Export the workbook from the Dashboard tab "
            "for a pre-formatted template."
        )
        uploaded_file = st.file_uploader(
            "Upload .xlsx",
            type=["xlsx", "xls"],
            key="excel_upload",
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            parsed_df, err = parse_excel_upload(uploaded_file)
            if err:
                st.error(f"❌ {err}")
            elif parsed_df is not None and not parsed_df.empty:
                st.session_state["uploaded_actuals"] = parsed_df
                st.success(
                    f"✅ Loaded **{len(parsed_df):,}** rows from "
                    f"`{uploaded_file.name}` — overlaying Gold data."
                )
            else:
                st.error("❌ File parsed but contained no valid rows.")

        if st.session_state.get("uploaded_actuals") is not None:
            n = len(st.session_state["uploaded_actuals"])
            st.info(f"📂 **{n:,} uploaded rows** are active.")
            if st.button("🗑️ Clear imported data", use_container_width=True):
                del st.session_state["uploaded_actuals"]
                st.rerun()

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

    unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label,
        f"{total_value:,.0f} {unit}",
        help=(
            "Sum of production across all selected regions for the chosen year. "
            "Shows 'actual' when EIA observed data exists; 'projected' when the "
            "year is beyond the latest data point (OLS linear trend)."
        ),
    )

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

    unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        label,
        f"{total_value:,.0f} {unit}",
        help=(
            "Sum of production across selected regions. "
            "'Actual' = EIA observed data; 'Projected' = OLS trend (slope × year + intercept). "
            f"Unit: {'thousand barrels/day' if ctrl['fuel'] == 'crude_oil' else 'billion cubic feet/month'}."
        ),
    )

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

    avg_score = ff[ff["region_id"].isin(selected)]["investment_score"].mean()
    c4.metric(
        "Avg Investment Score",
        f"{avg_score:.0f}/100" if pd.notna(avg_score) else "—",
        help=(
            "Composite 0–100: growth × 0.35 + (1−decline) × 0.25 + (1−volatility) × 0.20 + R² × 0.20. "
            "Higher = more attractive investment case. Averaged across selected regions."
        ),
    )

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


def tab_workspace(actuals, forecasts, annual_silver, ctrl):
    st.subheader("🗺️ Analytical Workspace")
    st.caption(
        "Click any region to focus the entire dashboard on it. "
        "Base overlay controls the color metric. Switch to **Bubble** mode "
        "for a size × color encoding (volume + growth in one glance)."
    )

    # ------------------------------------------------------------- focus banner
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
        banner, btn = st.columns([0.8, 0.2])
        with banner:
            st.info(
                f"📍 **Focused on {focus_name}** — every tab is filtered to "
                f"this region. Click the map again to switch focus."
            )
        with btn:
            if st.button("Clear focus", use_container_width=True):
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

    # Focus KPI strip — when a region is focused, show its headline numbers
    # directly below the map so the dashboard feels responsive to the click.
    if effective_focus:
        fa_focus = actuals[
            (actuals["region_id"] == effective_focus)
            & (actuals["fuel_type"] == ctrl["fuel"])
            & (actuals["year"] == ctrl["year"])
        ]
        ff_focus = forecasts[
            (forecasts["region_id"] == effective_focus)
            & (forecasts["fuel_type"] == ctrl["fuel"])
        ]
        if not ff_focus.empty:
            unit = "Mb/d" if ctrl["fuel"] == "crude_oil" else "MMcf"
            if not fa_focus.empty:
                prod = float(fa_focus["production"].iloc[0])
                kind = "actual"
                growth = fa_focus["growth_pct"].iloc[0]
            else:
                row = ff_focus.iloc[0]
                prod = float(row["slope"] * ctrl["year"] + row["intercept"])
                kind = "projected"
                growth = None
            name = ff_focus["region_name"].iloc[0]
            st.markdown(f"#### 📍 {name} — {ctrl['year']}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Production" if kind == "actual" else "Projected production",
                f"{prod:,.0f} {unit}",
            )
            if growth is not None and pd.notna(growth):
                k2.metric(
                    "YoY growth",
                    f"{growth:+.1f}%",
                    help="(Production_N − Production_N-1) / Production_N-1 × 100. Positive = growing, negative = declining.",
                )
            k3.metric(
                "Forecast R²",
                f"{ff_focus['r_squared'].iloc[0]:.2f}",
                help="OLS linear fit quality (0–1). ≥0.85 = reliable trend; <0.5 = noisy/non-linear.",
            )
            k4.metric(
                "Investment score",
                f"{ff_focus['investment_score'].iloc[0]:.0f}/100",
                help="Composite: growth × 0.35 + (1−decline) × 0.25 + (1−volatility) × 0.20 + R² × 0.20.",
            )
            # Status badge + hint to Compare tab for full chart
            label, emoji = classify_region(
                effective_focus, ctrl["fuel"], ctrl["year"], actuals, forecasts
            )
            st.caption(
                f"{emoji} **{label}** · "
                f"Forecast R²: {ff_focus['r_squared'].iloc[0]:.2f} · "
                f"Investment score: {ff_focus['investment_score'].iloc[0]:.0f}/100 · "
                f"Use the **🆚 Compare regions** expander below for full trend charts."
            )
            st.divider()

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
    region_id = st.selectbox(
        "Region",
        options=ff["region_id"].tolist(),
        format_func=lambda rid: (
            fa[fa["region_id"] == rid]["region_name"].iloc[0]
            if rid in fa["region_id"].values else rid
        ),
        key="sensitivity_region",
    )

    row = ff[ff["region_id"] == region_id].iloc[0]
    st.plotly_chart(
        sensitivity_heatmap(row, target_year=ctrl["year"]),
        use_container_width=True,
    )

    with st.expander("📖 How to read this"):
        st.markdown(
            """
            **X-axis:** WTI crude spot-price assumption.
            **Y-axis:** adjustment applied to the fitted slope — negative values simulate a
            decline (mature basin depletion), positive values simulate acceleration.
            **Cell value:** estimated annual revenue (in $B/yr) at the year selected in the sidebar.

            **Color:** red = weak revenue, green = strong. Use this to answer
            "at what WTI price does this region still make sense even under declining production?"
            """
        )


# ------------------------------------------------------------------ AI ANALYST


_DATA_BOX_STYLE = (
    "background:#0f2a44;padding:10px 14px;border-radius:6px;"
    "border-left:4px solid #4a90d9;margin:6px 0;color:#e8f1fb;"
)
_AI_BOX_STYLE = (
    "background:#3a2a05;padding:10px 14px;border-radius:6px;"
    "border-left:4px solid #f0a500;margin:6px 0;color:#fff5dc;"
)
_UNTAGGED_STYLE = (
    "padding:8px 12px;border-radius:6px;margin:6px 0;color:inherit;opacity:0.85;"
)


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
        client = GroqClient()
    except GroqUnavailable as exc:
        return {
            "role": "assistant",
            "raw": f"(AI Analysis) AI is unavailable — {exc}",
            "context": context,
            "artifact": artifact,
            "intent": intent,
        }

    with st.spinner("Analyzing live data with Llama 3.3 70B..."):
        resp = client.chat(messages)

    if resp.error:
        raw = f"(AI Analysis) {resp.error}. Please retry or check your GROQ_API_KEY."
    else:
        raw = resp.text or "(AI Analysis) Empty response from model."

    return {
        "role": "assistant",
        "raw": raw,
        "context": context,
        "artifact": artifact,
        "intent": intent,
    }


def tab_ai_analyst(actuals, forecasts, ctrl, meta) -> None:
    st.subheader("🤖 AI Analyst")
    st.caption(
        "Ask questions in plain English about regional production, forecasts, "
        "and KPIs. Answers are grounded in your **live Gold-layer data**. "
        "📊 blue blocks = data-backed claims · 🔮 amber blocks = AI inference."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Quick-action buttons — pre-fill common queries for the demo flow
    qa_cols = st.columns(4)
    quick_queries = [
        ("📊 Investment summary",
         f"Summarize the investment opportunity across all regions for {ctrl['year']} "
         f"based on current data. Highlight the top 2 and the highest-risk region."),
        ("🏆 Top region for 2027",
         "Which region has the highest projected production for 2027?"),
        ("⛽ Permian Basin",
         "Summarize the opportunity in the Permian Basin based on current data."),
        ("📉 What-if 15% decline",
         "What happens to my forecast if I assume a 15% steeper decline rate?"),
    ]
    pending_query: str | None = None
    for col, (label, q) in zip(qa_cols, quick_queries):
        if col.button(label, use_container_width=True, key=f"qa_{label}"):
            pending_query = q

    # Render existing chat history
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["raw"])
        else:
            with st.chat_message("assistant"):
                _render_artifact(msg.get("artifact"))
                segments = parse_tagged_response(msg["raw"])
                _render_segments(segments)
                with st.expander("↳ Data sent to AI (proves grounding)"):
                    st.code(msg.get("context", ""), language="text")

    # Chat input — accepts free-text questions; quick-action buttons override
    typed = st.chat_input("Ask about regions, forecasts, or KPIs…")
    query = pending_query or typed
    if not query:
        if not st.session_state["messages"]:
            st.info(
                "👆 Click a quick action above, or type a question below. "
                "Try: *'Which region has the highest projected production for 2027?'*"
            )
        return

    # Echo the user message immediately
    user_msg = {"role": "user", "raw": query, "context": "", "artifact": None}
    st.session_state["messages"].append(user_msg)
    with st.chat_message("user"):
        st.markdown(query)

    # Run the query (history excludes the just-appended user message)
    history_for_llm = st.session_state["messages"][:-1]
    assistant_msg = _ai_query(
        query, actuals, forecasts, ctrl, meta, history_for_llm,
    )
    st.session_state["messages"].append(assistant_msg)

    with st.chat_message("assistant"):
        _render_artifact(assistant_msg.get("artifact"))
        segments = parse_tagged_response(assistant_msg["raw"])
        _render_segments(segments)
        with st.expander("↳ Data sent to AI (proves grounding)"):
            st.code(assistant_msg.get("context", ""), language="text")


# ------------------------------------------------------------------ MAIN


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

    # Overlay any user-uploaded Excel data onto the Gold-layer actuals
    uploaded_actuals = st.session_state.get("uploaded_actuals")
    if uploaded_actuals is not None and not uploaded_actuals.empty:
        actuals = merge_uploaded_actuals(actuals, uploaded_actuals)

    # If the map has a focus active on the same fuel, narrow every tab to
    # that one region. The Map tab's "Clear focus" button removes the state.
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
        # 1. Top KPI strip
        with st.container(border=True):
            st.markdown("### 📌 Summary")
            render_kpi_strip(actuals, forecasts, ctrl)

        # 2. Map workspace — full width
        with st.container(border=True):
            tab_workspace(actuals, forecasts, annual_silver, ctrl)

        # 3. Sensitivity (crude only) — full width
        if ctrl["fuel"] == "crude_oil":
            with st.container(border=True):
                tab_sensitivity(actuals, forecasts, ctrl)

        # 4. Compare — expander so it doesn't clutter default view
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