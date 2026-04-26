"""Data provenance UI helpers.

Every figure shown in the dashboard should be traceable to its origin:
    - Source system (EIA endpoint or derived model)
    - Series ID for row-level provenance
    - Fetched-at timestamp
    - Short description

Consumed from `data/metadata.json` and row-level `source`/`series_id`
columns present on every Silver/Gold table.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st


def fresh_age(fetched_at: str | pd.Timestamp | None) -> str:
    """Human-readable 'N minutes/hours/days ago' string."""
    if fetched_at is None or (isinstance(fetched_at, float) and pd.isna(fetched_at)):
        return "unknown"
    try:
        ts = pd.Timestamp(fetched_at)
    except Exception:
        return "unknown"
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    delta = datetime.now(timezone.utc) - ts.to_pydatetime()
    mins = int(delta.total_seconds() / 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    hrs = mins // 60
    if hrs < 48:
        return f"{hrs} h ago"
    return f"{hrs // 24} days ago"


def provenance_caption(
    source: str | None,
    series_id: str | None = None,
    fetched_at: str | pd.Timestamp | None = None,
) -> str:
    """Small markdown string rendered under a metric card."""
    parts = []
    if source:
        parts.append(f"📡 `{source}`")
    if series_id and series_id not in ("None", "nan", ""):
        parts.append(f"id `{series_id}`")
    if fetched_at:
        parts.append(f"⏱ {fresh_age(fetched_at)}")
    return " · ".join(parts) if parts else ""


def provenance_popover(
    label: str,
    source: str | None,
    series_id: str | None = None,
    fetched_at: str | pd.Timestamp | None = None,
    description: str | None = None,
) -> None:
    """Detailed popover — use where a caption is too small."""
    with st.popover(label, use_container_width=False):
        if description:
            st.markdown(f"**{description}**")
        st.markdown(f"**Source:** {source or '—'}")
        if series_id and series_id not in ("None", "nan", ""):
            st.markdown(f"**Series ID:** `{series_id}`")
        if fetched_at:
            st.markdown(f"**Fetched:** {fetched_at} ({fresh_age(fetched_at)})")


_EIA_LINKS = {
    "crude_oil": {
        "label": "EIA Crude Oil Production by PADD",
        "url": "https://www.eia.gov/dnav/pet/pet_crd_crpdn_adc_mbblpd_m.htm",
        "api": "https://api.eia.gov/v2/petroleum/crd/crpdn/data/",
    },
    "natural_gas": {
        "label": "EIA Natural Gas Marketed Production by State",
        "url": "https://www.eia.gov/dnav/ng/ng_prod_sum_a_EPG0_VGM_mmcf_m.htm",
        "api": "https://api.eia.gov/v2/natural-gas/prod/sum/data/",
    },
    "wti": {
        "label": "EIA WTI Crude Spot Price (Cushing, OK)",
        "url": "https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm",
        "api": "https://api.eia.gov/v2/petroleum/pri/spt/data/",
    },
    "browser": {
        "label": "EIA Open Data Browser — verify any number",
        "url": "https://www.eia.gov/opendata/browser/",
        "api": None,
    },
}


def render_data_sources_panel(meta: dict) -> None:
    """Full data-sources footer with clickable links to the actual EIA data pages."""
    with st.expander("ℹ️ Data Sources & Provenance — click to verify", expanded=False):
        if not meta:
            st.caption("No metadata available. Run the pipeline first.")
        else:
            st.markdown(
                f"**Primary source:** {meta.get('source', '—')}  ·  "
                f"**Version:** {meta.get('version', '—')}  ·  "
                f"**Fetched:** {meta.get('fetched_at', '—')[:19]} UTC "
                f"({fresh_age(meta.get('fetched_at'))})"
            )
            rc = meta.get("row_counts", {})
            if rc:
                cols = st.columns(len(rc))
                for (k, v), col in zip(rc.items(), cols):
                    col.metric(k.replace("_", " ").title(), f"{v:,}")
            dqs = meta.get("dqs", {})
            if dqs:
                st.markdown(
                    f"**Data Quality Score:** {dqs.get('score', 0):.1f}/100 "
                    f"(completeness={dqs.get('completeness', 0):.0%} · "
                    f"consistency={dqs.get('consistency', 0):.0%} · "
                    f"freshness={dqs.get('freshness', 0):.0%})"
                )
            dr = meta.get("date_range", {})
            if dr:
                st.caption(f"Coverage: {dr.get('start', '?')} → {dr.get('end', '?')}")

        st.divider()
        st.markdown("**📡 Verify the underlying data directly at EIA:**")

        for key, info in _EIA_LINKS.items():
            col_label, col_view, col_api = st.columns([0.50, 0.25, 0.25])
            col_label.markdown(f"**{info['label']}**")
            col_view.link_button("🔗 View data", info["url"], use_container_width=True)
            if info["api"]:
                col_api.link_button(
                    "⚙️ API endpoint", info["api"], use_container_width=True
                )

        st.caption(
            "Every number in this dashboard is computed from the EIA Open Data API v2. "
            "Click **View data** to open the corresponding EIA table in your browser "
            "and cross-check any figure independently."
        )
