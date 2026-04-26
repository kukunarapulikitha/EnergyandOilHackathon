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


def render_data_sources_panel(meta: dict) -> None:
    """Full data-sources footer — shown at the bottom of every tab."""
    with st.expander("ℹ️ Data Sources & Provenance", expanded=False):
        if not meta:
            st.caption("No metadata available. Run the pipeline first.")
            return

        st.markdown(
            f"**Primary source:** {meta.get('source', '—')}  ·  "
            f"**Version:** {meta.get('version', '—')}  ·  "
            f"**Fetched:** {meta.get('fetched_at', '—')[:19]} UTC "
            f"({fresh_age(meta.get('fetched_at'))})"
        )

        endpoints = meta.get("endpoints", [])
        if endpoints:
            st.markdown("**Endpoints:**")
            for ep in endpoints:
                st.markdown(f"- `{ep}`")

        rc = meta.get("row_counts", {})
        if rc:
            st.markdown("**Row counts:**")
            cols = st.columns(len(rc))
            for (k, v), col in zip(rc.items(), cols):
                col.metric(k.replace("_", " ").title(), f"{v:,}")

        dqs = meta.get("dqs", {})
        if dqs:
            st.markdown(
                f"**Data Quality Score:** {dqs.get('score', 0):.1f} / 100 "
                f"(completeness={dqs.get('completeness', 0):.0%} · "
                f"consistency={dqs.get('consistency', 0):.0%} · "
                f"freshness={dqs.get('freshness', 0):.0%})"
            )

        dr = meta.get("date_range", {})
        if dr:
            st.caption(f"Coverage: {dr.get('start', '?')} → {dr.get('end', '?')}")
