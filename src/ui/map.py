"""Geographic choropleth map — new Tier 1 requirement.

Uses Plotly choropleth against built-in US-states locations.
No Mapbox token needed (carto-positron basemap is free).

Two views:
    crude_oil    — each state colored by ITS PADD's production total
                   (PADDs are groups of states; EIA reports crude at PADD level)
    natural_gas  — top 5 producing states colored individually; others grayed

Three overlays (user-toggleable):
    production   — raw volume at selected year
    growth_pct   — YoY % change
    rpi          — relative performance index (0-100)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

_REF_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


@lru_cache(maxsize=1)
def _load_basins() -> dict:
    path = _REF_DIR / "basins.geojson"
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def _load_rig_counts() -> dict:
    path = _REF_DIR / "rig_counts.json"
    if not path.exists():
        return {"by_state": {}, "as_of": "unknown", "total_us": 0}
    return json.loads(path.read_text())

# ---------------------------------------------------------------- mappings

PADD_TO_STATES: dict[str, list[str]] = {
    "R10": [  # East Coast
        "CT", "DC", "DE", "FL", "GA", "MA", "MD", "ME", "NC", "NH",
        "NJ", "NY", "PA", "RI", "SC", "VA", "VT", "WV",
    ],
    "R20": [  # Midwest
        "IA", "IL", "IN", "KS", "KY", "MI", "MN", "MO", "ND", "NE",
        "OH", "OK", "SD", "TN", "WI",
    ],
    "R30": ["AL", "AR", "LA", "MS", "NM", "TX"],                        # Gulf Coast
    "R40": ["CO", "ID", "MT", "UT", "WY"],                              # Rocky Mountain
    "R50": ["AK", "AZ", "CA", "HI", "NV", "OR", "WA"],                  # West Coast
}

# Reverse map: state → PADD
STATE_TO_PADD: dict[str, str] = {
    state: padd for padd, states in PADD_TO_STATES.items() for state in states
}

# A single "representative" state per PADD — used to place a value label in a
# sensible geographic center for crude-oil maps (where every state in a PADD
# shares the same number).
PADD_LABEL_STATE: dict[str, str] = {
    "R10": "PA",  # Pennsylvania — central to East Coast PADD
    "R20": "IL",  # Illinois — central Midwest
    "R30": "TX",  # Texas — dominant producer in Gulf Coast
    "R40": "WY",  # Wyoming — central Rocky Mtn
    "R50": "CA",  # California — central West Coast
}

# State centroids (approximate lat/lon) — used for placing value labels on the map
_STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "AL": (32.8, -86.8), "AK": (64.2, -149.5), "AZ": (34.2, -111.7),
    "AR": (34.7, -92.4), "CA": (36.7, -119.6), "CO": (39.0, -105.5),
    "CT": (41.6, -72.7), "DC": (38.9, -77.0), "DE": (39.0, -75.5),
    "FL": (27.8, -81.7), "GA": (32.9, -83.4), "HI": (20.8, -156.3),
    "ID": (44.2, -114.5), "IL": (40.0, -89.2), "IN": (39.9, -86.3),
    "IA": (42.1, -93.5), "KS": (38.5, -98.4), "KY": (37.6, -85.0),
    "LA": (31.2, -92.3), "ME": (45.3, -69.4), "MD": (39.1, -76.7),
    "MA": (42.4, -71.8), "MI": (44.3, -85.4), "MN": (46.3, -94.3),
    "MS": (32.7, -89.7), "MO": (38.6, -92.5), "MT": (46.9, -110.5),
    "NE": (41.5, -99.8), "NV": (38.5, -117.1), "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5), "NM": (34.3, -106.1), "NY": (42.9, -75.5),
    "NC": (35.5, -79.0), "ND": (47.5, -100.5), "OH": (40.3, -82.8),
    "OK": (35.5, -97.5), "OR": (44.0, -120.5), "PA": (40.9, -77.7),
    "RI": (41.7, -71.5), "SC": (34.0, -81.0), "SD": (44.3, -99.4),
    "TN": (35.9, -86.7), "TX": (31.1, -100.1), "UT": (39.3, -111.6),
    "VT": (44.0, -72.7), "VA": (37.8, -78.2), "WA": (47.4, -120.7),
    "WV": (38.5, -80.9), "WI": (44.3, -89.6), "WY": (43.0, -107.5),
}

# Natural gas state codes in our Silver data → US state abbreviation
NG_STATE_TO_ABBR: dict[str, str] = {
    "STX": "TX",
    "SPA": "PA",
    "SLA": "LA",
    "SOK": "OK",
    "SWV": "WV",
}
ABBR_TO_NG_STATE: dict[str, str] = {v: k for k, v in NG_STATE_TO_ABBR.items()}


# ---------------------------------------------------------------- metric selection


def _metric_config(metric: str, fuel: str) -> dict:
    unit = "Mb/d" if fuel == "crude_oil" else "MMcf"
    # Fixed colorscale ranges so that single-region selections don't collapse
    # the legend to a useless 0.4-unit window. None = auto-scale.
    return {
        "production": {
            "label": f"Production ({unit})",
            "scale": "Blues",
            "fmt": ",.0f",
            "zmin": None,
            "zmax": None,
            "zmid": None,
        },
        "growth_pct": {
            "label": "YoY Growth (%)",
            "scale": "RdYlGn",
            "fmt": "+.1f",
            "zmin": -10,
            "zmax": 10,
            "zmid": 0,   # red = decline, green = growth, yellow-white = flat
        },
        "relative_performance_index": {
            "label": "Relative Performance (0-100)",
            "scale": "RdYlGn",
            "fmt": ".0f",
            "zmin": 0,
            "zmax": 100,
            "zmid": 50,
        },
    }[metric]


def _project_for_region(
    region_id: str,
    fuel: str,
    target_year: int,
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
) -> float | None:
    """Actual if available for (region, fuel, year); otherwise forecast."""
    row = actuals[
        (actuals["region_id"] == region_id)
        & (actuals["fuel_type"] == fuel)
        & (actuals["year"] == target_year)
    ]
    if not row.empty:
        return float(row["production"].iloc[0])

    frow = forecasts[
        (forecasts["region_id"] == region_id) & (forecasts["fuel_type"] == fuel)
    ]
    if frow.empty:
        return None
    r = frow.iloc[0]
    return float(r["slope"] * target_year + r["intercept"])


# ---------------------------------------------------------------- public builder


def production_map(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected_year: int,
    fuel: str,
    metric: str = "production",
    selected_region_ids: list[str] | None = None,
    focused_region_id: str | None = None,
) -> go.Figure:
    """Choropleth of US states colored by the selected metric.

    Parameters
    ----------
    actuals, forecasts : Gold DataFrames
    selected_year      : year shown on the map; actuals if in range, otherwise projection
    fuel               : "crude_oil" | "natural_gas"
    metric             : "production" | "growth_pct" | "relative_performance_index"
    selected_region_ids: if given, only color these regions. Everything else
                         renders as an uncolored state outline (same blank look
                         as states outside the dataset). None = show all.
    """
    cfg = _metric_config(metric, fuel)
    is_crude = fuel == "crude_oil"

    rows: list[dict] = []

    def _overlay_value(region_id: str, metric_name: str) -> float | None:
        """Return the metric at selected_year if available, else the most recent
        actual value for that region (growth_pct and RPI are only defined on
        years where we have actual data)."""
        if metric_name == "production":
            return _project_for_region(region_id, fuel, selected_year, actuals, forecasts)

        rows_q = actuals[
            (actuals["region_id"] == region_id) & (actuals["fuel_type"] == fuel)
        ]
        if rows_q.empty:
            return None
        exact = rows_q[rows_q["year"] == selected_year]
        if not exact.empty and pd.notna(exact[metric_name].iloc[0]):
            return float(exact[metric_name].iloc[0])
        latest = rows_q.sort_values("year").dropna(subset=[metric_name])
        if latest.empty:
            return None
        return float(latest.iloc[-1][metric_name])

    def _kpi_bundle(region_id: str) -> dict:
        """Collect every KPI we want visible in the rich hover tooltip."""
        prod = _overlay_value(region_id, "production")
        growth = _overlay_value(region_id, "growth_pct")
        rpi = _overlay_value(region_id, "relative_performance_index")
        fr = forecasts[
            (forecasts["region_id"] == region_id) & (forecasts["fuel_type"] == fuel)
        ]
        r2 = float(fr["r_squared"].iloc[0]) if not fr.empty else None
        score = float(fr["investment_score"].iloc[0]) if not fr.empty else None
        return {
            "production": prod,
            "growth": growth,
            "rpi": rpi,
            "r_squared": r2,
            "investment_score": score,
        }

    allowed = set(selected_region_ids) if selected_region_ids is not None else None

    if is_crude:
        for padd_id, states in PADD_TO_STATES.items():
            if allowed is not None and padd_id not in allowed:
                continue
            val = _overlay_value(padd_id, metric)
            if val is None:
                continue
            kpi = _kpi_bundle(padd_id)
            for state in states:
                rows.append({
                    "state": state,
                    "region_id": padd_id,
                    "region_name": f"PADD {padd_id[1]} ({_padd_nickname(padd_id)})",
                    "value": val,
                    **kpi,
                })
    else:
        for state_abbr, ng_region_id in ABBR_TO_NG_STATE.items():
            if allowed is not None and ng_region_id not in allowed:
                continue
            val = _overlay_value(ng_region_id, metric)
            if val is None:
                continue
            kpi = _kpi_bundle(ng_region_id)
            rows.append({
                "state": state_abbr,
                "region_id": ng_region_id,
                "region_name": _ng_state_name(ng_region_id),
                "value": val,
                **kpi,
            })

    df = (
        pd.DataFrame(rows).dropna(subset=["value"])
        if rows
        else pd.DataFrame(columns=["state", "region_id", "region_name", "value"])
    )

    unit = "Mb/d" if is_crude else "MMcf"
    # Rich hover customdata: [region_id, region_name, production, growth, r², score, rpi]
    cd = df[[
        "region_id", "region_name", "production",
        "growth", "r_squared", "investment_score", "rpi",
    ]].values
    hovertemplate = (
        "<b>%{customdata[1]}</b><br>"
        f"Production: %{{customdata[2]:,.0f}} {unit}<br>"
        "YoY growth: %{customdata[3]:+.1f}%<br>"
        "Forecast R²: %{customdata[4]:.2f}<br>"
        "Investment score: %{customdata[5]:.0f}/100<br>"
        "Relative perf: %{customdata[6]:.0f}/100"
        "<extra></extra>"
    )

    choro_kwargs = dict(
        locations=df["state"],
        z=df["value"],
        locationmode="USA-states",
        colorscale=cfg["scale"],
        colorbar=dict(title=dict(text=cfg["label"], side="right")),
        customdata=cd,
        hovertemplate=hovertemplate,
        marker_line_color="white",
        marker_line_width=0.8,
    )
    if cfg["zmin"] is not None:
        choro_kwargs["zmin"] = cfg["zmin"]
    if cfg["zmax"] is not None:
        choro_kwargs["zmax"] = cfg["zmax"]
    if cfg["zmid"] is not None:
        choro_kwargs["zmid"] = cfg["zmid"]
    fig = go.Figure(go.Choropleth(**choro_kwargs))

    # Light-gray background state labels — helps orient users without relying
    # on hover alone. Subtle color so they don't compete with value labels.
    _STATES_TO_LABEL = [
        "TX", "CA", "NY", "FL", "IL", "PA", "OH", "GA", "NC",
        "MI", "NJ", "VA", "WA", "MA", "CO", "OK", "NM", "LA",
        "KY", "OR", "WY", "WV", "AL", "ND",
    ]
    lab_lats, lab_lons, lab_labels = [], [], []
    for s in _STATES_TO_LABEL:
        c = _STATE_CENTROIDS.get(s)
        if c:
            lab_lats.append(c[0] + 0.8)
            lab_lons.append(c[1])
            lab_labels.append(s)
    fig.add_trace(go.Scattergeo(
        lat=lab_lats, lon=lab_lons,
        text=lab_labels,
        mode="text",
        textfont=dict(size=9, color="rgba(180,180,190,0.55)", family="Arial"),
        showlegend=False, hoverinfo="skip",
    ))

    # Value labels on the colored states — makes numerical differences across
    # years visible at a glance rather than hidden behind hover tooltips.
    if not df.empty:
        if is_crude:
            # One label per PADD at a representative state
            label_map = (
                df.drop_duplicates(subset=["region_id"])
                .assign(label_state=lambda d: d["region_id"].map(PADD_LABEL_STATE))
            )
            lats = [_STATE_CENTROIDS.get(s, (None, None))[0] for s in label_map["label_state"]]
            lons = [_STATE_CENTROIDS.get(s, (None, None))[1] for s in label_map["label_state"]]
            values = label_map["value"]
        else:
            lats = [_STATE_CENTROIDS.get(s, (None, None))[0] for s in df["state"]]
            lons = [_STATE_CENTROIDS.get(s, (None, None))[1] for s in df["state"]]
            values = df["value"]

        labels = [format(v, cfg["fmt"]) for v in values]
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons,
            text=labels,
            mode="text",
            textfont=dict(size=12, color="#F0F0F0", family="Arial Black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Focus highlight — bright outline around the currently-focused region's states
    if focused_region_id and not df.empty:
        focus_rows = df[df["region_id"] == focused_region_id]
        if not focus_rows.empty:
            fig.add_trace(go.Choropleth(
                locations=focus_rows["state"],
                z=[1] * len(focus_rows),
                locationmode="USA-states",
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                marker_line_color="#FF3D00",
                marker_line_width=3.5,
                hoverinfo="skip",
                name="Focus",
                showlegend=False,
            ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="#0E1117",
            showland=True,
            landcolor="#1A1F28",
            showsubunits=True,
            subunitcolor="#3A4555",
            showcountries=False,
            bgcolor="rgba(0,0,0,0)",
        ),
        title=(
            f"U.S. {fuel.replace('_', ' ').title()} — {cfg['label']} "
            f"at {selected_year}"
        ),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------- click resolution


def regions_to_states(
    region_ids: list[str] | None, fuel: str
) -> set[str] | None:
    """Resolve a selected-region list to the set of U.S. state abbreviations
    they cover. Used so that Basin + Rig overlays can honor the same region
    filter as the base choropleth.

    None input = no filter (show all states).
    Empty list   = empty set (filter everything out).
    """
    if region_ids is None:
        return None
    if not region_ids:
        return set()
    states: set[str] = set()
    if fuel == "crude_oil":
        for rid in region_ids:
            states.update(PADD_TO_STATES.get(rid, []))
    else:
        for rid in region_ids:
            abbr = NG_STATE_TO_ABBR.get(rid)
            if abbr:
                states.add(abbr)
    return states


def resolve_clicked_region(clicked_state: str, fuel: str) -> str | None:
    """Translate a US state abbreviation from a map click to a region_id.

    Crude: return the PADD containing the state.
    Gas:   return the ng region_id only if the state is one of the top-5; else None.
    """
    if not clicked_state:
        return None
    state = clicked_state.upper()
    if fuel == "crude_oil":
        return STATE_TO_PADD.get(state)
    return ABBR_TO_NG_STATE.get(state)


# ---------------------------------------------------------------- helpers


_PADD_NICKS = {
    "R10": "East Coast",
    "R20": "Midwest",
    "R30": "Gulf Coast",
    "R40": "Rocky Mountain",
    "R50": "West Coast",
}


def _padd_nickname(padd_id: str) -> str:
    return _PADD_NICKS.get(padd_id, padd_id)


def _ng_state_name(ng_id: str) -> str:
    return {
        "STX": "Texas", "SPA": "Pennsylvania", "SLA": "Louisiana",
        "SOK": "Oklahoma", "SWV": "West Virginia",
    }.get(ng_id, ng_id)


# ============================================================ overlays


def add_basin_layer(
    fig: go.Figure,
    allowed_states: set[str] | None = None,
) -> go.Figure:
    """Overlay approximate EIA DPR basin polygons on top of the state choropleth.

    Each basin is a labeled polygon. All basins share a single color so the
    state choropleth underneath remains the primary visual channel.

    If `allowed_states` is provided, only basins whose `properties.states`
    intersects that set are rendered. Passing an empty set hides all basins.
    """
    geo = _load_basins()
    features = geo.get("features", [])
    if allowed_states is not None:
        features = [
            f for f in features
            if set(f.get("properties", {}).get("states", [])) & allowed_states
        ]
    if not features:
        return fig
    # Narrow the feature-collection payload too so Plotly doesn't receive
    # polygons for basins we aren't naming below.
    geo = {"type": "FeatureCollection", "features": features}

    names = [f["properties"]["name"] for f in features]
    # Assign each basin a z value = index so it picks up a distinct color from
    # a qualitative colorscale (nothing to do with production magnitude).
    z_values = list(range(len(names)))
    # ColorBrewer Set3 palette spelled out (Plotly doesn't ship a "Set3" name)
    qual_palette = [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
        "#80b1d3", "#fdb462", "#b3de69",
    ]

    fig.add_trace(go.Choropleth(
        geojson=geo,
        locations=names,
        z=z_values,
        featureidkey="properties.name",
        colorscale=qual_palette,
        showscale=False,
        marker_line_color="#333",
        marker_line_width=2.0,
        marker_opacity=0.45,
        customdata=[
            [
                f["properties"]["name"],
                ", ".join(f["properties"].get("states", [])),
                ", ".join(f["properties"].get("primary_products", [])),
                f["properties"].get("note", ""),
            ]
            for f in features
        ],
        hovertemplate=(
            "<b>%{customdata[0]} Basin</b><br>"
            "States: %{customdata[1]}<br>"
            "Products: %{customdata[2]}<br>"
            "%{customdata[3]}"
            "<extra></extra>"
        ),
        name="Basins",
    ))

    # Add basin name labels at approximate centroids
    centroid_lat, centroid_lon, labels = [], [], []
    for f in features:
        coords = f["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        centroid_lat.append(sum(lats) / len(lats))
        centroid_lon.append(sum(lons) / len(lons))
        labels.append(f["properties"]["name"])
    fig.add_trace(go.Scattergeo(
        lat=centroid_lat,
        lon=centroid_lon,
        text=labels,
        mode="text",
        textfont=dict(size=11, color="#111", family="Arial Black"),
        showlegend=False,
        hoverinfo="skip",
    ))
    return fig


def add_rig_layer(
    fig: go.Figure,
    allowed_states: set[str] | None = None,
) -> go.Figure:
    """Overlay active drilling-rig markers at state centroids.

    If `allowed_states` is provided, only rigs for those states are shown.
    Passing an empty set hides all rigs.
    """
    data = _load_rig_counts()
    by_state = data.get("by_state", {})
    if not by_state:
        return fig

    lats, lons, sizes, hovers, names = [], [], [], [], []
    for st_code, entry in by_state.items():
        if allowed_states is not None and st_code not in allowed_states:
            continue
        rigs = entry.get("rigs", 0)
        if rigs <= 0:
            continue
        # Offset markers slightly north-east of the state centroid so they
        # don't overlap the state's value label (which sits on the centroid).
        lats.append(entry["lat"] + 1.4)
        lons.append(entry["lon"] + 0.8)
        sizes.append(rigs)
        hovers.append(
            f"<b>{entry['name']}</b><br>"
            f"Active rigs: <b>{rigs}</b><br>"
            f"<i>Snapshot: {data.get('as_of', 'unknown')}</i>"
        )
        names.append(st_code)

    if not sizes:
        return fig

    # Scale marker size so Texas isn't overwhelming — sqrt keeps big states
    # visible but small states readable.
    max_rigs = max(sizes) if sizes else 1
    sized = [max(8, 8 + (r / max_rigs) ** 0.5 * 30) for r in sizes]

    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(
            size=sized,
            color="#000000",
            opacity=0.75,
            line=dict(color="#ffffff", width=1.5),
        ),
        text=[str(r) for r in sizes],
        textfont=dict(size=10, color="white"),
        textposition="middle center",
        hovertext=hovers,
        hoverinfo="text",
        showlegend=False,
        name="Active rigs",
    ))
    return fig


def production_bubble_map(
    actuals: pd.DataFrame,
    forecasts: pd.DataFrame,
    selected_year: int,
    fuel: str,
    selected_region_ids: list[str] | None = None,
    focused_region_id: str | None = None,
    color_metric: str = "growth_pct",
) -> go.Figure:
    """Bubble map: marker size ∝ production, marker color ∝ chosen metric.

    color_metric: "production" (Blues), "growth_pct" (RdYlGn −10..10),
                  or "relative_performance_index" (RdYlGn 0..100)
    """
    color_cfg = {
        "production": {"scale": "Blues", "cmin": None, "cmax": None, "cmid": None,
                        "label": "Production"},
        "growth_pct": {"scale": "RdYlGn", "cmin": -10, "cmax": 10, "cmid": 0,
                        "label": "YoY Growth (%)"},
        "relative_performance_index": {"scale": "RdYlGn", "cmin": 0, "cmax": 100, "cmid": 50,
                        "label": "Relative Perf."},
    }[color_metric]
    is_crude = fuel == "crude_oil"
    allowed = set(selected_region_ids) if selected_region_ids is not None else None

    rows: list[dict] = []
    if is_crude:
        for padd_id, states in PADD_TO_STATES.items():
            if allowed is not None and padd_id not in allowed:
                continue
            prod = _project_for_region(padd_id, fuel, selected_year, actuals, forecasts)
            if prod is None:
                continue
            label_state = PADD_LABEL_STATE.get(padd_id)
            lat, lon = _STATE_CENTROIDS.get(label_state, (None, None))
            if lat is None:
                continue
            act_rows = actuals[
                (actuals["region_id"] == padd_id) & (actuals["fuel_type"] == fuel)
            ]
            growth = _latest_notna(act_rows, "growth_pct", selected_year)
            rpi = _latest_notna(act_rows, "relative_performance_index", selected_year)
            fr = forecasts[
                (forecasts["region_id"] == padd_id) & (forecasts["fuel_type"] == fuel)
            ]
            r2 = float(fr["r_squared"].iloc[0]) if not fr.empty else None
            score = float(fr["investment_score"].iloc[0]) if not fr.empty else None
            rows.append({
                "region_id": padd_id,
                "region_name": f"PADD {padd_id[1]} ({_padd_nickname(padd_id)})",
                "lat": lat, "lon": lon,
                "production": prod,
                "growth": growth if growth is not None else 0.0,
                "r_squared": r2,
                "investment_score": score,
                "rpi": rpi,
            })
    else:
        for state_abbr, ng_id in ABBR_TO_NG_STATE.items():
            if allowed is not None and ng_id not in allowed:
                continue
            prod = _project_for_region(ng_id, fuel, selected_year, actuals, forecasts)
            if prod is None:
                continue
            lat, lon = _STATE_CENTROIDS.get(state_abbr, (None, None))
            if lat is None:
                continue
            act_rows = actuals[
                (actuals["region_id"] == ng_id) & (actuals["fuel_type"] == fuel)
            ]
            growth = _latest_notna(act_rows, "growth_pct", selected_year)
            rpi = _latest_notna(act_rows, "relative_performance_index", selected_year)
            fr = forecasts[
                (forecasts["region_id"] == ng_id) & (forecasts["fuel_type"] == fuel)
            ]
            r2 = float(fr["r_squared"].iloc[0]) if not fr.empty else None
            score = float(fr["investment_score"].iloc[0]) if not fr.empty else None
            rows.append({
                "region_id": ng_id,
                "region_name": _ng_state_name(ng_id),
                "lat": lat, "lon": lon,
                "production": prod,
                "growth": growth if growth is not None else 0.0,
                "r_squared": r2,
                "investment_score": score,
                "rpi": rpi,
            })

    df = pd.DataFrame(rows)

    fig = go.Figure()

    if not df.empty:
        max_prod = float(df["production"].max())
        # sqrt scaling so visual AREA (~size²) tracks production linearly
        sizes = [max(18.0, ((p / max_prod) ** 0.5) * 70) for p in df["production"]]

        unit = "Mb/d" if is_crude else "MMcf"
        focus_mask = df["region_id"] == (focused_region_id or "")
        border_widths = [4 if m else 1.5 for m in focus_mask]
        border_colors = ["#FF3D00" if m else "white" for m in focus_mask]

        # Pick color values by metric
        color_values = {
            "production": df["production"],
            "growth_pct": df["growth"],
            "relative_performance_index": df["rpi"],
        }[color_metric]

        marker = dict(
            size=sizes,
            color=color_values,
            colorscale=color_cfg["scale"],
            line=dict(color=border_colors, width=border_widths),
            colorbar=dict(title=dict(text=color_cfg["label"], side="right")),
            sizemode="diameter",
        )
        if color_cfg["cmin"] is not None:
            marker["cmin"] = color_cfg["cmin"]
        if color_cfg["cmax"] is not None:
            marker["cmax"] = color_cfg["cmax"]
        if color_cfg["cmid"] is not None:
            marker["cmid"] = color_cfg["cmid"]

        fig.add_trace(go.Scattergeo(
            lat=df["lat"], lon=df["lon"],
            mode="markers+text",
            text=[f"{p:,.0f}" for p in df["production"]],
            textposition="bottom center",
            textfont=dict(size=11, color="#F0F0F0", family="Arial Black"),
            marker=marker,
            customdata=df[[
                "region_id", "region_name", "production",
                "growth", "r_squared", "investment_score", "rpi",
            ]].values,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                f"Production: %{{customdata[2]:,.0f}} {unit}<br>"
                "YoY growth: %{customdata[3]:+.1f}%<br>"
                "Forecast R²: %{customdata[4]:.2f}<br>"
                "Investment score: %{customdata[5]:.0f}/100<br>"
                "Relative perf: %{customdata[6]:.0f}/100"
                "<extra></extra>"
            ),
            name="Regions",
            showlegend=False,
        ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            projection=go.layout.geo.Projection(type="albers usa"),
            showlakes=True,
            lakecolor="rgba(200,220,240,0.4)",
            showland=True,
            landcolor="#1A1F28",
            showsubunits=True,
            subunitcolor="#3A4555",
            bgcolor="rgba(0,0,0,0)",
        ),
        title=(
            f"U.S. {fuel.replace('_', ' ').title()} — "
            f"Bubble (size = production, color = YoY growth) at {selected_year}"
        ),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _latest_notna(df: pd.DataFrame, col: str, target_year: int) -> float | None:
    if df.empty or col not in df.columns:
        return None
    exact = df[df["year"] == target_year]
    if not exact.empty and pd.notna(exact[col].iloc[0]):
        return float(exact[col].iloc[0])
    latest = df.sort_values("year").dropna(subset=[col])
    if latest.empty:
        return None
    return float(latest.iloc[-1][col])


def rig_count_summary() -> str:
    """Short caption string describing the rig-count snapshot."""
    d = _load_rig_counts()
    return (
        f"Active rigs: **{d.get('total_us', '?')}** U.S. total · "
        f"snapshot {d.get('as_of', '?')} · "
        f"source: {d.get('source', 'Baker Hughes')}"
    )
