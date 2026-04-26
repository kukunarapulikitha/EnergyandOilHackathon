"""Region-level horizontal well defaults.

These are demo-grade typical-new-well averages, hand-tuned from published
EIA / operator disclosures. Real wells vary widely; the user is expected to
tune values in the Streamlit input panel. The point is to give a sensible
starting point per region so the demo flow (click region → see realistic
economics) feels grounded.
"""

from __future__ import annotations

REGION_DEFAULTS: dict[str, dict] = {
    # ------------------------------- Crude PADDs -------------------------------
    "R10": {
        "label": "PADD 1 (East Coast) — generic conventional",
        "fuel": "crude_oil", "ip": 200.0, "Di": 0.30, "b": 0.5,
        "capex": 4_000_000, "loe": 18.0, "price": 78.0, "severance_pct": 0.05,
        "well_life_months": 240,
    },
    "R20": {
        "label": "PADD 2 (Midwest) — Mississippi Lime / generic",
        "fuel": "crude_oil", "ip": 600.0, "Di": 0.55, "b": 0.9,
        "capex": 6_500_000, "loe": 12.0, "price": 78.0, "severance_pct": 0.07,
        "well_life_months": 240,
    },
    "R30": {
        "label": "PADD 3 (Gulf Coast) — Permian / Eagle Ford",
        "fuel": "crude_oil", "ip": 950.0, "Di": 0.62, "b": 1.1,
        "capex": 8_500_000, "loe": 10.0, "price": 78.0, "severance_pct": 0.075,
        "well_life_months": 360,
    },
    "R40": {
        "label": "PADD 4 (Rocky Mountain) — Bakken / Niobrara",
        "fuel": "crude_oil", "ip": 1100.0, "Di": 0.78, "b": 1.0,
        "capex": 8_000_000, "loe": 13.0, "price": 78.0, "severance_pct": 0.115,
        "well_life_months": 240,
    },
    "R50": {
        "label": "PADD 5 (West Coast) — California heavy / generic",
        "fuel": "crude_oil", "ip": 250.0, "Di": 0.20, "b": 0.3,
        "capex": 3_500_000, "loe": 22.0, "price": 78.0, "severance_pct": 0.085,
        "well_life_months": 360,
    },
    # ------------------------------- Natural gas -------------------------------
    "STX": {
        "label": "Texas — Haynesville / Eagle Ford gas",
        "fuel": "natural_gas", "ip": 18.0, "Di": 0.70, "b": 1.0,
        "capex": 9_500_000, "loe": 1.10, "price": 3.40, "severance_pct": 0.075,
        "well_life_months": 240,
    },
    "SPA": {
        "label": "Pennsylvania — Marcellus",
        "fuel": "natural_gas", "ip": 16.0, "Di": 0.55, "b": 1.2,
        "capex": 7_000_000, "loe": 0.95, "price": 3.40, "severance_pct": 0.05,
        "well_life_months": 360,
    },
    "SLA": {
        "label": "Louisiana — Haynesville",
        "fuel": "natural_gas", "ip": 22.0, "Di": 0.78, "b": 1.0,
        "capex": 10_500_000, "loe": 1.20, "price": 3.40, "severance_pct": 0.10,
        "well_life_months": 240,
    },
    "SOK": {
        "label": "Oklahoma — Anadarko / STACK",
        "fuel": "natural_gas", "ip": 12.0, "Di": 0.60, "b": 1.0,
        "capex": 7_500_000, "loe": 1.15, "price": 3.40, "severance_pct": 0.07,
        "well_life_months": 240,
    },
    "SWV": {
        "label": "West Virginia — Marcellus / Utica",
        "fuel": "natural_gas", "ip": 14.0, "Di": 0.58, "b": 1.2,
        "capex": 7_500_000, "loe": 1.00, "price": 3.40, "severance_pct": 0.05,
        "well_life_months": 360,
    },
}


GENERIC_DEFAULT: dict = {
    "label": "Generic horizontal well (no region selected)",
    "fuel": "crude_oil", "ip": 800.0, "Di": 0.60, "b": 1.0,
    "capex": 7_500_000, "loe": 12.0, "price": 78.0, "severance_pct": 0.075,
    "well_life_months": 240,
}


def get_defaults(region_id: str | None) -> dict:
    """Look up well defaults for a region_id; falls back to GENERIC_DEFAULT."""
    if region_id and region_id in REGION_DEFAULTS:
        return {**REGION_DEFAULTS[region_id], "region_id": region_id}
    return {**GENERIC_DEFAULT, "region_id": None}


def list_regions(fuel: str | None = None) -> list[tuple[str, str]]:
    """Return [(region_id, label), ...] for the preset selector.
    Optionally filter by fuel."""
    items = []
    for rid, d in REGION_DEFAULTS.items():
        if fuel is None or d["fuel"] == fuel:
            items.append((rid, d["label"]))
    return items
