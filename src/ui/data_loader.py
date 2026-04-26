"""Streamlit data-loading helpers with caching.

Reads from `data/gold/` and `data/silver/`. All heavy I/O is wrapped in
`@st.cache_data(ttl=3600)` to keep UI interaction snappy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold" / "csv"
SILVER = ROOT / "data" / "silver"
META = ROOT / "data" / "metadata.json"


@st.cache_data(ttl=3600)
def load_actuals() -> pd.DataFrame:
    path = GOLD / "regional_actuals.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_forecasts() -> pd.DataFrame:
    path = GOLD / "region_forecasts.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_annual_silver() -> pd.DataFrame:
    """Silver annual frame — used for live re-fitting when slider moves backward."""
    path = SILVER / "production_annual.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_metadata() -> dict:
    if not META.exists():
        return {}
    return json.loads(META.read_text())


def pipeline_ready() -> bool:
    return (GOLD / "regional_actuals.csv").exists() and (
        GOLD / "region_forecasts.csv"
    ).exists()


# ------------------------------------------------------------------ Excel import

_REQUIRED_UPLOAD_COLS = {"region_id", "fuel_type", "year", "production"}

_OPTIONAL_FILL: dict[str, object] = {
    "region_name": None,
    "growth_pct": None,
    "decline_rate_pct": None,
    "relative_performance_index": None,
    "revenue_potential_usd": None,
    "volatility_pct": None,
    "investment_score": None,
    "source": "uploaded",
    "fetched_at": None,
}


def parse_excel_upload(file_bytes) -> tuple[pd.DataFrame | None, str]:
    """Parse a user-uploaded Excel (.xlsx) file into a production DataFrame.

    The file is expected to contain a sheet named "Production" (matching the
    export template) or any sheet with at minimum these columns:
      region_id | fuel_type | year | production

    Returns (DataFrame, error_message). On success the error is "".
    On failure the DataFrame is None and error describes what went wrong.
    """
    try:
        xl = pd.ExcelFile(file_bytes)
    except Exception as exc:
        return None, f"Could not open file: {exc}"

    # Prefer the "Production" sheet; fall back to first available sheet
    sheet = "Production" if "Production" in xl.sheet_names else xl.sheet_names[0]
    try:
        df = xl.parse(sheet)
    except Exception as exc:
        return None, f"Could not parse sheet '{sheet}': {exc}"

    # Normalize headers: lowercase, strip spaces, replace spaces with underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    missing = _REQUIRED_UPLOAD_COLS - set(df.columns)
    if missing:
        return None, (
            f"Missing required columns: {sorted(missing)}. "
            f"Expected at minimum: {sorted(_REQUIRED_UPLOAD_COLS)}."
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["production"] = pd.to_numeric(df["production"], errors="coerce")
    df = df.dropna(subset=["year", "production"]).copy()
    if df.empty:
        return None, "No valid rows after parsing (check year and production columns)."

    df["year"] = df["year"].astype(int)

    # Fill optional columns with defaults where absent
    for col, default in _OPTIONAL_FILL.items():
        if col not in df.columns:
            df[col] = default
    df["source"] = df["source"].fillna("uploaded")
    if "region_name" not in df.columns or df["region_name"].isna().all():
        df["region_name"] = df["region_id"]

    return df, ""


def merge_uploaded_actuals(
    gold: pd.DataFrame,
    uploaded: pd.DataFrame,
) -> pd.DataFrame:
    """Overlay uploaded rows onto the Gold-layer actuals.

    For any (region_id, fuel_type, year) present in both, the uploaded row
    wins. Rows unique to either side are kept as-is.
    """
    key = ["region_id", "fuel_type", "year"]
    combined = pd.concat([gold, uploaded], ignore_index=True)
    # drop_duplicates keeps the *last* occurrence — put uploaded last so it wins
    combined = combined.drop_duplicates(subset=key, keep="last")
    return combined.reset_index(drop=True)
