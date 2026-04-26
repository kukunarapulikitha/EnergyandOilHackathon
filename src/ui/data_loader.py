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
