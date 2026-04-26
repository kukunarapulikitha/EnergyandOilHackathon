"""Silver layer — cleaned, schema-enforced production data.

Transforms raw Bronze payloads (EIAResponse) into DataFrames conforming
to SILVER_SCHEMA. One row per (region, date, fuel_type).

Cleaning steps (in order):
    1. Drop rows with null dates or null production values
    2. Standardize region identifiers + human-readable names
    3. Normalize timestamps to month-start
    4. Preserve units explicitly (no silent conversion)
    5. Deduplicate on (region_id, date, fuel_type) — keep last
    6. Assert conformance to schema contract
"""

from __future__ import annotations

import pandas as pd

from .eia_client import NG_STATES, PADD_REGIONS, EIAResponse
from .schema import SILVER_COLUMNS


def clean_crude_production(response: EIAResponse) -> pd.DataFrame:
    """Convert raw EIA crude oil payload → Silver DataFrame."""
    if not response.data:
        return _empty_frame()

    df = pd.DataFrame(response.data)

    # EIA returns two series per period (MBBL and MBBL/D). Keep daily-rate only
    # to avoid mixing units in the `production` column.
    if "units" in df.columns:
        df = df[df["units"] == "MBBL/D"].copy()

    return _assemble(
        df,
        fuel_type="crude_oil",
        unit_default="MBBL/D",
        region_map=PADD_REGIONS,
        response=response,
    )


def clean_natural_gas_production(response: EIAResponse) -> pd.DataFrame:
    """Convert raw EIA natural gas payload → Silver DataFrame."""
    if not response.data:
        return _empty_frame()

    df = pd.DataFrame(response.data)

    return _assemble(
        df,
        fuel_type="natural_gas",
        unit_default="MMCF",
        region_map=NG_STATES,
        response=response,
    )


def clean_wti_prices(response: EIAResponse) -> pd.DataFrame:
    """Convert raw WTI spot-price payload → daily price DataFrame.

    Note: WTI does NOT conform to SILVER_SCHEMA (no region, no fuel_type —
    it's a price series, not a production series). Separate helper.
    Schema: date, price_usd_per_bbl, source, fetched_at
    """
    if not response.data:
        return pd.DataFrame(
            columns=["date", "price_usd_per_bbl", "source", "fetched_at"]
        )

    df = pd.DataFrame(response.data)
    df["date"] = pd.to_datetime(df["period"], errors="coerce")
    df["price_usd_per_bbl"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "price_usd_per_bbl"])
    df["source"] = f"{response.source} / {response.endpoint}"
    df["fetched_at"] = response.fetched_at
    return (
        df[["date", "price_usd_per_bbl", "source", "fetched_at"]]
        .sort_values("date")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- internals


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SILVER_COLUMNS)


def _assemble(
    df: pd.DataFrame,
    fuel_type: str,
    unit_default: str,
    region_map: dict[str, str],
    response: EIAResponse,
) -> pd.DataFrame:
    """Shared pipeline used for both crude and natural gas sources."""
    df = df.copy()

    df["date"] = pd.to_datetime(df["period"], format="%Y-%m", errors="coerce")
    df["production"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    df = df.dropna(subset=["date", "production"])

    df["date"] = (df["date"] + pd.offsets.MonthBegin(0)).astype("datetime64[ns]")
    df["year"] = df["date"].dt.year.astype("int64")

    df["region_id"] = df["duoarea"].astype(str)
    df["region_name"] = df["region_id"].map(region_map).fillna(df["region_id"])

    if "units" in df.columns:
        df["unit"] = df["units"].astype(str)
    else:
        df["unit"] = unit_default

    df["fuel_type"] = fuel_type
    df["source"] = f"{response.source} / {response.endpoint}"
    df["series_id"] = (
        df.get("series", pd.Series([None] * len(df))).astype(str)
    )
    fetched = pd.Timestamp(response.fetched_at)
    if fetched.tz is None:
        fetched = fetched.tz_localize("UTC")
    df["fetched_at"] = pd.Series(
        [fetched] * len(df), dtype="datetime64[ns, UTC]", index=df.index
    )

    df = df[df["production"] >= 0]

    df = df.drop_duplicates(
        subset=["region_id", "date", "fuel_type"], keep="last"
    )

    df = df.sort_values(["region_id", "date"]).reset_index(drop=True)

    out = df[SILVER_COLUMNS].copy()
    for col in ("region_id", "region_name", "unit", "fuel_type", "source", "series_id"):
        out[col] = out[col].astype("string")
    return out


def aggregate_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Average monthly production → one row per (region, year, fuel_type)."""
    if monthly.empty:
        return monthly
    return (
        monthly.groupby(
            ["region_id", "region_name", "year", "fuel_type", "unit", "source"],
            as_index=False,
            dropna=False,
        )["production"]
        .mean()
        .sort_values(["fuel_type", "region_id", "year"])
        .reset_index(drop=True)
    )
