"""Interactive data exploration — 6 canned reports over Silver + Gold.

Run:  python scripts/explore_data.py

Uses pandas directly (no DuckDB dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold" / "csv"


def _hdr(title: str) -> None:
    print("\n" + "─" * 68)
    print(f"  {title}")
    print("─" * 68)


def main() -> int:
    monthly_path = SILVER / "production_monthly.csv"
    if not monthly_path.exists():
        print("No Silver data found. Run scripts/verify_pipeline.py first.")
        return 1

    monthly = pd.read_csv(monthly_path, parse_dates=["date"])
    actuals_path = GOLD / "regional_actuals.csv"
    forecasts_path = GOLD / "region_forecasts.csv"
    actuals = pd.read_csv(actuals_path) if actuals_path.exists() else pd.DataFrame()
    forecasts = pd.read_csv(forecasts_path) if forecasts_path.exists() else pd.DataFrame()

    # 1. Row counts per (region, fuel_type)
    _hdr("1. Row counts per (region_name, fuel_type)")
    counts = (
        monthly.groupby(["region_name", "fuel_type"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["fuel_type", "rows"], ascending=[True, False])
    )
    print(counts.to_string(index=False))

    # 2. Date-range coverage per region
    _hdr("2. Date range per region")
    coverage = (
        monthly.groupby(["region_name", "fuel_type"])["date"]
        .agg(["min", "max", "count"])
        .reset_index()
    )
    print(coverage.to_string(index=False))

    # 3. Duplicate detection
    _hdr("3. Duplicate (region_id, date, fuel_type)")
    dup_mask = monthly.duplicated(
        subset=["region_id", "date", "fuel_type"], keep=False
    )
    dup_count = int(dup_mask.sum())
    print(f"duplicates found: {dup_count}")
    if dup_count > 0:
        print(monthly[dup_mask].head().to_string(index=False))

    # 4. Null / zero production
    _hdr("4. Null or zero production rows")
    null_rows = int(monthly["production"].isna().sum())
    zero_rows = int((monthly["production"] == 0).sum())
    print(f"null  production: {null_rows}")
    print(f"zero  production: {zero_rows}")

    # 5. Top 10 months per region
    _hdr("5. Top 10 production months per region (crude oil)")
    crude = monthly[monthly["fuel_type"] == "crude_oil"]
    for region, g in crude.groupby("region_name"):
        top = g.nlargest(3, "production")[["date", "production", "unit"]]
        print(f"\n  {region}:")
        print(top.to_string(index=False))

    # 6. Monthly → annual reconciliation
    _hdr("6. Monthly vs Annual reconciliation (mean match check)")
    annual_path = SILVER / "production_annual.csv"
    if annual_path.exists():
        annual = pd.read_csv(annual_path)
        recomputed = (
            monthly.groupby(["region_id", "year", "fuel_type"])["production"]
            .mean()
            .reset_index()
            .rename(columns={"production": "recomputed"})
        )
        merged = annual.merge(
            recomputed, on=["region_id", "year", "fuel_type"], how="left"
        )
        merged["diff"] = (merged["production"] - merged["recomputed"]).abs()
        max_diff = float(merged["diff"].max())
        print(f"max |annual_mean - monthly_mean| = {max_diff:.6f}  "
              f"{'✅ match' if max_diff < 1e-6 else '⚠ mismatch'}")

    # Gold summary — actuals + forecast parameters (separated)
    if not actuals.empty:
        _hdr("Gold — regional_actuals.csv (first 10 rows)")
        print(actuals.head(10).to_string(index=False))
    if not forecasts.empty:
        _hdr("Gold — region_forecasts.csv (one row per region × fuel)")
        print(forecasts.to_string(index=False))
        _hdr("Example projection — slope × year + intercept")
        for _, row in forecasts.head(3).iterrows():
            for yr in (2027, 2030, 2035):
                proj = row["slope"] * yr + row["intercept"]
                print(f"  {row['region_name']:<30} {row['fuel_type']:<12} "
                      f"{yr}: {proj:>10,.1f}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
