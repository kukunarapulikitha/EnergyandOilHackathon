"""End-to-end Medallion pipeline: Bronze → Silver → validate → Gold.

Run:  python scripts/verify_pipeline.py

Produces:
    data/bronze/eia_crude_<ts>.json, eia_gas_<ts>.json, wti_spot_<ts>.json
    data/silver/production_monthly.csv, production_annual.csv, wti_prices.csv
    data/gold/regional_intelligence.csv, anomalies.csv
    data/metadata.json
    data/validation_report.json
    data/ingest.log
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import pandas as pd

from src.data.clean import (
    aggregate_annual,
    clean_crude_production,
    clean_natural_gas_production,
    clean_wti_prices,
)
from src.data.eia_client import EIAClient
from src.data.gold import build_region_forecasts, build_regional_actuals
from src.data.quality import compute_dqs
from src.data.validate import validate_dataset


def main() -> int:
    silver_dir = ROOT / "data" / "silver"
    gold_dir = ROOT / "data" / "gold" / "csv"
    silver_dir.mkdir(parents=True, exist_ok=True)
    gold_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("Medallion Pipeline — Bronze → Silver → validate → Gold")
    print("=" * 68)

    # ---------------------------------------------------------- BRONZE
    print("\n🟤 BRONZE — raw ingestion")
    client = EIAClient(bronze_dir=ROOT / "data" / "bronze",
                       log_path=ROOT / "data" / "ingest.log")
    crude_resp = client.fetch_crude_production_by_padd(start="2010-01")
    gas_resp = client.fetch_natural_gas_production(start="2010-01")
    wti_resp = client.fetch_wti_spot_price(start="2010-01-01")
    print(f"  crude records: {len(crude_resp.data):>5}  → {crude_resp.bronze_path.name}")
    print(f"  gas   records: {len(gas_resp.data):>5}  → {gas_resp.bronze_path.name}")
    print(f"  wti   records: {len(wti_resp.data):>5}  → {wti_resp.bronze_path.name}")

    # ---------------------------------------------------------- SILVER
    print("\n⚪ SILVER — cleaned, schema-enforced")
    crude = clean_crude_production(crude_resp)
    gas = clean_natural_gas_production(gas_resp)
    wti = clean_wti_prices(wti_resp)
    monthly = pd.concat([crude, gas], ignore_index=True)
    annual = aggregate_annual(monthly)

    (silver_dir / "production_monthly.csv").write_text(monthly.to_csv(index=False))
    (silver_dir / "production_annual.csv").write_text(annual.to_csv(index=False))
    (silver_dir / "wti_prices.csv").write_text(wti.to_csv(index=False))
    print(f"  monthly: {len(monthly):>5} rows  ({crude['fuel_type'].iloc[0] if not crude.empty else 'n/a'}+gas)")
    print(f"  annual : {len(annual):>5} rows")
    print(f"  wti    : {len(wti):>5} rows")

    # ---------------------------------------------------------- VALIDATE
    print("\n🔍 VALIDATION")
    report = validate_dataset(
        monthly, outliers_out=gold_dir / "anomalies.csv"
    )
    report.save(ROOT / "data" / "validation_report.json")
    status = "✅ PASSED" if report.passed else "❌ FAILED"
    print(f"  {status}  ({len(report.critical_failures)} critical, "
          f"{len(report.warnings)} warnings)")
    for f in report.critical_failures:
        print(f"    ✗ {f}")
    for w in report.warnings:
        print(f"    ⚠ {w}")

    if not report.passed:
        print("\nAborting before Gold layer — fix critical failures first.")
        return 1

    # ---------------------------------------------------------- DQS
    dqs = compute_dqs(monthly, report)
    print(f"\n📊 DQS: {dqs.score:.1f}/100")
    print(f"    completeness={dqs.completeness:.2%} "
          f"consistency={dqs.consistency:.2%} "
          f"freshness={dqs.freshness:.2%}  ({dqs.days_stale}d stale)")

    # ---------------------------------------------------------- GOLD
    print("\n🟡 GOLD — business-ready (actuals + forecasts separated)")
    actuals = build_regional_actuals(annual, wti_prices=wti)
    forecasts = build_region_forecasts(annual)
    (gold_dir / "regional_actuals.csv").write_text(actuals.to_csv(index=False))
    (gold_dir / "region_forecasts.csv").write_text(forecasts.to_csv(index=False))
    print(f"  actuals   : {len(actuals):>5} rows  "
          f"({actuals['fuel_type'].nunique()} fuels × "
          f"{actuals['region_id'].nunique()} regions × years)")
    print(f"  forecasts : {len(forecasts):>5} rows  "
          f"(one per region × fuel; live re-fit in UI for other cutoffs)")

    # ---------------------------------------------------------- METADATA
    metadata = {
        "source": "EIA API v2",
        "endpoints": [
            "petroleum/crd/crpdn",
            "natural-gas/prod/sum",
            "petroleum/pri/spt",
        ],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "version": "v1",
        "row_counts": {
            "crude_monthly": int((monthly["fuel_type"] == "crude_oil").sum()),
            "gas_monthly": int((monthly["fuel_type"] == "natural_gas").sum()),
            "wti_daily": len(wti),
            "gold_actuals": len(actuals),
            "gold_forecasts": len(forecasts),
        },
        "date_range": {
            "start": str(monthly["date"].min().date()) if not monthly.empty else None,
            "end": str(monthly["date"].max().date()) if not monthly.empty else None,
        },
        "dqs": dqs.to_dict(),
    }
    (ROOT / "data" / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print("\n" + "=" * 68)
    print("✅ Pipeline run complete. See data/metadata.json for summary.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
