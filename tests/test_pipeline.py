"""Pytest sanity tests for the data pipeline.

These tests use a synthetic EIAResponse so they run without hitting the
network. Run: pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean import (
    aggregate_annual,
    clean_crude_production,
    clean_natural_gas_production,
)
from src.data.eia_client import EIAResponse, PADD_REGIONS
from src.data.gold import build_region_forecasts, build_regional_actuals, project
from src.data.quality import compute_dqs
from src.data.schema import SILVER_COLUMNS, validate_schema
from src.data.validate import validate_dataset
from src.forecast.linear import fit_and_forecast
from src.kpi.calculations import decline_rate, relative_performance_index


# ------------------------------------------------------------ fixtures


def _synthetic_crude() -> EIAResponse:
    """Five PADDs × 36 months with known production values."""
    rows = []
    base = {"R10": 100, "R20": 400, "R30": 5000, "R40": 600, "R50": 500}
    for region, start in base.items():
        for i, month in enumerate(pd.date_range("2021-01-01", periods=36, freq="MS")):
            rows.append({
                "period": month.strftime("%Y-%m"),
                "duoarea": region,
                "product": "EPC0",
                "process": "FPF",
                "value": start + i * 2,
                "units": "MBBL/D",
                "series": f"{region}-EPC0-FPF",
            })
    return EIAResponse(
        data=rows,
        source="test",
        endpoint="petroleum/crd/crpdn/data/",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        bronze_path=None,
    )


def _synthetic_gas() -> EIAResponse:
    rows = []
    for region in ["STX", "SPA", "SLA"]:
        for i, month in enumerate(pd.date_range("2021-01-01", periods=36, freq="MS")):
            rows.append({
                "period": month.strftime("%Y-%m"),
                "duoarea": region,
                "process": "VGM",
                "value": 20000 + i * 50,
                "units": "MMCF",
                "series": f"{region}-NG",
            })
    return EIAResponse(
        data=rows,
        source="test",
        endpoint="natural-gas/prod/sum/data/",
        fetched_at=pd.Timestamp.now(tz="UTC"),
        bronze_path=None,
    )


@pytest.fixture
def monthly() -> pd.DataFrame:
    crude = clean_crude_production(_synthetic_crude())
    gas = clean_natural_gas_production(_synthetic_gas())
    return pd.concat([crude, gas], ignore_index=True)


# ------------------------------------------------------------ schema


def test_silver_has_all_contract_columns(monthly):
    assert set(SILVER_COLUMNS).issubset(monthly.columns)


def test_schema_validation_passes(monthly):
    assert validate_schema(monthly) == []


def test_all_five_padds_present(monthly):
    crude = monthly[monthly["fuel_type"] == "crude_oil"]
    assert set(crude["region_id"].unique()) == set(PADD_REGIONS.keys())


def test_no_duplicates(monthly):
    dup = monthly.duplicated(subset=["region_id", "date", "fuel_type"]).sum()
    assert dup == 0


def test_production_non_negative(monthly):
    assert (monthly["production"] >= 0).all()


# ------------------------------------------------------------ validation


def test_validation_report_passes(monthly):
    report = validate_dataset(monthly)
    assert report.passed, report.critical_failures
    assert report.duplicate_count == 0


# ------------------------------------------------------------ DQS


def test_dqs_in_zero_to_hundred(monthly):
    report = validate_dataset(monthly)
    dqs = compute_dqs(monthly, report)
    assert 0 <= dqs.score <= 100
    assert 0 <= dqs.completeness <= 1
    assert 0 <= dqs.consistency <= 1
    assert 0 <= dqs.freshness <= 1


# ------------------------------------------------------------ aggregation


def test_annual_mean_matches_monthly_mean(monthly):
    annual = aggregate_annual(monthly)
    # Recompute and compare
    manual = (
        monthly.groupby(["region_id", "year", "fuel_type"])["production"]
        .mean()
        .reset_index()
    )
    merged = annual.merge(
        manual, on=["region_id", "year", "fuel_type"], suffixes=("", "_manual")
    )
    assert (merged["production"] - merged["production_manual"]).abs().max() < 1e-9


# ------------------------------------------------------------ forecast


def test_forecast_r_squared_valid(monthly):
    annual = aggregate_annual(monthly)
    crude = annual[annual["fuel_type"] == "crude_oil"]
    fc = fit_and_forecast(crude, region_id="R30", selected_year=2022)
    assert 0 <= fc.r_squared <= 1
    assert not fc.actuals.empty
    assert not fc.forecast.empty


def test_forecast_split_boundary(monthly):
    annual = aggregate_annual(monthly)
    crude = annual[annual["fuel_type"] == "crude_oil"]
    fc = fit_and_forecast(crude, region_id="R30", selected_year=2022)
    assert fc.actuals["year"].max() <= 2022
    assert fc.forecast["year"].min() > 2022


# ----------------------------------------------- Gold: split tables


def test_gold_actuals_has_no_forecast_columns(monthly):
    annual = aggregate_annual(monthly)
    actuals = build_regional_actuals(annual)
    # Actuals is a facts-only table — must NOT carry model output columns
    assert "forecast" not in actuals.columns
    assert "slope" not in actuals.columns
    assert "is_forecast" not in actuals.columns
    # And every row must have a concrete production value
    assert actuals["production"].notna().all()


def test_gold_forecasts_has_no_projected_values(monthly):
    annual = aggregate_annual(monthly)
    forecasts = build_region_forecasts(annual)
    # Forecasts table stores only model parameters — not projected values
    assert "slope" in forecasts.columns
    assert "intercept" in forecasts.columns
    assert "r_squared" in forecasts.columns
    assert "production" not in forecasts.columns
    assert "actual" not in forecasts.columns
    # One row per (region, fuel)
    assert len(forecasts) == forecasts[["region_id", "fuel_type"]].drop_duplicates().shape[0]


# ----------------------------------------------- Tier 2 KPIs


def test_decline_rate_sign(monthly):
    """Declining series → positive decline_rate; rising series → negative."""
    annual = aggregate_annual(monthly)
    # The synthetic data has *rising* production (+2/month), so decline is negative
    crude = annual[annual["fuel_type"] == "crude_oil"]
    d = decline_rate(crude, region_id="R30")
    assert d < 0, f"expected negative (growing), got {d}"


def test_relative_performance_index_bounded(monthly):
    """RPI always ∈ [0, 100] or NaN when not enough peers."""
    annual = aggregate_annual(monthly)
    crude = annual[annual["fuel_type"] == "crude_oil"]
    for year in crude["year"].unique():
        for rid in crude["region_id"].unique():
            v = relative_performance_index(crude, rid, int(year))
            if not pd.isna(v):
                assert 0 <= v <= 100, f"{rid} {year}: RPI out of bounds: {v}"


def test_gold_actuals_has_new_kpi_columns(monthly):
    annual = aggregate_annual(monthly)
    actuals = build_regional_actuals(annual)
    for col in ("decline_rate_pct", "relative_performance_index"):
        assert col in actuals.columns, f"missing Tier 2 KPI column: {col}"


def test_project_helper_matches_fit_forecast(monthly):
    annual = aggregate_annual(monthly)
    forecasts = build_region_forecasts(annual)
    crude = annual[annual["fuel_type"] == "crude_oil"]
    r30_row = forecasts[
        (forecasts["region_id"] == "R30") & (forecasts["fuel_type"] == "crude_oil")
    ].iloc[0]

    # Hand-compute the 2030 projection from stored slope/intercept
    manual = project(r30_row, 2030)

    # Compare against a fresh fit_and_forecast with the same cutoff
    fc = fit_and_forecast(
        crude, region_id="R30", selected_year=int(r30_row["trained_through_year"])
    )
    engine = float(fc.slope * 2030 + fc.intercept)

    assert abs(manual - engine) < 1e-6
