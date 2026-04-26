"""Canonical Silver-layer schema contract.

Every cleaned DataFrame in this project must conform to `SILVER_SCHEMA`.
Downstream code (Gold layer, forecasting, KPIs, Streamlit UI, AI prompts)
reads against this contract. Changing it is a breaking change.

See docs/schema_contract.md for the human-readable spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

FuelType = Literal["crude_oil", "natural_gas"]


SILVER_SCHEMA: dict[str, str] = {
    "region_id": "string",
    "region_name": "string",
    "date": "datetime64[ns]",
    "year": "int64",
    "production": "float64",
    "unit": "string",
    "fuel_type": "string",
    "source": "string",
    "series_id": "string",
    "fetched_at": "datetime64[ns, UTC]",
}

SILVER_COLUMNS: list[str] = list(SILVER_SCHEMA.keys())

VALID_FUEL_TYPES: set[str] = {"crude_oil", "natural_gas"}
VALID_UNITS: set[str] = {"MBBL/D", "MMCF"}  # crude Mb/d; gas million cf

MIN_YEAR = 2000
MAX_YEAR = 2035


@dataclass
class SchemaViolation:
    rule: str
    detail: str


def validate_schema(df: pd.DataFrame) -> list[SchemaViolation]:
    """Return a list of schema contract violations. Empty list = valid."""
    violations: list[SchemaViolation] = []

    missing = [c for c in SILVER_COLUMNS if c not in df.columns]
    if missing:
        violations.append(
            SchemaViolation("missing_columns", f"missing: {missing}")
        )
        return violations  # no point checking dtypes if columns missing

    for col, expected in SILVER_SCHEMA.items():
        actual = str(df[col].dtype)
        if expected.startswith("datetime64[ns"):
            if not actual.startswith("datetime64[ns"):
                violations.append(
                    SchemaViolation(
                        "dtype_mismatch",
                        f"{col}: expected {expected}, got {actual}",
                    )
                )
        elif expected == "string":
            if actual not in ("string", "object"):
                violations.append(
                    SchemaViolation(
                        "dtype_mismatch",
                        f"{col}: expected string, got {actual}",
                    )
                )
        elif actual != expected:
            violations.append(
                SchemaViolation(
                    "dtype_mismatch",
                    f"{col}: expected {expected}, got {actual}",
                )
            )

    bad_fuels = set(df["fuel_type"].dropna().unique()) - VALID_FUEL_TYPES
    if bad_fuels:
        violations.append(
            SchemaViolation("invalid_fuel_type", f"unknown: {bad_fuels}")
        )

    return violations
