"""Validation layer — runs between Silver and Gold.

Six checks:
    1. Schema         — contract columns + dtypes
    2. Range          — production ≥ 0, year bounds
    3. Completeness   — % missing per (region, year)
    4. Uniqueness     — no duplicate (region_id, date, fuel_type)
    5. Consistency    — flag month-over-month swings > 50%
    6. Outliers       — 12-month rolling z-score > 3

`validate_dataset()` returns a `ValidationReport`. Fails loud on schema
or uniqueness issues; warns on everything else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import MAX_YEAR, MIN_YEAR, validate_schema

COMPLETENESS_THRESHOLD = 0.05     # warn if > 5% missing per region-year
CONSISTENCY_MOM_THRESHOLD = 0.50  # flag MoM change > 50%
OUTLIER_ZSCORE_THRESHOLD = 3.0
OUTLIER_WINDOW = 12


@dataclass
class ValidationReport:
    total_rows: int
    critical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    completeness: dict = field(default_factory=dict)
    consistency_flags: int = 0
    outlier_count: int = 0
    duplicate_count: int = 0

    @property
    def passed(self) -> bool:
        return len(self.critical_failures) == 0

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))


def validate_dataset(
    df: pd.DataFrame, outliers_out: Path | str | None = None
) -> ValidationReport:
    report = ValidationReport(total_rows=len(df))

    # 1. Schema
    violations = validate_schema(df)
    if violations:
        report.critical_failures.extend(
            [f"schema:{v.rule}: {v.detail}" for v in violations]
        )
        return report  # stop early — downstream checks assume valid schema

    # 2. Range
    if (df["production"] < 0).any():
        report.critical_failures.append(
            f"range: {int((df['production'] < 0).sum())} rows with negative production"
        )
    bad_years = df[~df["year"].between(MIN_YEAR, MAX_YEAR)]
    if not bad_years.empty:
        report.warnings.append(
            f"range: {len(bad_years)} rows with year outside [{MIN_YEAR}, {MAX_YEAR}]"
        )

    # 3. Completeness — expected month-ends per region per year
    expected = (
        df.groupby(["region_id", "year", "fuel_type"])
        .size()
        .rename("actual")
        .reset_index()
    )
    expected["missing_pct"] = (12 - expected["actual"]).clip(lower=0) / 12
    poor = expected[expected["missing_pct"] > COMPLETENESS_THRESHOLD]
    report.completeness = {
        "avg_missing_pct": float(expected["missing_pct"].mean()),
        "region_years_below_threshold": int(len(poor)),
    }
    if not poor.empty:
        report.warnings.append(
            f"completeness: {len(poor)} (region, year) buckets > "
            f"{COMPLETENESS_THRESHOLD:.0%} missing"
        )

    # 4. Uniqueness
    dup_mask = df.duplicated(subset=["region_id", "date", "fuel_type"])
    report.duplicate_count = int(dup_mask.sum())
    if report.duplicate_count > 0:
        report.critical_failures.append(
            f"uniqueness: {report.duplicate_count} duplicate (region, date, fuel_type)"
        )

    # 5. Consistency — MoM change flags
    df_sorted = df.sort_values(["region_id", "fuel_type", "date"])
    pct_change = (
        df_sorted.groupby(["region_id", "fuel_type"])["production"]
        .pct_change()
        .abs()
    )
    report.consistency_flags = int((pct_change > CONSISTENCY_MOM_THRESHOLD).sum())
    if report.consistency_flags > 0:
        report.warnings.append(
            f"consistency: {report.consistency_flags} month-over-month "
            f"changes > {CONSISTENCY_MOM_THRESHOLD:.0%} (review, not reject)"
        )

    # 6. Outliers — 12-month rolling z-score
    def _z(s: pd.Series) -> pd.Series:
        mu = s.rolling(OUTLIER_WINDOW, min_periods=3).mean()
        sd = s.rolling(OUTLIER_WINDOW, min_periods=3).std()
        return (s - mu) / sd.replace(0, np.nan)

    df_sorted = df_sorted.copy()
    df_sorted["zscore"] = (
        df_sorted.groupby(["region_id", "fuel_type"])["production"].transform(_z)
    )
    outliers = df_sorted[df_sorted["zscore"].abs() > OUTLIER_ZSCORE_THRESHOLD]
    report.outlier_count = len(outliers)
    if report.outlier_count > 0:
        report.warnings.append(
            f"outliers: {report.outlier_count} rows with |z| > "
            f"{OUTLIER_ZSCORE_THRESHOLD}"
        )

    if outliers_out and not outliers.empty:
        Path(outliers_out).parent.mkdir(parents=True, exist_ok=True)
        outliers.to_csv(outliers_out, index=False)

    return report
