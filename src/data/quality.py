"""Data Quality Score (DQS) — single 0–100 number with breakdown.

DQS = 0.4 · completeness + 0.3 · consistency + 0.3 · freshness

Where:
    completeness = 1 − (null_rows / total_rows)
    consistency  = 1 − (outlier_rows / total_rows)
    freshness    = clip(1 − days_since_last_data / 90, 0, 1)

Displayed in Streamlit header and persisted to data/metadata.json.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import pandas as pd

from .validate import ValidationReport

W_COMPLETENESS = 0.4
W_CONSISTENCY = 0.3
W_FRESHNESS = 0.3
FRESHNESS_HORIZON_DAYS = 120  # EIA monthly data has a ~2-3 month reporting lag


@dataclass
class QualityScore:
    score: float           # 0-100
    completeness: float    # 0-1
    consistency: float     # 0-1
    freshness: float       # 0-1
    most_recent_data: str  # ISO date
    days_stale: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_dqs(df: pd.DataFrame, report: ValidationReport) -> QualityScore:
    total = max(report.total_rows, 1)

    completeness = max(
        0.0, 1.0 - report.completeness.get("avg_missing_pct", 0.0)
    )
    consistency = max(0.0, 1.0 - (report.outlier_count / total))

    if df.empty:
        most_recent = pd.Timestamp("2000-01-01", tz="UTC")
    else:
        dates = pd.to_datetime(df["date"])
        if dates.dt.tz is None:
            dates = dates.dt.tz_localize("UTC")
        most_recent = dates.max()

    days_stale = (datetime.now(timezone.utc) - most_recent.to_pydatetime()).days
    freshness = max(0.0, min(1.0, 1.0 - days_stale / FRESHNESS_HORIZON_DAYS))

    score = 100.0 * (
        W_COMPLETENESS * completeness
        + W_CONSISTENCY * consistency
        + W_FRESHNESS * freshness
    )

    return QualityScore(
        score=round(score, 2),
        completeness=round(completeness, 4),
        consistency=round(consistency, 4),
        freshness=round(freshness, 4),
        most_recent_data=most_recent.isoformat(),
        days_stale=int(days_stale),
    )
