"""Linear regression forecasting engine.

Methodology (documented for `docs/kpi_definitions.md`):
    - Fits an ordinary least squares line on (year, production) pairs
    - Forecast = slope * year + intercept for any future year
    - R-squared reported as a transparent confidence indicator

We chose linear regression over more complex models (ARIMA, Prophet)
because a well-reasoned, explainable trend beats an unexplained black
box for business decision support. Forecast uncertainty grows with
horizon; users should not rely on forecasts more than ~5 years out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class ForecastResult:
    slope: float
    intercept: float
    r_squared: float
    actuals: pd.DataFrame   # columns: year, production, kind="actual"
    forecast: pd.DataFrame  # columns: year, production, kind="forecast"

    @property
    def methodology(self) -> str:
        return (
            f"Linear regression (OLS). "
            f"slope = {self.slope:.2f} Mb/d per year, "
            f"intercept = {self.intercept:.2f}, "
            f"R² = {self.r_squared:.3f}."
        )


def fit_and_forecast(
    annual_df: pd.DataFrame,
    region_id: str,
    selected_year: int,
    horizon_end: int = 2035,
) -> ForecastResult:
    """Fit a linear model on actual data up to selected_year, then project.

    Args:
        annual_df: DataFrame from normalize.aggregate_annual()
        region_id: PADD region identifier (e.g. "R30")
        selected_year: year boundary — actuals up to and including this,
                       forecasts from selected_year + 1 onward
        horizon_end: last year to forecast

    Returns:
        ForecastResult with actuals, forecast frames and model parameters.
    """
    region_df = annual_df[annual_df["region_id"] == region_id].copy()
    region_df = region_df.sort_values("year").reset_index(drop=True)

    historical = region_df[region_df["year"] <= selected_year]
    if len(historical) < 2:
        raise ValueError(
            f"Need at least 2 historical points for region {region_id}; "
            f"got {len(historical)}"
        )

    X = historical[["year"]].to_numpy()
    y = historical["production"].to_numpy()

    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, y))

    actuals = historical[["year", "production"]].copy()
    actuals["kind"] = "actual"

    future_years = np.arange(selected_year + 1, horizon_end + 1)
    if len(future_years) == 0:
        forecast = pd.DataFrame(columns=["year", "production", "kind"])
    else:
        preds = model.predict(future_years.reshape(-1, 1))
        forecast = pd.DataFrame(
            {
                "year": future_years,
                "production": preds,
                "kind": "forecast",
            }
        )

    return ForecastResult(
        slope=slope,
        intercept=intercept,
        r_squared=r2,
        actuals=actuals.reset_index(drop=True),
        forecast=forecast.reset_index(drop=True),
    )
