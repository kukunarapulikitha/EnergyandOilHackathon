# Schema Contract — Silver Layer

The Silver layer is the canonical clean dataset. Everything downstream — Gold, forecasting, KPIs, Streamlit UI, AI prompts — reads against this contract. Enforced in [`src/data/schema.py`](../src/data/schema.py).

## Canonical columns

| Column | Type | Notes |
|--------|------|-------|
| `region_id`   | string        | EIA identifier. PADDs (`R10`…`R50`) for crude; state codes (`STX`, `SPA`, …) for natural gas |
| `region_name` | string        | Human-readable (e.g. "PADD 3 (Gulf Coast)", "Texas") |
| `date`        | datetime      | Month-start timestamp |
| `year`        | int64         | Derived from `date`; range `[2000, 2035]` |
| `production`  | float64       | Non-negative. Unit depends on `fuel_type` |
| `unit`        | string        | `"MBBL/D"` (crude, thousand bbl/day) or `"MMCF"` (gas, million cubic feet) |
| `fuel_type`   | string        | `"crude_oil"` \| `"natural_gas"` |
| `source`      | string        | `"EIA API v2 / <endpoint>"` for provenance |
| `series_id`   | string        | EIA series identifier for row-level provenance |
| `fetched_at`  | datetime UTC  | When the Bronze payload was fetched |

## Rules

1. **Unique key:** `(region_id, date, fuel_type)` is unique. Deduplication keeps the last-seen row.
2. **Units preserved:** crude and gas are different physical quantities. Never auto-convert — surface the mismatch via the `unit` column.
3. **Non-negative production:** negative values are rejected at the cleaning stage.
4. **Month-start dates:** `date` is always the first of the month (`pd.offsets.MonthBegin(0)`).
5. **Everything downstream reads Gold, not Silver.** Silver is the intermediate canonical form; Gold is the business-ready join.

## Validation checks (in [`src/data/validate.py`](../src/data/validate.py))

| Check | Severity | Rule |
|-------|----------|------|
| Schema   | critical | All contract columns present with correct dtypes |
| Range    | critical | `production ≥ 0` |
| Range    | warning  | `year ∈ [2000, 2035]` |
| Uniqueness | critical | No duplicate `(region_id, date, fuel_type)` |
| Completeness | warning | ≤ 5% missing months per (region, year) |
| Consistency  | warning | Month-over-month change > 50% flagged for review |
| Outliers     | info    | 12-month rolling z-score > 3 → `data/gold/anomalies.csv` |

Critical failures abort the pipeline before the Gold layer is written.

## Data Quality Score

See [`src/data/quality.py`](../src/data/quality.py).

```
DQS = 0.4 · completeness + 0.3 · consistency + 0.3 · freshness
```

Exposed in `data/metadata.json` and the Streamlit header in Phase 2.

---

## Gold layer — two separate tables

Gold is **deliberately split** into two files so that facts and model outputs stay structurally apart. The problem statement requires the system to "distinguish data-backed claims from model-generated inference"; splitting the tables makes that boundary physical, not just cosmetic.

### `data/gold/regional_actuals.csv` — FACTS only

One row per `(region_id, year, fuel_type)`.

| Column | Notes |
|--------|-------|
| region_id, region_name, year, fuel_type, production, unit | Observed from EIA |
| growth_pct         | YoY percentage change (derived from observed data) |
| volatility_pct     | Coefficient of variation across historical years |
| revenue_potential_usd | Production × latest WTI spot. `None` for natural gas |
| source, fetched_at | Provenance — everything here traces to EIA |

No forecast columns, no model parameters. Every value is reproducible from Silver without invoking any model.

### `data/gold/region_forecasts.csv` — MODEL PARAMETERS only

One row per `(region_id, fuel_type)`.

| Column | Notes |
|--------|-------|
| slope, intercept, r_squared | Linear-regression parameters from scikit-learn |
| trained_through_year | The cutoff year used for fitting (latest actual) |
| horizon_end | Max year valid for projection (default 2035) |
| method | `"linear_ols"` |
| investment_score | Composite 0-100 score (uses forecast + stability + growth) |
| source | e.g. `"model:linear_ols trained on EIA annual 2010–2024"` |

Projected values are **not stored** — they're computed at read time:

```python
projection = slope * target_year + intercept
```

For alternate cutoffs (when the UI slider moves backward), the Streamlit app calls `fit_and_forecast(annual, region_id, selected_year=Y)` live — see `src/forecast/linear.py`. The pre-baked table is a default-view snapshot + the input for the AI prompt's forecast section.

### Consumption pattern

Streamlit year-selector logic:

```python
# On slider change to year Y:
actuals   = pd.read_csv("data/gold/regional_actuals.csv")
forecasts = pd.read_csv("data/gold/region_forecasts.csv")

# Historical ribbon (solid line)
history = actuals.query("region_id == @r and year <= @Y")

# Forward projection (dashed line)
row = forecasts.query("region_id == @r").iloc[0]
future_years = range(Y + 1, row.horizon_end + 1)
projected = [(y, row.slope * y + row.intercept) for y in future_years]
```

AI-prompt serialization can tag sections as `(Data)` (from actuals) and `(AI Analysis)` (from forecasts) with zero ambiguity.
