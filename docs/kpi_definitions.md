# KPI Definitions

All KPIs are computed in the Gold layer (`src/data/gold.py` and `src/kpi/calculations.py`) and are available per region × year for each fuel type. They update dynamically when the year selector or fuel selector changes.

---

## 1. Projected Production Estimate *(Mandatory KPI)*

**Definition:**
```
Production(year) = slope × year + intercept
```
where `slope` and `intercept` are the OLS linear regression coefficients fitted per (region, fuel type) pair on the annual production history up to the selected year.

**Unit:** Mb/d (crude oil) | MMcf/month (natural gas)

**Data source:** EIA Open Data API v2 — monthly production aggregated annually

**Business purpose:** Core investment signal. Answers: *"How much will this region produce in my target year?"*

**Verifiable at:** https://www.eia.gov/dnav/pet/pet_crd_crpdn_adc_mbblpd_m.htm (crude) | https://www.eia.gov/dnav/ng/ng_prod_sum_a_EPG0_VGM_mmcf_m.htm (gas)

---

## 2. Production Growth Rate (YoY)

**Definition:**
```
growth_pct = (Production_N − Production_N-1) / Production_N-1 × 100
```

**Unit:** Percentage (%)

**Data source:** Computed from EIA annual aggregates

**Business purpose:** Distinguishes growing basins from plateaued or declining ones. A consistently positive growth rate signals an investable trend.

**Interpretation:**
- > +5% → strong growth
- 0 to +5% → stable
- < 0% → declining basin — requires additional diligence

---

## 3. Production Decline Rate (5-Year Annualized)

**Definition:**
```
decline_rate_pct = ((Production_N − Production_N-5) / Production_N-5) / 5 × 100
```
Annualized rate of change over the trailing 5-year window.

**Unit:** Percentage per year (%/yr)

**Data source:** Computed from EIA annual aggregates; requires ≥ 6 years of history

**Business purpose:** Critical for mature basins. A steep 5-year annualized decline indicates a basin past peak; a mild or recovering rate may signal secondary development opportunity.

**Note:** A negative value means production declined over the 5-year window. Displayed with an "inverse" delta color in the UI (red = declining = caution).

---

## 4. Revenue Potential

**Definition:**
```
revenue_potential_usd = Projected_Production × WTI_price × days_in_year × unit_conversion
```
- Crude: `Mb/d × 1,000 × WTI ($/bbl) × 365`
- Gas: `MMcf × Henry_Hub ($/Mcf) × 1,000,000`

**Unit:** USD per year

**Data source:**
- Production: EIA Open Data API
- WTI price: EIA petroleum spot price series (`PET.RWTC.D`) — same API key, no additional credentials

**Business purpose:** Translates production volume into dollar terms using the current commodity price. Allows direct comparison of oil vs. gas opportunity in a common unit.

**Verifiable at:** https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm

---

## 5. Volatility Score

**Definition:**
```
volatility_pct = std(production_trailing_10yr) / mean(production_trailing_10yr) × 100
```
Coefficient of variation of annual production over the trailing 10 years.

**Unit:** Percentage (%)

**Data source:** Computed from EIA annual aggregates; requires ≥ 10 years of history

**Business purpose:** Measures operational predictability. High volatility may indicate weather disruption, infrastructure constraints, price-driven curtailments, or regulatory exposure. Lower volatility is preferable for long-term investment underwriting.

---

## 6. Relative Performance Index (RPI)

**Definition:**
```
composite = growth_rate × 0.4 + revenue_potential_normalized × 0.4 − volatility_normalized × 0.2
RPI = percentile_rank(composite, all_regions) × 100
```
The composite score is computed across all regions for the same fuel type and year; then percentile-ranked and scaled 0–100.

**Unit:** Score (0–100)

**Business purpose:** A single comparable number for ranking regions. An RPI of 87 means the region outperforms 87% of peers on the composite metric. Designed to directly support the *"Is this region worth pursuing?"* decision.

---

## 7. Investment Score

**Definition:**
```
investment_score = (
    growth_rate_normalized    × 0.35 +
    (1 − decline_rate_norm)   × 0.25 +
    (1 − volatility_norm)     × 0.20 +
    forecast_r_squared        × 0.20
) × 100
```
Each component is min-max normalized across all regions for the same fuel type before weighting.

**Unit:** Score (0–100)

**Business purpose:** Single-number investment attractiveness signal. Penalizes high volatility and weak forecast confidence; rewards strong growth and stable decline profiles.

**Weight rationale:**
| Component | Weight | Why |
|-----------|--------|-----|
| Growth rate | 35% | Primary signal for an expanding market |
| Decline rate (inverse) | 25% | Mature basins with steep declines need heavy reinvestment to maintain output |
| Volatility (inverse) | 20% | Predictable production is easier to underwrite financially |
| Forecast R² | 20% | Low R² = noisy trend = uncertain projection = higher risk |

---

## 8. Forecast R²

**Definition:**
```
R² = 1 − SS_res / SS_tot
```
Standard coefficient of determination from the OLS linear regression fit.

**Unit:** Dimensionless (0–1)

**Business purpose:** Transparency metric — tells the analyst how much to trust the Projected Production Estimate. Surfaced on every chart and in the AI analyst context so all projections carry an explicit confidence signal.

**Interpretation:**
- ≥ 0.85 → reliable linear trend; projection is grounded
- 0.60–0.84 → moderate fit; treat projection as directional
- < 0.60 → noisy or non-linear trend; projection is speculative

---

## Well Economics KPIs (Well Economics Calculator Tab)

### EUR — Estimated Ultimate Recovery
```
EUR = Σ q(t) × days_per_month   over all months of the well life
```
where `q(t)` is from the Arps hyperbolic decline curve: `q(t) = qi / (1 + b × Di × t)^(1/b)`.

**Unit:** bbl (crude) | Mcf (gas)

### NPV — Net Present Value
```
NPV = Σ CF_t / (1 + r_monthly)^t
r_monthly = (1 + r_annual)^(1/12) − 1
```

**Unit:** USD

### IRR — Internal Rate of Return
Monthly rate that sets NPV = 0, annualized: `IRR_annual = (1 + IRR_monthly)^12 − 1`.
Computed via `numpy-financial.irr` with `scipy.optimize.brentq` fallback.

### Payback Period
First month at which cumulative cashflow ≥ 0. Undiscounted.

**Unit:** Months

### Monthly Cashflow
```
CF_t = (Volume_t × price − Volume_t × LOE − Revenue_t × severance_pct)
CF_0 = CF_0 − capex   (capex booked at t = 0)
Volume_t = q(t) × 30.4375 days/month
```
