# Data Exploration — Step 0 Notes

Manual understanding of the data **before** building the pipeline. Companion to the schema contract.

## Mental model

```
Region → Time → Production value
```

Two commodities are in scope per the problem statement ("oil & gas"):
- **Crude oil** (primary driver of the investment KPIs)
- **Natural gas** (regional diversification signal)

## Granularity

| Dimension | Choice | Why |
|-----------|--------|-----|
| Time      | monthly (ingested) → annual (forecast) | EIA publishes monthly; annual averages smooth noise for year-selector UX |
| Region    | PADD 1–5 for crude; top-5 states for gas | PADDs = industry-standard geographic comparison. Gas has no PADD equivalent — state-level is the finest public regional granularity |
| Fuel      | crude + gas, two separate rows per (region, date, fuel) | Different units, different regions — never collapse |

## Dimensions confirmed

- **Regions (crude):** R10 East Coast, R20 Midwest, R30 Gulf Coast, R40 Rocky Mtn, R50 West Coast. R30 dominates (Permian + Eagle Ford basin).
- **Regions (gas):** Texas (STX), Pennsylvania (SPA), Louisiana (SLA), Oklahoma (SOK), West Virginia (SWV). These 5 cover ~80% of U.S. marketed production.
- **Time range:** 2010-01 through most-recent-published (~2 month lag on EIA data).
- **Units:** `MBBL/D` for crude, `MMCF` for natural gas. Preserved, never converted.

## Known issues / watchouts

1. **Dual-series returns from EIA crude:** the `petroleum/crd/crpdn` endpoint returns both `MBBL` (monthly total) and `MBBL/D` (daily rate) for every period. We filter to `MBBL/D` only in `clean_crude_production()`.
2. **Naming inconsistencies:** `R30` vs `"PADD 3"` vs `"Gulf Coast"` — mapped explicitly in `PADD_REGIONS`.
3. **Data lag:** EIA publishes with a ~2-month lag. The `freshness` component of DQS captures this.
4. **Unit mismatch crude vs gas:** Mb/d (a rate) vs MMcf (a volume). Do not sum. Filter by `fuel_type` before aggregating.
5. **Natural gas is state-level, not PADD-level:** makes apples-to-apples geographic comparison with crude imperfect. Document this in the UI when both fuels are shown.

## Data sources used vs. considered

| Source | Status | Rationale |
|--------|--------|-----------|
| EIA API — crude | ✅ in use | Primary spec requirement |
| EIA API — natural gas | ✅ in use | "oil & gas" in problem statement |
| EIA API — WTI spot | ✅ in use | Live price for revenue KPI |
| Texas RRC, NDIC, COGCC | ❌ deferred | Overlaps with PADD granularity; state-only scrapers add setup cost without new signal |
| FRED | ❌ skipped | EIA Spot Prices already covers WTI |
| OpenWeatherMap | ❌ skipped | Weather is correlational noise for this decision |

## Open questions

- Do we want basin-level crude (Permian, Bakken, Eagle Ford) in addition to PADD? EIA's Drilling Productivity Report has these. Defer to Phase 2 if time permits.
- Should gas production be converted to barrel-of-oil-equivalent (BOE) for a unified view? Keeps units consistent but loses physical meaning. Decision: **no** — show side-by-side in native units.
