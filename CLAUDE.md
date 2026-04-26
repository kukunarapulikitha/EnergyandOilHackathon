# CLAUDE.md

Context file for Claude Code. Read this before making changes.

---

## Project

**CDF Energy AI Hackathon — U.S. Oil & Gas Energy Intelligence System**

A decision-support dashboard for business development analysts to evaluate regional U.S. oil/gas production opportunities, forecast future output, and identify high-potential regions. 5-day hackathon build.

Deadline: **April 12, 2026**. Submission = repo state at deadline + live hosted URL + 5-min walkthrough video.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | Python + Streamlit |
| Data | pandas, requests |
| Forecasting | scikit-learn `LinearRegression` (explainable, documented) |
| Charts | Plotly |
| AI | Llama 3.3 70B Instruct via **Groq API** (free, open-source) |
| Hosting | Streamlit Community Cloud |
| Data Source | EIA Open Data API v2 (free, public) |

**Decisions made:**
- Python-only (no Next.js, no separate backend) — one language, one deploy.
- Linear regression over Prophet/ARIMA — explainability beats sophistication per rubric.
- Groq + open-source Llama over OpenAI — free, fast, reproducible.
- `@st.cache_data(ttl=3600)` on EIA fetches to avoid rate limits.

---

## File Structure

```
energy-intelligence-system-kukunarapulikitha-main/
├── CLAUDE.md                  # This file
├── README.md                  # Submission checklist + live URL (fill before deadline)
├── problem_statement.md       # Full hackathon brief
├── requirements.txt           # Python dependencies
├── .env.example               # Template — copy to .env and fill keys
├── .gitignore                 # Predefined
│
├── planning/
│   └── planning.md            # COMPLETE — tech stack, phases, priorities
│
├── docs/
│   ├── schema_contract.md     # Silver-layer canonical schema (enforced)
│   ├── data_exploration.md    # Step 0 findings — mental model + sources used
│   ├── architecture.md        # TODO: system design overview
│   ├── kpi_definitions.md     # TODO: KPI formulas and business rationale
│   ├── reflection.md          # TODO: fill before submission
│   └── walkthrough.md         # TODO: video link
│
├── src/
│   ├── data/
│   │   ├── schema.py          # Silver schema contract + validate_schema()
│   │   ├── eia_client.py      # Bronze: crude + natural gas + WTI; raw JSON → data/bronze/
│   │   ├── clean.py           # Silver: cleaned, schema-enforced DataFrames
│   │   ├── normalize.py       # Thin back-compat shim → clean.py
│   │   ├── validate.py        # 6 validation checks + ValidationReport
│   │   ├── quality.py         # DQS 0–100 (completeness + consistency + freshness)
│   │   └── gold.py            # Gold: builds two tables — actuals + forecast params
│   ├── forecast/
│   │   └── linear.py          # OLS regression + ForecastResult dataclass
│   ├── kpi/
│   │   └── calculations.py    # 5 KPIs incl. Projected Production (required)
│   ├── ai/                    # TODO Phase 3: Groq client + prompts
│   └── ui/
│       ├── charts.py          # Plotly helpers (actuals + forecast split, heat map)
│       ├── data_loader.py     # Cached Gold/Silver/metadata readers
│       ├── provenance.py      # Provenance tooltips + data sources panel
│       ├── export.py          # Formula-driven Excel workbook builder
│       └── map.py             # Phase 2.6: Plotly choropleth + PADD state mapping
│
├── scripts/
│   ├── verify_pipeline.py     # End-to-end: Bronze → Silver → validate → DQS → Gold
│   └── explore_data.py        # 6 pandas reports for manual data inspection
│
├── tests/
│   └── test_pipeline.py       # 10 pytest sanity tests (all green)
│
├── data/                      # (mostly gitignored, regenerable)
│   ├── bronze/                # raw API payloads, timestamped audit trail
│   ├── silver/                # cleaned CSVs (production_monthly, annual, wti_prices)
│   ├── gold/
│   │   └── csv/               # regional_actuals.csv + region_forecasts.csv + anomalies.csv
│   ├── reference/             # STATIC, committed — map overlays
│   │   ├── basins.geojson     # 7 EIA DPR basin polygons (approximate)
│   │   └── rig_counts.json    # Baker Hughes rig-count snapshot
│   ├── metadata.json          # source, endpoints, row counts, DQS
│   ├── validation_report.json # full 6-check report
│   └── ingest.log             # per-fetch log
│
└── app.py                     # TODO Phase 2: Streamlit entry point
```

---

## Build Phases

### Phase 1 — Data & Foundation  ✅ COMPLETE
- [x] `requirements.txt`, `.env.example`, folder structure
- [x] `src/data/eia_client.py` — EIA API v2 client
- [x] `src/data/normalize.py` — pandas normalization + annual aggregation
- [x] `src/forecast/linear.py` — scikit-learn linear regression
- [x] `src/kpi/calculations.py` — Projected Production + 4 KPIs + Investment Score
- [x] `scripts/verify_pipeline.py` — end-to-end smoke test

### Phase 1.5 — Medallion Data Pipeline  ✅ COMPLETE
- [x] Schema contract (`src/data/schema.py` + `docs/schema_contract.md`)
- [x] Bronze layer: raw JSON persistence + natural gas + WTI endpoints
- [x] Silver layer: `clean.py` with unit preservation, dedup, month-start dates
- [x] Validation: 6 checks (schema, range, completeness, uniqueness, consistency, outliers)
- [x] Data Quality Score (DQS) with completeness/consistency/freshness breakdown
- [x] Gold layer: `regional_intelligence.csv` — single source of truth for UI/AI
- [x] `scripts/explore_data.py` — 6 pandas reports for manual inspection
- [x] `tests/test_pipeline.py` — 10 tests, all passing
- [x] `docs/data_exploration.md` — Step 0 findings

### Phase 2 — Core System Build (Day 3–4)  ✅ COMPLETE
- [x] `app.py` — Streamlit entry point with sidebar + 4 tabs
- [x] Sidebar: year slider (2010–2035) + region multi-select + fuel radio
- [x] Tab 1 **Overview**: KPI cards, top-regions bar chart, forecast params table
- [x] Tab 2 **Regional Detail**: Plotly chart — actuals (solid) + forecast (dashed + shaded zone)
- [x] Tab 3 **Compare**: side-by-side overlay + small multiples + comparison table
- [x] `src/ui/charts.py` — Plotly helpers for actuals/forecast split
- [x] `src/ui/data_loader.py` — cached readers for Gold + Silver + metadata
- [x] Documented methodology section rendered inline
- [x] Live re-fit when slider < baked cutoff
- [x] Forecast horizon extended to 2036 so 2035 is always a visible projection

### Phase 2.5 — Tier 2 Stretch Goals  ✅ COMPLETE
- [x] Custom KPIs: `decline_rate` + `relative_performance_index`
- [x] Data Provenance UI tooltips (`src/ui/provenance.py`) + data sources page
- [x] Live Refresh button in header + graceful stale-data fallback
- [x] Excel Export: formula-driven `.xlsx` download (`src/ui/export.py`)
- [x] Sensitivity Analysis heat map (new tab): WTI × decline rate → revenue potential

### Phase 2.6 — Geographic Map  (NEW Tier 1 — Hackathon_Scope_Update.pdf)  ✅ COMPLETE
- [x] `src/ui/map.py` — Plotly choropleth (USA-states, no Mapbox token needed)
- [x] State-to-PADD mapping constant
- [x] Crude view: states colored by their PADD's production
- [x] Natural gas view: 5 producing states colored; others grayed
- [x] Overlay toggle: Production / YoY growth / Relative performance
- [x] Clickable regions wired to `st.session_state["map_selected_region"]`
- [x] Regional Detail tab auto-selects the clicked region (integration)
- [x] New first tab: "🗺️ Map"
- [x] **Basin polygons** overlay (EIA DPR: Permian, Eagle Ford, Bakken, Anadarko, Appalachia, Haynesville, Niobrara) — `data/reference/basins.geojson`
- [x] **Active rig counts** overlay (Baker Hughes snapshot) — `data/reference/rig_counts.json`
- [x] Layer multiselect so Basins + Rigs toggle independently of base metric

### Phase 3 — Conversational AI Agent  (NEW Tier 1 — was Phase 3)
- [ ] `src/ai/client.py` — Groq client for Llama 3.3 70B
- [ ] `src/ai/prompts.py` — `build_regional_context(df, year)` + data/inference tagging
- [ ] Tab 4 **AI Analyst**: `st.chat_input` chat + "Generate Investment Summary" button
- [ ] UI: amber badge for AI inference, blue for data-backed claims
- [ ] Collapsible "Data sent to AI" per response
- [ ] Deploy to Streamlit Community Cloud (set `EIA_API_KEY` + `GROQ_API_KEY` secrets)
- [ ] Fill `docs/architecture.md`, `docs/kpi_definitions.md`, `docs/reflection.md`
- [ ] Record 5-min walkthrough video, update README with live URL

### Tier 3 — Stretch (if time permits)
- [ ] Tab 5 **Anomalies**: rolling ±2σ outliers, Llama-generated one-sentence explanations
- [ ] Investment Score gauge on Overview + "Explain this score" AI button

---

## Rubric (Judging Weights)

| Dimension | Weight | Key drivers |
|-----------|--------|-------------|
| AI Integration & Usage | 25% | Data grounding, data/inference distinction, decision value |
| Technical Architecture | 25% | Clean modular code, reproducibility |
| UI/UX & Usability | 20% | Interactive year selector, clear actuals/forecast split |
| Data Engineering | 15% | Pipeline quality, normalization, null handling |
| Project Management | 15% | Clean git history, documentation completeness |

---

## Tier 1 Requirements (MANDATORY — must all work)

1. **Data collection** from real public source (EIA API)
2. **Multiple U.S. regions** for geographic comparison (PADD 1-5)
3. **Interactive year selector** — historical up to year, forecast beyond
4. **Documented forecasting methodology** (linear regression with R²)
5. **Projected Production Estimate** KPI (required, by region + year)
6. **Hosted web app with live URL** (Streamlit Cloud) — NO local-only submissions
7. **At least one AI feature** with live data access, clear data/inference boundary
8. **Documentation**: data sources, forecasting approach, KPI definitions, architecture

## Cut List (if time runs short)
1. **First cut**: Excel export, sensitivity analysis heat map
2. **Second cut**: additional custom KPIs beyond Projected Production + growth rate
3. **NEVER cut**: year selector, Projected Production KPI, at least one AI feature, live deployment

---

## Architecture — Medallion (Bronze → Silver → Gold)

**One-liner (use in docs):**
> Lightweight Medallion architecture with batch ingestion from EIA APIs, strict schema enforcement, data validation checks, and a deterministic transformation pipeline supporting downstream forecasting, KPI computation, and AI-driven decision support.

| Layer | Purpose | Rules |
|-------|---------|-------|
| 🟤 **Bronze** | Raw API payloads | No transforms. Timestamped JSON files. Append-only audit trail. |
| ⚪ **Silver** | Cleaned, schema-enforced | Conforms to `SILVER_SCHEMA`. Units preserved. Idempotent (overwritten per run). |
| 🟡 **Gold** | Business-ready tables — **split into actuals vs. forecasts** | Actuals = observed facts + derived KPIs. Forecasts = model parameters only (slope/intercept/R²). This structural separation preserves the data-vs-inference boundary. UI and AI read only from here. |

**Validation runs between Silver and Gold.** Critical failures (schema, duplicates) abort the pipeline; warnings (completeness, consistency, outliers) are logged but don't block.

**DQS** is the 0-100 data quality score — shown in Streamlit header (Phase 2).

## How to Run

```bash
# One-time setup
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with real EIA_API_KEY (https://www.eia.gov/opendata/register.php)
# and GROQ_API_KEY (https://console.groq.com/keys)

# Run full Bronze → Silver → validate → DQS → Gold pipeline
python scripts/verify_pipeline.py

# Manually inspect the data (6 canned reports)
python scripts/explore_data.py

# Run test suite
pytest tests/ -v

# Run Streamlit app (once Phase 2 is built)
streamlit run app.py
```

---

## Conventions

- **Imports**: absolute (`from src.data.eia_client import EIAClient`)
- **Type hints**: use them everywhere (3.10+ syntax: `str | None`, `list[dict]`)
- **Caching**: `@st.cache_data(ttl=3600)` on any function that calls EIA or does heavy pandas work
- **Forecasting methodology**: always documented inline + in `docs/kpi_definitions.md`
- **AI outputs**: tag with `(Data)` for data-backed claims and `(AI Analysis)` for model inference — parse in UI for visual distinction
- **Commits**: conventional format (`feat:`, `docs:`, `chore:`), meaningful granularity, no squashing. Target ~15+ commits across phases. Never commit `.env` or API keys.

---

## Data Model

```
EIA API (crude + gas + WTI)
         ↓ EIAClient.fetch_*()
🟤 data/bronze/*.json              (raw, timestamped, audit trail)
         ↓ clean_crude_production / clean_natural_gas_production / clean_wti_prices
⚪ data/silver/                    (SILVER_SCHEMA enforced)
    ├─ production_monthly.csv      [region_id, region_name, date, year,
    │                               production, unit, fuel_type, source,
    │                               series_id, fetched_at]
    ├─ production_annual.csv       (monthly → annual mean per region-year-fuel)
    └─ wti_prices.csv              [date, price_usd_per_bbl, source, fetched_at]
         ↓ validate_dataset()  →  data/validation_report.json
         ↓ compute_dqs()       →  data/metadata.json
         ↓ build_regional_actuals() / build_region_forecasts()
🟡 data/gold/
    ├─ regional_actuals.csv    (FACTS only — observed production + derived KPIs)
    │  [region_id, region_name, year, fuel_type, production, unit,
    │   growth_pct, volatility_pct, revenue_potential_usd,
    │   source, fetched_at]
    │
    └─ region_forecasts.csv    (MODEL PARAMS only — no projected values)
       [region_id, region_name, fuel_type, slope, intercept, r_squared,
        trained_through_year, horizon_end, method, investment_score, source]
         ↓
Streamlit reads BOTH → stitches chart live on slider change
 • production up to slider_year → solid line (from actuals)
 • slope × year + intercept beyond slider_year → dashed line (from forecasts)
 • For alternate cutoffs: fit_and_forecast(annual, region, Y) re-runs live
```

**PADD region IDs (crude):** `R10` East Coast, `R20` Midwest, `R30` Gulf Coast, `R40` Rocky Mtn, `R50` West Coast. `R30` is the largest producer (Permian/Eagle Ford).

**Natural gas state IDs:** `STX` Texas, `SPA` Pennsylvania, `SLA` Louisiana, `SOK` Oklahoma, `SWV` West Virginia.

**Units are preserved, never auto-converted.** Crude = `MBBL/D`, gas = `MMCF`.

---

## AI Grounding Strategy

System prompt injection — at request time, serialize current KPI snapshot into a text block:

```
REGIONAL DATA SNAPSHOT (source: EIA API v2, fetched: <timestamp>)
Selected year: <year>

Region: PADD 3 (Gulf Coast)
  Historical: [(2015, 8321), (2016, 8560), ...]
  Projected 2027: 11,420 Mb/d
  YoY growth: +4.2%
  Volatility: 12.3%
  Forecast R²: 0.94
...
```

Instruct Llama to:
- Prefix every data-derived claim with `(Data)`
- Prefix every interpretive/inferred statement with `(AI Analysis)`
- Cite specific region + year for every number

UI parses tags → blue box for Data, amber box + "Model Inference" badge for AI Analysis.

---

## Key Links

- EIA Open Data docs: https://www.eia.gov/opendata/documentation.php
- EIA API v2 browser: https://www.eia.gov/opendata/browser/
- Groq console: https://console.groq.com/
- Streamlit Cloud: https://streamlit.io/cloud
- Problem statement: [`problem_statement.md`](problem_statement.md)
- Plan: [`planning/planning.md`](planning/planning.md)
