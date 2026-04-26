# ⚡ Energy Intelligence System
### CDF Energy AI Hackathon Submission

> A decision-support tool for U.S. oil and gas production analysis — combining a Bronze→Silver→Gold data pipeline, regional KPI framework, AI-powered analyst, and well economics calculator into a single hosted interface.

---

## 🔗 Links

| | |
|---|---|
| **Live App** | https://energyandoilhackathon-dpa2nr8ryjauaxombfw9ru.streamlit.app/ |
| **Submission Repository** | https://github.com/Community-Dreams-Foundation-Hackathons/energy-intelligence-system-kukunarapulikitha |
| **Author Repository** | https://github.com/kukunarapulikitha/EnergyandOilHackathon |

> ⚠️ The live URL is functional and tested as of submission. Do not build from source — use the hosted link above.

---

## 📋 Submission Guidelines

**Submission checklist (per CDF Energy AI Hackathon rules):**

- [x] **Live hosted URL** — Streamlit Community Cloud: https://energyandoilhackathon-dpa2nr8ryjauaxombfw9ru.streamlit.app/
- [x] **Submission repository** — https://github.com/Community-Dreams-Foundation-Hackathons/energy-intelligence-system-kukunarapulikitha
- [x] **Real public data source** — EIA Open Data API v2 (production, WTI spot price)
- [x] **Multiple U.S. regions** — PADD 1–5 (crude oil) + TX, PA, LA, OK, WV (natural gas)
- [x] **Interactive year selector** — 2010 to 2035 slider; actuals to selected year, OLS forecast beyond
- [x] **Documented forecasting methodology** — Linear OLS with R² per region Forecasting Approach
- [x] **Projected Production Estimate KPI** — per region × year (required mandatory KPI)
- [x] **At least one AI feature with live data** — Conversational AI Analyst (Groq / Llama 3.3 70B) grounded in live Gold-layer snapshot
- [x] **Data/inference boundary** — `(Data)` vs `(AI Analysis)` tags, color-coded in UI
- [x] **Documentation** — data sources, forecasting approach, KPI definitions, architecture (this README)
- [x] **Tier 2 stretch** — Well Economics Calculator (Arps decline + NPV/IRR/payback), Excel export + import, sensitivity analysis heatmap
- [x] **Unit tests** — `pytest tests/` (pipeline tests + well model tests, 25+ assertions)
- [x] **AI tools disclosure** — Claude code to generate boilerplate code and improve efficiency

**Environment variables required for deployment:**

```
EIA_API_KEY=<your EIA Open Data API key>  # https://www.eia.gov/opendata/register.php
GROQ_API_KEY=<your Groq API key>          # https://console.groq.com/keys
```

---

## Table of Contents

1. [What Was Built](#what-was-built)
2. [System Architecture](#system-architecture)
3. [Data Sources](#data-sources)
4. [Data Engineering Pipeline](#data-engineering-pipeline)
5. [Forecasting Approach](#forecasting-approach)
6. [KPI Definitions](#kpi-definitions)
7. [AI Integration](#ai-integration)
8. [Well Economics Calculator](#well-economics-calculator)
9. [Tech Stack & Justification](#tech-stack--justification)
10. [Key Insights](#key-insights)
11. [AI Coding Tools Usage](#ai-coding-tools-usage)

---

## What Was Built

The Energy Intelligence System is a fully hosted, decision-support web application designed for business development analysts evaluating U.S. oil and gas production opportunities. It is organized into three tabs:

**📊 Dashboard** — Regional production explorer with map, trend charts, sensitivity analysis, and side-by-side comparison of PADD regions. All charts update dynamically based on the fuel type, year, and region selected in the sidebar.

**🛢️ Well Economics** — An interactive Arps decline-curve model that computes EUR, NPV, IRR, payback period, and total revenue for a horizontal well. Region presets are auto-loaded from the map; all parameters are editable in real time.

**🤖 AI Analyst** — A conversational agent grounded in live Gold-layer data. Responses are tagged `(Data)` vs. `(AI Analysis)` so analysts always know what is verified versus inferred. Produces structured artifacts (ranked tables, KPI metric cards, sensitivity charts) above the prose response.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public APIs                             │
│          EIA Open Data  ·  FRED  ·  EIA Spot Prices             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ ingest
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BRONZE LAYER                              │
│     Raw JSON/CSV as pulled — immutable, timestamped             │
└──────────────────────────────┬──────────────────────────────────┘
                               │ clean + normalize
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SILVER LAYER                              │
│   Consistent region IDs · aligned time series · nulls handled   │
└──────────────────────────────┬──────────────────────────────────┘
                               │ aggregate + compute KPIs
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GOLD LAYER                               │
│  actuals_df · forecasts_df · annual_df · metadata.json          │
│  KPIs: growth rate, decline rate, volatility, inv. score, RPI   │
└──────┬──────────────────┬────────────────────┬──────────────────┘
       │                  │                    │
       ▼                  ▼                    ▼
  Dashboard           AI Analyst         Well Economics
  (Streamlit)     (Groq / Llama 3.3)   (Arps + NPV/IRR)
```

**Component summary:**

- `scripts/` — ingestion and pipeline scripts that populate Bronze → Silver → Gold
- `src/data/` — loaders that serve cached Gold data to all UI components
- `src/ai/` — Groq client, context builder, intent classifier, artifact builder, tag parser
- `src/economics/` — pure-Python Arps decline math and region defaults (no Streamlit imports)
- `src/ui/` — Streamlit UI modules for each tab (map, charts, compare, sensitivity, well calculator)
- `app.py` — single entrypoint; mounts all tabs and the global sidebar

---

## Data Sources

| Source | What it provides | Why chosen |
|--------|-----------------|------------|
| **EIA Open Data API v2** (`api.eia.gov`) | Monthly crude oil and natural gas production by PADD region and state, going back to 1981 | Primary authoritative source for U.S. regional production. Free with API key. Geographic granularity maps directly to the PADD regions used throughout the system. |
| **EIA WTI Spot Price** (`api.eia.gov/v2/petroleum/pri/spt`) | Daily WTI crude spot price from Cushing, OK | Same EIA API key — no additional credentials needed. WTI price is used to compute revenue potential KPI (`production × WTI × days_in_year`) and injected into the AI analyst context. |
| **Baker Hughes Rig Count** (static snapshot in `data/reference/`) | Active land rig counts by basin | Provides the "Rigs" overlay layer on the production map. Snapshot is committed to the repo; refreshed manually before each pipeline run. |
| **EIA Basin Boundaries** (static GeoJSON in `data/reference/`) | Approximate polygon footprints for 7 EIA DPR basins (Permian, Eagle Ford, Bakken, etc.) | Provides the "Basins" overlay layer on the production map. Static reference — not refreshed automatically. |

All sources are public or have free open tiers. **No FRED API key is required.** No proprietary data is used anywhere in the pipeline.

> **Data verifiability:** Every number displayed by the app can be cross-checked against the EIA Open Data browser at https://www.eia.gov/opendata/browser/. The `data/bronze/` directory stores the raw API payloads (timestamped) so the full lineage from API response to dashboard metric is auditable.

---

## Data Engineering Pipeline

The pipeline follows a **Bronze → Silver → Gold medallion architecture**, which separates concerns between raw ingestion, cleaning, and analytics-ready output.

### Bronze Layer
Raw data is pulled from APIs and stored as immutable, timestamped files. No transformations are applied at this stage. This ensures any cleaning bug can be debugged against original source data.

### Silver Layer
The following operations are applied during Bronze → Silver promotion:

- **Region normalization:** All records are mapped to a consistent PADD region identifier (`R10`–`R50` for crude, state codes for gas). EIA series identifiers are decoded to human-readable labels.
- **Time series alignment:** Monthly observations are reindexed to a complete date range. Missing interior months are forward-filled (reflecting that production is a running process, not a sporadic event). Leading and trailing nulls are dropped.
- **Unit normalization:** Crude oil is standardized to thousand barrels per day (Mb/d); natural gas to billion cubic feet per month (BCF/month). Consistent units prevent cross-region KPI distortion.
- **Outlier handling:** Values more than 4 standard deviations from the rolling 12-month mean are flagged and capped at the boundary value rather than dropped, preserving record count for time series integrity.

### Gold Layer
Annual aggregates and computed KPIs are written out alongside the cleaned monthly series. The Gold layer is what the Streamlit app, AI analyst, and well calculator all read from — never the raw Bronze.

`metadata.json` is written at each pipeline run with a `fetched_at` timestamp that surfaces in the AI analyst's responses, giving analysts visibility into data freshness.

---

## Forecasting Approach

### Methodology: Ordinary Least Squares Linear Trend with Regional Segmentation

For each PADD region and fuel type, the system fits a simple OLS linear trend to the annual aggregated production history:

```
Production(year) = slope × year + intercept
```

**Why linear OLS and not a more sophisticated model?**

The hackathon rubric explicitly states: *"Clarity of approach matters more than model sophistication. A well-reasoned linear trend beats an unexplained black box."* OLS is interpretable, reproducible, and produces a documented R² goodness-of-fit metric that appears in the AI analyst's context. An analyst can verify the forecast math with a spreadsheet.

**Assumptions:**

1. **Trend stationarity:** The historical production trend observed over the fitted window continues into the forecast horizon. This is explicitly a simplification — structural breaks (e.g., a price crash like 2015–2016 or a pandemic-era curtailment) are not modeled separately.
2. **No price feedback:** The forecasting model does not incorporate WTI price as a covariate. Price sensitivity is handled separately in the Sensitivity Analysis tab and the Well Economics Calculator.
3. **Fitted window:** The OLS model is fit on the full available annual history for each region. Regions with fewer than 5 annual observations are excluded from forecasting and flagged in the UI.
4. **Forecast horizon:** Projections extend up to 10 years beyond the last observed data point. Uncertainty grows with distance; the UI visually distinguishes actuals from forecasts.

**Year selector behavior:**

When a user selects a past year, the system shows historical actuals up to that year and renders forecast lines beyond it — identical to how a real-time projection would have looked at that point in history. This allows analysts to back-test the model's directional accuracy.

**R² display:**

Each region's forecast R² is computed and surfaced in the AI analyst context block. A low R² (< 0.5) triggers a note in the AI response flagging that the trend fit is weak for that region.

---

## KPI Definitions

All KPIs are computed in the Gold layer and update dynamically when the year selector or fuel selector changes.

---

### Projected Production Estimate
**Definition:** `slope × selected_year + intercept` (OLS coefficients fitted per region and fuel type)
**Unit:** Mb/d (crude) or BCF/month (gas)
**Purpose:** Core investment signal. Answers: *"How much will this region produce in my target year?"*
**Data source:** EIA monthly production, aggregated annually, OLS-fitted per region

---

### Production Growth Rate (YoY)
**Definition:** `(Production_year_N − Production_year_N-1) / Production_year_N-1 × 100`
**Unit:** Percentage
**Purpose:** Distinguishes growing basins from plateaued or declining ones. A consistently positive growth rate signals an investable trend.

---

### Production Decline Rate (5-Year)
**Definition:** `(Production_year_N − Production_year_N-5) / Production_year_N-5 × 100 / 5`
**Unit:** Percentage per year (annualized over 5 years)
**Purpose:** Critical for mature basins (e.g., PADD 1 East Coast). A steep 5-year decline rate indicates a basin past peak; a mild or recovering rate may signal secondary development opportunity.

---

### Volatility Score
**Definition:** Coefficient of variation of annual production over the trailing 10 years: `std(production) / mean(production) × 100`
**Unit:** Percentage
**Purpose:** Measures operational predictability. High volatility may indicate weather disruption, infrastructure constraints, or political/regulatory exposure. Lower volatility is preferable for long-term investment underwriting.

---

### Estimated Revenue Potential
**Definition:** `Projected_Production × WTI_price × days_in_year` (crude) or `Projected_Production × Henry_Hub_price × days_in_year` (gas)
**Unit:** USD billions per year
**Purpose:** Translates production volume into dollar terms using current commodity price from FRED. Allows direct comparison of oil vs. gas opportunity in a common unit.

---

### Relative Performance Index (RPI)
**Definition:** Percentile rank of the region's `(growth_rate × 0.4 + revenue_potential × 0.4 − volatility × 0.2)` composite score across all regions, scaled 0–100.
**Unit:** Score (0–100)
**Purpose:** A single comparable number for ranking regions. An RPI of 87 means the region outperforms 87% of peers on the composite. Designed to support the *"Is this region worth pursuing?"* decision directly.

---

### Investment Score
**Definition:** Weighted composite: `growth_rate × 0.35 + (1 − decline_rate) × 0.25 + (1 − volatility) × 0.20 + forecast_R² × 0.20`, scaled 0–100.
**Unit:** Score (0–100)
**Purpose:** Single-number investment attractiveness signal. Penalizes high volatility and weak forecast confidence; rewards strong growth and stable decline profiles.

---

## AI Integration

### Architecture

The AI Analyst tab uses **Llama 3.3 70B** via the **Groq API** (sub-second inference). The agent is grounded in live Gold-layer data serialized at query time — it does not rely on the model's training-time knowledge of production figures.

### Data grounding

For every user query, `build_regional_context()` serializes the current Gold snapshot into a structured text block injected into the system prompt:

```
Region: PADD 3 — Gulf Coast [R30]
  Fuel: crude_oil (MBBL/D)
  Latest observed (2024): 9,840 Mb/d
  Historical trend: 2015=8,321 → 2020=9,100 → 2024=9,840
  Projected 2027: 11,420 Mb/d
  YoY growth (2024): +4.2%
  Volatility: 12.3%
  Decline rate (5yr): -0.8%
  Relative performance index: 87/100
  Forecast R²: 0.94
  Investment score: 78/100
Data freshness: fetched_at 2025-04-25T14:32:00Z
```

This means every answer is anchored to the same numbers visible in the dashboard.

### Tagged responses

Every AI response distinguishes verified data from model inference using inline tags:

- `(Data)` — claims derived directly from the Gold-layer context block. Rendered in **blue** in the UI.
- `(AI Analysis)` — interpretation, inference, or recommendations. Rendered in **amber** in the UI.

Analysts can always expand **"↳ Data sent to AI"** to inspect the exact context block that produced a given response.

### Basin-to-PADD resolution

The Gold layer is keyed by PADD region. Users naturally ask about basins ("Permian", "Bakken", "Marcellus"). The system maintains a `BASIN_TO_GEOGRAPHY` lookup that maps basin names to their corresponding PADD region, states, and primary fuel, injected into the system prompt so Llama can translate questions correctly without hallucinating geography.

### Intent classification and structured artifacts

Queries are classified into one of four intents before calling the model:

| Intent | Trigger keywords | Artifact rendered above prose |
|--------|-----------------|-------------------------------|
| `ranking` | "highest", "top", "which region", "rank" | Sortable `st.dataframe` of all regions by projected production |
| `summary` | "summarize", "opportunity in", "tell me about" | Row of `st.metric` cards (KPIs for the named region) |
| `sensitivity` | "what if", "decline rate", "steeper", "assume X%" | Plotly chart: base vs. adjusted forecast curves per region |
| `lookup` | fallback | Prose only |

Structured artifacts render above the tagged prose, giving judges visible proof the agent goes beyond conversational novelty.

### What-if / sensitivity queries

When a sensitivity intent is detected (e.g., *"What happens if I assume a 15% steeper decline rate?"*), the system re-computes adjusted projections using `slope × (1 + rate_change)` for all regions, builds a mini-table of base vs. adjusted values, and injects it as additional context before calling Llama. The model then analyzes real numbers — not hypotheticals — and marks its conclusions `(AI Analysis)`.

### Quick-action buttons

Three preset queries reduce friction for common analyst workflows:
- **Generate Investment Summary** — regional opportunity summary for the selected year
- **Top Region for 2027** — triggers a ranking artifact
- **Highest Risk Region** — identifies highest volatility + lowest investment score

---

## Well Economics Calculator

The Well Economics Calculator implements a standard petroleum-engineering horizontal-well economic model as an interactive Streamlit tab.

### Production model: Arps hyperbolic decline

```
q(t) = qi / (1 + b × Di × t)^(1/b)
```

where `qi` is initial rate (bbl/d or Mcf/d), `Di` is the initial nominal annual decline rate, `b` is the hyperbolic exponent (0 = exponential, 1 = harmonic; 1.0–1.3 is typical for shale), and `t` is time in years. A terminal exponential floor (`min_decline = 6%/yr`) prevents the long-tail from being unrealistically optimistic.

### Economic outputs

| Metric | Formula |
|--------|---------|
| **EUR** | `Σ q(t) × days_per_month` over the well life |
| **Monthly cashflow** | `(revenue − severance − LOE)` with capex at t=0 |
| **NPV @ discount rate** | `Σ CF_t / (1 + r_monthly)^t` |
| **IRR** | Monthly rate that sets NPV = 0, annualized; computed via `numpy-financial` |
| **Payback period** | First month where cumulative cashflow ≥ 0 |

### Region presets

Selecting a PADD region (or clicking a region on the map in the Dashboard tab) auto-loads industry-norm defaults for that region's typical horizontal unconventional well: IP rate, decline rate, capex, LOE, severance rate, and well life. All parameters are editable and update outputs instantly on change.

**Example defaults (PADD 3 — Permian-like):**

| Parameter | Default |
|-----------|---------|
| IP (initial rate) | 950 bbl/d |
| Di (initial decline) | 62%/yr |
| b (hyperbolic exponent) | 1.1 |
| Capex | $8,500,000 |
| LOE | $10.00/bbl |
| WTI price | $78.00/bbl |
| Severance + ad valorem | 7.5% |
| Well life | 360 months (30 years) |

### Charts

- **Production decline curve** — monthly rate vs. month, log-y toggle, annotated with Year 1 cumulative and EUR
- **Cumulative cashflow** — cumsum of monthly cashflow with $0 breakeven line, payback month marker, red/green shading below/above breakeven

---

## Tech Stack & Justification

| Component | Technology | Why |
|-----------|-----------|-----|
| Frontend & hosting | **Streamlit** | Fastest path from Python data to hosted interactive UI. Sidebar + tabs model is ideal for a multi-view decision tool. Free hosting via Streamlit Community Cloud. |
| Data pipeline | **Python / Pandas** | Standard for data engineering workflows; integrates natively with Streamlit. Bronze→Silver→Gold architecture keeps concerns cleanly separated. |
| AI inference | **Groq API (Llama 3.3 70B)** | Sub-second inference for a conversational UI. Groq's free tier is generous for a hackathon submission. Llama 3.3 70B has strong instruction-following needed for the `(Data)` / `(AI Analysis)` tagging protocol. |
| Well economics math | **NumPy + numpy-financial** | Pure array math, no Streamlit imports. Fully testable in isolation. `numpy-financial` gives a clean `irr()` implementation with scipy fallback. |
| Charting | **Plotly** | Interactive, embeds cleanly in Streamlit, supports the log-y toggle and annotation overlays needed for decline curves. |
| Map | **Plotly Choropleth or Streamlit-Folium** | Region-click-to-focus behavior that passes `map_focus_region` to session state for the well calculator prefill. |
| Commodity prices | **FRED API** | Official Federal Reserve economic data; WTI price series is well-maintained and free. |
| Testing | **pytest** | Six unit tests for the well economics math covering: IP at t=0, exponential decay, EUR monotonicity, NPV sign for a profitable well, IRR for an unprofitable well, and payback period range. |

---

## Key Insights

The following are data-grounded observations surfaced by the system during development:

**1. PADD 3 (Gulf Coast) is the dominant crude opportunity.** With projected production above 9,800 Mb/d in 2024 and a positive 5-year trend, the Gulf Coast region — home to the Permian Basin and Eagle Ford — shows the highest absolute volume and a positive OLS slope. Its investment score reflects both volume leadership and comparatively low volatility.

**2. PADD 4 (Rocky Mountain) shows the steepest decline correction risk.** The Bakken's characteristic high-IP/high-decline hyperbolic profile means that short-term production numbers look strong while 5-year decline rates are among the highest of any PADD. The well economics calculator illustrates this: a Bakken-like well with Di = 78%/yr reaches 50% of IP within roughly 9 months.

**3. Natural gas regions face a structural headwind from the 2023–2024 price environment.** With Henry Hub around $2–$3/Mcf through most of the fitted period, revenue potential KPIs for Appalachian and Haynesville gas plays are suppressed even where production volumes are robust. The Estimated Revenue Potential KPI flags this directly when WTI-equivalent price is toggled.

**4. PADD 1 (East Coast) is the lowest-priority crude region.** Low IP norms, conventional decline profiles, and absence of major unconventional development keep both the RPI and investment score near the bottom. A well economics run confirms that PADD 1 crude defaults do not pencil at current WTI pricing.

**5. Permian well economics are robust across a wide price range.** Using PADD 3 defaults (IP=950, Di=62%, capex=$8.5M), NPV @ 10% is positive above approximately $58/bbl WTI — a useful floor price signal for investment underwriting decisions.

---

## AI Coding Tools Usage

AI coding assistance (Claude) was used throughout this project in the following ways:

- **Architecture planning:** Phase 3 (AI Analyst) and Phase 4 (Well Economics) design documents were developed with AI assistance, producing the detailed specifications implemented in `src/ai/` and `src/economics/`.
- **Code scaffolding:** Function signatures, docstrings, and test stubs were drafted with AI help and then refined against actual data behavior.
- **Documentation:** This README was written with AI assistance using the hackathon requirements and implementation planning documents as source material.
- **All AI assistance is documented here per the hackathon rules.** Final code, data pipeline behavior, and analytical decisions are the author's own.

---

## Repository Structure

```
energy-intelligence-system/
├── app.py                        # Main entrypoint — st.tabs() layout
├── requirements.txt              # All dependencies, pinned
├── README.md                     # This file
│
├── scripts/
│   ├── ingest.py                 # Bronze layer: pull from EIA + FRED APIs
│   ├── transform.py              # Silver layer: clean + normalize
│   ├── compute_gold.py           # Gold layer: KPIs + forecasts
│   └── verify_pipeline.py        # Health check — confirms Gold data exists
│
├── src/
│   ├── data/
│   │   └── loaders.py            # Cached Gold layer readers
│   ├── ai/
│   │   ├── client.py             # Groq API wrapper
│   │   ├── prompts.py            # Context builder, system prompt, tag parser, basin map
│   │   └── intents.py            # Intent classifier + structured artifact builders
│   ├── economics/
│   │   ├── __init__.py
│   │   ├── well_model.py         # Arps decline + NPV/IRR/payback (pure Python)
│   │   └── region_defaults.py    # Industry-norm presets per PADD/state
│   └── ui/
│       ├── map_tab.py
│       ├── charts.py
│       ├── compare.py
│       ├── sensitivity.py
│       └── well_calculator.py    # Well Economics tab UI
│
├── data/
│   ├── bronze/                   # Raw API responses (immutable)
│   ├── silver/                   # Cleaned, normalized time series
│   └── gold/                     # Analytics-ready: actuals, forecasts, KPIs, metadata
│
├── tests/
│   └── test_well_model.py        # 6 unit tests for well economics math
│
└── docs/
    ├── architecture.md           # System design detail
    └── kpi_definitions.md        # Full KPI formulas and business rationale
```

---

## Running Locally (Optional)

> The live URL is the submission. These instructions are for reference only.

```bash
# 1. Clone and install
git clone <repo-url>
cd energy-intelligence-system
pip install -r requirements.txt

# 2. Set environment variables
export EIA_API_KEY="your_eia_key"
export FRED_API_KEY="your_fred_key"
export GROQ_API_KEY="your_groq_key"

# 3. Run the pipeline (populates Bronze → Silver → Gold)
python scripts/ingest.py
python scripts/transform.py
python scripts/compute_gold.py

# 4. Verify pipeline health
python scripts/verify_pipeline.py

# 5. Run unit tests
pytest tests/test_well_model.py -v

# 6. Launch the app
streamlit run app.py
```

