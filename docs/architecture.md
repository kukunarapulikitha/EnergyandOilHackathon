# Architecture Overview

> This documents how the app was actually implemented, including all phases through final polish.
> Compare against `planning/PLANNING.md` to see where the plan changed and why.

---

## Final Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| App framework | Python + Streamlit | Frontend, UI state, 3-tab layout |
| Data processing | Pandas | Cleaning, normalization, KPI computation |
| Forecasting | scikit-learn (LinearRegression) | OLS per region and fuel type |
| Visualization | Plotly | Choropleth map, trend charts, heatmaps, decline curves |
| AI inference | Groq → xAI → Gemini (fallback chain) | Conversational analyst, grounded in Gold data |
| Commodity prices | EIA WTI Spot Price (`petroleum/pri/spt`) | Live WTI context for revenue KPIs |
| Production data | EIA Open Data API v2 | Primary source for oil and gas production by state/PADD |
| Well economics | Pure-Python Arps decline + NPV/IRR/payback | Standalone well underwriting calculator |
| Hosting | Streamlit Community Cloud | Live deployment, environment variable management |

**What was dropped from the plan:**
- `DuckDB` — Pandas was sufficient; DuckDB added complexity with no benefit at this data volume
- `FastAPI` — Not needed; Streamlit's rerun model handles all interactivity natively
- `Prophet` — Replaced by scikit-learn OLS. OLS R² is directly interpretable and maps to rubric clarity preference
- `LangChain` — Replaced by custom `src/ai/`. Direct SDK gives full control over prompt structure and context injection

---

## Folder Structure

```
energy-hackathon-fresh/
├── app.py                        # Entrypoint — st.tabs() layout (Dashboard / Well Economics / AI Analyst)
├── requirements.txt
├── .env.example                  # Template — copy to .env, fill API keys
│
├── scripts/
│   ├── verify_pipeline.py        # End-to-end Bronze → Silver → validate → DQS → Gold health check
│   └── explore_data.py           # 6 canned pandas reports for manual inspection
│
├── src/
│   ├── data/
│   │   ├── schema.py             # Silver schema contract + validate_schema()
│   │   ├── eia_client.py         # Bronze: EIA API v2 — crude, gas, WTI; raw JSON audit trail
│   │   ├── clean.py              # Silver: normalized, schema-enforced DataFrames
│   │   ├── validate.py           # 6 validation checks + ValidationReport
│   │   ├── quality.py            # DQS 0–100 (completeness + consistency + freshness)
│   │   └── gold.py               # Gold: regional_actuals.csv + region_forecasts.csv
│   │
│   ├── forecast/
│   │   └── linear.py             # OLS regression per (region, fuel) + ForecastResult dataclass
│   │
│   ├── kpi/
│   │   └── calculations.py       # 7 KPIs: Projected Production, YoY Growth, Decline Rate,
│   │                             # Revenue Potential, Volatility, RPI, Investment Score, Forecast R²
│   │
│   ├── ai/
│   │   ├── client.py             # Three-provider fallback: Groq → xAI Grok → Gemini
│   │   ├── prompts.py            # build_regional_context(), build_system_prompt(),
│   │   │                         # parse_tagged_response(), resolve_basin(), detect_what_if(),
│   │   │                         # compute_sensitivity_context(), BASIN_TO_GEOGRAPHY
│   │   └── intents.py            # classify_intent(), build_artifact() → table/metrics/chart
│   │
│   ├── economics/
│   │   ├── well_model.py         # Arps hyperbolic decline + EUR + monthly_cashflow + NPV + IRR + payback
│   │   └── region_defaults.py    # Industry-norm presets for R10–R50 + TX/PA/LA/OK/WV gas states
│   │
│   └── ui/
│       ├── charts.py             # Plotly: actuals/forecast split, sensitivity heatmap, small multiples
│       ├── data_loader.py        # st.cache_data readers for Gold + Silver + metadata
│       ├── provenance.py         # Provenance tooltips + EIA data sources panel (with DQS)
│       ├── export.py             # Formula-driven 4-sheet Excel workbook builder
│       ├── map.py                # Choropleth + bubble map, basin/rig overlays, region-click focus
│       ├── rankings.py           # Ranked bar chart (sidebar)
│       ├── badges.py             # Region status badge classifier
│       └── well_calculator.py    # Well Economics Streamlit UI
│
├── tests/
│   ├── test_pipeline.py          # 25+ pipeline sanity tests
│   └── test_well_model.py        # 13 well model math tests
│
└── docs/
    ├── architecture.md           # This file
    ├── kpi_definitions.md        # Full KPI formulas + business rationale
    ├── reflection.md             # Build retrospective + future roadmap
    ├── walkthrough.md            # Demo video link
    ├── schema_contract.md        # Silver-layer canonical schema
    └── data_exploration.md       # Step 0 data exploration findings
```

---

## Cross-Tab Data Flow

```
EIA Open Data API v2
       │
       ▼ src/data/eia_client.py
  data/bronze/          ← raw JSON, immutable, timestamped audit trail
       │
       ▼ src/data/clean.py
  data/silver/          ← normalized region IDs, aligned monthly time series, nulls handled
    ├── production_monthly.csv
    ├── production_annual.csv
    └── wti_prices.csv
       │
       ▼ validate_dataset() → src/data/quality.py (DQS)
       │
       ▼ src/data/gold.py
  data/gold/
    ├── regional_actuals.csv    ← FACTS: observed production + derived KPIs
    └── region_forecasts.csv    ← MODEL PARAMS: slope/intercept/R² per (region, fuel)
       │
       ▼ src/ui/data_loader.py  [st.cache_data — loaded once per session]
       │
  ┌────┼──────────────────────────┬─────────────────────────┐
  │    │                          │                         │
  ▼    ▼                          ▼                         ▼
Dashboard Tab              Well Economics Tab         AI Analyst Tab
 Map (click → focus)        Arps decline curve         classify_intent()
 Forecast chart             NPV / IRR / payback          │
 Sensitivity heatmap        Region preset auto-fill       ▼
 Compare panel                                        build_artifact()
 Excel export                                         [table/metrics/chart]
                                                          │
                                                          ▼
                                                      build_regional_context()
                                                      + resolve_basin()
                                                      + detect_what_if()
                                                      + compute_sensitivity_context()
                                                          │
                                                          ▼
                                                      build_system_prompt()
                                                          │
                                                          ▼
                                                      GroqClient.chat()
                                                      [Groq → xAI → Gemini fallback]
                                                          │
                                                          ▼
                                                      parse_tagged_response()
                                                      → (Data) blue blocks
                                                      → (AI Analysis) amber blocks
                                                      → follow-up question chips
                                                      → "Context sent to AI" expander
```

---

## Dashboard Layout (Final)

```
st.tabs(["📊 Dashboard", "🛢️ Well Economics", "🤖 AI Analyst"])

Dashboard tab order:
  1. Summary KPI strip (Total Production, Avg YoY Growth, Avg R², Avg Investment Score)
     + visible insight captions (top region, growth count, reliability signal)
     + 📥 Export to Excel button
  2. 🗺️ Production Map  ← map click updates (3) and (4) simultaneously
  3. 📈 Regional Forecast Chart  ← auto-selects clicked region
  4. 🎛️ Sensitivity Analysis  ← auto-selects clicked region
  5. 🆚 Compare regions (expander, collapsed by default)

Sidebar (global, affects all dashboard components):
  - Fuel type toggle (crude / gas)
  - Year slider (data_min → 2035)
  - Region multiselect
  - Regional Rankings bar chart (live)
```

---

## Map → Chart Integration (Click Flow)

```
User clicks state on map
       │
       ▼ resolve_clicked_region() in src/ui/map.py
st.session_state["map_focus_region"] = region_id
st.session_state["map_focus_fuel"]   = fuel
       │
       ▼ On next Streamlit rerun:
  Regional Forecast Chart  → default_idx set to clicked region
  Sensitivity Analysis     → default_idx set to clicked region
  Well Economics tab       → loads region's industry-norm preset defaults
```

All three components read from the same `map_focus_region` session state key.

---

## AI Integration Design

### Grounding
At every query, `build_regional_context()` serializes the live Gold-layer KPI snapshot into a structured text block injected into the system prompt. The model cannot fall back on training-time figures — it can only cite numbers present in the context. The "Context sent to AI" expander makes this verifiable by anyone.

### Transparency — tagged responses
The system prompt instructs the model to prefix every data-derived claim with `(Data)` and every interpretive statement with `(AI Analysis)`. `parse_tagged_response()` splits on these tags:
- **Blue blocks** — verified, traceable to the Gold layer
- **Amber blocks** — model inference or recommendation

### Structured artifacts — beyond prose
Before calling the model, `classify_intent()` pre-computes a structured widget:

| Intent | Trigger keywords | Artifact |
|--------|-----------------|---------|
| `ranking` | "highest", "top", "which region" | `st.dataframe` — regions ranked by projected production |
| `summary` | "summarize", "opportunity in" | `st.metric` row — live KPI cards for the named region |
| `sensitivity` | "what if", "decline rate", "15%" | Plotly chart — base vs. adjusted forecast curves |
| `lookup` | fallback | Prose only |

Artifacts render above the prose response — visible proof the agent produces structured decision-support, not just chat.

### Follow-up question chips
After each AI response, 3 contextual follow-up questions are shown as clickable buttons. Intent-aware: ranking responses suggest comparison questions; sensitivity responses suggest resilience questions. Clicking fires the question through the same pipeline with full context grounding.

### Three-provider AI fallback chain
```
Groq (Llama 3.3 70B)  →  xAI (Grok: grok-2-1212 / grok-beta)  →  Gemini (2.0-flash / 1.5-flash / pro)
```
Each provider is skipped if its key is missing. On API error, the chain falls through to the next. `GroqUnavailable` is raised only when all configured providers fail — caught in the UI and shown as a friendly amber message, never crashing the app.

### What-if handling
`detect_what_if()` scans for decline-rate scenario language. When detected, `compute_sensitivity_context()` re-runs `slope × (1 + rate_change)` for all regions, builds an adjustment table, and injects real numbers into context — the model analyzes actual figures, not hypotheticals.

### Basin-to-PADD resolution
`BASIN_TO_GEOGRAPHY` in `prompts.py` maps "Permian", "Bakken", "Marcellus" etc. to the correct PADD and fuel type. Injected into the system prompt so Llama translates basin questions to the correct data slice without hallucinating geography.

---

## Well Economics Calculator

A standalone tab implementing standard petroleum-engineering well economics:

| Component | Implementation |
|-----------|---------------|
| Production decline | Arps hyperbolic `q(t) = qi / (1 + b·Di·t)^(1/b)` with terminal exponential switch |
| EUR | Cumulative production over well life |
| Monthly cashflow | Revenue − severance − LOE − CAPEX at t=0 |
| NPV | Monthly-discounted at user-specified annual rate |
| IRR | `numpy-financial.irr` with `scipy.optimize.brentq` fallback |
| Payback | First month cumulative cashflow ≥ 0 |
| Region presets | Industry-norm defaults for R10–R50 (crude) + TX/PA/LA/OK/WV (gas) |
| Map prefill | Clicking a region on the Dashboard map auto-loads that region's well defaults |

---

## What Changed From the Plan

| Area | Planned | Actual | Why |
|------|---------|--------|-----|
| **Forecasting** | Prophet or OLS | OLS only | Annual aggregates don't benefit from Prophet; OLS R² is directly interpretable |
| **Data layer** | Pandas + DuckDB | Pandas only | Data volume fit in memory; DuckDB added overhead with no gain |
| **API layer** | FastAPI optional | Not used | Streamlit rerun model handled all interactivity natively |
| **AI orchestration** | LangChain | Custom `src/ai/` | LangChain abstractions obscure the prompt structure that needed precise control |
| **AI fallback** | Single provider | Groq → xAI → Gemini chain | Groq org restriction prompted multi-provider fallback; now more robust |
| **Sensitivity analysis** | Listed as first-to-cut | Shipped | Trivial extension of the OLS model; became a Tier 2 differentiator |
| **Well Economics** | Phase 4 stretch | Fully shipped | Arps curve + NPV/IRR/payback + map prefill + 13 unit tests — all complete |
| **Data Quality Score** | In header | Moved to Data Provenance panel | Header was too cluttered; DQS is still accessible but not primary navigation |
| **Excel import** | Added in polish | Removed | User only needed Excel export; import added complexity with no use case |
| **Theme toggle** | Added in polish | Removed | CSS injection via `unsafe_allow_html` didn't cover all Streamlit internal components |

---

## Source Repository

- **Author repo:** https://github.com/kukunarapulikitha/EnergyandOilHackathon
- **Submission repo:** https://github.com/Community-Dreams-Foundation-Hackathons/energy-intelligence-system-kukunarapulikitha
