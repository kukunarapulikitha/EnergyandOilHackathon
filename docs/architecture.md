# Architecture Overview

> This documents how the app was actually implemented.
> Compare against `planning/PLANNING.md` to see where the plan changed and why.

---

## Final Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| App framework | Python + Streamlit | Frontend, UI state, tab layout |
| Data processing | Pandas | Cleaning, normalization, KPI computation |
| Forecasting | scikit-learn (LinearRegression) | OLS per region and fuel type |
| Visualization | Plotly | Choropleth map, trend charts, sensitivity charts |
| AI inference | Groq API — Llama 3.3 70B Versatile | Conversational analyst, grounded in Gold data |
| Commodity prices | FRED API | Live WTI price context for revenue KPIs |
| Production data | EIA Open Data API | Primary source for oil and gas production by state/PADD |
| Hosting | Streamlit Community Cloud | Live deployment, environment variable management |

**What was dropped from the plan:**
- `DuckDB` — Pandas was sufficient for the data volume; DuckDB added complexity with no benefit at this scale
- `FastAPI` — Not needed; Streamlit's rerun model handles all interactivity natively
- `Prophet` — Replaced by scikit-learn OLS. Prophet is overkill for annual regional aggregates and harder to explain; OLS R² is interpretable and maps directly to the rubric's preference for clarity over sophistication
- `LangChain` — Replaced by a custom `src/ai/` module. LangChain abstraction was unnecessary overhead for a single-model, single-provider setup; direct Groq SDK gives full control over prompt structure and context injection

---

## Folder Structure

```
src/
├── app.py                        # Entrypoint — st.tabs() layout, sidebar, global state
├── requirements.txt
│
├── scripts/
│   ├── ingest.py                 # Bronze: pulls EIA + FRED via API, writes raw files
│   ├── transform.py              # Silver: normalizes regions, aligns time series, handles nulls
│   ├── compute_gold.py           # Gold: computes KPIs, fits OLS per region, writes forecasts
│   └── verify_pipeline.py        # Health check — confirms Gold layer exists and is fresh
│
├── data/
│   └── loaders.py                # Cached st.cache_data readers for actuals, forecasts, annual, metadata
│
├── ai/
│   ├── client.py                 # Thin Groq SDK wrapper
│   ├── prompts.py                # build_regional_context(), build_system_prompt(),
│   │                             # parse_tagged_response(), resolve_basin(), detect_what_if(),
│   │                             # compute_sensitivity_context(), BASIN_TO_GEOGRAPHY lookup
│   └── intents.py                # classify_intent(), build_artifact() — table / metrics / chart
│
└── ui/
    ├── map_tab.py                # Choropleth + bubble map, region-click focus
    ├── charts.py                 # Production trend charts with actual/forecast split
    ├── compare.py                # Side-by-side regional comparison panel
    └── sensitivity.py            # Sensitivity heat map — decline rate vs. projected production

docs/
├── walkthrough.md
├── architecture.md               # This file
├── kpi_definitions.md
└── reflection.md

data/
├── bronze/                       # Raw API JSON — immutable, timestamped
├── silver/                       # Cleaned Parquet — consistent IDs, aligned dates
└── gold/                         # actuals_df, forecasts_df, annual_df, metadata.json
```

---

## Cross-Tab Data Flow

```
EIA API + FRED API
       │
       ▼ scripts/ingest.py
  data/bronze/          ← raw JSON, never modified after write
       │
       ▼ scripts/transform.py
  data/silver/          ← normalized region IDs, aligned monthly time series, nulls handled
       │
       ▼ scripts/compute_gold.py
  data/gold/            ← actuals_df, forecasts_df (OLS slope/intercept per region),
                           annual_df (KPIs), metadata.json (fetched_at timestamp)
       │
       ▼ src/data/loaders.py  [st.cache_data — loaded once per session]
       │
  ┌────┴────────────────────────────────────┐
  │                                         │
  ▼                                         ▼
Dashboard Tab                          AI Analyst Tab
  sidebar controls (fuel, year,          classify_intent(query)
  regions) → filter actuals_df             │
  → render map, charts, KPI strip          ▼
  → compare panel                      build_artifact()
  → sensitivity heat map                [table / metrics / chart]
                                          │
                                          ▼
                                      resolve_basin() + detect_what_if()
                                          │
                                          ▼
                                      build_regional_context(actuals_df,
                                        forecasts_df, annual_df)
                                          │
                                          ▼
                                      build_system_prompt(context, year)
                                          │
                                          ▼
                                      GroqClient.chat()  [Llama 3.3 70B]
                                          │
                                          ▼
                                      parse_tagged_response()
                                      → (Data) blue blocks
                                      → (AI Analysis) amber blocks
                                      → "Data sent to AI" expander
```

The sidebar (fuel type, year selector, region multiselect) is global — it controls the data slice passed to every Dashboard component. The AI Analyst reads the same Gold layer independently and serializes its own context snapshot at query time, so AI answers always reflect the current sidebar state.

---

## AI Integration Design

The AI module is built around three principles: **grounding, transparency, and structured output.**

### Grounding
At every query, `build_regional_context()` serializes the live Gold-layer KPI snapshot into a structured text block injected into the system prompt. The model has no access to the internet and cannot fall back on training-time production figures — it can only cite numbers present in the context block. The "Data sent to AI" expander in the UI makes this verifiable.

### Transparency — tagged responses
The system prompt instructs Llama to prefix every data-derived claim with `(Data)` and every interpretive statement with `(AI Analysis)`. `parse_tagged_response()` splits on these tags and the UI renders them as distinct colored blocks:
- **Blue** — verified, traceable to the Gold layer
- **Amber** — model inference or recommendation

### Structured artifacts — beyond prose
Before calling the model, `classify_intent()` identifies the query type and `build_artifact()` pre-computes a structured widget:

| Intent | Trigger | Artifact |
|--------|---------|---------|
| `ranking` | "highest", "top", "which region" | `st.dataframe` — regions ranked by projected production |
| `summary` | "summarize", "opportunity in" | `st.metric` row — KPI cards for the named region |
| `sensitivity` | "what if", "decline rate", "15%" | Plotly chart — base vs. adjusted forecast curves |
| `lookup` | fallback | Prose only |

Artifacts render above the prose response — visible proof the agent produces structured decision-support output, not just chat.

### What-if handling
When a sensitivity intent is detected, `compute_sensitivity_context()` re-runs `slope × (1 + rate_change)` for all regions, builds a mini adjustment table, and injects it as additional context before the Groq call. The model then analyzes real adjusted figures rather than reasoning about a hypothetical.

### Basin-to-PADD resolution
Gold data is keyed by PADD region (`R10`–`R50`, state codes for gas). Users ask about basins. A `BASIN_TO_GEOGRAPHY` lookup in `prompts.py` maps "Permian", "Bakken", "Marcellus" etc. to the correct PADD, states, and fuel type. This mapping is injected into the system prompt so Llama translates basin questions to the correct data slice without hallucinating geography.

---

## What Changed From the Plan

| Area | Planned | Actual | Why it changed |
|------|---------|--------|----------------|
| **Forecasting model** | Prophet or scikit-learn | scikit-learn OLS only | Prophet requires daily/weekly data to show its strengths; annual regional aggregates don't benefit from it. OLS is more interpretable and maps directly to the rubric's stated preference. |
| **Data layer** | Pandas + DuckDB | Pandas only | Data volume (~5 states × ~20 years of monthly records) fit comfortably in memory. DuckDB added setup overhead with no performance gain at this scale. |
| **API / modularity** | Optional FastAPI | Not used | Streamlit's rerun model handled all interactivity natively. FastAPI would have added a client-server boundary with no user-facing benefit. |
| **AI orchestration** | LangChain | Custom `src/ai/` module | LangChain's abstractions obscure prompt structure and context injection — exactly what needed to be controlled precisely for the `(Data)` / `(AI Analysis)` tagging protocol. Direct Groq SDK took 30 lines and gave full visibility. |
| **AI fallback** | Mixtral 8x7B or local Ollama | Not implemented | Groq's free tier was reliable throughout the build window. A fallback path was not needed in practice. |
| **Sensitivity analysis** | Listed as "first to cut" | Shipped | Sensitivity was planned as a cut candidate but turned out to be a natural extension of the OLS model — adjusting the slope coefficient is trivial once the forecasting layer is built. It became a Tier 2 differentiator rather than a cut. |
| **Well Economics Calculator** | Not in original plan (added later as Phase 4) | Designed, not shipped | Full spec was written — Arps decline curve, NPV/IRR/payback, region presets, Streamlit UI, 6 unit tests — but not completed within the submission window. This is the primary gap between planned and delivered scope. |
| **LangChain for AI** | Planned as optional orchestration layer | Replaced entirely | See AI orchestration row above. The custom module is simpler, more transparent, and easier to debug. |
| **Three-phase delivery** | Day 1–2: data, Day 3–4: UI, Day 5: AI + polish | Roughly followed | The Bronze→Silver→Gold pipeline took longer than expected on Day 1 due to EIA API response format inconsistencies across series. This compressed Day 5 AI polish time, which is why the Well Economics tab was not completed. Honestly I would present it much neatly 


##Link for actual git - https://github.com/kukunarapulikitha/EnergyandOilHackathon
