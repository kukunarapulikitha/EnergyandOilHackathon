# Reflection

## What I Built

### Dashboard Tab
A regional production explorer with a Plotly choropleth map at the top, year selector, and sidebar controls for fuel type and region selection. The map supports three overlay modes — production volume, YoY growth, and relative performance — plus a bubble mode that encodes both volume and growth simultaneously. Clicking a state on the map simultaneously pre-selects that region in the Regional Forecast Chart below it and in the Sensitivity Analysis heatmap, so a single click updates the full analytical view.

The summary strip surfaces four live KPIs at a glance — Total Production, Avg YoY Growth, Avg Forecast R², Avg Investment Score — with visible insight captions below each card (top-producing region, growth count, R² reliability signal, investment signal). All values update when the year selector is dragged. An Excel export button is always visible in the summary panel.

A Compare Regions panel and Sensitivity Analysis view are included. The sensitivity heatmap shows how Projected Production Estimate and Revenue Potential change as WTI price and decline-rate assumptions shift — color-coded from weak to strong opportunity, with the map-clicked region pre-selected automatically.

### Well Economics Tab
A standalone single-well underwriting calculator implementing the full Arps hyperbolic decline model with a terminal exponential switch. Users set initial production rate (IP), decline rate (Di), hyperbolic exponent (b), CAPEX, LOE, commodity price, severance, and discount rate. The tab outputs five metric cards (EUR, NPV, IRR, Payback, Total Revenue), a production decline curve, and a cumulative cashflow chart with payback annotation and red/green shading. Clicking a state on the dashboard map auto-loads that region's industry-norm defaults into the calculator. Covered by 13 unit tests.

### AI Analyst Tab
A conversational agent powered by a three-provider fallback chain (Groq Llama 3.3 70B → xAI Grok → Google Gemini), grounded in live Gold-layer data serialized at query time. Key features:

- `(Data)` vs `(AI Analysis)` tagging on every response — blue for verified EIA figures, amber for inference — rendered as color-coded blocks
- Intent classification pre-computes structured artifacts (ranked tables, KPI metric cards, sensitivity charts) above the prose response before the model is called
- Basin-to-PADD resolution: users ask about "Permian" or "Marcellus", get answers keyed to the correct PADD region
- What-if sensitivity queries that recompute adjusted slopes for all regions and inject real adjusted numbers into context rather than answering hypothetically
- Follow-up question chips: 3 contextual next-step questions appear after each response, intent-aware (ranking → comparison questions; sensitivity → resilience questions)
- Quick-action panel with 4 pre-built queries for instant demo flow
- "Context sent to AI" expander on every response: verifiable proof of grounding
- Clear chat button, legend inline in header, clean empty state

### Data Pipeline
Full Bronze → Silver → Gold medallion architecture pulling from EIA Open Data API v2 (crude, natural gas, WTI spot price — no external dependencies). Six validation checks run between Silver and Gold; the Data Quality Score (completeness, consistency, freshness) is available in the Data Provenance panel. Live Refresh button re-runs ingestion on demand; degrades gracefully to cached data if EIA API key is missing.

---

## What Was Improved (Post-Initial Build)

- **Map click → multi-component sync**: clicking a state now pre-selects the region in the Forecast Chart, Sensitivity Analysis, and Well Economics tab simultaneously
- **Number formatting**: large natural gas figures formatted with K/M suffix and unit labels (e.g. "2.6M MMcf") instead of raw integers that truncated in metric cards
- **KPI insight captions**: visible text below each metric card showing top region, growth count, R² reliability label — not just tooltips
- **AI fallback chain**: Groq → xAI → Gemini with per-provider model candidate lists; resilient to model deprecations and org restrictions
- **Follow-up question chips**: contextual next-step questions auto-generated per AI response intent
- **Data Quality Score**: removed from header (clutter) and kept in the Data Provenance expander
- **Map and forecast order**: map moved above forecast chart for more natural top-down exploration flow

---

## What I'd Do Differently

### 1. UI in TypeScript + Next.js
Streamlit is fast to build in but has real UI ceiling. For the next iteration, the frontend should be a proper Next.js app consuming a FastAPI backend:
- Full layout control — grid, panels, custom chart components
- React state management for instant cross-component updates without full-page reruns
- shadcn/ui + Recharts or D3.js for production-quality visualizations
- Separate deploy cycles for frontend and API — easier to scale and test

### 2. Additional Forecasting Methods
The current OLS linear model is interpretable and satisfies the rubric, but the next version should add:
- **ARIMA / SARIMA** — captures seasonal patterns in monthly production data
- **Prophet** — handles trend breaks (e.g., COVID production drops) and uncertainty bands automatically
- **Exponential smoothing (Holt-Winters)** — better for basins with recent inflection points
- **Model comparison panel**: show all four forecasts side-by-side per region with R², RMSE, and MAE so analysts can choose the most appropriate model for each basin

### 3. Persistent AI Analyst Memory
The current AI Analyst has session-level memory only — the conversation resets on page refresh. The next iteration should include:
- **Cross-session conversation history** persisted to a lightweight database (SQLite or Supabase)
- **User preferences stored as memory**: preferred regions, risk tolerance, typical investment horizon
- **Analyst profile**: "you previously showed interest in Permian Basin for crude oil at $75 WTI" injected into context automatically
- **Saved analyses**: ability to name and recall a previous AI session ("Q2 2025 Bakken review")
- Implementation: store message history + metadata per user session ID; inject a summary of past sessions as an additional context block in the system prompt

### 4. Wider Data Coverage
- State-level crude data (Texas RRC, NDIC for Bakken) for sub-PADD granularity
- International comparisons: Permian vs. Vaca Muerta, Eagle Ford vs. North Sea
- Rig count time series (not just Baker Hughes snapshot) for leading indicator signals
- Well-level production data from FracFocus / DrillingInfo for bottom-up EUR validation

### 5. Real-Time Price Integration
Replace the cached EIA WTI snapshot with a live streaming price feed (CME Group API or Bloomberg) so revenue potential KPIs update intraday rather than on pipeline refresh.

---

## AI Tools Used

**Claude (Anthropic)** was the primary development tool throughout.

- **Architecture and planning**: Bronze → Silver → Gold medallion design, KPI formulas, three-tab app layout, Phase 3 AI agent spec and Phase 4 Well Economics spec
- **AI Analyst module**: `build_regional_context()` serializer, `(Data)` / `(AI Analysis)` tagging protocol, basin-to-PADD lookup, `parse_tagged_response()`, intent classifier, structured artifact builders, follow-up question chip system, three-provider fallback chain
- **Well Economics**: Arps decline curve implementation, monthly cashflow + NPV/IRR/payback math, region preset defaults, 13 unit tests
- **UI polish**: number formatting helper, KPI insight captions, map-click multi-component sync, sensitivity analysis integration
- **Documentation**: README, architecture.md, kpi_definitions.md, this reflection, walkthrough script

All AI assistance is documented here per hackathon rules. Data pipeline behavior, analytical decisions, and final code are my own.
