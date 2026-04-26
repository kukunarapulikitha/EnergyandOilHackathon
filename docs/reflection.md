# Reflection

## What I Built

**Dashboard Tab**
A regional production explorer with a choropleth map, year selector, and sidebar controls for fuel type and region selection. The map supports three overlay modes — production volume, YoY growth, and relative performance — plus a bubble mode that encodes both volume and growth simultaneously. Clicking a region focuses the entire dashboard on it. A regional rankings panel updates dynamically alongside the map.

The summary strip surfaces four live KPIs at a glance: Total Production, Average YoY Growth, Average Forecast R², and Average Investment Score. All values update when the year selector is dragged.

A Compare Regions panel and Sensitivity Analysis view are also included. The sensitivity tab shows a heat map of how Projected Production Estimate changes as the decline rate assumption shifts — color-coded from weak to strong opportunity.

**AI Analyst Tab**
A conversational agent powered by Llama 3.3 70B via the Groq API, grounded in live Gold-layer data serialized at query time. Key features that work:
- `(Data)` vs `(AI Analysis)` tagging on every response — blue for verified figures, amber for inference — rendered as color-coded blocks in the UI
- Intent classification that auto-generates structured artifacts (ranked tables, KPI metric cards, sensitivity charts) above the prose response before the model is even called
- Basin-to-PADD resolution so users can ask about "Permian" or "Marcellus" and get answers keyed to the correct PADD region
- What-if sensitivity queries that recompute adjusted slopes for all regions and inject real numbers into the model's context rather than answering hypothetically
- A "Data sent to AI" expander on every response so judges and analysts can verify exactly what grounded each answer
- Four quick-action buttons for the most common analyst queries

**Data Pipeline**
A full Bronze → Silver → Gold medallion architecture pulling from the EIA Open Data API and FRED. The Gold layer writes `metadata.json` with a `fetched_at` timestamp that surfaces in the UI header and in AI responses. A live refresh button re-runs ingestion on demand and degrades gracefully to cached data if the API is unavailable.

**What doesn't work / known gaps**
- The Well Economics Calculator (Phase 4) was designed and fully specced out — Arps decline curve, NPV/IRR/payback math, region presets, and Streamlit UI — but was not completed within the submission window. The tab is not present in the live app.
- The Data Quality score shown in the header (69%) reflects missing coverage for some PADD regions in the crude oil dataset; this is documented in the Data Sources panel.
- Natural gas coverage is scoped to the top 5 producing states (TX, PA, LA, OK, WV), representing ~80% of U.S. marketed production. Unshaded states are out of scope, not zero — this is noted in the UI.

---

## What I'd Do Differently

**Ship the Well Economics Calculator.** This was the highest-priority Tier 2 item and the one I'm most disappointed not to have completed. The Arps decline curve math, NPV/IRR/payback functions, and region preset defaults were all fully designed. With one more day I would have built and presented it cleanly — a proper input panel with sliders, five metric output cards, a decline curve chart, and a cumulative cashflow chart with payback annotation. That tab would have made the investment decision loop complete: from regional screening on the dashboard, to AI-assisted comparison, to single-well underwriting in the calculator.

**Cleaner UI separation between actual and forecast values.** The year selector works, but the visual distinction between historical actuals and OLS-projected values could be sharper — a dashed line style and a labeled "Forecast →" annotation at the cutoff point would make the forecasting story clearer during a live demo.

**Wider crude oil regional coverage.** The current crude data covers PADD regions at a high level. Pulling state-level crude data from Texas RRC and NDIC for the Bakken would have added granularity that makes the map more actionable for analysts targeting specific basins.

**Tighter AI response formatting.** The (Data) / (AI Analysis) tag parsing occasionally leaves backtick artifacts visible in the rendered output. A more robust regex parser and a post-processing cleanup pass would eliminate those before display.

---

## AI Tools Used

**Claude (Anthropic)** was the primary tool used throughout this project.

- **Architecture and planning:** Used Claude to design the full Bronze → Silver → Gold medallion pipeline, define the six KPI formulas, and structure the three-tab app layout before writing any code. The Phase 3 and Phase 4 planning documents were developed with Claude, covering every component down to function signatures and data flow.
- **AI Analyst module:** Claude scaffolded `src/ai/prompts.py` — specifically the `build_regional_context()` serializer, the `(Data)` / `(AI Analysis)` tagging protocol in the system prompt, the basin-to-PADD lookup table, and the `parse_tagged_response()` tag splitter. The intent classifier and structured artifact builders in `src/ai/intents.py` were also designed in collaboration with Claude.
- **Well Economics math:** Claude drafted the Arps decline curve implementation, the monthly cashflow function, and the NPV/IRR/payback logic in `src/economics/well_model.py`, along with the six pytest unit tests.
- **Documentation:** The README, this reflection, `docs/walkthrough.md`, and the walkthrough video script were all written with Claude using the actual app screenshots and planning documents as source material.
- **Debugging:** Used Claude to debug surrogate key issues in the Gold-layer KPI join logic and to trace a rendering issue where Streamlit was re-running the Groq call on every widget interaction instead of reading from session state.

All AI assistance is documented here per the hackathon rules. Data pipeline behavior, analytical decisions, and final code are my own.
