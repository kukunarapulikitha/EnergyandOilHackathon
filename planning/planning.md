# Planning Document

---

## Tech Stack

**Framework / Language:**

* **Python + Streamlit (frontend + app layer)**
* **Pandas / DuckDB (data layer)**
* **Optional FastAPI (if needed for modularity)**

**Why did you choose this stack?**
I chose Python and Streamlit to prioritize **speed of development and tight integration between data processing, forecasting, and UI**. This allows me to build a fully functional decision-support system within the 5-day constraint without overhead from complex frontend frameworks.
Streamlit also enables **real-time interaction (year selector, region comparison)**, which is critical for the forecasting requirement.

---

**Key Libraries:**

* `pandas` – data cleaning and transformation
* `requests` – API ingestion (EIA data)
* `prophet` or `scikit-learn` – forecasting
* `plotly` – interactive visualizations
* `numpy` – numerical computations
* `langchain` (optional) – AI orchestration

---

**AI Provider:**

* **Llama 3.3 70B Instruct (open-source) — served via Groq API (free tier)**
* Fallback: **Mixtral 8x7B Instruct** or local **Ollama (Llama 3.1 8B)** if rate-limited

**Why?**
Llama 3.3 70B is a fully open-source model with strong reasoning performance on structured numerical tasks, comparable to proprietary GPT-class models on financial/analytical benchmarks. Serving it through Groq gives extremely fast inference (sub-second) on a free tier — ideal for an interactive dashboard where AI responses must feel snappy. Keeping the model open-source also means the system is **reproducible, transparent, and cost-free**, which aligns with the hackathon's free-source constraint.

It enables:

* Automated regional investment summaries
* Context-aware Q&A grounded in live data
* Clear data-backed vs. model-inference distinction via structured prompting

The focus is on **augmenting decision-making**, not just chat functionality.

---

## Phases & Priorities

> I am prioritizing a **fully working Tier 1 system first**, then layering AI and advanced features.

| Phase                     | Target Dates | Goals                                                                                                                                 |
| ------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **1 – Data & Foundation** | Day 1–2      | Ingest EIA data, clean and normalize datasets, design structured data layer, implement baseline forecasting model                     |
| **2 – Core System Build** | Day 3–4      | Build Streamlit UI, integrate forecasting with year selector, implement required KPI (Projected Production), enable region comparison |
| **3 – AI + Polish**       | Day 5        | Add AI-powered investment summaries + Q&A, refine UI/UX, deploy app, create walkthrough video and documentation                       |

---

## What I'll Cut If Time Is Short

* **First to cut:**

  * Advanced AI features (e.g., anomaly detection, multi-agent workflows)
  * Sensitivity analysis

* **Second to cut:**

  * Additional custom KPIs beyond 1–2 strong ones

* **Last to cut (must remain):**

  * Clean data pipeline
  * Forecasting engine with year selector
  * Projected Production KPI
  * At least one meaningful AI feature (investment summary)

---

## Open Questions / Risks

* **Data consistency risk:**
  EIA datasets may have inconsistent region naming or missing values, requiring additional preprocessing.

* **Forecast reliability:**
  Simpler models (linear/Prophet) may not capture all production dynamics, but clarity is prioritized over complexity.

* **AI grounding risk:**
  Ensuring AI outputs are based on **actual computed data rather than hallucination** will require careful prompt design and structured inputs.

* **Deployment risk:**
  Ensuring the hosted app remains stable and responsive under time constraints (Streamlit Cloud limitations).
