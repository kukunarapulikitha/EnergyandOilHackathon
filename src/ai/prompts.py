"""Prompt builders, context serialization, and tag parsing for the AI Analyst.

Design principles:
- Live-data grounding: every numeric claim Llama makes must come from the
  injected context block (never from training data).
- Data vs. inference boundary: Llama is instructed to prefix every claim with
  either `(Data)` or `(AI Analysis)`. The UI parses these tags into colored
  segments (blue for Data, amber for AI Analysis).
- Basin → PADD bridge: users ask about basins (Permian, Bakken, Marcellus...)
  but the data is keyed by PADD/state. The basin lookup table is injected so
  Llama can translate.
- Sensitivity grounding: when a what-if query is detected, we re-run the
  forecast in Python and inject the computed numbers — Llama only narrates,
  it never does math.
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------- BASIN MAP

BASIN_TO_GEOGRAPHY: dict[str, dict] = {
    "permian":      {"padd": "R30", "padd_name": "Gulf Coast",     "states": ["TX", "NM"], "fuel": "crude_oil"},
    "eagle ford":   {"padd": "R30", "padd_name": "Gulf Coast",     "states": ["TX"],       "fuel": "crude_oil"},
    "bakken":       {"padd": "R40", "padd_name": "Rocky Mountain", "states": ["ND"],       "fuel": "crude_oil"},
    "anadarko":     {"padd": "R30", "padd_name": "Gulf Coast",     "states": ["OK", "TX"], "fuel": "crude_oil"},
    "appalachia":   {"padd": "R10", "padd_name": "East Coast",     "states": ["PA", "WV", "OH"], "fuel": "natural_gas"},
    "marcellus":    {"padd": "R10", "padd_name": "East Coast",     "states": ["PA", "WV"], "fuel": "natural_gas"},
    "utica":        {"padd": "R10", "padd_name": "East Coast",     "states": ["OH", "PA"], "fuel": "natural_gas"},
    "haynesville":  {"padd": "R30", "padd_name": "Gulf Coast",     "states": ["LA", "TX"], "fuel": "natural_gas"},
    "niobrara":     {"padd": "R40", "padd_name": "Rocky Mountain", "states": ["CO", "WY"], "fuel": "crude_oil"},
}

STATE_TO_GAS_REGION_ID = {
    "TX": "STX", "PA": "SPA", "LA": "SLA", "OK": "SOK", "WV": "SWV",
}


def resolve_basin(query: str) -> dict | None:
    """Scan a user query for a basin name; return the geography dict or None."""
    q = query.lower()
    for basin, geo in BASIN_TO_GEOGRAPHY.items():
        if basin in q:
            return {"basin": basin, **geo}
    return None


def _basin_reference_block() -> str:
    lines = ["BASIN → REGION LOOKUP (use to translate basin names to data keys):"]
    for basin, geo in BASIN_TO_GEOGRAPHY.items():
        states = ", ".join(geo["states"])
        lines.append(
            f"  - {basin.title()}: {geo['padd_name']} ({geo['padd']}) — "
            f"states: {states} — primary fuel: {geo['fuel']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- CONTEXT BUILDER

def _project(slope: float, intercept: float, year: int) -> float:
    return float(slope * year + intercept)


def build_regional_context(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    controls: dict,
    metadata: dict | None = None,
) -> str:
    """Serialize the live Gold-layer snapshot into a text block for the LLM.

    Includes: per-region history sample, projected production at the user's
    selected year, KPIs, model R², and a basin lookup table.
    """
    fuel = controls["fuel"]
    year = controls["year"]
    selected = controls.get("regions") or []

    fa = actuals_df[actuals_df["fuel_type"] == fuel].copy()
    ff = forecasts_df[forecasts_df["fuel_type"] == fuel].copy()
    if selected:
        ff = ff[ff["region_id"].isin(selected)]
        fa = fa[fa["region_id"].isin(selected)]

    unit = "Mb/d (thousand barrels per day)" if fuel == "crude_oil" else "MMcf (million cubic feet)"

    fetched = (metadata or {}).get("fetched_at", "unknown")
    dqs = (metadata or {}).get("dqs", {}).get("score", "n/a")

    blocks = [
        f"=== LIVE DATA SNAPSHOT (source: EIA API v2; fetched_at: {fetched}; DQS: {dqs}/100) ===",
        f"User-selected year: {year}",
        f"Fuel: {fuel} (unit: {unit})",
        f"Selected regions: {', '.join(selected) if selected else 'all'}",
        "",
    ]

    for _, frow in ff.sort_values("region_id").iterrows():
        rid = frow["region_id"]
        name = frow["region_name"]
        slope = float(frow["slope"])
        intercept = float(frow["intercept"])
        r2 = float(frow["r_squared"])
        score = frow.get("investment_score")

        rhist = fa[fa["region_id"] == rid].sort_values("year")
        if rhist.empty:
            continue

        # Sample 5 evenly-spaced historical points so the prompt stays compact
        sample = rhist.iloc[::max(1, len(rhist) // 5)][["year", "production"]]
        hist_str = " → ".join(
            f"{int(r['year'])}={r['production']:,.0f}" for _, r in sample.iterrows()
        )

        latest = rhist.iloc[-1]
        latest_year = int(latest["year"])
        latest_prod = float(latest["production"])

        proj = _project(slope, intercept, year)

        # Most-recent observed KPIs for this region
        latest_growth = latest.get("growth_pct")
        latest_vol = latest.get("volatility_pct")
        latest_decline = latest.get("decline_rate_pct")
        latest_rpi = latest.get("relative_performance_index")
        latest_rev = latest.get("revenue_potential_usd")

        block = [
            f"Region: {name} [{rid}]",
            f"  Latest observed ({latest_year}): {latest_prod:,.1f}",
            f"  Historical sample: {hist_str}",
            f"  Projected production at {year}: {proj:,.1f}  "
            f"[formula: slope({slope:.4f}) × {year} + intercept({intercept:,.2f})]",
            f"  Forecast R²: {r2:.3f}",
        ]
        if pd.notna(latest_growth):
            block.append(f"  YoY growth ({latest_year}): {latest_growth:+.2f}%")
        if pd.notna(latest_vol):
            block.append(f"  Volatility (CV %): {latest_vol:.2f}%")
        if pd.notna(latest_decline):
            block.append(f"  Decline rate (5yr CAGR): {latest_decline:+.2f}%/yr")
        if pd.notna(latest_rpi):
            block.append(f"  Relative performance index: {latest_rpi:.0f}/100")
        if pd.notna(latest_rev):
            block.append(f"  Revenue potential (annual, latest): ${latest_rev/1e9:,.2f}B")
        if score is not None and pd.notna(score):
            block.append(f"  Investment score: {float(score):.0f}/100")
        blocks.append("\n".join(block))
        blocks.append("")

    blocks.append(_basin_reference_block())
    return "\n".join(blocks)


def _anchored_projection(
    actuals_df: pd.DataFrame,
    region_id: str,
    fuel: str,
    slope: float,
    intercept: float,
    rate_change: float,
    year: int,
) -> tuple[float, float, int, float]:
    """Compute base + adjusted projections anchored at the latest actual.

    Approach: project the base trend to `year` using the OLS line, then for the
    adjusted scenario, anchor at the latest observed actual and walk the
    *adjusted* slope forward. This keeps the math interpretable —
    "15% steeper decline" means the slope from today onward, not a re-fit of
    the intercept that produces absurd numbers.

    Returns: (base_value, adjusted_value, anchor_year, anchor_value).
    """
    base = _project(slope, intercept, year)
    region_hist = actuals_df[
        (actuals_df["region_id"] == region_id) & (actuals_df["fuel_type"] == fuel)
    ].sort_values("year")
    if region_hist.empty:
        # Fall back to projecting backward from base to anchor at year-1
        adjusted_slope = slope * (1 + rate_change)
        return base, _project(adjusted_slope, intercept, year), year, base
    anchor = region_hist.iloc[-1]
    anchor_year = int(anchor["year"])
    anchor_val = float(anchor["production"])
    adjusted_slope = slope * (1 + rate_change)
    if year >= anchor_year:
        adjusted = anchor_val + adjusted_slope * (year - anchor_year)
    else:
        # User picked a year before the anchor — fall back to OLS-line behavior
        adjusted = _project(slope * (1 + rate_change), intercept, year)
    return base, adjusted, anchor_year, anchor_val


def compute_sensitivity_context(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    rate_change: float,
    fuel: str,
    year: int,
    selected_regions: list[str] | None = None,
) -> str:
    """Pre-compute adjusted projections under a slope adjustment.

    rate_change: e.g. -0.15 = "15% steeper decline" (slope multiplied by 0.85).
    Returns a text block injected into the system prompt so Llama narrates
    real numbers (it never does the math itself).

    The adjusted projection is anchored at the latest observed actual to
    produce interpretable numbers — see `_anchored_projection`.
    """
    ff = forecasts_df[forecasts_df["fuel_type"] == fuel].copy()
    if selected_regions:
        ff = ff[ff["region_id"].isin(selected_regions)]
    if ff.empty:
        return ""

    direction = "steeper decline" if rate_change < 0 else "accelerated growth"
    lines = [
        "=== SENSITIVITY ANALYSIS (PRE-COMPUTED — use these numbers; do not recompute) ===",
        f"Scenario: forward slope adjusted by {rate_change:+.0%} ({direction}); "
        f"adjusted curve anchored at latest observed actual.",
        f"Target year: {year}",
        "",
    ]
    for _, r in ff.sort_values("region_id").iterrows():
        rid = r["region_id"]
        slope = float(r["slope"])
        intercept = float(r["intercept"])
        base, adjusted, anchor_year, anchor_val = _anchored_projection(
            actuals_df, rid, fuel, slope, intercept, rate_change, year,
        )
        delta_pct = (adjusted - base) / base * 100 if base else 0.0
        lines.append(
            f"  {r['region_name']} [{rid}]: "
            f"anchor {anchor_year}={anchor_val:,.0f} → "
            f"base {year}={base:,.0f} → adjusted {year}={adjusted:,.0f}  "
            f"({delta_pct:+.1f}% vs base)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- WHAT-IF DETECTION

_PCT_PATTERN = re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%")


def detect_what_if(query: str) -> dict | None:
    """Detect sensitivity / what-if intent and extract a rate change.

    Returns {"rate_change": -0.15} for "15% steeper decline".
    Returns {"rate_change": +0.10} for "10% acceleration".
    Returns None if no scenario detected.
    """
    q = query.lower()
    triggers = ("what if", "decline rate", "steeper", "assume", "scenario", "stress test", "stress-test")
    if not any(t in q for t in triggers):
        return None
    m = _PCT_PATTERN.search(q)
    if not m:
        return None
    pct = float(m.group(1)) / 100.0
    # Direction: "steeper", "decline", "drop", "fall", "lower" → negative
    negative_words = ("steeper", "decline", "drop", "fall", "lower", "worse", "decrease")
    is_negative = any(w in q for w in negative_words)
    return {"rate_change": -pct if is_negative else pct}


# ---------------------------------------------------------------- SYSTEM PROMPT

SYSTEM_INSTRUCTIONS = """You are an Energy Intelligence Analyst embedded in a U.S. oil & gas \
decision-support dashboard. Your job: answer business-development questions \
about regional production, forecasts, and KPIs using ONLY the data block below.

CRITICAL RULES:
1. Every numeric claim or factual statement MUST be prefixed with `(Data)` and \
cite the source region + year. Example: `(Data) PADD 3 (Gulf Coast) projected \
production for 2027 is 11,420 Mb/d.`
2. Every interpretive, comparative, or recommendation statement MUST be \
prefixed with `(AI Analysis)`. Example: `(AI Analysis) This positions the \
Gulf Coast as the strongest near-term opportunity given its R² of 0.94.`
3. Never invent numbers. If the data block doesn't contain a value the user \
asked about, say so explicitly: `(Data) The dataset does not include X.`
4. When a user asks about a basin (Permian, Bakken, Marcellus, etc.), use the \
BASIN → REGION LOOKUP at the end of the data block to find the corresponding \
PADD/state, then answer using that region's data. Note the mapping in your \
response: `(AI Analysis) Permian Basin output is captured under PADD 3 \
(Gulf Coast) — TX + NM.`
5. For what-if / sensitivity questions, a SENSITIVITY ANALYSIS block will be \
pre-computed for you. Use those exact numbers — DO NOT recompute. Tag the \
underlying scenario as `(AI Analysis)` since it is hypothetical, but cite \
the pre-computed adjusted values as `(Data)` since they came from the system.
6. Keep responses tight: lead with the direct answer, then 2–4 supporting \
bullet points. No filler.

FORECAST METHODOLOGY (cite when asked): linear OLS regression fit on annual \
production. Projection = slope × year + intercept. R² reports historical fit.

The user's selected year and fuel filter are noted in the data block — \
your answer should respect those filters.
"""


def build_system_prompt(context: str) -> str:
    return f"{SYSTEM_INSTRUCTIONS}\n\n{context}"


# ---------------------------------------------------------------- TAG PARSER

_TAG_RE = re.compile(r"\((Data|AI Analysis)\)\s*", re.IGNORECASE)


def parse_tagged_response(text: str) -> list[dict[str, str]]:
    """Split a tagged response into ordered segments.

    Each segment: {"type": "data"|"inference"|"untagged", "text": str}.
    Untagged prose appearing before the first tag is preserved as
    type="untagged" so nothing is silently dropped.
    """
    if not text:
        return []

    segments: list[dict[str, str]] = []
    matches = list(_TAG_RE.finditer(text))

    if not matches:
        return [{"type": "untagged", "text": text.strip()}]

    # Preamble before the first tag
    preamble = text[: matches[0].start()].strip()
    if preamble:
        segments.append({"type": "untagged", "text": preamble})

    for i, m in enumerate(matches):
        tag = m.group(1).lower()
        seg_type = "data" if tag == "data" else "inference"
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            segments.append({"type": seg_type, "text": chunk})
    return segments
