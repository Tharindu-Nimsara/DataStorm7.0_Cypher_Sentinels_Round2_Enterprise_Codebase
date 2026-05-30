# Latent Outlet Potential — Data Storm v7.0 (Team Cypher Sentinels)

Estimate the **latent maximum monthly volume** (liters) each of our 20,000 Sri Lankan
retail outlets *could* sell in **January 2026**, then turn that estimate into trade
decisions: a 5,000,000-LKR budget allocation and a user-facing Outlet Intelligence app.

The hard part is **left-censoring**. Observed monthly volume is
`min(true_demand, supply_capacity)` — so a naive regression on `Volume_Liters` learns the
*ceiling* an outlet keeps hitting, not the demand behind it. A busy town-centre kade can
look "average" only because it runs out of stock or fridge space every month. Our pipeline
estimates that ceiling and lifts the prediction toward true potential, so trade spend can
chase opportunity instead of rewarding outlets that are already maxed out.

---

## What's new in the final round

The preliminary pipeline already does Bronze → Silver → Gold, a Stage-A LightGBM demand
model, a 3-stage potential formula, and a resumable POI scrape (16,423 of 20,000 outlets
cached so far, ~82%). The final round **extends** that pipeline — it does not rebuild it —
with five new modules:

| Module | File | What it adds |
|---|---|---|
| Distance-decay spatial signals | `src/poi_decay.py` | BallTree haversine lookup, per-type exponential/gravity decay weights, and competitive catchment density + a market-share proxy — replacing flat radius counts with gravity signals |
| Formal censoring + physical limits | `src/censoring_model.py` | A Tobit censored-regression cross-check against LightGBM, plus a physical `Cooler_Count`-based ceiling so Stage B's constraint logic is physical, not only statistical |
| Per-outlet explainability (XAI) | `src/xai_explain.py` | SHAP local attributions + an LLM layer that explains each outlet in plain business language, grounded only in the real numbers and cached to JSON |
| Budget optimizer | `src/optimize_budget.py` | Allocates LKR 5M across Western-province outlets to maximize *incremental* volume under a diminishing-returns response function |
| Outlet Intelligence web app | `app/app.py` | Streamlit app: browse/search all 20,000 predictions, filter by province/distributor, drill into one outlet (SHAP chart + map + LLM explanation), and a Western budget view |

**Prelim baseline (recompute for finals):** 20,000 predictions, 77 gold features, mean
potential ≈ 433.2 L, median ≈ 219.5 L, range 77–2955 L (POI features imputed for ~18% of
outlets not yet scraped). Outlets by province: Western 9,000 · Central 4,000 ·
North-Western 4,000 · Southern 3,000.

---

## Repository layout

```
project/
├── data/                         # gitignored, regenerated from bronze CSVs
│   ├── bronze/                   # raw CSVs, untouched
│   ├── silver/                   # cleaned, monthly-aggregated
│   ├── gold/                     # model-ready, 1 row per outlet
│   ├── rejected_records/         # quarantined rows + failure_reason
│   └── external/poi_cache/       # Overpass API responses
├── src/
│   ├── bronze_ingest.py          # read raw CSVs, log schema
│   ├── dq_checks.py              # 7 reusable DQ functions
│   ├── silver_clean.py           # clean + quarantine + monthly agg
│   ├── poi_scrape.py             # scrape OpenStreetMap POIs (resumable)
│   ├── poi_decay.py              # NEW: distance-decay + competitor density
│   ├── gold_features.py          # build features per outlet
│   ├── censoring_model.py        # NEW: Tobit cross-check + physical_max
│   ├── model.py                  # Stage A LightGBM, 5-fold GroupKFold
│   ├── predict.py                # 3-stage formula -> submission
│   ├── xai_explain.py            # NEW: SHAP + LLM explanations -> JSON
│   ├── optimize_budget.py        # NEW: 5M Western allocation -> CSV
│   └── validate_predictions.py   # sanity checks on all outputs
├── app/
│   └── app.py                    # NEW: Streamlit Outlet Intelligence app
├── reports/
│   ├── cypher_sentinels_predictions.csv
│   └── cypher_sentinels_budget_allocations.csv
├── GENAI_LOG.md                  # NEW: prompts + accepted/rejected/fixed
├── README.md
└── requirements.txt
```

---

## End-to-end run order

```bash
pip install -r requirements.txt
python src/bronze_ingest.py
python src/silver_clean.py
python src/poi_scrape.py          # long-running, resumable; separate terminal
python src/poi_decay.py           # NEW: distance-decay + competitor density
python src/gold_features.py
python src/censoring_model.py     # NEW: Tobit cross-check + physical_max
python src/model.py
python src/predict.py
python src/xai_explain.py         # NEW: SHAP + LLM explanations -> JSON
python src/optimize_budget.py     # NEW: 5M Western allocation -> budget CSV
python src/validate_predictions.py
streamlit run app/app.py          # NEW: web app
```

`data/` is gitignored and fully reproducible from the raw CSVs in `data/bronze/`. Every
stage is **idempotent**: re-running overwrites its outputs and never appends. No row is
ever silently dropped — invalid rows are quarantined to `data/rejected_records/` with a
`failure_reason`. All randomness is seeded with `random_state=42`.
