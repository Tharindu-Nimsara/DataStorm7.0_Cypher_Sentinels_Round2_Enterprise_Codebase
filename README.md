# Latent Outlet Potential — Data Storm 7.0 (Team Cypher Sentinels)

Estimate the **latent maximum monthly volume** (liters) each of 20,000 Sri Lankan retail
outlets *could* sell in **January 2026**, then turn that estimate into trade decisions: a
5,000,000-LKR Western-province budget allocation and a user-facing Outlet Intelligence app.

The hard part is **left-censoring**. Observed monthly volume is
`min(true_demand, supply_capacity)` — so a naive regression on `Volume_Liters` learns the
*ceiling* an outlet keeps hitting, not the demand behind it. A busy town-centre kade can look
"average" only because it runs out of stock or fridge space every month. This pipeline
estimates that ceiling and lifts the prediction toward true potential, so trade spend can
chase opportunity instead of rewarding outlets that are already maxed out.

---

## What's new in the final round

The preliminary pipeline already did Bronze → Silver → Gold, a Stage-A LightGBM demand model,
a 3-stage potential formula, and a resumable POI scrape (16,423 of 20,000 outlets cached,
~82 %). The final round **extends** that pipeline — it does not rebuild it — with five new
capabilities:

| Capability | File | What it adds |
|---|---|---|
| Distance-decay spatial signals | `src/poi_decay.py` | BallTree haversine lookup, per-type exponential/gravity decay weights, and competitive catchment density + a market-share proxy — gravity signals instead of flat radius counts |
| Formal censoring + physical limits | `src/censoring_model.py` | Tobit and Weibull-AFT censored-regression cross-checks against LightGBM, plus a physical `Cooler_Count`-based ceiling so the constraint logic is physical, not only statistical |
| Per-outlet explainability (XAI) | `src/xai_explain.py` | SHAP local attributions + an LLM layer that explains each outlet in plain business language, grounded only in the real numbers and cached to JSON |
| Budget optimizer | `src/optimize_budget.py` | Allocates LKR 5M across Western-province outlets to maximize *incremental* volume under a diminishing-returns response function (cvxpy, with a greedy cross-check) |
| Outlet Intelligence web app | `app/app.py` | Streamlit app: browse/search all 20,000 predictions, filter by province/distributor, drill into one outlet (SHAP chart + map + LLM explanation), and a Western budget view |

The web app is also published as a **standalone repository** (with the precomputed artifacts it
needs bundled) so it can be cloned and run on its own; `app/app.py` here is the same app within
the pipeline.

**Finals results (20,000 outlets, 105 Gold features):** predicted potential mean **430.5 L**,
median **214.8 L**, range **77–2,955 L**. Total latent potential **8.61 M L/mo** vs **4.19 M
L/mo** sold today — a **2.05× headroom**. The 5M-LKR Western plan funds **805 outlets** for a
projected **+154,524 L/mo**. Outlets by province: Western 9,000 · Central 4,000 ·
North-Western 4,000 · Southern 3,000.

---

## Repository layout

```
project-folder/
├── data/                         # gitignored, regenerated from bronze CSVs
│   ├── bronze/                   # raw CSVs, untouched
│   ├── silver/                   # cleaned, monthly-aggregated
│   ├── gold/                     # model-ready, 1 row per outlet
│   ├── rejected_records/         # quarantined rows + failure_reason
│   └── external/poi_cache/       # Overpass API responses
├── src/
│   ├── bronze_ingest.py          # read raw CSVs, log schema
│   ├── dq_checks.py              # 7 reusable data-quality functions
│   ├── silver_clean.py           # clean + quarantine + monthly aggregation
│   ├── poi_scrape.py             # scrape OpenStreetMap POIs (resumable)
│   ├── poi_decay.py              # distance-decay + competitor density
│   ├── gold_features.py          # build the 105 features per outlet
│   ├── censoring_model.py        # Tobit/Weibull cross-checks + physical_max
│   ├── model.py                  # Stage A LightGBM, 5-fold GroupKFold
│   ├── predict.py                # combining formula -> submission CSV
│   ├── xai_explain.py            # SHAP + LLM explanations -> JSON
│   ├── optimize_budget.py        # 5M Western allocation -> budget CSV
│   ├── validate_predictions.py   # sanity checks on predictions/budget/explanations
│   └── finals_stats.py           # aggregate headline numbers -> reports/finals_stats.*
├── app/
│   └── app.py                    # Streamlit Outlet Intelligence app
├── notebooks/
│   ├── eda.ipynb                 # exploratory analysis
│   └── report_charts.py          # regenerate report figures from outputs
├── reports/
│   ├── cypher_sentinels_predictions.csv          # Latent Potential output
│   ├── cypher_sentinels_budget_allocations.csv   # Western 5M allocation
│   ├── finals_stats.{json,md}                    # headline numbers
│   └── figures/                                  # generated charts
├── make_repro_zip.py             # build the reproducible-codebase zip
├── .env.example                  # config template (GITHUB_TOKEN for the XAI layer)
├── GENAI_LOG.md                  # GenAI transparency log: prompts + decisions
├── requirements.txt
└── README.md
```

---

## Run it end to end

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt

python src/bronze_ingest.py
python src/silver_clean.py
python src/poi_scrape.py          # long-running, resumable; run in a separate terminal
python src/poi_decay.py           # distance-decay + competitor density
python src/gold_features.py
python src/censoring_model.py     # Tobit/Weibull cross-checks + physical_max
python src/model.py               # train Stage A LightGBM
python src/predict.py             # apply the combining formula -> predictions CSV
python src/xai_explain.py         # SHAP + LLM explanations -> JSON  (see Configuration)
python src/optimize_budget.py     # 5M Western allocation -> budget CSV
python src/validate_predictions.py  # validates all outputs; exits non-zero on any failure
python src/finals_stats.py        # headline numbers -> reports/finals_stats.{json,md}

streamlit run app/app.py          # the web app
```

`data/` is gitignored and fully reproducible from the raw CSVs in `data/bronze/`. Every stage
is **idempotent**: re-running overwrites its outputs and never appends — a full double-run
produces byte-identical features, predictions, budget, and rejected-records files. **No row is
ever silently dropped**: invalid rows are quarantined to `data/rejected_records/` with a
`failure_reason`. All randomness is seeded with `random_state=42`.

### Configuration

The XAI layer (`src/xai_explain.py`) is the only stage that uses a secret. Copy `.env.example`
to `.env` and set `GITHUB_TOKEN` (a GitHub PAT with the **Models** permission) to generate live
LLM explanations via GitHub Models. **Without a token it falls back to a deterministic,
grounded offline template**, so the full pipeline still runs end to end with no key. `.env` is
gitignored.

### Reproducible-codebase zip

`python make_repro_zip.py` builds `dist/cypher_sentinels_reproducible_codebase.zip` — the full
repo **plus** the raw data, the POI cache, and all precomputed outputs (≈141 MB zipped) so a
reviewer can run everything in minutes without the ~5-hour Overpass scrape. It excludes secrets
and local scaffolding.

---

## Method in one paragraph

Stage A (LightGBM, `log1p(volume)`, 5-fold GroupKFold by outlet) is trained **only on
uncensored outlet-months** so it learns demand, not the ceiling. The prediction is then
`max(historical_peak × 1.05, Stage A, peer-85th)`, scaled by January seasonality and a
constraint uplift, and clipped to a ceiling that is the smaller of a peer bound and the physical
cooler capacity — with the historical floor always winning last so no prediction ever falls
below proven history. Tobit and Weibull-AFT censored regressions independently confirm the
ranking of latent headroom (Spearman ≈ 0.90). Full detail and the GenAI workflow are in
[`GENAI_LOG.md`](GENAI_LOG.md).
