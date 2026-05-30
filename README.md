# Latent Outlet Potential — Sri Lanka

Predicting `Maximum_Monthly_Liters` for ~20,000 traditional retail outlets in
Sri Lanka for January 2026. The output is the latent monthly volume ceiling for
each outlet — what it *could* sell if supply weren't binding — rather than a
naive forecast of next month's sales.

The core methodological challenge is **left-censoring**: observed historical
volume is `min(true_demand, supply_capacity)`. A direct regression on
`Volume_Liters` would learn the ceiling rather than the demand. We address
this with a three-stage framework anchored to outlet history.

## Repo layout — Bronze → Silver → Gold

```
project/
├── data/                                        ← (gitignored, regenerated)
│   ├── bronze/                                  ← raw CSVs, untouched
│   ├── silver/                                  ← cleaned, monthly-aggregated
│   ├── gold/                                    ← model-ready, 1 row per outlet
│   ├── rejected_records/                        ← quarantined rows + failure_reason
│   └── external/poi_cache/                      ← Overpass API responses
│
├── src/
│   ├── bronze_ingest.py                         ← BRONZE: read raw CSVs, log schema
│   ├── dq_checks.py                             ← SILVER: 7 reusable DQ functions
│   ├── silver_clean.py                          ← SILVER: clean + quarantine + monthly agg
│   ├── poi_scrape.py                            ← GOLD: scrape OpenStreetMap POIs
│   ├── gold_features.py                         ← GOLD: build 77 features per outlet
│   ├── model.py                                 ← MODEL: Stage A LightGBM
│   ├── predict.py                               ← MODEL: 3-stage formula → submission
│   └── validate_predictions.py                  ← 8 sanity checks on submission
│
├── notebooks/
│   ├── eda.ipynb                                ← exploratory data analysis
│   ├── report_charts.py                         ← chart generation for report
│   └── build_report_html.py                     ← markdown → HTML for PDF render
│
├── reports/
│   ├── cypher_sentinels_predictions.csv         ← FINAL SUBMISSION (20k × 2)
│   ├── report.pdf                               ← FINAL REPORT (5 pages)
│   └── figures/                                 ← chart PNGs embedded in PDF
│
├── README.md
└── requirements.txt
```

## Pipeline flow

```
data/bronze/  →  silver_clean.py  →  data/silver/  →  gold_features.py  →  data/gold/  →  model.py  →  predict.py  →  reports/cypher_sentinels_predictions.csv
```

| Stage | Script | Purpose |
|---|---|---|
| Bronze | `src/bronze_ingest.py` | Read raw CSVs, log schema/dtypes/row counts |
| Silver | `src/silver_clean.py` | 7 DQ checks, quarantine rejects with `failure_reason`, monthly aggregation |
| Gold | `src/gold_features.py` | One row per outlet, 77 features across 5 buckets |
| Model | `src/model.py` | LightGBM on uncensored months, 5-fold GroupKFold CV by Outlet_ID |
| Predict | `src/predict.py` | Apply three-stage formula, write submission CSV |
| Validate | `src/validate_predictions.py` | 8 sanity checks; exits non-zero on failure |

## Three-stage framework

```
Stage A — LightGBM trained only on outlet-months where volume < outlet_P99 × 0.95
Stage B — Constraint_Uplift = 1.0 + 0.25 × (0.5 × censoring + 0.5 × plateau)
Stage C — 85th percentile of monthly volume within peer cluster

raw       = max(Historical_Peak × 1.05, Stage_A_Pred, Peer_85th)
scaled    = raw × Seasonality_Jan2026 × Constraint_Uplift
ceiled    = min(scaled, Peer_99th × 1.5)        # sanity ceiling
potential = max(ceiled, Historical_Peak × 1.05) # sanity floor (always)
```

## Data quality

Every check returns `(passed_df, rejected_df)`. The rejected_df always carries
a `failure_reason` column. **No row is ever silently dropped.** Quarantined
rows land in `data/rejected_records/{dataset}_rejected.csv`.

Per-outlet outlet-master fixes (case normalization, typo repair, null
imputation) are soft — the row stays in the pipeline so the submission can
contain a prediction for every master outlet — but every fix is logged in the
audit trail.

## Run

```bash
pip install -r requirements.txt

# 1. Place Kaggle CSVs in data/bronze/:
#    transactions_history_final.csv
#    outlet_master.csv
#    outlet_coordinates.csv
#    distributor_seasonality_details.csv
#    holiday_list.csv

python src/bronze_ingest.py
python src/silver_clean.py
python src/poi_scrape.py            # long-running; run in a separate terminal
python src/gold_features.py
python src/model.py
python src/predict.py
python src/validate_predictions.py
```

`poi_scrape.py` queries the Overpass API (no key required, free for academic
use) and caches per-outlet responses to `data/external/poi_cache/`. It is
resumable — re-running picks up where it left off. Outlets without cache are
handled via cluster-median imputation in `gold_features.py` with a
`poi_imputed` flag preserved as a model feature.

## Output

`reports/cypher_sentinels_predictions.csv` — 20,000 rows, exact columns
`Outlet_ID, Maximum_Monthly_Liters`. Validation enforces:

- Row count equals outlet master count
- No nulls, no zeros, no negatives
- Every prediction ≥ outlet's historical peak × 1.05
- Large/Extra Large outlet mean > Small outlet mean

## Reproducibility

- Python 3.10+ (tested on 3.12)
- All randomness seeded with `random_state=42`
- All paths are relative via `pathlib.Path`
- Logs to both stdout and `logs/{stage}.log`
