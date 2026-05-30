"""Phase 7 — finals statistics dump for the technical paper + pitch deck.

Single source of truth for every headline number the two PDF deliverables cite, so the paper
and deck quote *real finals figures* (not prelim baselines) and every claim is traceable to a
codebase output. Reads only precomputed artifacts; writes:
  * ``reports/finals_stats.json`` — machine-readable, for embedding in figures/tables
  * ``reports/finals_stats.md``   — human-readable summary to copy into the documents

Pure read + aggregate; idempotent (overwrites).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    log = logging.getLogger("finals_stats")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(sh)
    return log


def main() -> None:
    log = setup_logging()
    log.info("Phase 7 — finals statistics dump start.")

    preds = pd.read_csv(REPORTS / "cypher_sentinels_predictions.csv")
    diag = pd.read_parquet(GOLD / "_predictions_diagnostic.parquet")
    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    d = diag.merge(preds, on="Outlet_ID")

    s = preds["Maximum_Monthly_Liters"]
    stats: dict = {"n_outlets": int(len(preds))}

    # ── Prediction distribution ──
    stats["prediction"] = {
        "mean": round(float(s.mean()), 1), "median": round(float(s.median()), 1),
        "min": round(float(s.min()), 1), "max": round(float(s.max()), 1),
        "p95": round(float(s.quantile(0.95)), 1), "p99": round(float(s.quantile(0.99)), 1),
    }

    # ── Per-size segment means (paper §4 / deck §4) ──
    seg = d.groupby("Outlet_Size")["Maximum_Monthly_Liters"].mean().round(0)
    stats["segment_mean_potential"] = {k: float(v) for k, v in seg.items()}

    # ── Formula mechanics ──
    stats["constraint_uplift_mean"] = round(float(diag["constraint_uplift"].mean()), 3)
    bw = np.argmax(np.column_stack(
        [diag["floor"], diag["stage_a_pred"], diag["peer_85th"]]), axis=1)
    names = ["historical_floor", "stage_a", "peer_85th"]
    split = pd.Series([names[i] for i in bw]).value_counts(normalize=True)
    stats["max_branch_split_pct"] = {k: round(float(v) * 100, 1) for k, v in split.items()}

    # ── Urban vs rural ──
    if "urban_flag" in gold.columns:
        j = preds.merge(gold[["Outlet_ID", "urban_flag"]], on="Outlet_ID")
        stats["urban_vs_rural_mean"] = {
            "urban": round(float(j[j.urban_flag == 1]["Maximum_Monthly_Liters"].mean()), 0),
            "rural": round(float(j[j.urban_flag == 0]["Maximum_Monthly_Liters"].mean()), 0),
        }

    # ── Upside gap (all outlets) ──
    tot_pot = float(s.sum())
    tot_act = float(gold["vol_mean"].sum())
    stats["upside_gap"] = {
        "total_potential_L": round(tot_pot, 0),
        "total_recent_actual_L": round(tot_act, 0),
        "upside_gap_L": round(tot_pot - tot_act, 0),
        "upside_multiple": round(tot_pot / max(tot_act, 1), 2),
    }

    # ── Censoring cross-check (Phase 2), if present ──
    cc = GOLD / "_censoring_crosscheck.json"
    if cc.exists():
        stats["censoring_crosscheck"] = json.loads(cc.read_text(encoding="utf-8")).get("agreement", {})

    # ── Budget optimiser (Phase 5), if present ──
    bud_path = GOLD / "budget_allocation_detail.parquet"
    if bud_path.exists():
        bud = pd.read_parquet(bud_path)
        funded = bud[bud["Trade_Spend_Allocation_LKR"] > 1.0]
        spent = float(bud["Trade_Spend_Allocation_LKR"].sum())
        inc = float(bud["projected_incremental_L"].sum())
        by_dist = (funded.groupby("primary_distributor")
                   .agg(outlets=("Outlet_ID", "count"),
                        spend_LKR=("Trade_Spend_Allocation_LKR", "sum"),
                        incremental_L=("projected_incremental_L", "sum")).round(0))
        stats["budget"] = {
            "total_spent_LKR": round(spent, 2),
            "outlets_funded": int(len(funded)),
            "incremental_L_per_month": round(inc, 0),
            "efficiency_L_per_1000_LKR": round(inc / max(spent, 1) * 1000, 2),
            "by_distributor": {k: {kk: float(vv) for kk, vv in row.items()}
                               for k, row in by_dist.iterrows()},
            "by_spend_type": {k: int(v) for k, v in funded["spend_type"].value_counts().items()}
            if "spend_type" in funded.columns else {},
        }

    (REPORTS / "finals_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Wrote %s", REPORTS / "finals_stats.json")

    # ── Markdown summary ──
    md = _to_markdown(stats)
    (REPORTS / "finals_stats.md").write_text(md, encoding="utf-8")
    log.info("Wrote %s", REPORTS / "finals_stats.md")
    log.info("Finals stats:\n%s", md)


def _to_markdown(s: dict) -> str:
    p = s["prediction"]
    seg = s.get("segment_mean_potential", {})
    L = [
        "# Finals statistics — Cypher Sentinels (auto-generated)",
        "",
        f"- **Outlets:** {s['n_outlets']:,}",
        f"- **Prediction (L/month):** mean {p['mean']}, median {p['median']}, "
        f"P95 {p['p95']}, P99 {p['p99']}, min {p['min']}, max {p['max']}",
        f"- **Segment mean potential (L):** "
        + " · ".join(f"{k} {seg[k]:.0f}" for k in
                     ["Extra Large", "Large", "Medium", "Small"] if k in seg),
        f"- **Constraint uplift mean:** {s['constraint_uplift_mean']}×",
        f"- **max() branch split:** "
        + " · ".join(f"{k} {v}%" for k, v in s["max_branch_split_pct"].items()),
    ]
    if "urban_vs_rural_mean" in s:
        uv = s["urban_vs_rural_mean"]
        L.append(f"- **Urban vs rural mean potential:** {uv['urban']:.0f} L vs {uv['rural']:.0f} L")
    ug = s["upside_gap"]
    L.append(f"- **Upside gap:** total potential {ug['total_potential_L']:,.0f} L vs recent "
             f"actual {ug['total_recent_actual_L']:,.0f} L ({ug['upside_multiple']}× headroom)")
    if "censoring_crosscheck" in s:
        for model, a in s["censoring_crosscheck"].items():
            L.append(f"- **{model} vs LightGBM:** Pearson {a['pearson']}, Spearman {a['spearman']}, "
                     f"mean diff {a['mean_diff_L']} L")
    if "budget" in s:
        b = s["budget"]
        L += [
            f"- **5M budget:** {b['total_spent_LKR']:,.0f} LKR across {b['outlets_funded']} "
            f"outlets → **+{b['incremental_L_per_month']:,.0f} L/month** "
            f"({b['efficiency_L_per_1000_LKR']} L per 1,000 LKR)",
        ]
        for dist, r in b["by_distributor"].items():
            L.append(f"  - {dist}: {int(r['outlets'])} outlets, {r['spend_LKR']:,.0f} LKR, "
                     f"+{r['incremental_L']:,.0f} L")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()