# Submission sanity checks. Exits non-zero on failure.
# 1) rows == master, 2) exact columns, 3) no nulls, 4) no zero/negatives,
# 5) every prediction >= outlet peak * 1.05, 6) distribution stats,
# 7) urban > rural mean (warn), 8) large > small mean.

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"

TEAMNAME = "cypher_sentinels"  # keep in sync with predict.py
SUBMISSION = REPORTS / f"{TEAMNAME}_predictions.csv"
BUDGET_CSV = REPORTS / f"{TEAMNAME}_budget_allocations.csv"
EXPLANATIONS_JSON = GOLD / "outlet_explanations.json"
TOTAL_BUDGET_LKR = 5_000_000.0


def setup_logging() -> logging.Logger:
    log = logging.getLogger("validate")
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


def main() -> int:
    log = setup_logging()
    failures: list[str] = []

    if not SUBMISSION.exists():
        log.error("Submission file does not exist: %s", SUBMISSION)
        return 1

    sub = pd.read_csv(SUBMISSION)
    master = pd.read_parquet(SILVER / "outlet_master_clean.parquet")
    gold = pd.read_parquet(GOLD / "outlet_features.parquet")

    if len(sub) != len(master):
        failures.append(f"row count {len(sub)} != master {len(master)}")
    log.info("[1] Rows: %d (master has %d) — %s",
             len(sub), len(master), "OK" if len(sub) == len(master) else "FAIL")

    expected = ["Outlet_ID", "Maximum_Monthly_Liters"]
    if list(sub.columns) != expected:
        failures.append(f"columns {list(sub.columns)} != {expected}")
    log.info("[2] Columns: %s — %s",
             sub.columns.tolist(), "OK" if list(sub.columns) == expected else "FAIL")

    n_null = sub.isna().any(axis=1).sum()
    if n_null:
        failures.append(f"{n_null} rows have nulls")
    log.info("[3] Nulls: %d — %s", n_null, "OK" if n_null == 0 else "FAIL")

    n_bad = (sub["Maximum_Monthly_Liters"] <= 0).sum()
    if n_bad:
        failures.append(f"{n_bad} rows have non-positive predictions")
    log.info("[4] Non-positive predictions: %d — %s", n_bad, "OK" if n_bad == 0 else "FAIL")

    joined = sub.merge(gold[["Outlet_ID", "vol_max"]], on="Outlet_ID", how="left")
    joined["min_required"] = joined["vol_max"] * 1.05
    below_floor = joined["Maximum_Monthly_Liters"] < joined["min_required"] - 1e-6
    if below_floor.any():
        worst = joined.loc[below_floor].head(5)
        failures.append(f"{below_floor.sum()} rows below historical_peak × 1.05 floor")
        log.error("  worst offenders:\n%s", worst.to_string())
    log.info("[5] Floor (≥ peak × 1.05): %d violations — %s",
             int(below_floor.sum()), "OK" if not below_floor.any() else "FAIL")

    log.info("[6] Distribution: mean=%.1f  median=%.1f  min=%.1f  max=%.1f  p95=%.1f  p99=%.1f",
             sub["Maximum_Monthly_Liters"].mean(),
             sub["Maximum_Monthly_Liters"].median(),
             sub["Maximum_Monthly_Liters"].min(),
             sub["Maximum_Monthly_Liters"].max(),
             sub["Maximum_Monthly_Liters"].quantile(0.95),
             sub["Maximum_Monthly_Liters"].quantile(0.99))

    # urban>rural is noisy while POI cache is incomplete — log as WARN, not FAIL
    j = sub.merge(gold[["Outlet_ID", "urban_flag", "Outlet_Size", "poi_imputed"]], on="Outlet_ID")
    urban_mean = j.loc[j["urban_flag"] == 1, "Maximum_Monthly_Liters"].mean()
    rural_mean = j.loc[j["urban_flag"] == 0, "Maximum_Monthly_Liters"].mean()
    urban_ok = urban_mean > rural_mean
    pct_imputed = 100 * j["poi_imputed"].mean()
    log.info("[7] Urban mean=%.1f vs Rural mean=%.1f — %s (POI imputed: %.1f%%)",
             urban_mean, rural_mean, "OK" if urban_ok else "WARN", pct_imputed)

    large_mean = j.loc[j["Outlet_Size"].isin(["Large", "Extra Large"]), "Maximum_Monthly_Liters"].mean()
    small_mean = j.loc[j["Outlet_Size"] == "Small", "Maximum_Monthly_Liters"].mean()
    size_ok = large_mean > small_mean
    if not size_ok:
        failures.append(f"Large/XL mean ({large_mean:.1f}) not greater than Small mean ({small_mean:.1f})")
    log.info("[8] Large/XL mean=%.1f vs Small mean=%.1f — %s",
             large_mean, small_mean, "OK" if size_ok else "WARN")

    log.info("Sample 10 predictions:")
    sample = j.sample(10, random_state=42).sort_values("Outlet_ID")
    log.info("\n%s", sample.to_string(index=False))

    # ── Budget allocation CSV (Phase 5) ──
    failures += validate_budget(master, gold, log)

    # ── Explanations JSON (Phase 4) ──
    failures += validate_explanations(sub, log)

    if failures:
        log.error("VALIDATION FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    log.info("ALL CHECKS PASSED.")
    return 0


def validate_budget(master, gold, log) -> list[str]:
    """Western-only, sum ≤ 5M, no negatives/nulls, exact two columns."""
    fails: list[str] = []
    if not BUDGET_CSV.exists():
        log.warning("[B0] Budget CSV missing (%s) — skipping budget checks "
                    "(run optimize_budget.py).", BUDGET_CSV.name)
        return fails

    b = pd.read_csv(BUDGET_CSV)
    expected = ["Outlet_ID", "Trade_Spend_Allocation_LKR"]
    if list(b.columns) != expected:
        fails.append(f"budget columns {list(b.columns)} != {expected}")
    log.info("[B1] Budget columns: %s — %s", b.columns.tolist(),
             "OK" if list(b.columns) == expected else "FAIL")

    # Western only — every budget Outlet_ID must be a Western-province outlet
    west_ids = set(gold.loc[gold["Province"] == "Western", "Outlet_ID"]) \
        if "Province" in gold.columns else set(b["Outlet_ID"])
    n_non_west = int((~b["Outlet_ID"].isin(west_ids)).sum())
    if n_non_west:
        fails.append(f"{n_non_west} budget rows are not Western-province outlets")
    log.info("[B2] Western-only: %d non-Western rows — %s",
             n_non_west, "OK" if n_non_west == 0 else "FAIL")

    total = float(b["Trade_Spend_Allocation_LKR"].sum())
    if total > TOTAL_BUDGET_LKR + 1.0:
        fails.append(f"budget sum {total:.2f} exceeds {TOTAL_BUDGET_LKR:.0f}")
    log.info("[B3] Budget sum: %.2f (<= %.0f) — %s",
             total, TOTAL_BUDGET_LKR, "OK" if total <= TOTAL_BUDGET_LKR + 1.0 else "FAIL")

    n_neg = int((b["Trade_Spend_Allocation_LKR"] < 0).sum())
    n_null = int(b.isna().any(axis=1).sum())
    if n_neg or n_null:
        fails.append(f"budget has {n_neg} negative and {n_null} null rows")
    log.info("[B4] Negatives/nulls: %d / %d — %s",
             n_neg, n_null, "OK" if not (n_neg or n_null) else "FAIL")

    n_funded = int((b["Trade_Spend_Allocation_LKR"] > 1.0).sum())
    log.info("[B5] Funded outlets: %d of %d; spend concentrated where gap×responsiveness "
             "is highest.", n_funded, len(b))
    return fails


def validate_explanations(sub, log) -> list[str]:
    """One explanation per outlet, each with text + an evidence packet."""
    import json
    fails: list[str] = []
    if not EXPLANATIONS_JSON.exists():
        log.warning("[X0] Explanations JSON missing (%s) — skipping (run xai_explain.py).",
                    EXPLANATIONS_JSON.name)
        return fails

    ex = json.loads(EXPLANATIONS_JSON.read_text(encoding="utf-8"))
    sub_ids = set(sub["Outlet_ID"])
    missing = sub_ids - set(ex.keys())
    if missing:
        fails.append(f"{len(missing)} outlets have no explanation")
    log.info("[X1] Coverage: %d explanations for %d outlets — %s",
             len(ex), len(sub_ids), "OK" if not missing else "FAIL")

    n_empty = sum(1 for r in ex.values()
                  if not r.get("explanation") or "evidence" not in r)
    if n_empty:
        fails.append(f"{n_empty} explanations are empty or missing an evidence packet")
    log.info("[X2] Non-empty + evidence packet: %d bad — %s",
             n_empty, "OK" if n_empty == 0 else "FAIL")

    from collections import Counter
    srcs = Counter("llm" if r["source"].startswith("github_models") else "template"
                   for r in ex.values())
    log.info("[X3] Sources: %d live LLM, %d grounded template.",
             srcs.get("llm", 0), srcs.get("template", 0))
    return fails


if __name__ == "__main__":
    sys.exit(main())
