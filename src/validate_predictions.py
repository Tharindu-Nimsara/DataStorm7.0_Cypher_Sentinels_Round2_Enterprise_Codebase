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

    if failures:
        log.error("VALIDATION FAILED:\n  - " + "\n  - ".join(failures))
        return 1
    log.info("ALL CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
