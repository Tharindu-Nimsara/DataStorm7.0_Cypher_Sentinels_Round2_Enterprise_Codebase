# Apply Stage B (constraint uplift) and Stage C (peer 85th), combine with
# Stage A via the formula:
#   raw = max(peak*1.05, stage_a, peer_85)
#   scaled = raw * seas * uplift
#   ceiled = min(scaled, peer_99 * 1.5)
#   final  = max(ceiled, peak * 1.05)        # floor wins on outlier outlets
# Writes reports/{teamname}_predictions.csv with cols Outlet_ID, Maximum_Monthly_Liters.

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
REPORTS = ROOT / "reports"
LOG_DIR = ROOT / "logs"
REPORTS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_FLOOR = 1.05      # Sanity floor: never below outlet's max × 1.05
SANITY_CEIL_MULT = 1.5       # Sanity ceiling: never above peer P99 × 1.5
UPLIFT_MAX = 0.25            # Constraint uplift coefficient (range [1.00, 1.25])

TEAMNAME = "cypher_sentinels"
OUTPUT_CSV = REPORTS / f"{TEAMNAME}_predictions.csv"


def setup_logging() -> logging.Logger:
    log = logging.getLogger("predict")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / "predict.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def stage_b_constraint_uplift(gold):
    # severity is a 50/50 mix of (P95-loose) censoring rate and high-volume plateau ratio
    severity = (0.5 * gold["censoring_score"] + 0.5 * gold["plateau_norm"]).clip(0, 1)
    return 1.0 + UPLIFT_MAX * severity


def main() -> None:
    log = setup_logging()
    log.info("Phase 4 predict start.")

    gold = pd.read_parquet(GOLD / "outlet_features.parquet")
    stage_a = pd.read_parquet(GOLD / "stage_a_pred.parquet")

    log.info("Gold: %d outlets. Stage A pred: %d rows.", len(gold), len(stage_a))
    assert len(gold) == len(stage_a), "Stage A pred count must match gold."

    df = gold.merge(stage_a, on="Outlet_ID", how="left")
    assert df["stage_a_pred"].notna().all(), "Missing Stage A predictions for some outlets."

    df["constraint_uplift"] = stage_b_constraint_uplift(df)
    log.info("Stage B uplift — min=%.3f median=%.3f max=%.3f",
             df["constraint_uplift"].min(),
             df["constraint_uplift"].median(),
             df["constraint_uplift"].max())

    df["peer_85th"] = df["peer_p85_monthly"]
    df["peer_99th"] = df["peer_p99_monthly"]
    df["historical_peak"] = df["vol_max"]
    df["floor"] = df["historical_peak"] * HISTORICAL_FLOOR
    df["sanity_ceiling"] = df["peer_99th"] * SANITY_CEIL_MULT
    df["seas"] = df["seasonality_jan_mult"]

    log.info("Applying formula: max(floor, stage_a, peer_85) * seas * uplift, clipped.")
    raw = np.maximum.reduce([
        df["floor"].values,
        df["stage_a_pred"].values,
        df["peer_85th"].values,
    ])
    df["potential_raw"] = raw
    df["potential_scaled"] = raw * df["seas"].values * df["constraint_uplift"].values
    df["potential_ceiled"] = np.minimum(df["potential_scaled"].values, df["sanity_ceiling"].values)
    # floor wins on outliers where ceiling < floor — we always trust history
    df["potential_final"] = np.maximum(df["potential_ceiled"].values, df["floor"].values)

    branch_winner = np.argmax(
        np.column_stack([df["floor"].values, df["stage_a_pred"].values, df["peer_85th"].values]),
        axis=1,
    )
    branch_names = ["floor", "stage_a", "peer_85th"]
    win_counts = pd.Series([branch_names[i] for i in branch_winner]).value_counts()
    log.info("Max() branch winners:\n%s", win_counts.to_string())

    n_floor_clamps = int((df["potential_ceiled"] < df["floor"]).sum())
    n_ceil_clamps = int((df["potential_scaled"] > df["sanity_ceiling"]).sum())
    n_floor_overrides_ceiling = int((df["potential_final"] > df["sanity_ceiling"] + 1e-6).sum())
    log.info("Sanity clamps — floor activated %d times, ceiling activated %d times.",
             n_floor_clamps, n_ceil_clamps)
    log.info("Floor overrides ceiling (outlier outlets in sparse peers): %d",
             n_floor_overrides_ceiling)

    log.info("Final Maximum_Monthly_Liters distribution:")
    log.info("  mean=%.1f  median=%.1f  min=%.1f  max=%.1f",
             df["potential_final"].mean(), df["potential_final"].median(),
             df["potential_final"].min(), df["potential_final"].max())

    # ceil-round to 2dp so we never round a prediction below the peak*1.05 floor
    submission = df[["Outlet_ID", "potential_final"]].rename(
        columns={"potential_final": "Maximum_Monthly_Liters"}
    )
    submission["Maximum_Monthly_Liters"] = (
        np.ceil(submission["Maximum_Monthly_Liters"] * 100) / 100
    )
    submission.to_csv(OUTPUT_CSV, index=False)
    log.info("Wrote %s — %d rows, %d cols", OUTPUT_CSV, *submission.shape)
    log.info("Submission columns: %s", submission.columns.tolist())

    diag_path = GOLD / "_predictions_diagnostic.parquet"
    df[[
        "Outlet_ID", "Outlet_Type", "Outlet_Size", "Province",
        "vol_mean", "vol_max", "vol_p95", "vol_p99",
        "stage_a_pred", "peer_85th", "peer_99th", "floor", "sanity_ceiling",
        "censoring_score", "plateau_score", "plateau_norm",
        "constraint_uplift", "seas",
        "potential_raw", "potential_scaled", "potential_ceiled", "potential_final",
    ]].to_parquet(diag_path, index=False)
    log.info("Wrote diagnostic frame: %s", diag_path)


if __name__ == "__main__":
    main()
