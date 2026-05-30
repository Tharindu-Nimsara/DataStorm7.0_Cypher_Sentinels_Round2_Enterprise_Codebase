# Silver: clean all 5 bronze datasets via dq_checks, quarantine rejects with
# reasons (data/rejected_records/), monthly-aggregate transactions to one row
# per (outlet, year, month). Outlet_master uses SOFT imputation so we keep
# all 20k rows (the submission needs one prediction per outlet).

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dq_checks import (
    check_duplicates,
    check_format,
    check_geo_bounds,
    check_nulls,
    check_price_consistency,
    check_referential_integrity,
    check_value_range,
)

ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data" / "bronze"
SILVER = ROOT / "data" / "silver"
REJECTS = ROOT / "data" / "rejected_records"
LOG_DIR = ROOT / "logs"

SILVER.mkdir(parents=True, exist_ok=True)
REJECTS.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

VALID_DISTRIBUTORS = {
    "DIST_W_01", "DIST_W_02", "DIST_W_03",
    "DIST_C_01", "DIST_C_02", "DIST_C_03",
    "DIST_NW_01", "DIST_NW_02",
    "DIST_S_01", "DIST_S_02",
}


def _setup_logging() -> logging.Logger:
    log = logging.getLogger("silver_clean")
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

    fh = logging.FileHandler(LOG_DIR / "silver_clean.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def _accumulate(rejects_bucket: list[pd.DataFrame], rejected: pd.DataFrame, stage: str) -> None:
    if len(rejected) == 0:
        return
    r = rejected.copy()
    r["stage"] = stage
    rejects_bucket.append(r)


def clean_transactions(
    tx: pd.DataFrame, outlet_master_ids: set[str], log: logging.Logger
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rejects: list[pd.DataFrame] = []
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["input"] = len(tx)

    # Product_Name is absent from source — exclude from required cols
    required = ["Outlet_ID", "Year", "Month", "Distributor_ID", "SKU_ID",
                "Volume_Liters", "Total_Bill_Value"]
    tx, rej = check_nulls(tx, required)
    _accumulate(rejects, rej, "nulls"); counts["after_nulls"] = len(tx)

    tx, rej = check_format(tx, "Year", allowed_values=[2023, 2024, 2025])
    _accumulate(rejects, rej, "year_format"); counts["after_year"] = len(tx)
    tx, rej = check_format(tx, "Month", allowed_values=list(range(1, 13)))
    _accumulate(rejects, rej, "month_format"); counts["after_month"] = len(tx)

    tx, rej = check_value_range(tx, "Volume_Liters", min_val=1e-9)
    _accumulate(rejects, rej, "volume_le_zero"); counts["after_vol_pos"] = len(tx)
    tx, rej = check_value_range(tx, "Total_Bill_Value", min_val=1e-9)
    _accumulate(rejects, rej, "bill_le_zero"); counts["after_bill_pos"] = len(tx)

    tx, rej = check_referential_integrity(tx, "Outlet_ID", outlet_master_ids, "outlet_master")
    _accumulate(rejects, rej, "outlet_not_in_master"); counts["after_outlet_ref"] = len(tx)
    tx, rej = check_referential_integrity(tx, "Distributor_ID", VALID_DISTRIBUTORS, "valid_distributors")
    _accumulate(rejects, rej, "distributor_invalid"); counts["after_dist_ref"] = len(tx)

    key = ["Outlet_ID", "Year", "Month", "SKU_ID", "Distributor_ID"]
    tx, rej = check_duplicates(tx, key)
    _accumulate(rejects, rej, "duplicate_key"); counts["after_dedup"] = len(tx)

    # P99.9-per-SKU volume outliers — hunting decimal-error rows
    p999 = tx.groupby("SKU_ID")["Volume_Liters"].quantile(0.999)
    threshold = tx["SKU_ID"].map(p999)
    outlier_mask = tx["Volume_Liters"] > threshold
    if outlier_mask.any():
        out = tx.loc[outlier_mask].copy()
        out["failure_reason"] = "volume > P99.9 per SKU (suspected decimal error)"
        _accumulate(rejects, out, "vol_p999")
        tx = tx.loc[~outlier_mask].copy()
    counts["after_vol_p999"] = len(tx)

    # ghost entries (bill==0 vs vol>0) already filtered above; keep counter for the report
    counts["ghost_entries_caught_above"] = counts["after_bill_pos"] - counts["after_vol_pos"]

    tx, rej = check_price_consistency(tx, "Volume_Liters", "Total_Bill_Value", "SKU_ID")
    _accumulate(rejects, rej, "price_outlier"); counts["after_price"] = len(tx)

    # flag outlets served by multiple distributors — keep them, just count
    multi_dist = tx.groupby("Outlet_ID")["Distributor_ID"].nunique()
    multi_dist_outlets = set(multi_dist[multi_dist > 1].index)
    if multi_dist_outlets:
        log.warning(
            "  %d outlets served by multiple distributors (flag, not drop).",
            len(multi_dist_outlets),
        )
    counts["outlets_multi_distributor"] = len(multi_dist_outlets)

    all_rejects = (
        pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=["failure_reason", "stage"])
    )
    counts["final_passed"] = len(tx)
    counts["total_rejected"] = len(all_rejects)
    return tx, all_rejects, dict(counts)


def clean_outlet_master(om: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Soft-impute, don't reject — submission needs all 20k outlets.
    # Touched rows are mirrored to the audit trail with their reason.
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["input"] = len(om)
    om = om.copy()
    audit_rows: list[pd.DataFrame] = []

    def _audit(mask: pd.Series, reason: str) -> None:
        if mask.any():
            r = om.loc[mask].copy()
            r["failure_reason"] = reason
            audit_rows.append(r)

    # Outlet_Size
    raw_size = om["Outlet_Size"].copy()
    om["Outlet_Size"] = om["Outlet_Size"].astype(str).str.strip().str.title()
    canonical_sizes = ["Small", "Medium", "Large", "Extra Large"]
    # astype(str) turns NaN into the literal "Nan" — restore it
    om.loc[om["Outlet_Size"] == "Nan", "Outlet_Size"] = pd.NA
    case_fixed = (raw_size != om["Outlet_Size"]) & om["Outlet_Size"].notna()
    _audit(case_fixed, "outlet_size_case_normalized")
    counts["outlet_size_case_normalized"] = int(case_fixed.sum())

    invalid_size = ~om["Outlet_Size"].isin(canonical_sizes) & om["Outlet_Size"].notna()
    _audit(invalid_size, "outlet_size_invalid_value_imputed_Medium")
    om.loc[invalid_size, "Outlet_Size"] = "Medium"
    counts["outlet_size_invalid_imputed"] = int(invalid_size.sum())

    null_size = om["Outlet_Size"].isna()
    _audit(null_size, "outlet_size_null_imputed_Medium")
    om.loc[null_size, "Outlet_Size"] = "Medium"
    counts["outlet_size_null_imputed"] = int(null_size.sum())

    # Outlet_Type
    raw_type = om["Outlet_Type"].copy()
    om["Outlet_Type"] = om["Outlet_Type"].astype(str).str.strip()
    typos = {
        "Grocry": "Grocery", "Grocries": "Grocery",
        "Bakry": "Bakery",
        "Pharm": "Pharmacy",
    }
    type_typo_fixed = om["Outlet_Type"].isin(typos)
    _audit(type_typo_fixed, "outlet_type_typo_fixed")
    om["Outlet_Type"] = om["Outlet_Type"].replace(typos)
    whitespace_fixed = (raw_type.astype(str) != om["Outlet_Type"]) & ~type_typo_fixed
    _audit(whitespace_fixed, "outlet_type_whitespace_stripped")
    counts["outlet_type_typos_fixed"] = int(type_typo_fixed.sum())
    counts["outlet_type_whitespace_stripped"] = int(whitespace_fixed.sum())

    # Cooler_Count
    om["Cooler_Count"] = pd.to_numeric(om["Cooler_Count"], errors="coerce")
    null_cooler = om["Cooler_Count"].isna()
    _audit(null_cooler, "cooler_count_null_imputed_1")
    om.loc[null_cooler, "Cooler_Count"] = 1
    neg_cooler = om["Cooler_Count"] < 0
    _audit(neg_cooler, "cooler_count_negative_clamped_0")
    om.loc[neg_cooler, "Cooler_Count"] = 0
    om["Cooler_Count"] = om["Cooler_Count"].astype(int)
    counts["cooler_count_null_imputed"] = int(null_cooler.sum())
    counts["cooler_count_negative_clamped"] = int(neg_cooler.sum())

    # null Outlet_ID is unrecoverable — hard reject
    null_id = om["Outlet_ID"].isna()
    if null_id.any():
        r = om.loc[null_id].copy()
        r["failure_reason"] = "outlet_id_null_unrecoverable"
        audit_rows.append(r)
        om = om.loc[~null_id].copy()
    counts["outlet_id_null_rejected"] = int(null_id.sum())

    type_dupes = om.groupby("Outlet_ID")["Outlet_Type"].nunique()
    conflicting = set(type_dupes[type_dupes > 1].index)
    counts["outlets_conflicting_type"] = len(conflicting)

    dup_mask = om.duplicated(subset=["Outlet_ID"], keep="first")
    if dup_mask.any():
        r = om.loc[dup_mask].copy()
        r["failure_reason"] = "duplicate_outlet_id_kept_first"
        audit_rows.append(r)
        om = om.loc[~dup_mask].copy()
    counts["duplicate_outlet_id_dropped"] = int(dup_mask.sum())

    # master-data decay flag — count only
    decay_mask = (om["Cooler_Count"] == 0) & om["Outlet_Size"].isin(["Large", "Extra Large"])
    counts["master_data_decay_flag"] = int(decay_mask.sum())
    if decay_mask.any():
        log.warning("  %d outlets: Cooler_Count==0 but Outlet_Size Large/XL (master-data decay).",
                    int(decay_mask.sum()))

    audit_df = (
        pd.concat(audit_rows, ignore_index=True)
        if audit_rows else pd.DataFrame(columns=list(om.columns) + ["failure_reason"])
    )
    counts["final_passed"] = len(om)
    counts["audit_trail_rows"] = len(audit_df)
    return om, audit_df, dict(counts)


def clean_coordinates(coords: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rejects: list[pd.DataFrame] = []
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["input"] = len(coords)

    coords, rej = check_nulls(coords, ["Outlet_ID"])
    _accumulate(rejects, rej, "nulls_outlet_id"); counts["after_id_nulls"] = len(coords)

    # missing lat/lon: flag only, impute downstream
    missing_coords = coords[["Latitude", "Longitude"]].isna().any(axis=1)
    counts["missing_coords_flag"] = int(missing_coords.sum())

    has_coords = coords.dropna(subset=["Latitude", "Longitude"])
    _, geo_rej = check_geo_bounds(has_coords, "Latitude", "Longitude")
    _accumulate(rejects, geo_rej, "outside_sri_lanka")
    counts["outside_sri_lanka"] = len(geo_rej)

    coords, rej = check_duplicates(coords, ["Outlet_ID"])
    _accumulate(rejects, rej, "duplicate_outlet_id"); counts["after_dedup"] = len(coords)

    all_rejects = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=["failure_reason"])
    counts["final_passed"] = len(coords)
    counts["total_rejected"] = len(all_rejects)
    return coords, all_rejects, dict(counts)


def clean_seasonality(seas: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rejects: list[pd.DataFrame] = []
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["input"] = len(seas)

    seas, rej = check_nulls(seas, ["Distributor_ID", "Year", "Month", "Seasonality_Index"])
    _accumulate(rejects, rej, "nulls"); counts["after_nulls"] = len(seas)

    seas, rej = check_referential_integrity(seas, "Distributor_ID", VALID_DISTRIBUTORS, "valid_distributors")
    _accumulate(rejects, rej, "distributor_invalid"); counts["after_dist_ref"] = len(seas)

    seas, rej = check_format(seas, "Seasonality_Index",
                             allowed_values=["Favorable", "Moderate", "Un-Favorable"])
    _accumulate(rejects, rej, "seasonality_invalid"); counts["after_seas_fmt"] = len(seas)

    seas, rej = check_duplicates(seas, ["Distributor_ID", "Year", "Month"])
    _accumulate(rejects, rej, "duplicate_key"); counts["after_dedup"] = len(seas)

    all_rejects = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=["failure_reason"])
    counts["final_passed"] = len(seas)
    counts["total_rejected"] = len(all_rejects)
    return seas, all_rejects, dict(counts)


def clean_holidays(h: pd.DataFrame, log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rejects: list[pd.DataFrame] = []
    counts: "OrderedDict[str, int]" = OrderedDict()
    counts["input"] = len(h)

    h, rej = check_nulls(h, ["Date", "Holiday_Name", "Holiday_Type"])
    _accumulate(rejects, rej, "nulls"); counts["after_nulls"] = len(h)

    h, rej = check_format(h, "Date", dtype="datetime")
    _accumulate(rejects, rej, "date_format"); counts["after_date_fmt"] = len(h)

    h = h.copy()
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce")

    # dedup on the full triple — same holiday legitimately appears under
    # multiple Holiday_Type labels (Public / Bank / Mercantile / Poya Day)
    h, rej = check_duplicates(h, ["Date", "Holiday_Name", "Holiday_Type"])
    _accumulate(rejects, rej, "duplicate_key"); counts["after_dedup"] = len(h)

    all_rejects = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(columns=["failure_reason"])
    counts["final_passed"] = len(h)
    counts["total_rejected"] = len(all_rejects)
    return h, all_rejects, dict(counts)


def build_monthly_outlet(tx: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """SKU-level monthly transactions → one row per (Outlet_ID, Year, Month)."""
    log.info("Building monthly_outlet aggregation...")
    agg = tx.groupby(["Outlet_ID", "Year", "Month"]).agg(
        Volume_Liters=("Volume_Liters", "sum"),
        Total_Bill_Value=("Total_Bill_Value", "sum"),
        n_unique_skus=("SKU_ID", "nunique"),
        n_transactions=("SKU_ID", "size"),
        Distributor_ID=("Distributor_ID", lambda s: s.mode().iloc[0] if len(s) else None),
    ).reset_index()
    log.info("  monthly_outlet shape: %s", agg.shape)
    return agg


def main() -> None:
    log = _setup_logging()
    log.info("Silver clean start.")
    log.info("NOTE: transactions_history_final.csv lacks Product_Name column "
             "(documented schema deviation; SKU_ID alone identifies the product).")

    # Load bronze
    tx = pd.read_csv(BRONZE / "transactions_history_final.csv")
    om = pd.read_csv(BRONZE / "outlet_master.csv")
    coords = pd.read_csv(BRONZE / "outlet_coordinates.csv")
    seas = pd.read_csv(BRONZE / "distributor_seasonality_details.csv")
    hol = pd.read_csv(BRONZE / "holiday_list.csv")

    # master first — its IDs are the reference set for transactions
    log.info("Cleaning outlet_master...")
    om_clean, om_rej, om_counts = clean_outlet_master(om, log)
    om_rej.to_csv(REJECTS / "outlet_master_rejected.csv", index=False)
    log.info("  outlet_master: %s", om_counts)

    outlet_ids = set(om_clean["Outlet_ID"])

    log.info("Cleaning outlet_coordinates...")
    coords_clean, coords_rej, coords_counts = clean_coordinates(coords, log)
    coords_rej.to_csv(REJECTS / "outlet_coordinates_rejected.csv", index=False)
    log.info("  outlet_coordinates: %s", coords_counts)

    log.info("Cleaning distributor_seasonality...")
    seas_clean, seas_rej, seas_counts = clean_seasonality(seas, log)
    seas_rej.to_csv(REJECTS / "distributor_seasonality_rejected.csv", index=False)
    log.info("  distributor_seasonality: %s", seas_counts)

    log.info("Cleaning holiday_list...")
    hol_clean, hol_rej, hol_counts = clean_holidays(hol, log)
    hol_rej.to_csv(REJECTS / "holiday_list_rejected.csv", index=False)
    log.info("  holiday_list: %s", hol_counts)

    log.info("Cleaning transactions (the heavy one)...")
    tx_clean, tx_rej, tx_counts = clean_transactions(tx, outlet_ids, log)
    tx_rej.to_csv(REJECTS / "transactions_rejected.csv", index=False)
    log.info("  transactions: %s", tx_counts)

    active_outlets = set(tx_clean["Outlet_ID"])
    ghost_outlets = outlet_ids - active_outlets
    log.warning("  GHOST OUTLETS (in master, zero transactions): %d", len(ghost_outlets))

    monthly = build_monthly_outlet(tx_clean, log)

    log.info("Writing silver artifacts...")
    tx_clean.to_parquet(SILVER / "transactions_clean.parquet", index=False)
    om_clean.to_parquet(SILVER / "outlet_master_clean.parquet", index=False)
    coords_clean.to_parquet(SILVER / "outlet_coordinates_clean.parquet", index=False)
    seas_clean.to_parquet(SILVER / "distributor_seasonality_clean.parquet", index=False)
    hol_clean.to_parquet(SILVER / "holiday_list_clean.parquet", index=False)
    monthly.to_parquet(SILVER / "monthly_outlet.parquet", index=False)

    summary_rows = []
    for name, counts in [
        ("transactions", tx_counts),
        ("outlet_master", om_counts),
        ("outlet_coordinates", coords_counts),
        ("distributor_seasonality", seas_counts),
        ("holiday_list", hol_counts),
    ]:
        for k, v in counts.items():
            summary_rows.append({"dataset": name, "metric": k, "value": v})
    summary_rows.append({"dataset": "transactions", "metric": "ghost_outlets", "value": len(ghost_outlets)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SILVER / "_summary.csv", index=False)

    log.info("=" * 60)
    log.info("SILVER SUMMARY")
    log.info("=" * 60)
    log.info("\n%s", summary.to_string(index=False))
    log.info("Silver clean done.")


if __name__ == "__main__":
    main()
