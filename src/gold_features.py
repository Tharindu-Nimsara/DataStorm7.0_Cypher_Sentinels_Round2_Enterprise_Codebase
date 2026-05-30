# Gold: one row per outlet, ~77 features.
# Buckets: structural, transactional (incl. censoring + plateau), spatial (POI),
# temporal (Jan-2026 seasonality + holidays), peer (cluster percentiles).
# Outlets without a POI cache get cluster-median imputation, with poi_imputed=1.

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
POI_CACHE = ROOT / "data" / "external" / "poi_cache"
LOG_DIR = ROOT / "logs"
GOLD.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

POI_TYPES = ["school", "university", "bus_station", "hospital",
             "place_of_worship", "marketplace", "restaurant", "tourism"]
RADII = [500, 1000, 2000]

# Seasonality multipliers — empirically derived from silver monthly_outlet:
# the mean log-volume difference between Favorable / Moderate / Un-Favorable
# months across 2023-2025 implied multipliers of 1.44 / 1.00 / 0.61. We use
# slightly conservative values (cap the upside at +30%) because Jan 2026
# tagging is unknown and we don't want to overshoot on extrapolation.
SEASONALITY_MULT = {"Favorable": 1.30, "Moderate": 1.00, "Un-Favorable": 0.70}

# POI weighted-density score: schools×2, bus_stations×3, hospitals×2, others×1
POI_WEIGHTS = {
    "school": 2, "university": 1, "bus_station": 3, "hospital": 2,
    "place_of_worship": 1, "marketplace": 1, "restaurant": 1, "tourism": 1,
}

PROVINCE_FROM_PREFIX = {
    "DIST_W_": "Western", "DIST_C_": "Central",
    "DIST_NW_": "North-Western", "DIST_S_": "Southern",
}


def setup_logging() -> logging.Logger:
    log = logging.getLogger("gold_features")
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
    fh = logging.FileHandler(LOG_DIR / "gold_features.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    return log


def build_structural(master: pd.DataFrame, tx_stats: pd.DataFrame) -> pd.DataFrame:
    out = master[["Outlet_ID", "Cooler_Count"]].copy()

    size_dummies = pd.get_dummies(master["Outlet_Size"], prefix="size").astype(int)
    type_dummies = pd.get_dummies(master["Outlet_Type"], prefix="type").astype(int)
    out = pd.concat([out, size_dummies, type_dummies], axis=1)

    size_order = {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4}
    out["size_ordinal"] = master["Outlet_Size"].map(size_order)

    out = out.merge(tx_stats[["Outlet_ID", "vol_mean"]], on="Outlet_ID", how="left")
    out["volume_per_cooler"] = out["vol_mean"] / out["Cooler_Count"].clip(lower=1)
    out["cooler_per_volume"] = out["Cooler_Count"] / out["vol_mean"].clip(lower=1)
    out = out.drop(columns=["vol_mean"])
    return out


def build_spatial(coords: pd.DataFrame, master: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """POI counts per radius/type, density score, urban flag.

    Outlets without an Overpass cache file get NaN here; we impute below
    by (Outlet_Type + Province) median, with a flag column.
    """
    cached_ids = {p.stem for p in POI_CACHE.glob("*.json")}
    log.info("  POI cache files available: %d / %d outlets",
             len(cached_ids), len(coords))

    rows = []
    for outlet_id in master["Outlet_ID"]:
        if outlet_id not in cached_ids:
            rows.append({"Outlet_ID": outlet_id, "poi_imputed": 1})
            continue
        try:
            data = json.loads((POI_CACHE / f"{outlet_id}.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            rows.append({"Outlet_ID": outlet_id, "poi_imputed": 1})
            continue
        counts = data.get("counts", {})
        row = {"Outlet_ID": outlet_id, "poi_imputed": 0}
        weighted = 0
        for poi in POI_TYPES:
            for r in RADII:
                v = counts.get(poi, {}).get(str(r), 0)
                row[f"poi_{poi}_{r}m"] = v
                if r == 1000:
                    weighted += v * POI_WEIGHTS[poi]
        row["poi_density_weighted_1km"] = weighted
        rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith("poi_") and c != "poi_imputed"]
    return df, feature_cols


def add_province(features: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    # Province from each outlet's most-frequent Distributor_ID prefix
    primary_dist = (
        monthly.groupby("Outlet_ID")["Distributor_ID"]
        .agg(lambda s: s.mode().iloc[0] if len(s) else None)
        .rename("primary_distributor")
        .reset_index()
    )

    def to_province(d):
        if pd.isna(d):
            return "Unknown"
        for prefix, prov in PROVINCE_FROM_PREFIX.items():
            if d.startswith(prefix):
                return prov
        return "Unknown"

    primary_dist["Province"] = primary_dist["primary_distributor"].map(to_province)
    return features.merge(primary_dist, on="Outlet_ID", how="left")


def impute_spatial(features: pd.DataFrame, poi_cols: list[str], log: logging.Logger) -> pd.DataFrame:
    # fill missing POI features with (Outlet_Type, Province) cluster median;
    # fall back to overall median for clusters with no cached observations
    needs_imputation = features["poi_imputed"] == 1
    n_needs = int(needs_imputation.sum())
    log.info("  Imputing POI features for %d outlets (no cache yet).", n_needs)

    cluster_medians = (
        features.loc[features["poi_imputed"] == 0]
        .groupby(["Outlet_Type", "Province"])[poi_cols]
        .median()
    )
    overall_median = features.loc[features["poi_imputed"] == 0, poi_cols].median()

    for idx in features.index[needs_imputation]:
        key = (features.at[idx, "Outlet_Type"], features.at[idx, "Province"])
        if key in cluster_medians.index:
            vals = cluster_medians.loc[key]
        else:
            vals = overall_median
        for c in poi_cols:
            features.at[idx, c] = vals[c] if not pd.isna(vals[c]) else 0
    features[poi_cols] = features[poi_cols].fillna(0)
    return features


def build_temporal(seas: pd.DataFrame, hol: pd.DataFrame, master: pd.DataFrame,
                   monthly: pd.DataFrame) -> pd.DataFrame:
    # Jan-2026 seasonality multiplier, holiday counts, working days
    primary = (
        monthly.groupby("Outlet_ID")["Distributor_ID"]
        .agg(lambda s: s.mode().iloc[0] if len(s) else None)
        .rename("primary_distributor")
        .reset_index()
    )

    # prefer Jan 2026 label if present, else latest available Jan (2025)
    seas_jan = seas[seas["Month"] == 1].copy()
    pick = (
        seas_jan.sort_values("Year", ascending=False)
        .drop_duplicates("Distributor_ID")[["Distributor_ID", "Seasonality_Index"]]
        .rename(columns={"Seasonality_Index": "seasonality_jan_label"})
    )
    out = primary.merge(pick, left_on="primary_distributor", right_on="Distributor_ID", how="left")
    out["seasonality_jan_label"] = out["seasonality_jan_label"].fillna("Moderate")
    out["seasonality_jan_mult"] = out["seasonality_jan_label"].map(SEASONALITY_MULT).fillna(1.0)

    h = hol.copy()
    h["Date"] = pd.to_datetime(h["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    jan26 = h[(h["Date"].dt.year == 2026) & (h["Date"].dt.month == 1)]
    holiday_counts = jan26["Holiday_Type"].value_counts().to_dict()
    n_public = holiday_counts.get("Public", 0)
    n_bank = holiday_counts.get("Bank", 0)
    n_poya = holiday_counts.get("Poya Day", 0)
    n_merc = holiday_counts.get("Mercantile", 0)
    n_total = len(jan26["Date"].unique())

    # working days = 31 - weekends - public holidays falling on weekdays
    days = pd.date_range("2026-01-01", "2026-01-31", freq="D")
    weekend_mask = days.dayofweek >= 5
    public_dates = set(jan26[jan26["Holiday_Type"] == "Public"]["Date"].dt.normalize())
    holiday_mask = pd.Series(days.normalize(), index=range(len(days))).isin(public_dates)
    n_working = int((~weekend_mask & ~holiday_mask.values).sum())

    out["jan26_n_public_holidays"] = n_public
    out["jan26_n_bank_holidays"] = n_bank
    out["jan26_n_poya"] = n_poya
    out["jan26_n_mercantile"] = n_merc
    out["jan26_n_total_holidays"] = n_total
    out["jan26_working_days"] = n_working

    return out[["Outlet_ID", "primary_distributor",
                "seasonality_jan_label", "seasonality_jan_mult",
                "jan26_n_public_holidays", "jan26_n_bank_holidays", "jan26_n_poya",
                "jan26_n_mercantile", "jan26_n_total_holidays", "jan26_working_days"]]


def build_transactional(monthly: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Per-outlet aggregates over the 2023-2025 monthly_outlet table."""
    log.info("  Computing per-outlet stats over %d outlet-months...", len(monthly))

    # Basic stats
    agg = monthly.groupby("Outlet_ID").agg(
        vol_mean=("Volume_Liters", "mean"),
        vol_median=("Volume_Liters", "median"),
        vol_max=("Volume_Liters", "max"),
        vol_std=("Volume_Liters", "std"),
        vol_p95=("Volume_Liters", lambda s: s.quantile(0.95)),
        vol_p99=("Volume_Liters", lambda s: s.quantile(0.99)),
        bill_mean=("Total_Bill_Value", "mean"),
        months_active=("Volume_Liters", "size"),
        sku_diversity=("n_unique_skus", "mean"),
    ).reset_index()

    agg["vol_std"] = agg["vol_std"].fillna(0)
    agg["vol_cv"] = agg["vol_std"] / agg["vol_mean"].clip(lower=1e-9)
    agg["price_per_liter_avg"] = agg["bill_mean"] / agg["vol_mean"].clip(lower=1e-9)

    m = monthly.sort_values(["Outlet_ID", "Year", "Month"]).copy()
    m["time_idx"] = m["Year"] * 12 + m["Month"]

    # recent_trend = mean(last 3 months) / mean(prior 3 months)
    def _trend(g):
        v = g["Volume_Liters"].values
        if len(v) < 6:
            return np.nan
        last3 = v[-3:].mean()
        prior3 = v[-6:-3].mean()
        return last3 / prior3 if prior3 > 0 else np.nan

    trend = m.groupby("Outlet_ID").apply(_trend, include_groups=False).rename("recent_trend").reset_index()

    yoy_2024 = monthly[monthly["Year"] == 2024].groupby("Outlet_ID")["Volume_Liters"].mean()
    yoy_2025 = monthly[monthly["Year"] == 2025].groupby("Outlet_ID")["Volume_Liters"].mean()
    yoy = (yoy_2025 / yoy_2024.replace(0, np.nan)).rename("yoy_growth").reset_index()

    # censoring threshold loosened from "within 2% of peak" (zero hits in EDA)
    # to "within 2% of P95" (graded signal across most outlets)
    m2 = m.merge(agg[["Outlet_ID", "vol_p95"]], on="Outlet_ID")
    m2["at_ceiling"] = m2["Volume_Liters"] >= m2["vol_p95"] * 0.98
    censoring = m2.groupby("Outlet_ID")["at_ceiling"].mean().rename("censoring_score").reset_index()

    # plateau: longest run of consecutive months where pairwise diff < 5% at high volume (>P75)
    def _plateau(g):
        if len(g) < 3:
            return 0
        v = g["Volume_Liters"].values
        p75 = np.percentile(v, 75)
        run, best = 0, 0
        for i in range(1, len(v)):
            if v[i] >= p75 and v[i-1] >= p75:
                rel_diff = abs(v[i] - v[i-1]) / max(v[i-1], 1e-9)
                if rel_diff < 0.05:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            else:
                run = 0
        return best

    plateau = m.groupby("Outlet_ID").apply(_plateau, include_groups=False).rename("plateau_score").reset_index()

    out = (
        agg.merge(trend, on="Outlet_ID", how="left")
        .merge(yoy, on="Outlet_ID", how="left")
        .merge(censoring, on="Outlet_ID", how="left")
        .merge(plateau, on="Outlet_ID", how="left")
    )
    out["recent_trend"] = out["recent_trend"].fillna(1.0)
    out["yoy_growth"] = out["yoy_growth"].fillna(1.0)
    out["plateau_norm"] = out["plateau_score"] / out["months_active"].clip(lower=1)
    return out


def build_peer(features: pd.DataFrame, monthly: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    # Cluster on (Outlet_Type, Outlet_Size, Province, POI_density_tier, urban_flag).
    f = features.copy()
    f["poi_density_tier"] = pd.qcut(
        f["poi_density_weighted_1km"].rank(method="first"),
        q=4, labels=["q1", "q2", "q3", "q4"]
    ).astype(str)
    f["urban_flag"] = (f["poi_density_weighted_1km"] > f["poi_density_weighted_1km"].median()).astype(int)

    f["peer_cluster"] = (
        f["Outlet_Type"].astype(str) + "|" +
        f["Outlet_Size"].astype(str) + "|" +
        f["Province"].astype(str) + "|" +
        f["poi_density_tier"].astype(str) + "|" +
        f["urban_flag"].astype(str)
    )

    cluster_size = f.groupby("peer_cluster")["Outlet_ID"].count().rename("peer_cluster_size")
    f = f.merge(cluster_size, on="peer_cluster", how="left")

    f["peer_pct_vol_mean"] = (
        f.groupby("peer_cluster")["vol_mean"]
        .rank(pct=True, method="average")
    )

    # peer-85 / peer-99 are taken over monthly volumes (not per-outlet means)
    # so Stage C sees the same distribution the model would
    outlet_to_cluster = f[["Outlet_ID", "peer_cluster"]]
    monthly_with_cluster = monthly.merge(outlet_to_cluster, on="Outlet_ID")
    cluster_p85 = monthly_with_cluster.groupby("peer_cluster")["Volume_Liters"].quantile(0.85).rename("peer_p85_monthly")
    cluster_p99 = monthly_with_cluster.groupby("peer_cluster")["Volume_Liters"].quantile(0.99).rename("peer_p99_monthly")
    f = f.merge(cluster_p85, on="peer_cluster", how="left")
    f = f.merge(cluster_p99, on="peer_cluster", how="left")

    log.info("  Peer clusters: %d (median size %d)",
             f["peer_cluster"].nunique(),
             int(f["peer_cluster_size"].median()))
    return f


def merge_poi_decay(features: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Merge Phase 1 distance-decay + competitor features (data/gold/poi_decay_features.parquet).

    poi_decay.py guarantees one row per master outlet with no nulls, so this is a plain
    left join. If the file is missing we warn and return features unchanged so this script
    still runs standalone (the model just won't see the decay features that run).
    """
    decay_path = GOLD / "poi_decay_features.parquet"
    if not decay_path.exists():
        log.warning("  poi_decay_features.parquet not found — run poi_decay.py first. "
                    "Continuing without distance-decay / competitor features.")
        return features

    decay = pd.read_parquet(decay_path)
    before = features.shape[1]
    merged = features.merge(decay, on="Outlet_ID", how="left")
    added = merged.shape[1] - before
    n_missing = int(merged["decayed_density_weighted"].isna().sum()) if "decayed_density_weighted" in merged else -1
    log.info("  Merged %d distance-decay/competitor features (%d outlets unmatched).",
             added, n_missing)
    # poi_decay covers every master outlet, but guard against a stale file
    if n_missing > 0:
        decay_cols = [c for c in decay.columns if c != "Outlet_ID"]
        merged[decay_cols] = merged[decay_cols].fillna(0)
    return merged


def add_physical_max(features: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Attach the Phase-2 physical cooler ceiling + headroom signals as features.

    Reuses censoring_model.build_physical_max so the constants and stale-cooler-data flag
    live in exactly one place. Pure function of Cooler_Count + vol_max + vol_mean, all of
    which are already on `features`, so no file dependency.
    """
    try:
        from censoring_model import build_physical_max
    except Exception as e:
        log.warning("  could not import censoring_model.build_physical_max (%s) — "
                    "skipping physical_max features.", e)
        return features

    phys = build_physical_max(features, log)
    merged = features.merge(phys, on="Outlet_ID", how="left")
    log.info("  Added %d physical-ceiling features (physical_max, headroom, breach flag).",
             phys.shape[1] - 1)
    return merged


def main() -> None:
    log = setup_logging()
    log.info("Gold features start.")

    log.info("Loading silver...")
    master = pd.read_parquet(SILVER / "outlet_master_clean.parquet")
    coords = pd.read_parquet(SILVER / "outlet_coordinates_clean.parquet")
    seas = pd.read_parquet(SILVER / "distributor_seasonality_clean.parquet")
    hol = pd.read_parquet(SILVER / "holiday_list_clean.parquet")
    monthly = pd.read_parquet(SILVER / "monthly_outlet.parquet")
    log.info("  master=%d coords=%d seas=%d hol=%d monthly=%s",
             len(master), len(coords), len(seas), len(hol), f"{len(monthly):,}")

    # transactional first because its vol_mean feeds the structural cooler-density features
    log.info("Building transactional features...")
    tx_feat = build_transactional(monthly, log)

    log.info("Building structural features...")
    struct_feat = build_structural(master, tx_feat)

    log.info("Building spatial features (POI)...")
    spatial_raw, poi_cols = build_spatial(coords, master, log)

    log.info("Building temporal features (Jan 2026)...")
    temporal_feat = build_temporal(seas, hol, master, monthly)

    log.info("Merging all feature buckets...")
    features = (
        master[["Outlet_ID", "Outlet_Size", "Outlet_Type"]]
        .merge(struct_feat, on="Outlet_ID", how="left")
        .merge(tx_feat, on="Outlet_ID", how="left")
        .merge(spatial_raw, on="Outlet_ID", how="left")
        .merge(temporal_feat, on="Outlet_ID", how="left")
    )

    # add_province re-attaches primary_distributor — drop the duplicate first
    if "primary_distributor" in features.columns:
        features = features.drop(columns=["primary_distributor"])
    features = add_province(features, monthly)

    features = impute_spatial(features, poi_cols, log)

    # peer features must run after spatial + province so the cluster key is complete
    features = build_peer(features, monthly, log)

    # Phase 1 spatial upgrade: merge the distance-decay + competitor features built by
    # poi_decay.py. Kept in a separate module/parquet (not inlined here) so the heavy
    # BallTree step can be rerun independently of the rest of the gold build. If the file
    # is absent (poi_decay.py not yet run) we warn and continue with the prelim features,
    # so gold_features stays runnable on its own.
    features = merge_poi_decay(features, log)

    # Phase 2 physical cooler ceiling: computed inline from Cooler_Count + vol_max (both
    # already in `features`) by reusing censoring_model's pure builder. Computing it here —
    # rather than reading censoring_model's parquet — avoids a circular dependency
    # (censoring_model reads gold), so physical_max is always present for the model + predict.
    features = add_physical_max(features, log)

    assert len(features) == len(master), \
        f"Row count drift: features={len(features)} master={len(master)}"
    assert features["Outlet_ID"].is_unique, "Duplicate Outlet_IDs in gold features"

    null_counts = features.isna().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols):
        log.warning("Columns with nulls after build:\n%s", null_cols.to_string())
    else:
        log.info("No nulls in gold features.")

    out_path = GOLD / "outlet_features.parquet"
    features.to_parquet(out_path, index=False)
    log.info("Wrote %s — shape=%s, cols=%d", out_path, features.shape, features.shape[1])

    log.info("=" * 60)
    log.info("FEATURE INVENTORY (%d columns)", features.shape[1])
    log.info("=" * 60)
    for c in features.columns:
        log.info("  %-40s %s", c, features[c].dtype)


if __name__ == "__main__":
    main()
